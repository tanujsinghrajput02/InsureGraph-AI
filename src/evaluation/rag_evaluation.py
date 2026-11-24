
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.agent.model_manager import LazyModelLoader
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import VectorSearch
from src.extraction.entity_extractor import EntityExtractor, ExtractionResult
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery
from src.agent.rag_agent import RAGAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden_dataset" / "rag_golden.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "rag_evaluation_results.json"

POLICY_PATTERN = re.compile(r"\bPOL-\d+\b", re.IGNORECASE)


def load_golden_dataset():
    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_application():

    load_dotenv(PROJECT_ROOT / ".env")

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCUMENTS_DIR}"
        )

    documents = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(str(pdf_file)))

    chunks = chunk_documents(documents)

    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    embedding_model = EmbeddingModel()

    vector_search = VectorSearch(embedding_model)
    vector_search.index_documents(chunks)

    extractor = EntityExtractor()

    all_entities = {}
    all_relationships = {}

    for chunk in chunks:

        extraction_result = extractor.extract(chunk.page_content)

        for entity in extraction_result.entities:
            key = (entity.name, entity.entity_type)
            all_entities[key] = entity

        for relationship in extraction_result.relationships:
            key = (
                relationship.source,
                relationship.relation,
                relationship.target
            )
            all_relationships[key] = relationship

    extraction = ExtractionResult(
        entities=list(all_entities.values()),
        relationships=list(all_relationships.values())
    )

    graph_builder = KnowledgeGraphBuilder()
    graph_builder.build(extraction)

    graph_query = GraphQuery(graph_builder)

    model_name = os.getenv(
        "HF_LLM_MODEL",
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    model_loader = LazyModelLoader(model_name)

    agent = RAGAgent(vector_search, graph_query)

    print("Lazy model loading configured.")
    print("The local LLM will load only if a question requires it.")

    return agent, model_loader, None


def contains_all_terms(text, required_terms):

    normalized_text = text.casefold()

    return all(
        term.casefold() in normalized_text
        for term in required_terms
    )


def resolve_expected_policy_id(agent, question):

    policy_match = POLICY_PATTERN.search(question)

    if policy_match:
        return policy_match.group(0).upper()

    question_lower = question.casefold()
    graph = agent.graph_query.graph

    for entity, attributes in graph.nodes(data=True):

        if attributes.get("entity_type") != "Customer":
            continue

        customer_name = str(entity)

        if customer_name.casefold() not in question_lower:
            continue

        relationships = agent.graph_query.get_connected_entities(
            customer_name
        )

        for relationship in relationships:

            if (
                relationship["relation"] == "OWNS"
                and relationship["source"].casefold()
                == customer_name.casefold()
            ):
                policy_id = relationship["target"]

                if POLICY_PATTERN.fullmatch(str(policy_id)):
                    return str(policy_id).upper()

    return None


def get_all_known_policy_ids(agent):

    return {
        str(entity).upper()
        for entity in agent.graph_query.graph.nodes
        if POLICY_PATTERN.fullmatch(str(entity))
    }


def is_aggregate_question(question):
    """
    Detect questions whose answer depends on multiple policy records.
    """

    question = question.casefold()

    aggregate_phrases = (
        "highest",
        "lowest",
        "rank all",
        "ranking",
        "compare all",
        "compare policies",
        "all policies",
        "list all customers",
        "list all insurers",
        "list all insurance providers",
        "all customers",
        "all providers",
        "all insurance providers"
    )

    return any(
        phrase in question
        for phrase in aggregate_phrases
    )


def get_expected_policy_ids(agent, test_case):
    """
    Resolve the complete policy scope for each test.

    Aggregate questions expect all known policies.
    Individual questions expect one policy.
    """

    question = test_case["question"]

    # Ranking, comparison and list-all questions must cover
    # the complete policy collection, even if the expected
    # answer terms mention only the winning policy.
    if is_aggregate_question(question):
        return get_all_known_policy_ids(agent)

    expected_terms = test_case["expected_answer_terms"]

    expected_ids = {
        policy_id.upper()
        for term in expected_terms
        for policy_id in POLICY_PATTERN.findall(term)
    }

    if expected_ids:
        return expected_ids

    single_policy_id = resolve_expected_policy_id(
        agent,
        question
    )

    if single_policy_id:
        return {single_policy_id}

    # Use the agent's existing multi-policy intent detection
    # as a secondary fallback.
    if agent.is_multi_policy_question(question):
        return get_all_known_policy_ids(agent)

    return set()


def group_documents_by_source(documents):

    grouped = {}

    for document in documents:

        metadata = document.get("metadata", {})
        source = metadata.get("source", "unknown")

        if source not in grouped:
            grouped[source] = []

        grouped[source].append(document["content"])

    return {
        source: "\n".join(contents)
        for source, contents in grouped.items()
    }


def extract_policy_ids(text):

    return {
        policy_id.upper()
        for policy_id in POLICY_PATTERN.findall(text)
    }


def evaluate_rag(agent, model_loader, tokenizer, test_cases):

    results = []

    for index, test_case in enumerate(test_cases, start=1):

        question = test_case["question"]

        expected_policy_ids = get_expected_policy_ids(
            agent,
            test_case
        )

        if not expected_policy_ids:
            raise ValueError(
                f"Could not resolve expected policy scope "
                f"for question: {question}"
            )

        print(f"\nTest {index}/{len(test_cases)}")
        print("Question:", question)

        if len(expected_policy_ids) > 1:
            print(
                "Expected policies:",
                ", ".join(sorted(expected_policy_ids))
            )
        else:
            print(
                "Expected policy:",
                next(iter(expected_policy_ids))
            )

        # Retrieve once for each question.
        retrieved = agent.retrieve(question)

        retrieved_documents = retrieved["documents"]

        grouped_sources = group_documents_by_source(
            retrieved_documents
        )

        source_policy_ids = {}

        for source, content in grouped_sources.items():
            source_policy_ids[source] = extract_policy_ids(content)

        retrieved_policy_ids = set()

        for policy_ids in source_policy_ids.values():
            retrieved_policy_ids.update(policy_ids)

        # Exact scope check:
        # single-policy questions must retrieve one policy;
        # aggregate questions must retrieve all expected policies.
        policy_retrieval_correct = (
            retrieved_policy_ids == expected_policy_ids
        )

        # Collect evidence from all expected policy documents.
        expected_policy_contents = [
            content
            for content in grouped_sources.values()
            if extract_policy_ids(content) & expected_policy_ids
        ]

        expected_policy_content = "\n".join(
            expected_policy_contents
        )

        # Reuse retrieved context. No second retrieval.
        response = agent.answer_from_context(
            question,
            retrieved,
            model_loader,
            tokenizer
        )

        answer = response["answer"]

        answer_correct = contains_all_terms(
            answer,
            test_case["expected_answer_terms"]
        )

        evidence_grounded = (
            bool(expected_policy_content)
            and contains_all_terms(
                expected_policy_content,
                test_case["required_evidence_terms"]
            )
        )

        source_grounded_answer = (
            answer_correct
            and policy_retrieval_correct
            and evidence_grounded
        )

        result = {
            "question": question,
            "expected_policy_ids": sorted(expected_policy_ids),
            "answer": answer,
            "answer_source": response.get("answer_source"),
            "answer_correct": answer_correct,
            "policy_retrieval_correct": policy_retrieval_correct,
            "evidence_grounded": evidence_grounded,
            "source_grounded_answer": source_grounded_answer,
            "retrieved_policy_ids": sorted(retrieved_policy_ids),
            "sources": response["sources"],
            "graph_evidence": response.get("graph_evidence", [])
        }

        results.append(result)

        print("Answer:", answer)
        print("Answer source:", response.get("answer_source"))
        print("Answer correct:", answer_correct)
        print("Policy retrieval correct:", policy_retrieval_correct)
        print("Evidence grounded:", evidence_grounded)
        print("Source-grounded answer:", source_grounded_answer)

    total = len(results)

    def calculate_rate(field):

        if total == 0:
            return 0.0

        correct = sum(
            1 for item in results if item[field]
        )

        return round(correct / total, 4)

    return {
        "total_questions": total,
        "answer_accuracy": calculate_rate("answer_correct"),
        "policy_retrieval_accuracy": calculate_rate(
            "policy_retrieval_correct"
        ),
        "evidence_grounding_rate": calculate_rate(
            "evidence_grounded"
        ),
        "source_grounded_answer_accuracy": calculate_rate(
            "source_grounded_answer"
        ),
        "llm_loaded_during_evaluation": model_loader.loaded,
        "results": results
    }


def main():

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_FILE}"
        )

    test_cases = load_golden_dataset()

    print("Building InsureGraph AI...")

    agent, model_loader, tokenizer = build_application()

    print("\nStarting stronger RAG evaluation...")

    evaluation = evaluate_rag(
        agent,
        model_loader,
        tokenizer,
        test_cases
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            evaluation,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== RAG EVALUATION ==========")
    print("Total questions:", evaluation["total_questions"])
    print("Answer accuracy:", evaluation["answer_accuracy"])

    print(
        "Policy retrieval accuracy:",
        evaluation["policy_retrieval_accuracy"]
    )

    print(
        "Evidence grounding rate:",
        evaluation["evidence_grounding_rate"]
    )

    print(
        "Source-grounded answer accuracy:",
        evaluation["source_grounded_answer_accuracy"]
    )

    print(
        "LLM loaded during evaluation:",
        evaluation["llm_loaded_during_evaluation"]
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
