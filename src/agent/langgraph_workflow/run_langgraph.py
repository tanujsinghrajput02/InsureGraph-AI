
import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.agent.model_manager import LazyModelLoader
from src.agent.langgraph_workflow.workflow import (
    build_insurance_workflow
)
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import VectorSearch
from src.extraction.entity_extractor import (
    EntityExtractor,
    ExtractionResult
)
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery
from src.agent.rag_agent import RAGAgent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
METRICS_FILE = PROJECT_ROOT / "outputs" / "performance_metrics.json"


def build_application():

    load_dotenv(PROJECT_ROOT / ".env")

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No insurance PDFs found.")

    documents = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(str(pdf_file)))

    chunks = chunk_documents(documents)

    print("PDF files:", len(pdf_files))
    print("Pages:", len(documents))
    print("Chunks:", len(chunks))

    embedding_model = EmbeddingModel()

    vector_search = VectorSearch(embedding_model)
    vector_search.index_documents(chunks)

    extractor = EntityExtractor()

    all_entities = {}
    all_relationships = {}

    for chunk in chunks:

        result = extractor.extract(chunk.page_content)

        for entity in result.entities:
            all_entities[
                (entity.name, entity.entity_type)
            ] = entity

        for relationship in result.relationships:

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

    rag_agent = RAGAgent(
        vector_search,
        graph_query
    )

    app = build_insurance_workflow(
        rag_agent,
        model_loader,
        None
    )

    print("Local LLM loading is deferred until required.")

    return app


def save_performance_metrics(record):

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if METRICS_FILE.exists():

        try:
            with METRICS_FILE.open("r", encoding="utf-8") as file:
                records = json.load(file)

            if not isinstance(records, list):
                records = []

        except (json.JSONDecodeError, OSError):
            records = []

    else:
        records = []

    records.append(record)

    with METRICS_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )


def ask_question(app, question):

    start_time = time.perf_counter()

    result = app.invoke({
        "question": question
    })

    total_seconds = time.perf_counter() - start_time

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": result.get("answer", ""),
        "answer_source": result.get("answer_source"),
        "vector_search_calls": result.get(
            "vector_search_calls",
            0
        ),
        "retrieval_seconds": result.get(
            "retrieval_seconds",
            0.0
        ),
        "answer_generation_seconds": result.get(
            "answer_generation_seconds",
            0.0
        ),
        "total_response_seconds": round(total_seconds, 6),
        "recovery_attempts": result.get(
            "recovery_attempts",
            0
        ),
        "answer_valid": result.get("answer_valid"),
        "source_count": len(result.get("sources", []))
    }

    save_performance_metrics(record)

    print("\n========== ANSWER ==========")
    print(result.get("answer", "No answer generated."))

    print("\nAnswer source:")
    print(result.get("answer_source"))

    print("\nSources:")

    for source in result.get("sources", []):
        print(source)

    print("\n========== PERFORMANCE ==========")
    print("Vector search calls:", record["vector_search_calls"])
    print("Retrieval time:", record["retrieval_seconds"], "seconds")
    print(
        "Answer generation time:",
        record["answer_generation_seconds"],
        "seconds"
    )
    print(
        "Total response time:",
        record["total_response_seconds"],
        "seconds"
    )
    print("Recovery attempts:", record["recovery_attempts"])

    print("\nMetrics saved to:", METRICS_FILE)
    print("============================")


def main():

    print("\nBuilding InsureGraph AI LangGraph Agent...")

    app = build_application()

    print("\nLangGraph agent is ready.")
    print("\nType 'exit' to stop.")

    while True:

        question = input(
            "\nAsk insurance question: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Agent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            ask_question(app, question)

        except Exception as error:
            print("\nError while processing question:", error)


if __name__ == "__main__":
    main()
