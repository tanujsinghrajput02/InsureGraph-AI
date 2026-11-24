
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

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


load_dotenv()


def main():

    pdf_files = list(Path("data/documents").glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF documents found.")

    print("\nLoading insurance documents...")

    documents = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(str(pdf_file)))

    chunks = chunk_documents(documents)

    print(f"Documents: {len(pdf_files)}")
    print(f"Chunks: {len(chunks)}")

    print("\nCreating embeddings...")

    embedding_model = EmbeddingModel()
    vector_search = VectorSearch(embedding_model)
    vector_search.index_documents(chunks)

    print("\nExtracting entities and relationships...")

    extractor = EntityExtractor()

    all_entities = {}
    all_relationships = {}

    for chunk in chunks:

        result = extractor.extract(chunk.page_content)

        for entity in result.entities:
            all_entities[(entity.name, entity.entity_type)] = entity

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

    print(f"Entities: {len(extraction.entities)}")
    print(f"Relationships: {len(extraction.relationships)}")

    print("\nLoading local language model...")

    model_name = os.getenv(
        "HF_LLM_MODEL",
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    agent = RAGAgent(vector_search, graph_query)

    print("\nInsureGraph AI is ready.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nAsk an insurance question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = agent.answer(question, model, tokenizer)

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")
            print(result["sources"])

            print("\nGRAPH EVIDENCE:")
            print(result["graph_evidence"])

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
