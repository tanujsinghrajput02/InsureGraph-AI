
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents

from src.extraction.entity_extractor import (
    EntityExtractor,
    ExtractionResult
)

from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery


def build_graph():

    print("\nLoading insurance PDFs...")

    pdf_files = sorted(
        Path("data/documents").glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No insurance PDF files found."
        )

    documents = []

    for pdf_file in pdf_files:

        print("Loading:", pdf_file.name)

        documents.extend(
            load_pdf(str(pdf_file))
        )

    chunks = chunk_documents(documents)

    print("PDF files:", len(pdf_files))
    print("Pages:", len(documents))
    print("Chunks:", len(chunks))

    extractor = EntityExtractor()

    all_entities = {}
    all_relationships = {}

    for chunk in chunks:

        result = extractor.extract(
            chunk.page_content
        )

        for entity in result.entities:

            key = (
                entity.name,
                entity.entity_type
            )

            all_entities[key] = entity

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

    return GraphQuery(graph_builder)


def main():

    query = build_graph()

    print("\n========== ALL POLICIES ==========")

    policies = query.get_all_policy_details()

    for policy in policies:

        print("\nPolicy:", policy["policy"])
        print("Customer:", policy["customer"])
        print("Insurer:", policy["insurer"])
        print("Premium:", policy["premium"])
        print("Coverage:", policy["coverage"])

    print("\n========== RANKED BY COVERAGE ==========")

    coverage_ranking = query.rank_policies(
        field="coverage"
    )

    for position, policy in enumerate(
        coverage_ranking,
        start=1
    ):

        print(
            position,
            policy["policy"],
            policy["customer"],
            policy["coverage"],
            policy["numeric_value"]
        )

    print("\n========== RANKED BY PREMIUM ==========")

    premium_ranking = query.rank_policies(
        field="premium"
    )

    for position, policy in enumerate(
        premium_ranking,
        start=1
    ):

        print(
            position,
            policy["policy"],
            policy["customer"],
            policy["premium"],
            policy["numeric_value"]
        )

    print("\n========== TEST COMPLETED ==========")


if __name__ == "__main__":
    main()
