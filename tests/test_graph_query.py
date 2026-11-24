
from src.extraction.entity_extractor import (
    Entity,
    Relationship,
    ExtractionResult
)
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery


def test_graph_query():

    extraction = ExtractionResult(
        entities=[
            Entity(name="Customer A", entity_type="Customer", evidence="Customer A"),
            Entity(name="Policy A", entity_type="Policy", evidence="Policy A"),
            Entity(name="Insurer A", entity_type="Insurer", evidence="Insurer A")
        ],
        relationships=[
            Relationship(
                source="Customer A",
                relation="OWNS",
                target="Policy A",
                evidence="Customer A owns Policy A"
            ),
            Relationship(
                source="Policy A",
                relation="ISSUED_BY",
                target="Insurer A",
                evidence="Policy A issued by Insurer A"
            )
        ]
    )

    builder = KnowledgeGraphBuilder()
    builder.build(extraction)

    query = GraphQuery(builder)

    policies = query.get_customer_policies("Customer A")
    details = query.get_policy_details("Policy A")

    print("\nCustomer policies:", policies)
    print("\nPolicy details:", details)

    assert policies == ["Policy A"]
    assert len(details["relationships"]) == 1
