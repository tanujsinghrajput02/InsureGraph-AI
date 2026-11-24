
from src.extraction.entity_extractor import Entity, Relationship, ExtractionResult
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder


def test_graph_builder():

    extraction = ExtractionResult(
        entities=[
            Entity(
                name="Customer A",
                entity_type="Customer",
                evidence="Customer A"
            ),
            Entity(
                name="Policy A",
                entity_type="Policy",
                evidence="Policy A"
            )
        ],
        relationships=[
            Relationship(
                source="Customer A",
                relation="OWNS",
                target="Policy A",
                evidence="Customer A owns Policy A"
            )
        ]
    )

    builder = KnowledgeGraphBuilder()
    graph = builder.build(extraction)

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    print("\nGraph nodes:", builder.get_graph_data()["nodes"])
    print("\nGraph edges:", builder.get_graph_data()["edges"])
    print("\nKnowledge graph test passed!")
