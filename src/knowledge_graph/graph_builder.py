
import networkx as nx

from src.extraction.entity_extractor import ExtractionResult


class KnowledgeGraphBuilder:

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def build(self, extraction: ExtractionResult):

        self.graph.clear()

        for entity in extraction.entities:
            self.graph.add_node(
                entity.name,
                entity_type=entity.entity_type,
                evidence=entity.evidence
            )

        for relationship in extraction.relationships:

            if (
                relationship.source not in self.graph
                or relationship.target not in self.graph
            ):
                continue

            self.graph.add_edge(
                relationship.source,
                relationship.target,
                relation=relationship.relation,
                evidence=relationship.evidence
            )

        return self.graph

    def get_graph_data(self):

        return {
            "nodes": [
                {
                    "id": node,
                    **attributes
                }
                for node, attributes in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **attributes
                }
                for source, target, attributes
                in self.graph.edges(data=True)
            ]
        }

    def get_relationships(self, entity_name: str):

        if entity_name not in self.graph:
            return []

        return [
            {
                "source": source,
                "target": target,
                **attributes
            }
            for source, target, attributes
            in self.graph.edges(entity_name, data=True)
        ]
