
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.extraction.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder


class InsuranceGraphPipeline:

    def __init__(self):
        self.extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()

    def run(self, pdf_path: str):

        path = Path(pdf_path)

        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        documents = load_pdf(str(path))
        chunks = chunk_documents(documents)

        all_entities = {}
        all_relationships = {}

        for chunk in chunks:
            result = self.extractor.extract(chunk.page_content)

            for entity in result.entities:
                key = (entity.name, entity.entity_type)
                all_entities[key] = entity

            for relationship in result.relationships:
                key = (
                    relationship.source,
                    relationship.relation,
                    relationship.target
                )
                all_relationships[key] = relationship

        from src.extraction.entity_extractor import ExtractionResult

        extraction = ExtractionResult(
            entities=list(all_entities.values()),
            relationships=list(all_relationships.values())
        )

        graph = self.graph_builder.build(extraction)

        return {
            "source": path.name,
            "pages": len(documents),
            "chunks": len(chunks),
            "entities": len(extraction.entities),
            "relationships": len(extraction.relationships),
            "graph": self.graph_builder.get_graph_data()
        }
