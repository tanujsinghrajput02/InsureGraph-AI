
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.extraction.entity_extractor import EntityExtractor


def test_entity_extraction():

    pdf_files = list(Path("data/documents").glob("*.pdf"))

    assert pdf_files, "No PDF documents found."

    all_text = []

    for pdf_file in pdf_files:
        documents = load_pdf(str(pdf_file))

        all_text.extend(
            document.page_content
            for document in documents
        )

    document_text = "\n".join(all_text)

    extractor = EntityExtractor()
    result = extractor.extract(document_text)

    assert len(result.entities) > 0

    print("\nEXTRACTED ENTITIES")

    for entity in result.entities:
        print(entity.model_dump())

    print("\nEXTRACTED RELATIONSHIPS")

    for relationship in result.relationships:
        print(relationship.model_dump())


