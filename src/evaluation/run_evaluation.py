
import json
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.extraction.entity_extractor import EntityExtractor
from src.evaluation.metrics import evaluate_extraction


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden_dataset"
    / "extraction_golden.json"
)


def main():
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_FILE}")

    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        expected = json.load(file)

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCUMENTS_DIR}"
        )

    extractor = EntityExtractor()

    all_entities = {}
    all_relationships = {}

    for pdf_file in pdf_files:
        print(f"\nEvaluating: {pdf_file.name}")

        documents = load_pdf(str(pdf_file))
        chunks = chunk_documents(documents)

        for chunk in chunks:
            result = extractor.extract(chunk.page_content)

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

    predicted = ExtractionResult(
        entities=list(all_entities.values()),
        relationships=list(all_relationships.values())
    )

    metrics = evaluate_extraction(expected, predicted)

    print("\n========== ENTITY EVALUATION ==========")
    print(metrics["entities"])

    print("\n======= RELATIONSHIP EVALUATION =======")
    print(metrics["relationships"])

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "evaluation_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
