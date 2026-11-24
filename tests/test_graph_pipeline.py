
from pathlib import Path

from src.knowledge_graph.graph_pipeline import InsuranceGraphPipeline


def test_graph_pipeline():

    pdf_files = list(Path("data/documents").glob("*.pdf"))

    assert pdf_files, "No PDF files found."

    pipeline = InsuranceGraphPipeline()

    result = pipeline.run(str(pdf_files[0]))

    print("\nPIPELINE RESULTS")
    print("Source:", result["source"])
    print("Pages:", result["pages"])
    print("Chunks:", result["chunks"])
    print("Entities:", result["entities"])
    print("Relationships:", result["relationships"])

    print("\nGRAPH DATA")
    print(result["graph"])

    assert result["entities"] > 0
    assert len(result["graph"]["nodes"]) > 0
