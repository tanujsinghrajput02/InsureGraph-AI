from src.ingestion.pdf_loader import load_pdf


def test_pdf_loading():
    documents = load_pdf("data/documents/sample_policy.pdf")

    assert len(documents) > 0
    assert "POL-1001" in documents[0].page_content
    assert documents[0].metadata["page"] == 1

    print("\nPDF extraction successful!")
    print(documents[0].page_content)