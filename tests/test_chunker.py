
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents


def test_document_chunking():

    documents = load_pdf("data/documents/sample_policy.pdf")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=40
    )

    assert len(chunks) > 0

    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.metadata["source"] == "sample_policy.pdf"

    print(f"\nTotal chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.page_content)


