
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import VectorSearch

def test_vector_search():

    pdf_files = list(Path("data/documents").glob("*.pdf"))

    assert pdf_files, "No PDF documents found."

    documents = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(str(pdf_file)))

    chunks = chunk_documents(documents)

    embedding_model = EmbeddingModel()

    vector_search = VectorSearch(embedding_model)
    vector_search.index_documents(chunks)

    query = input("\nEnter your insurance question: ")

    results = vector_search.search(query, top_k=3)

    for result in results:
        print("\nRelevance score:", round(result["score"], 4))
        print("Source:", result["metadata"]["source"])
        print("Content:", result["content"])