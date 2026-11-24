
import numpy as np
from langchain_core.documents import Document

from src.embeddings.embedding_model import EmbeddingModel


class VectorSearch:

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.documents: list[Document] = []
        self.vectors: np.ndarray | None = None

    def index_documents(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        texts = [doc.page_content for doc in documents]

        self.vectors = self.embedding_model.embed_documents(texts)
        self.documents = documents

    def search(self, query: str, top_k: int = 3) -> list[dict]:

        if self.vectors is None or not self.documents:
            raise ValueError("Index documents before searching.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.embedding_model.embed_text(query)

        scores = np.dot(self.vectors, query_vector)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "content": self.documents[index].page_content,
                "metadata": self.documents[index].metadata,
                "score": float(scores[index])
            })

        return results
