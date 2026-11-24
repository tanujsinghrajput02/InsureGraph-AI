
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return float(np.dot(vector1, vector2))
