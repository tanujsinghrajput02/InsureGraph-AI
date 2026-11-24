
from src.embeddings.embedding_model import EmbeddingModel


def test_embeddings():

    embedding_model = EmbeddingModel()

    texts = [
        "Rahul Sharma owns insurance policy POL-1001.",
        "The annual insurance premium is INR 15000.",
        "The policy coverage amount is INR 500000."
    ]

    vectors = embedding_model.embed_documents(texts)

    assert vectors.shape == (3, 384)

    similarity = embedding_model.similarity(
        vectors[0],
        vectors[1]
    )

    print("\nEmbedding shape:", vectors.shape)
    print("Similarity score:", similarity)
    print("Embedding test passed!")


