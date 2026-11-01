
import numpy as np

from src.embeddings import EmbeddingGenerator


def main():
    generator = EmbeddingGenerator()

    sentences = [
        "Machine learning is a branch of artificial intelligence.",
        "Artificial intelligence includes machine learning.",
        "I enjoy playing cricket.",
    ]

    embeddings = generator.generate_embeddings(sentences)

    print("\nEmbedding results")
    print("-----------------")

    print("Number of sentences:", len(sentences))
    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", generator.dimension)

    # Calculate cosine similarities.
    similarity = embeddings @ embeddings.T

    print("\nSimilarity matrix:")
    print(np.round(similarity, 3))

    assert embeddings.shape == (3, 384)

    assert np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-5,
    )

    assert similarity[0][1] > similarity[0][2]

    print("\nEmbedding tests passed!")


if __name__ == "__main__":
    main()
