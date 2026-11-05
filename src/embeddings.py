
from sentence_transformers import SentenceTransformer

from src.text_chunker import TextChunk


class EmbeddingGenerator:
    """Generate embeddings using a local Sentence Transformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.dimension = self.model.get_embedding_dimension()
        print(f"Loaded embedding model: {model_name}")
        print(f"Embedding dimension: {self.dimension}")

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
    ):
        """Convert a list of text passages into normalized vectors."""

        if not texts:
            raise ValueError("Input texts cannot be empty")

        if any(not text.strip() for text in texts):
            raise ValueError("Input texts cannot contain empty strings")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings

    def embed_chunks(self, chunks: list[TextChunk]):
        """Generate embeddings while retaining the original chunks."""

        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        embeddings = self.generate_embeddings(texts)

        return [
            {
                "chunk": chunk,
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
