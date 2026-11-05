from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


class DocumentRetriever:
    """Retrieve relevant document chunks using semantic search."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k: int = 3,
        max_distance: float | None = 0.60,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance cannot be negative")

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_distance = max_distance

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict]:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        selected_top_k = (
            top_k if top_k is not None else self.top_k
        )

        if selected_top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = (
            self.embedding_generator.generate_embeddings(
                [question.strip()]
            )[0]
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=selected_top_k,
        )

        if self.max_distance is not None:
            results = [
                result
                for result in results
                if result["distance"] <= self.max_distance
            ]

        for result in results:
            result["similarity"] = 1.0 - result["distance"]

        return results