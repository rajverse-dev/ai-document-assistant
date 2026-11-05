
import hashlib
from pathlib import Path

import chromadb

from src.text_chunker import TextChunk


class VectorStore:
    """Manage persistent document embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "document_collection",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _generate_id(chunk: TextChunk) -> str:
        """Generate a stable ID for each document chunk."""

        unique_text = (
            f"{chunk.source}|{chunk.file_type}|"
            f"{chunk.page_number}|{chunk.chunk_index}|{chunk.text}"
        )

        return hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

    def add_chunks(self, chunks, embeddings):
        """Store chunks, their embeddings, and metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match"
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for chunk, embedding in zip(chunks, embeddings):

            ids.append(self._generate_id(chunk))

            documents.append(chunk.text)

            metadata = {
                "source": chunk.source,
                "filename": chunk.source.split(":", 1)[-1],
                "document_id": chunk.source.split(":", 1)[0],
                 "file_type": chunk.file_type,
                "chunk_index": chunk.chunk_index,
}

            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number

            metadatas.append(metadata)

            vectors.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

        print(f"Successfully stored {len(ids)} chunks.")

    def search(self, query_embedding, top_k: int = 3):
        """Retrieve the most relevant document chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        documents = results.get("documents") or [[]]
        metadatas = results["metadatas"] or [[]]
        distances = results.get("distances") or [[]]

        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "text": documents[0][i],
                "metadata": metadatas[0][i],
                "distance": distances[0][i] if distances[0] else None,
            })

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self.collection.count()
