from pathlib import Path

from src.document_loader import load_document
from src.text_chunker import chunk_documents


def index_document(
    file_path: str | Path,
    document_id: str,
    embedding_generator,
    vector_store,
):
    """Extract, chunk, embed and store an uploaded document."""

    # 1. Extract document text
    documents = load_document(file_path)

    if not documents:
        return {
            "document_id": document_id,
            "chunks_added": 0,
            "message": "No readable text found.",
        }

    # 2. Chunk document text
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    if not chunks:
        return {
            "document_id": document_id,
            "chunks_added": 0,
            "message": "No chunks generated.",
        }

    # 3. Add unique document identity to each chunk
    for chunk in chunks:
        chunk.source = f"{document_id}:{Path(file_path).name}"

    # 4. Generate embeddings
    embedded_chunks = embedding_generator.embed_chunks(chunks)

    embeddings = [
        item["embedding"]
        for item in embedded_chunks
    ]

    # 5. Store in ChromaDB
    vector_store.add_chunks(chunks, embeddings)

    return {
        "document_id": document_id,
        "chunks_added": len(chunks),
        "message": "Document successfully indexed.",
    }