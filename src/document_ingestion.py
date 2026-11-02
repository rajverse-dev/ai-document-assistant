
from pathlib import Path

from src.document_loader import load_document
from src.text_chunker import chunk_documents


def ingest_document(
    file_path,
    embedding_generator,
    vector_store,
):
    # 1. Load document
    documents = load_document(file_path)

    if not documents:
        return {
            "filename": Path(file_path).name,
            "chunks_added": 0,
            "message": "No readable text found.",
        }

    # 2. Split into chunks
    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    if not chunks:
        return {
            "filename": Path(file_path).name,
            "chunks_added": 0,
            "message": "No chunks generated.",
        }

    # 3. Generate embeddings
    embedded_chunks = embedding_generator.embed_chunks(chunks)

    embeddings = [
        item["embedding"]
        for item in embedded_chunks
    ]

    # 4. Store in ChromaDB
    vector_store.add_chunks(chunks, embeddings)

    return {
        "filename": Path(file_path).name,
        "chunks_added": len(chunks),
        "message": "Document successfully indexed.",
    }
