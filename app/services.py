from functools import lru_cache

from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_rag_services():

    embedding_generator = EmbeddingGenerator(
        model_name=EMBEDDING_MODEL
    )

    vector_store = VectorStore(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    retriever = DocumentRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        top_k=5,
        max_distance=None,
    )

    llm_client = OllamaClient(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=180,
    )

    rag = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    return {
        "rag": rag,
        "embedding_generator": embedding_generator,
        "vector_store": vector_store,
    }