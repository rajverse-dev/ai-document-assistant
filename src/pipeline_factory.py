
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline


def create_rag_components():
    embedding_generator = EmbeddingGenerator()

    vector_store = VectorStore(
        persist_directory="chroma_db",
        collection_name="document_collection",
    )

    retriever = DocumentRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        top_k=3,
        max_distance=None,
    )

    llm_client = OllamaClient(
        model="qwen2.5:1.5b"
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
