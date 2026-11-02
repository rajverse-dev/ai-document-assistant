
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline


def test_rag_pipeline():
    # 1. Load the embedding model
    embedding_generator = EmbeddingGenerator()

    # 2. Connect to the existing ChromaDB database
    vector_store = VectorStore(
        persist_directory="chroma_db"
    )

    # 3. Create the retriever
    retriever = DocumentRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        top_k=3,
        max_distance=None,
    )

    # 4. Connect to the local Ollama model
  
    llm_client = OllamaClient(
        model="qwen2.5:1.5b"
    )

    # 5. Build the complete RAG pipeline
    rag = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    # 6. Ask a question
    result = rag.ask(
        "How to make cake?"
    )

    print("\nQUESTION:")
    print(result["question"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")
    for source in result["sources"]:
        print(source)

    assert result["answer"].strip()
    assert result["sources"]

    print("\nRAG pipeline test passed!")


if __name__ == "__main__":
    test_rag_pipeline()
