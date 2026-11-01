
from src.document_loader import load_document
from src.text_chunker import chunk_documents
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever


def main():

    documents = load_document("data/sample.md")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(
        [chunk.text for chunk in chunks]
    )

    store = VectorStore()

    store.add_chunks(chunks, embeddings)

    retriever = DocumentRetriever(
        embedding_generator=generator,
        vector_store=store,
        top_k=5,
        max_distance=0.60,
    )

    questions = [
    "What is artificial intelligence?",
    "Explain retrieval augmented generation.",
    "How does RAG use external documents?",
    "What is the purpose of embeddings?",
]

    results = [
        result
        for question in questions
        for result in retriever.retrieve(question)
    ]

    print("\nFiltered retrieval results:")
    print("=" * 40)

    for result in results:
        print("Source:", result["metadata"]["source"])
        print("Distance:", round(result["distance"], 4))
        print("Text:", result["text"])
        print("-" * 40)

    assert all(
        result["distance"] <= 0.60
        for result in results
    )

    print("Threshold test passed!")


if __name__ == "__main__":
    main()
