
from src.document_loader import load_document
from src.text_chunker import chunk_documents
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever


def main():

    # Step 1: Load documents.
    documents = load_document("data/sample.md")

    # Step 2: Generate chunks.
    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    # Step 3: Generate embeddings.
    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(
        [chunk.text for chunk in chunks]
    )

    # Step 4: Initialize ChromaDB.
    store = VectorStore()

    # Step 5: Store document chunks.
    store.add_chunks(chunks, embeddings)

    # Step 6: Initialize retriever.
    retriever = DocumentRetriever(
        embedding_generator=generator,
        vector_store=store,
        top_k=3,
        max_distance=None,
    )

    # Step 7: Ask a question.
    question = "What is Retrieval-Augmented Generation?"

    results = retriever.retrieve(question)

    print("\nUser Question:")
    print(question)

    print("\nRetrieved Documents:")
    print("=" * 50)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Source:", result["metadata"]["source"])

        print("Chunk:", result["metadata"]["chunk_index"])

        print("Distance:", round(result["distance"], 4))

        print("Similarity:", round(result["similarity"], 4))

        print("Content:", result["text"])

    assert len(results) > 0, "No documents retrieved"

    assert len(results) <= 3

    print("\nRetriever test passed!")


if __name__ == "__main__":
    main()
