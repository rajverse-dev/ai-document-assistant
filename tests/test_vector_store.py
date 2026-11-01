
from src.document_loader import load_document
from src.text_chunker import chunk_documents
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


def main():

    # Step 1: Load documents
    documents = load_document("data/sample.md")

    # Step 2: Generate chunks
    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    # Step 3: Generate embeddings
    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(
        [chunk.text for chunk in chunks]
    )

    # Step 4: Initialize vector database
    vector_store = VectorStore()

    # Step 5: Store embeddings
    vector_store.add_chunks(chunks, embeddings)

    print("\nTotal stored chunks:", vector_store.count())

    # Step 6: Search using a question
    question = "What is Retrieval Augmented Generation?"

    query_embedding = generator.generate_embeddings(
        [question]
    )[0]

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print("\nSearch Results")
    print("=" * 50)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("Source:", result["metadata"]["source"])
        print("Chunk:", result["metadata"]["chunk_index"])
        print("Distance:", round(result["distance"], 4))
        print("Text:", result["text"])


if __name__ == "__main__":
    main()
