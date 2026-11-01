
from src.document_loader import load_document
from src.text_chunker import chunk_documents
from src.embeddings import EmbeddingGenerator


def main():
    # Step 1: Load document
    documents = load_document("data/sample.md")

    # Step 2: Generate chunks
    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    # Step 3: Generate embeddings
    generator = EmbeddingGenerator()

    results = generator.embed_chunks(chunks)

    print("\nComplete RAG preprocessing pipeline")
    print("------------------------------------")

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))
    print("Embeddings:", len(results))

    for item in results:
        chunk = item["chunk"]
        embedding = item["embedding"]

        print("\nSource:", chunk.source)
        print("Chunk index:", chunk.chunk_index)
        print("Embedding shape:", embedding.shape)
        print("First five values:", embedding[:5])


if __name__ == "__main__":
    main()
