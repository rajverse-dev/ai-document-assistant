
from src.document_loader import load_document
from src.text_chunker import chunk_documents


def main():
    documents = load_document("data/sample.md")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40,
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "-" * 40)
        print(f"Source: {chunk.source}")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Text: {chunk.text[:150]}")


if __name__ == "__main__":
    main()
