
from src.document_loader import DocumentRecord
from src.text_chunker import chunk_documents, split_text


def test_split_text():
    text = "word " * 300

    chunks = split_text(
        text,
        chunk_size=200,
        overlap=40,
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 200 for chunk in chunks)

    print(f"Text splitting passed: {len(chunks)} chunks")


def test_metadata_preservation():
    document = DocumentRecord(
        text="Attendance is required. " * 50,
        source="college_rules.pdf",
        file_type="pdf",
        page_number=4,
    )

    chunks = chunk_documents(
        [document],
        chunk_size=200,
        overlap=40,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.source == "college_rules.pdf"
        assert chunk.file_type == "pdf"
        assert chunk.page_number == 4

    assert [c.chunk_index for c in chunks] == list(
        range(len(chunks))
    )

    print("Metadata preservation passed!")


if __name__ == "__main__":
    test_split_text()
    test_metadata_preservation()
