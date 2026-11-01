
from pathlib import Path

from src.document_loader import load_document


def test_load_markdown():
    path = Path("data/sample.md")

    records = load_document(path)

    assert len(records) == 1
    assert "Retrieval-Augmented Generation" in records[0].text
    assert records[0].source == "sample.md"
    assert records[0].file_type == "md"
    assert records[0].page_number is None

    print("Markdown loader test passed!")


if __name__ == "__main__":
    test_load_markdown()
