from pathlib import Path

from src.document_loader import load_document


def extract_document(file_path: str | Path):
    """Extract text and metadata from an uploaded document."""

    records = load_document(file_path)

    total_characters = sum(len(record.text) for record in records)

    return {
        "records": records,
        "record_count": len(records),
        "total_characters": total_characters,
    }