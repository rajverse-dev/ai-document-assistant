
from dataclasses import dataclass

from src.document_loader import DocumentRecord


@dataclass
class TextChunk:
    """A chunk of text with metadata from its source document."""

    text: str
    source: str
    file_type: str
    page_number: int | None
    chunk_index: int


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer a nearby whitespace boundary to avoid cutting a word.
        if end < text_length:
            boundary = text.rfind(" ", start, end)

            # Avoid producing very small chunks because of an early space.
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Ensure forward progress even when whitespace is unusual.
        next_start = max(start + 1, end - overlap)

        # Move the next start to a word boundary when possible.
        while next_start < end and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """Chunk loaded documents and preserve source metadata."""

    all_chunks = []

    for document in documents:
        pieces = split_text(
            document.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, piece in enumerate(pieces):
            all_chunks.append(
                TextChunk(
                    text=piece,
                    source=document.source,
                    file_type=document.file_type,
                    page_number=document.page_number,
                    chunk_index=index,
                )
            )

    return all_chunks
