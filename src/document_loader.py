
from dataclasses import dataclass
from pathlib import Path

from docx import Document as WordDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}


@dataclass
class DocumentRecord:
    """A piece of extracted document text with source metadata."""

    text: str
    source: str
    file_type: str
    page_number: int | None = None


def _make_record(
    text: str,
    path: Path,
    page_number: int | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        text=text.strip(),
        source=path.name,
        file_type=path.suffix.lower().lstrip("."),
        page_number=page_number,
    )


def load_pdf(path: Path) -> list[DocumentRecord]:
    """Extract text separately from each PDF page."""
    reader = PdfReader(str(path))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            records.append(
                _make_record(text, path, page_number)
            )

    return records


def load_text_file(path: Path) -> list[DocumentRecord]:
    """Load TXT or Markdown text."""
    text = path.read_text(encoding="utf-8-sig")

    if not text.strip():
        return []

    return [_make_record(text, path)]


def load_docx(path: Path) -> list[DocumentRecord]:
    """Extract paragraphs from a Word document."""
    document = WordDocument(str(path))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [_make_record(text, path)]


def load_document(file_path: str | Path) -> list[DocumentRecord]:
    """Load a supported document into structured text records."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        return load_pdf(path)

    if extension in {".md", ".txt"}:
        return load_text_file(path)

    if extension == ".docx":
        return load_docx(path)

    raise ValueError(f"Unsupported file type: {extension}")
