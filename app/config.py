import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "RAG Assistant API")
APP_VERSION = "1.0.0"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(BASE_DIR / "chroma_db")
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "document_collection"
)

UPLOAD_DIR = BASE_DIR / "data" / "uploads"

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)