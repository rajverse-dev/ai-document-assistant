import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ingestion.document_manager import list_documents
from app.api.schemas import QueryRequest
from app.config import (
    APP_NAME,
    APP_VERSION,
    UPLOAD_DIR,
    MAX_UPLOAD_MB,
)

from app.services import get_rag_services
from app.ingestion.indexing_service import index_document
from src.document_loader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(400, "Filename is required.")

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            "Supported formats: PDF, DOCX, TXT and MD."
        )

    content = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    if len(content) > max_bytes:
        raise HTTPException(
            413,
            f"Maximum upload size is {MAX_UPLOAD_MB} MB."
        )

    document_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{document_id}{extension}"

    try:
        file_path.write_bytes(content)

        services = get_rag_services()

        result = index_document(
            file_path=file_path,
            document_id=document_id,
            embedding_generator=services["embedding_generator"],
            vector_store=services["vector_store"],
        )

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "file_type": extension.lstrip("."),
            "chunks_added": result["chunks_added"],
            "message": result["message"],
        }

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            422,
            f"Document indexing failed: {str(error)}"
        ) from error

    finally:
        await file.close()

@router.get("/documents")
def get_documents():

    try:
        services = get_rag_services()

        documents = list_documents(
            services["vector_store"]
        )

        return {
            "status": "success",
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception as error:
        raise HTTPException(
            500,
            f"Unable to retrieve documents: {str(error)}"
        ) from error

@router.post("/query")
def query_documents(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        services = get_rag_services()

        result = services["rag"].ask(
            question=question,
            top_k=request.top_k,
        )

        sources = []

        for source in result["sources"]:
            original_source = source.get("source", "Unknown")

            # Stored source format: document_id:filename
            filename = original_source.split(":", 1)[-1]

            sources.append({
                "reference": source["reference"],
                "filename": filename,
                "page_number": source.get("page_number"),
                "chunk_index": source.get("chunk_index"),
                "similarity": source.get("similarity"),
            })

        return {
            "status": "success",
            "question": question,
            "answer": result["answer"],
            "sources": sources,
            "retrieved_chunks": len(sources),
        }

    except ConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(error)}",
        ) from error