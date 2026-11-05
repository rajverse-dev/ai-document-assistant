from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_upload_unsupported_file():
    response = client.post(
        "/api/upload",
        files={
            "file": (
                "example.exe",
                b"invalid content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400


def test_upload_empty_file():
    response = client.post(
        "/api/upload",
        files={
            "file": (
                "empty.md",
                b"",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 400


def test_upload_success():
    mock_result = {
        "document_id": "test123",
        "chunks_added": 2,
        "message": "Document successfully indexed.",
    }

    with patch(
        "app.api.routes.get_rag_services"
    ) as mock_services, patch(
        "app.api.routes.index_document",
        return_value=mock_result,
    ):

        mock_services.return_value = {
            "embedding_generator": object(),
            "vector_store": object(),
        }

        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "test.md",
                    b"This is a sample document for testing.",
                    "text/markdown",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["chunks_added"] == 2