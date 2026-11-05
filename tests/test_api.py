from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to RAG Assistant API"


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_empty_question():
    response = client.post(
        "/api/query",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_query_invalid_top_k():
    response = client.post(
        "/api/query",
        json={
            "question": "Explain RAG",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_success():
    mock_result = {
        "question": "Explain RAG",
        "answer": "RAG retrieves relevant information. [1]",
        "sources": [
            {
                "reference": 1,
                "source": "abc123:sample.md",
                "page_number": None,
                "chunk_index": 0,
                "similarity": 0.85,
            }
        ],
    }

    with patch(
        "app.api.routes.get_rag_services"
    ) as mock_services:

        mock_services.return_value = {
            "rag": type(
                "MockRAG",
                (),
                {"ask": lambda self, question, top_k: mock_result},
            )()
        }

        response = client.post(
            "/api/query",
            json={
                "question": "Explain RAG",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["answer"] == mock_result["answer"]
    assert data["retrieved_chunks"] == 1
    assert data["sources"][0]["filename"] == "sample.md"