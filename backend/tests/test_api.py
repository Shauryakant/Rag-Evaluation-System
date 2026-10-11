"""API integration tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "rag-eval-lab"
    assert data["status"] == "online"


def test_health_endpoint():
    """Test health check and corpus status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["indexed_documents"] >= 8
    assert data["total_chunks"] >= 20
    assert "bge-small" in data["embedder_model"]


def test_query_endpoint():
    """Test live ad-hoc query endpoint."""
    payload = {
        "query": "What is IndexFlatIP in FAISS?",
        "retriever": "dense",
        "top_k": 3,
        "use_reranker": False,
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["retrieved_chunks"]) == 3
    assert "answer" in data
    assert "latency" in data
    assert "retrieval_ms" in data["latency"]
    assert "total_ms" in data["latency"]
    # Critical design rule: No accuracy scores fabricated on query page
    assert "accuracy" not in data
    assert "precision" not in data


def test_evaluation_run_endpoint():
    """Test evaluation run endpoint over benchmark testset."""
    payload = {
        "retriever": "sparse",
        "top_k": 3,
        "use_reranker": False,
        "include_details": True,
    }
    response = client.post("/api/evaluation/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["retriever"] == "sparse"
    assert data["top_k"] == 3
    assert data["total_queries"] >= 40
    assert data["recall_at_k"] > 0.0
    assert data["mrr"] > 0.0
    assert "structurally capped" in data["note"]
    assert len(data["query_details"]) == data["total_queries"]


def test_benchmarks_matrix_endpoint():
    """Test retrieving precomputed benchmark matrix."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert data["total_configurations"] == 18
    assert len(data["rows"]) == 18
    assert data["dataset_size"] >= 40


def test_benchmarks_csv_download():
    """Test downloading benchmark results CSV."""
    response = client.get("/api/benchmarks/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "retriever,top_k,reranker" in response.text
