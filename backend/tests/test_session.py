"""Unit and integration tests for ephemeral document session management and API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
import pypdf

from backend.generation.groq_generator import GenerationResult, GroqGenerator
from backend.main import app
from backend.session.manager import SessionManager, get_session_manager


def _create_dummy_pdf_bytes(text: str = "This is a test PDF document for RAG evaluation.") -> bytes:
    """Create a valid in-memory single-page PDF byte stream."""
    writer = pypdf.PdfWriter()
    # Add a blank page with text annotation or create with canvas if reportlab exists
    # Simple valid PDF binary with pypdf:
    page = writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_session_manager_text_extraction():
    """Verify session manager extracts text from markdown and text files."""
    mgr = SessionManager()
    content = "# Introduction\n\nArtificial Intelligence and RAG systems are powerful tools for knowledge retrieval."
    text, pages = mgr.extract_text_from_bytes(content.encode("utf-8"), "test.md")
    assert "Artificial Intelligence" in text
    assert pages == 1


def test_session_creation_and_query():
    """Verify session creates isolated FAISS and BM25 indices and retrieves chunks."""
    mgr = SessionManager()
    content = (
        "# Quantum Computing\n\n"
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.\n\n"
        "# Classical vs Quantum\n\n"
        "Unlike classical computers that use bits, quantum computers use qubits which can exist in superposition."
    )
    session = mgr.create_session(
        file_bytes=content.encode("utf-8"),
        filename="quantum.md",
        chunk_size=150,
        chunk_overlap=20,
    )

    assert session.session_id is not None
    assert session.chunk_count >= 2
    assert session.filename == "quantum.md"

    # Test retrieval against session's isolated service
    retrieved, latency = session.service.retrieve("What are qubits?", retriever_type="sparse", top_k=2)
    assert len(retrieved) > 0
    assert any("qubits" in r.chunk.text for r in retrieved)

    # Test get_session and touch
    retrieved_session = mgr.get_session(session.session_id)
    assert retrieved_session is not None
    assert retrieved_session.session_id == session.session_id

    # Test delete session
    deleted = mgr.delete_session(session.session_id)
    assert deleted is True
    assert mgr.get_session(session.session_id) is None


def test_session_api_endpoints(monkeypatch):
    """Verify session API routes: upload, get detail, query, compare, delete."""
    # Mock generator to avoid real Groq network calls during test
    def mock_generate(self, query, retrieved_chunks, temperature=0.1, max_tokens=1024):
        return GenerationResult(
            answer="Mocked generation answer for test.",
            model="mock-model",
            generation_ms=12.5,
            prompt_tokens=40,
            completion_tokens=20,
            total_tokens=60,
        )

    monkeypatch.setattr(GroqGenerator, "generate", mock_generate)
    client = TestClient(app)

    # 1. Upload a text document
    sample_doc = (
        "# Machine Learning Architecture\n\n"
        "Machine learning models require robust feature pipelines and continuous evaluation.\n\n"
        "# Neural Networks\n\n"
        "Deep learning leverages multi-layer perceptrons and backpropagation to learn representations."
    )
    files = {"file": ("ml_arch.txt", sample_doc.encode("utf-8"), "text/plain")}
    data = {"chunk_size": "200", "chunk_overlap": "20"}

    res = client.post("/api/session/upload", files=files, data=data)
    assert res.status_code == 200, res.text
    session_data = res.json()
    session_id = session_data["session_id"]
    assert session_data["chunk_count"] >= 2
    assert len(session_data["sample_questions"]) > 0

    # 2. Get session details
    detail_res = client.get(f"/api/session/{session_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["session_id"] == session_id
    assert len(detail_res.json()["chunks"]) == session_data["chunk_count"]

    # 3. Query the session
    query_payload = {
        "query": "What does deep learning leverage?",
        "retriever": "sparse",
        "top_k": 2,
        "use_reranker": False,
    }
    query_res = client.post(f"/api/session/{session_id}/query", json=query_payload)
    assert query_res.status_code == 200
    query_json = query_res.json()
    assert len(query_json["retrieved_chunks"]) > 0
    assert query_json["latency"]["retrieval_ms"] >= 0

    # 4. Compare two configurations
    compare_payload = {
        "query": "How do neural networks learn representations?",
        "config_a": {
            "label": "Sparse BM25",
            "retriever": "sparse",
            "top_k": 2,
            "use_reranker": False,
        },
        "config_b": {
            "label": "Dense FAISS",
            "retriever": "dense",
            "top_k": 2,
            "use_reranker": False,
        },
    }
    compare_res = client.post(f"/api/session/{session_id}/compare", json=compare_payload)
    assert compare_res.status_code == 200
    comp_json = compare_res.json()
    assert "result_a" in comp_json
    assert "result_b" in comp_json
    assert "chunk_overlap_percentage" in comp_json
    assert "latency_difference_ms" in comp_json

    # 5. Delete session
    del_res = client.delete(f"/api/session/{session_id}")
    assert del_res.status_code == 200

    # 6. Verify deleted
    verify_res = client.get(f"/api/session/{session_id}")
    assert verify_res.status_code == 404
