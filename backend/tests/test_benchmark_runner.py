"""Unit tests for BenchmarkRunner execution and reporting."""

import pytest

from backend.chunking.splitter import Chunk
from backend.evaluation.benchmark_runner import BenchmarkRunner
from backend.evaluation.ground_truth import ResolvedTestCase
from backend.retrieval.service import RetrievalService


@pytest.fixture
def mock_benchmark_env():
    """Create a lightweight deterministic test environment for benchmark running."""
    c1 = Chunk(
        chunk_id="doc1_c0",
        doc_id="doc1.md",
        text="BM25 is a term frequency keyword ranking function.",
        start_char=0,
        end_char=52,
    )
    c2 = Chunk(
        chunk_id="doc2_c0",
        doc_id="doc2.md",
        text="FAISS supports dense vector indexing using IndexFlatIP.",
        start_char=0,
        end_char=56,
    )
    chunks = [c1, c2]

    test_cases = [
        ResolvedTestCase(
            id="q1",
            question="What is BM25?",
            document="doc1.md",
            expected_answer_spans=["BM25 is a term frequency keyword ranking function."],
            relevant_chunk_ids={"doc1_c0"},
        ),
        ResolvedTestCase(
            id="q2",
            question="What is FAISS?",
            document="doc2.md",
            expected_answer_spans=["FAISS supports dense vector indexing using IndexFlatIP."],
            relevant_chunk_ids={"doc2_c0"},
        ),
    ]

    service = RetrievalService(chunks=chunks)
    runner = BenchmarkRunner(service=service, test_cases=test_cases)
    return runner


def test_benchmark_runner_evaluation(mock_benchmark_env):
    """Test single configuration evaluation with per-query details."""
    runner = mock_benchmark_env
    res = runner.evaluate_configuration(
        retriever_type="sparse",
        top_k=2,
        use_reranker=False,
        include_details=True,
    )

    assert res.total_queries == 2
    assert res.hit_rate == 1.0
    assert res.recall_at_k == 1.0
    assert res.query_details is not None
    assert len(res.query_details) == 2
    assert res.query_details[0].is_hit is True


def test_benchmark_runner_sweep_csv(mock_benchmark_env, tmp_path):
    """Test sweep execution and CSV creation."""
    runner = mock_benchmark_env
    out_csv = tmp_path / "test_benchmark.csv"

    results = runner.run_full_sweep(
        retrievers=["sparse"],
        top_ks=[1],
        rerankers=[False],
        output_csv=out_csv,
    )

    assert len(results) == 1
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "retriever,top_k,reranker" in content
    assert "sparse,1,off" in content
