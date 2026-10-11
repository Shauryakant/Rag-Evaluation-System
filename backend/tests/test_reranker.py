"""Unit tests for CrossEncoder reranker."""

import pytest

from backend.chunking.splitter import Chunk
from backend.retrieval.faiss_store import RetrievedChunk
from backend.retrieval.reranker import CrossEncoderReranker, load_cross_encoder


@pytest.fixture
def candidate_chunks():
    """Fixture providing candidate chunks with varying relevance."""
    chunk_irrelevant = Chunk(
        chunk_id="doc1_chunk_0001",
        doc_id="doc1.md",
        text="The weather in Seattle is typically rainy during the winter months.",
        start_char=0,
        end_char=70,
        metadata={},
    )
    chunk_relevant = Chunk(
        chunk_id="doc2_chunk_0001",
        doc_id="doc2.md",
        text="Reciprocal Rank Fusion combines sparse BM25 and dense FAISS vectors into a unified ranking.",
        start_char=0,
        end_char=93,
        metadata={},
    )

    # Simulate an initial retrieval where the irrelevant chunk was ranked 1st
    return [
        RetrievedChunk(chunk=chunk_irrelevant, score=0.88, rank=1),
        RetrievedChunk(chunk=chunk_relevant, score=0.72, rank=2),
    ]


def test_reranker_reordering(candidate_chunks):
    """Test that CrossEncoder promotes the truly relevant passage to rank 1."""
    reranker = CrossEncoderReranker()
    query = "How does hybrid search combine BM25 and FAISS?"

    reranked, latency = reranker.rerank(query=query, candidates=candidate_chunks, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == "doc2_chunk_0001"
    assert reranked[0].rank == 1
    assert reranked[1].chunk.chunk_id == "doc1_chunk_0001"
    assert reranked[1].rank == 2
    assert reranked[0].score > reranked[1].score
    assert latency > 0.0


def test_reranker_edge_cases(candidate_chunks):
    """Test empty queries and empty candidates."""
    reranker = CrossEncoderReranker()
    assert reranker.rerank(query="", candidates=candidate_chunks) == ([], 0.0)
    assert reranker.rerank(query="   ", candidates=candidate_chunks) == ([], 0.0)
    assert reranker.rerank(query="What is RRF?", candidates=[]) == ([], 0.0)


def test_reranker_model_caching():
    """Verify that load_cross_encoder returns cached instances."""
    model_a = load_cross_encoder()
    model_b = load_cross_encoder()
    assert model_a is model_b
