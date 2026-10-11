"""Unit tests for Dense (FAISS), Sparse (BM25), and Hybrid (RRF) retrieval systems."""

import numpy as np
import pytest

from backend.chunking.splitter import Chunk
from backend.embeddings.embedder import BGEEmbedder
from backend.retrieval.bm25_store import BM25SparseStore, tokenize_text
from backend.retrieval.faiss_store import FAISSDenseStore, RetrievedChunk
from backend.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


@pytest.fixture
def sample_chunks():
    """Create a sample fixture of chunks for testing retrieval."""
    return [
        Chunk(
            chunk_id="doc1_chunk_0000",
            doc_id="doc1.md",
            text="Reciprocal Rank Fusion (RRF) combines rankings from multiple retrieval algorithms.",
            start_char=0,
            end_char=83,
            metadata={"headings": "# Hybrid Search > ## RRF"},
        ),
        Chunk(
            chunk_id="doc1_chunk_0001",
            doc_id="doc1.md",
            text="BM25Okapi is a probabilistic lexical ranking function based on TF-IDF term saturation.",
            start_char=84,
            end_char=172,
            metadata={"headings": "# Lexical Search > ## BM25"},
        ),
        Chunk(
            chunk_id="doc2_chunk_0000",
            doc_id="doc2.md",
            text="FAISS provides dense vector similarity search with exact inner product IndexFlatIP.",
            start_char=0,
            end_char=84,
            metadata={"headings": "# Dense Search > ## FAISS"},
        ),
    ]


def test_tokenize_text():
    """Test BM25 text tokenization."""
    text = "RAG-Eval-Lab: Advanced Retrieval!"
    tokens = tokenize_text(text)
    assert "rag" in tokens
    assert "eval" in tokens
    assert "lab" in tokens
    assert "advanced" in tokens
    assert "retrieval" in tokens


def test_bm25_sparse_store(sample_chunks):
    """Test BM25 keyword matching and ranking."""
    store = BM25SparseStore(chunks=sample_chunks)
    results = store.search(query="BM25Okapi lexical saturation", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc1_chunk_0001"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_bm25_edge_cases(sample_chunks):
    """Test BM25 empty queries and top_k bounds."""
    store = BM25SparseStore(chunks=sample_chunks)
    assert store.search(query="", top_k=5) == []
    assert store.search(query="   ", top_k=5) == []
    assert store.search(query="test", top_k=0) == []


def test_faiss_dense_store(sample_chunks, tmp_path):
    """Test FAISS IndexFlatIP dense vector store."""
    embedder = BGEEmbedder(cache_dir=tmp_path / "embeddings")
    dense_store = FAISSDenseStore(chunks=sample_chunks, embedder=embedder)

    results = dense_store.search(query="vector similarity inner product search", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc2_chunk_0000"
    assert results[0].rank == 1
    assert results[0].score >= -1.0 and results[0].score <= 1.0


def test_reciprocal_rank_fusion_math(sample_chunks):
    """Test Reciprocal Rank Fusion mathematical accuracy."""
    c0 = sample_chunks[0]
    c1 = sample_chunks[1]
    c2 = sample_chunks[2]

    # Retriever 1 ranks: [c0 (rank 1), c1 (rank 2)]
    list_dense = [
        RetrievedChunk(chunk=c0, score=0.9, rank=1),
        RetrievedChunk(chunk=c1, score=0.8, rank=2),
    ]
    # Retriever 2 ranks: [c1 (rank 1), c2 (rank 2)]
    list_sparse = [
        RetrievedChunk(chunk=c1, score=5.0, rank=1),
        RetrievedChunk(chunk=c2, score=4.0, rank=2),
    ]

    fused = reciprocal_rank_fusion([list_dense, list_sparse], k_rrf=60, top_k=3)

    # c1 appears in both: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.016129 + 0.016393 = 0.032522
    # c0 appears in dense: 1/(60+1) = 1/61 ~= 0.016393
    # c2 appears in sparse: 1/(60+2) = 1/62 ~= 0.016129
    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == c1.chunk_id
    assert fused[1].chunk.chunk_id == c0.chunk_id
    assert fused[2].chunk.chunk_id == c2.chunk_id
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[2].rank == 3

    # Validate exact mathematical formula
    expected_c1_score = round(1.0 / 62 + 1.0 / 61, 6)
    assert abs(fused[0].score - expected_c1_score) < 1e-5


def test_hybrid_retriever_end_to_end(sample_chunks, tmp_path):
    """Test HybridRetriever combining dense and sparse search."""
    embedder = BGEEmbedder(cache_dir=tmp_path / "embed_cache")
    dense_store = FAISSDenseStore(chunks=sample_chunks, embedder=embedder)
    bm25_store = BM25SparseStore(chunks=sample_chunks)

    hybrid = HybridRetriever(dense_store=dense_store, bm25_store=bm25_store, k_rrf=60)
    results = hybrid.search(query="Reciprocal Rank Fusion hybrid algorithms", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc1_chunk_0000"
    assert results[0].rank == 1
