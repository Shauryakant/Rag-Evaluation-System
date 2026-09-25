"""Unit tests for retrieval implementations (Vector, BM25, Hybrid)."""

import pytest
from src.chunking import Chunk
from src.retrievers import BM25Retriever, VectorRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="doc1_c0", doc_id="doc1.txt", text="Python programming language", start_char=0, end_char=27),
        Chunk(chunk_id="doc1_c1", doc_id="doc1.txt", text="Machine learning with PyTorch", start_char=28, end_char=57),
        Chunk(chunk_id="doc2_c0", doc_id="doc2.txt", text="Cooking recipes and baking cakes", start_char=0, end_char=32),
    ]


def test_vector_retriever(sample_chunks):
    retriever = VectorRetriever(chunks=sample_chunks, model_name="all-MiniLM-L6-v2")
    results = retriever.retrieve(query="machine learning", top_k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert "Machine learning" in results[0].chunk.text


def test_bm25_retriever(sample_chunks):
    retriever = BM25Retriever(chunks=sample_chunks)
    results = retriever.retrieve(query="baking cakes", top_k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert "Cooking recipes" in results[0].chunk.text

