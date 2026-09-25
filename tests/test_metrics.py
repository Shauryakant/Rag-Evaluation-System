"""Unit tests for metrics module: span-overlap matching, Recall@k, and MRR."""

import pytest
from src.chunking import Chunk
from src.metrics import is_hit, is_span_overlap, mrr, recall_at_k
from src.retrievers import RetrievedChunk


def test_is_span_overlap():
    # Overlapping intervals
    assert is_span_overlap(0, 100, 50, 150) is True
    assert is_span_overlap(50, 150, 0, 100) is True
    assert is_span_overlap(10, 90, 20, 80) is True  # Contained

    # Non-overlapping adjacent or distant intervals
    assert is_span_overlap(0, 50, 50, 100) is False
    assert is_span_overlap(0, 40, 50, 100) is False


def test_is_hit():
    chunk = Chunk(chunk_id="doc1_c0", doc_id="doc1.txt", text="sample text", start_char=100, end_char=300)

    # Correct doc_id and overlapping span
    assert is_hit(chunk, gt_doc_id="doc1.txt", gt_start_char=200, gt_end_char=400) is True

    # Wrong doc_id
    assert is_hit(chunk, gt_doc_id="doc2.txt", gt_start_char=200, gt_end_char=400) is False

    # Correct doc_id but non-overlapping span
    assert is_hit(chunk, gt_doc_id="doc1.txt", gt_start_char=400, gt_end_char=500) is False


def test_recall_at_k_and_mrr():
    c1 = Chunk("c1", "doc1.txt", "text1", 0, 100)
    c2 = Chunk("c2", "doc1.txt", "text2", 100, 200)
    c3 = Chunk("c3", "doc1.txt", "text3", 200, 300)

    retrieved = [
        RetrievedChunk(chunk=c1, score=0.9, rank=1),  # span [0..100]
        RetrievedChunk(chunk=c2, score=0.8, rank=2),  # span [100..200]
        RetrievedChunk(chunk=c3, score=0.7, rank=3),  # span [200..300]
    ]

    # Ground truth is at span [150..250] in doc1.txt (overlaps c2 and c3, first hit is c2 at rank 2)
    gt_doc = "doc1.txt"
    gt_start = 150
    gt_end = 250

    # top_k = 1: c1 does not overlap [150..250] -> Recall@1 = 0.0, MRR@1 = 0.0
    assert recall_at_k(retrieved, gt_doc, gt_start, gt_end, top_k=1) == 0.0
    assert mrr(retrieved, gt_doc, gt_start, gt_end, top_k=1) == 0.0

    # top_k = 2: c2 overlaps -> Recall@2 = 1.0, MRR@2 = 1/2 = 0.5
    assert recall_at_k(retrieved, gt_doc, gt_start, gt_end, top_k=2) == 1.0
    assert pytest.approx(mrr(retrieved, gt_doc, gt_start, gt_end, top_k=2)) == 0.5

    # top_k = 3: first hit remains at rank 2 -> MRR = 0.5
    assert recall_at_k(retrieved, gt_doc, gt_start, gt_end, top_k=3) == 1.0
    assert pytest.approx(mrr(retrieved, gt_doc, gt_start, gt_end, top_k=3)) == 0.5
