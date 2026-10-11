"""Unit tests for Information Retrieval metrics."""

import pytest

from backend.evaluation.metrics import (
    calculate_query_metrics,
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    """Test Precision@K and verify the mathematical structural cap."""
    ground_truth = {"chunk_01"}
    retrieved = ["chunk_01", "chunk_02", "chunk_03", "chunk_04", "chunk_05"]

    # When |GT| = 1 and k = 5, maximum possible Precision@5 is 1/5 = 0.20
    p5 = precision_at_k(retrieved, ground_truth, k=5)
    assert p5 == 0.20

    # At k = 1, precision is 1/1 = 1.0
    p1 = precision_at_k(retrieved, ground_truth, k=1)
    assert p1 == 1.0

    # When no ground truth is retrieved
    p_miss = precision_at_k(["chunk_99"], ground_truth, k=1)
    assert p_miss == 0.0


def test_recall_at_k():
    """Test Recall@K calculation."""
    ground_truth = {"chunk_01", "chunk_02"}
    retrieved = ["chunk_01", "chunk_03", "chunk_04"]

    # 1 out of 2 ground truth chunks found in top 3
    r3 = recall_at_k(retrieved, ground_truth, k=3)
    assert r3 == 0.50

    # When all ground truth chunks are retrieved
    r_all = recall_at_k(["chunk_01", "chunk_02"], ground_truth, k=2)
    assert r_all == 1.0

    # When none are retrieved
    r_none = recall_at_k(["chunk_05"], ground_truth, k=1)
    assert r_none == 0.0


def test_mrr():
    """Test Mean Reciprocal Rank (first relevant item rank)."""
    ground_truth = {"chunk_target"}

    # Relevant at rank 1 -> 1 / 1 = 1.0
    assert mrr(["chunk_target", "chunk_other"], ground_truth) == 1.0

    # Relevant at rank 2 -> 1 / 2 = 0.5
    assert mrr(["chunk_other", "chunk_target"], ground_truth) == 0.5

    # Relevant at rank 4 -> 1 / 4 = 0.25
    assert mrr(["c1", "c2", "c3", "chunk_target"], ground_truth) == 0.25

    # Not retrieved within k=2
    assert mrr(["c1", "c2", "chunk_target"], ground_truth, k=2) == 0.0

    # Not in list at all
    assert mrr(["c1", "c2"], ground_truth) == 0.0


def test_hit_rate():
    """Test Hit Rate binary indicator."""
    ground_truth = {"target"}
    assert hit_rate(["target", "c2"], ground_truth, k=1) == 1.0
    assert hit_rate(["c1", "target"], ground_truth, k=1) == 0.0
    assert hit_rate(["c1", "target"], ground_truth, k=2) == 1.0
    assert hit_rate(["c1", "c2"], ground_truth, k=5) == 0.0


def test_edge_cases():
    """Test boundary and empty inputs."""
    assert precision_at_k([], {"c1"}, k=5) == 0.0
    assert precision_at_k(["c1"], set(), k=5) == 0.0
    assert precision_at_k(["c1"], {"c1"}, k=0) == 0.0

    assert recall_at_k([], {"c1"}, k=5) == 0.0
    assert recall_at_k(["c1"], set(), k=5) == 0.0
    assert recall_at_k(["c1"], {"c1"}, k=0) == 0.0

    assert mrr([], {"c1"}) == 0.0
    assert mrr(["c1"], set()) == 0.0

    assert hit_rate([], {"c1"}, k=3) == 0.0
    assert hit_rate(["c1"], set(), k=3) == 0.0
    assert hit_rate(["c1"], {"c1"}, k=0) == 0.0


def test_calculate_query_metrics():
    """Test all metrics combined dictionary generation."""
    metrics = calculate_query_metrics(
        retrieved_chunk_ids=["c1", "c2", "c3"],
        ground_truth_chunk_ids={"c2"},
        k=3,
    )
    assert metrics["precision"] == round(1 / 3, 4)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["hit_rate"] == 1.0
