"""Tests for span-based ground truth resolution and re-chunking invariance."""

import pytest

from backend.chunking.splitter import Chunk, RecursiveHeadingSplitter
from backend.evaluation.ground_truth import (
    GroundTruthResolutionError,
    load_testset,
    resolve_ground_truth,
)
from backend.ingestion.loader import Document, load_corpus


def test_span_resolution_rechunking_invariance():
    """Prove that changing chunk sizes does not break ground truth,

    dynamically resolving to the newly formed chunks containing the target span.
    """
    doc_text = (
        "# Dense Retrieval\n\n"
        "Dense retrieval maps queries and documents to continuous vector embeddings.\n\n"
        "IndexFlatIP performs exact brute-force cosine similarity without quantization errors.\n\n"
        "Hierarchical Navigable Small World (HNSW) graphs offer approximate nearest neighbors."
    )
    doc = Document(doc_id="dense.md", content=doc_text)
    span = "IndexFlatIP performs exact brute-force cosine similarity without quantization errors."

    test_case = [
        {
            "id": "q_hnsw",
            "question": "What does IndexFlatIP do?",
            "document": "dense.md",
            "expected_answer_spans": [span],
        }
    ]

    # Strategy A: Small chunks (size 120)
    splitter_a = RecursiveHeadingSplitter(chunk_size=120, chunk_overlap=20)
    chunks_a = splitter_a.split_documents([doc])
    resolved_a = resolve_ground_truth(chunks=chunks_a, testset_data=test_case)
    assert len(resolved_a) == 1
    chunk_ids_a = resolved_a[0].relevant_chunk_ids
    assert len(chunk_ids_a) == 1
    # Verify the resolved chunk actually contains the span
    target_chunk_a = next(c for c in chunks_a if c.chunk_id in chunk_ids_a)
    assert span in target_chunk_a.text

    # Strategy B: Large chunks (size 350)
    splitter_b = RecursiveHeadingSplitter(chunk_size=350, chunk_overlap=30)
    chunks_b = splitter_b.split_documents([doc])
    resolved_b = resolve_ground_truth(chunks=chunks_b, testset_data=test_case)
    assert len(resolved_b) == 1
    chunk_ids_b = resolved_b[0].relevant_chunk_ids
    assert len(chunk_ids_b) == 1
    target_chunk_b = next(c for c in chunks_b if c.chunk_id in chunk_ids_b)
    assert span in target_chunk_b.text

    # Proves chunk IDs may differ across configurations, but ground truth resolves correctly!
    assert target_chunk_a.span != target_chunk_b.span or len(chunks_a) != len(chunks_b)


def test_zero_matches_fails_loudly():
    """Verify that a missing or stale text span raises GroundTruthResolutionError."""
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc1.md",
            text="This is text discussing topic A.",
            start_char=0,
            end_char=32,
        )
    ]
    stale_testset = [
        {
            "id": "q_stale",
            "question": "What about topic B?",
            "document": "doc1.md",
            "expected_answer_spans": ["This nonexistent span for topic B"],
        }
    ]

    with pytest.raises(GroundTruthResolutionError) as exc_info:
        resolve_ground_truth(chunks=chunks, testset_data=stale_testset, strict=True)

    assert "Zero chunks contain expected span" in str(exc_info.value)
    assert "q_stale" in str(exc_info.value)


def test_non_discriminative_span_fails_loudly():
    """Verify that an ambiguous span appearing in multiple chunks raises GroundTruthResolutionError."""
    ambiguous_span = "Common repeated header."
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc1.md",
            text=f"{ambiguous_span} Section 1 details.",
            start_char=0,
            end_char=40,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc1.md",
            text=f"{ambiguous_span} Section 2 details.",
            start_char=41,
            end_char=81,
        ),
    ]
    ambiguous_testset = [
        {
            "id": "q_ambig",
            "question": "What is the header?",
            "document": "doc1.md",
            "expected_answer_spans": [ambiguous_span],
        }
    ]

    with pytest.raises(GroundTruthResolutionError) as exc_info:
        resolve_ground_truth(chunks=chunks, testset_data=ambiguous_testset, strict=True)

    assert "Expected span is not discriminative" in str(exc_info.value)
    assert "Matches 2 chunks" in str(exc_info.value)


def test_live_corpus_and_testset_resolution():
    """Validate that ALL 40 benchmark questions in testset.json resolve unambiguously against the default corpus index."""
    documents = load_corpus()
    splitter = RecursiveHeadingSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    resolved_cases = resolve_ground_truth(chunks=chunks, strict=True)

    assert len(resolved_cases) >= 40
    for case in resolved_cases:
        assert len(case.relevant_chunk_ids) >= 1, f"Question {case.id} resolved to 0 chunks"
