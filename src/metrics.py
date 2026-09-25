"""Evaluation metrics for RAG retrieval performance: Recall@k and Mean Reciprocal Rank (MRR)."""

from typing import List

from src.chunking import Chunk
from src.retrievers import RetrievedChunk


def is_span_overlap(
    chunk_start: int,
    chunk_end: int,
    gt_start: int,
    gt_end: int,
) -> bool:
    """Determine whether two character index intervals overlap.

    Args:
        chunk_start: Start character offset of retrieved chunk.
        chunk_end: End character offset of retrieved chunk.
        gt_start: Start character offset of ground-truth span.
        gt_end: End character offset of ground-truth span.

    Returns:
        True if intervals overlap, False otherwise.
    """
    return max(chunk_start, gt_start) < min(chunk_end, gt_end)


def is_hit(
    chunk: Chunk,
    gt_doc_id: str,
    gt_start_char: int,
    gt_end_char: int,
) -> bool:
    """Check if a retrieved chunk matches ground truth document ID and span overlap.

    Args:
        chunk: Candidate Chunk object.
        gt_doc_id: Ground truth document filename ID.
        gt_start_char: Ground truth start character offset.
        gt_end_char: Ground truth end character offset.

    Returns:
        True if doc_id matches and character spans overlap.
    """
    if chunk.doc_id != gt_doc_id:
        return False
    return is_span_overlap(chunk.start_char, chunk.end_char, gt_start_char, gt_end_char)


def recall_at_k(
    retrieved_chunks: List[RetrievedChunk],
    gt_doc_id: str,
    gt_start_char: int,
    gt_end_char: int,
    top_k: int,
) -> float:
    """Compute Recall@k score (1.0 if at least one hit in top_k, else 0.0).

    Args:
        retrieved_chunks: List of RetrievedChunk instances.
        gt_doc_id: Ground truth document ID.
        gt_start_char: Ground truth start character index.
        gt_end_char: Ground truth end character index.
        top_k: Target k evaluation depth.

    Returns:
        1.0 for hit present in top_k, 0.0 otherwise.
    """
    if not retrieved_chunks or top_k <= 0:
        return 0.0

    eval_chunks = retrieved_chunks[:top_k]
    for item in eval_chunks:
        if is_hit(item.chunk, gt_doc_id, gt_start_char, gt_end_char):
            return 1.0
    return 0.0


def mrr(
    retrieved_chunks: List[RetrievedChunk],
    gt_doc_id: str,
    gt_start_char: int,
    gt_end_char: int,
    top_k: int,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a query (1 / rank of first hit in top_k).

    Args:
        retrieved_chunks: List of RetrievedChunk instances.
        gt_doc_id: Ground truth document ID.
        gt_start_char: Ground truth start character index.
        gt_end_char: Ground truth end character index.
        top_k: Target k evaluation depth.

    Returns:
        Reciprocal rank float value (1.0, 0.5, 0.333, etc.) or 0.0 if no hit.
    """
    if not retrieved_chunks or top_k <= 0:
        return 0.0

    eval_chunks = retrieved_chunks[:top_k]
    for item in eval_chunks:
        if is_hit(item.chunk, gt_doc_id, gt_start_char, gt_end_char):
            return 1.0 / float(item.rank)
    return 0.0
