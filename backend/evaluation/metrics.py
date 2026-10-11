"""Evaluation metrics for Information Retrieval in RAG pipelines.

Pure, deterministic evaluation functions:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Hit Rate@K
"""

from typing import Dict, List, Optional, Set


def precision_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Set[str],
    k: int,
) -> float:
    """Calculate Precision@K: proportion of retrieved chunks in top K that are relevant.

    NOTE ON STRUCTURAL CAP:
    When questions only have 1 or 2 relevant ground-truth chunks in the corpus,
    Precision@K is structurally capped at |GT| / K (e.g. 1 / 5 = 0.20 max at K=5).
    Recall@K and MRR are the primary signals of retrieval effectiveness.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Set of relevant ground-truth chunk IDs.
        k: Cutoff rank.

    Returns:
        Float value in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in ground_truth_chunk_ids)
    return round(hits / k, 4)


def recall_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Set[str],
    k: int,
) -> float:
    """Calculate Recall@K: proportion of relevant chunks retrieved in top K.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Set of relevant ground-truth chunk IDs.
        k: Cutoff rank.

    Returns:
        Float value in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in ground_truth_chunk_ids)
    return round(hits / len(ground_truth_chunk_ids), 4)


def mrr(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Set[str],
    k: Optional[int] = None,
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the FIRST relevant chunk.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Set of relevant ground-truth chunk IDs.
        k: Optional cutoff rank.

    Returns:
        Float value in [0.0, 1.0]. Returns 0.0 if no relevant chunk found within rank K.
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return 0.0

    eval_list = retrieved_chunk_ids[:k] if k is not None and k > 0 else retrieved_chunk_ids

    for rank_idx, cid in enumerate(eval_list, start=1):
        if cid in ground_truth_chunk_ids:
            return round(1.0 / rank_idx, 4)

    return 0.0


def hit_rate(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Set[str],
    k: int,
) -> float:
    """Calculate Hit Rate@K: binary indicator (1.0 or 0.0) whether at least one relevant chunk was retrieved in top K.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Set of relevant ground-truth chunk IDs.
        k: Cutoff rank.

    Returns:
        1.0 if hit, else 0.0.
    """
    if k <= 0 or not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    for cid in top_k_retrieved:
        if cid in ground_truth_chunk_ids:
            return 1.0

    return 0.0


def calculate_query_metrics(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Set[str],
    k: int,
) -> Dict[str, float]:
    """Calculate all standard metrics for a single query.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Set of relevant ground-truth chunk IDs.
        k: Cutoff rank.

    Returns:
        Dictionary with precision_at_k, recall_at_k, mrr, hit_rate.
    """
    return {
        "precision": precision_at_k(retrieved_chunk_ids, ground_truth_chunk_ids, k=k),
        "recall": recall_at_k(retrieved_chunk_ids, ground_truth_chunk_ids, k=k),
        "mrr": mrr(retrieved_chunk_ids, ground_truth_chunk_ids, k=k),
        "hit_rate": hit_rate(retrieved_chunk_ids, ground_truth_chunk_ids, k=k),
    }
