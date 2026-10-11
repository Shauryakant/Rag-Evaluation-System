"""Hybrid retrieval combining dense vector search and BM25 lexical search with Reciprocal Rank Fusion."""

from typing import Dict, List, Optional

from backend.chunking.splitter import Chunk
from backend.retrieval.bm25_store import BM25SparseStore
from backend.retrieval.faiss_store import FAISSDenseStore, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
    k_rrf: int = 60,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """Merge multiple ranked lists of chunks using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in M} (1 / (k_rrf + rank_m(d)))

    Args:
        ranked_lists: List of retrieved lists from different retrievers (e.g., [dense, sparse]).
        k_rrf: Smoothing constant to balance high vs lower ranked documents (standard = 60).
        top_k: Number of fused results to return.

    Returns:
        Consolidated List[RetrievedChunk] sorted by descending RRF score.
    """
    if not ranked_lists or top_k <= 0:
        return []

    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            c_id = item.chunk.chunk_id
            if c_id not in chunk_map:
                chunk_map[c_id] = item.chunk

            # Rank is 1-based index in the candidate list
            rrf_contribution = 1.0 / (k_rrf + item.rank)
            scores[c_id] = scores.get(c_id, 0.0) + rrf_contribution

    # Sort chunks by RRF score descending
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    fused_results: List[RetrievedChunk] = []
    for rank_idx, (c_id, score) in enumerate(sorted_items, start=1):
        fused_results.append(
            RetrievedChunk(
                chunk=chunk_map[c_id],
                score=round(score, 6),
                rank=rank_idx,
            )
        )

    return fused_results


class HybridRetriever:
    """Coordinates dense vector and sparse lexical retrieval engines."""

    def __init__(
        self,
        dense_store: FAISSDenseStore,
        bm25_store: BM25SparseStore,
        k_rrf: int = 60,
    ) -> None:
        """Initialize hybrid retriever with underlying dense and sparse stores.

        Args:
            dense_store: FAISSDenseStore instance.
            bm25_store: BM25SparseStore instance.
            k_rrf: RRF smoothing constant (default 60).
        """
        self.dense_store = dense_store
        self.bm25_store = bm25_store
        self.k_rrf = k_rrf

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: Optional[int] = None,
        sparse_candidates: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Perform hybrid search over dense and sparse indices.

        Args:
            query: User query string.
            top_k: Final number of ranked chunks to return.
            dense_candidates: Number of initial candidates to fetch from dense store (default: 2 * top_k).
            sparse_candidates: Number of initial candidates to fetch from sparse store (default: 2 * top_k).

        Returns:
            List[RetrievedChunk] ranked by fused score.
        """
        if not query.strip() or top_k <= 0:
            return []

        cand_dense = dense_candidates if dense_candidates is not None else max(top_k * 2, 20)
        cand_sparse = sparse_candidates if sparse_candidates is not None else max(top_k * 2, 20)

        dense_results = self.dense_store.search(query, top_k=cand_dense)
        sparse_results = self.bm25_store.search(query, top_k=cand_sparse)

        return reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k_rrf=self.k_rrf,
            top_k=top_k,
        )
