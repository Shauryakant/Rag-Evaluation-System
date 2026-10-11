"""CrossEncoder reranking module using ms-marco-MiniLM-L-6-v2 (opt-in per query)."""

import time
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from backend.config import RERANKER_MODEL_NAME
from backend.retrieval.faiss_store import RetrievedChunk

_LOADED_RERANKERS: Dict[str, CrossEncoder] = {}


def load_cross_encoder(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Lazy-load and cache CrossEncoder models in memory.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        CrossEncoder instance.
    """
    if model_name not in _LOADED_RERANKERS:
        _LOADED_RERANKERS[model_name] = CrossEncoder(model_name)
    return _LOADED_RERANKERS[model_name]


class CrossEncoderReranker:
    """Reranks candidate chunks using full cross-attention between query and passage."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace cross-encoder identifier.
        """
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Get or lazily initialize the CrossEncoder model."""
        if self._model is None:
            self._model = load_cross_encoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], float]:
        """Rerank candidate chunks using cross-attention scores.

        Args:
            query: User query text.
            candidates: Pre-retrieved candidate chunks.
            top_k: Number of reranked chunks to return (defaults to len(candidates)).

        Returns:
            Tuple of (reranked_chunks, latency_ms).
        """
        if not candidates or not query.strip():
            return [], 0.0

        start_time = time.perf_counter()

        # Build query-passage pairs for CrossEncoder
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = self.model.predict(pairs)

        # Pair candidates with their new cross-encoder scores
        scored_candidates = []
        for item, score in zip(candidates, scores):
            scored_candidates.append((item.chunk, float(score)))

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        k = top_k if top_k is not None and top_k > 0 else len(candidates)
        selected = scored_candidates[:k]

        reranked_results: List[RetrievedChunk] = []
        for rank_idx, (chunk, score) in enumerate(selected, start=1):
            reranked_results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(score, 6),
                    rank=rank_idx,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return reranked_results, round(latency_ms, 2)
