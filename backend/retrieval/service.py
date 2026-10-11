"""Unified retrieval service ensuring identical retrieval paths across live Query and Evaluation Lab."""

import time
from typing import Dict, List, Literal, Optional, Tuple

from backend.chunking.splitter import Chunk
from backend.embeddings.embedder import BGEEmbedder
from backend.retrieval.bm25_store import BM25SparseStore
from backend.retrieval.faiss_store import FAISSDenseStore, RetrievedChunk
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker

RetrieverType = Literal["dense", "sparse", "hybrid"]


class RetrievalService:
    """Production retrieval coordinator managing FAISS, BM25, Hybrid RRF, and CrossEncoder.

    CRITICAL DESIGN RULE:
    Query mode and Evaluation Lab mode call the exact same retrieve() method.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: Optional[BGEEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        k_rrf: int = 60,
    ) -> None:
        """Initialize all retrieval components for the provided chunks.

        Args:
            chunks: List of indexed document chunks.
            embedder: BGEEmbedder instance (lazily created if None).
            reranker: CrossEncoderReranker instance (lazily created if None).
            k_rrf: Reciprocal Rank Fusion smoothing constant.
        """
        self.chunks = chunks
        self.embedder = embedder or BGEEmbedder()
        self.dense_store = FAISSDenseStore(chunks=self.chunks, embedder=self.embedder)
        self.bm25_store = BM25SparseStore(chunks=self.chunks)
        self.hybrid_retriever = HybridRetriever(
            dense_store=self.dense_store,
            bm25_store=self.bm25_store,
            k_rrf=k_rrf,
        )
        self._reranker = reranker

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Lazy-loaded CrossEncoder reranker instance."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        retriever_type: RetrieverType = "dense",
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """Retrieve top_k chunks matching query with granular latency telemetry.

        Args:
            query: User or benchmark question.
            retriever_type: 'dense' | 'sparse' | 'hybrid'.
            top_k: Number of final chunks to return.
            use_reranker: Opt-in reranking via CrossEncoder.

        Returns:
            Tuple of (List[RetrievedChunk], latency_dict_in_ms).
        """
        if not query.strip() or top_k <= 0 or not self.chunks:
            return [], {"retrieval_ms": 0.0, "reranker_ms": 0.0, "total_ms": 0.0}

        total_start = time.perf_counter()

        # If reranking is enabled, fetch a wider candidate pool before reranking
        candidate_k = max(top_k * 3, 20) if use_reranker else top_k

        # 1. Base Retrieval Phase
        retrieval_start = time.perf_counter()
        if retriever_type == "dense":
            candidates = self.dense_store.search(query, top_k=candidate_k)
        elif retriever_type == "sparse":
            candidates = self.bm25_store.search(query, top_k=candidate_k)
        elif retriever_type == "hybrid":
            candidates = self.hybrid_retriever.search(query, top_k=candidate_k)
        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}. Expected 'dense', 'sparse', or 'hybrid'.")
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 2. Opt-in CrossEncoder Reranking Phase
        reranker_ms = 0.0
        if use_reranker and candidates:
            final_chunks, reranker_ms = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            final_chunks = candidates[:top_k]

        total_ms = (time.perf_counter() - total_start) * 1000.0

        latency = {
            "retrieval_ms": round(retrieval_ms, 2),
            "reranker_ms": round(reranker_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        return final_chunks, latency
