"""Retrieval algorithms: Vector search, BM25, and Hybrid RRF."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from src.chunking import Chunk
from src.embed_index import get_embeddings, get_or_build_faiss_index


@dataclass
class RetrievedChunk:
    """Container for a retrieved chunk alongside its similarity score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


class VectorRetriever:
    """Dense vector retriever utilizing FAISS index cosine similarity search."""

    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str,
        index: Optional[faiss.IndexFlatIP] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize VectorRetriever with chunks and a FAISS index.

        Args:
            chunks: List of indexed Chunk objects.
            model_name: HuggingFace sentence transformer model identifier.
            index: Optional pre-built FAISS index. Built automatically if None.
            cache_dir: Optional directory path for disk caching embeddings.
        """
        self.chunks = chunks
        self.model_name = model_name
        if index is not None:
            self.index = index
        else:
            self.index, _ = get_or_build_faiss_index(chunks, model_name, cache_dir=cache_dir)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve top_k most similar chunks for a given query.

        Args:
            query: Query string.
            top_k: Number of top chunks to retrieve.

        Returns:
            List of RetrievedChunk objects ordered by similarity rank.
        """
        if not self.chunks or top_k <= 0 or not query.strip():
            return []

        query_emb = get_embeddings([query], self.model_name)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_emb, k)

        results: List[RetrievedChunk] = []
        for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank_idx,
                )
            )
        return results
