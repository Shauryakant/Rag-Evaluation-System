"""BM25Okapi sparse lexical store for exact keyword retrieval."""

import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.chunking.splitter import Chunk
from backend.retrieval.faiss_store import RetrievedChunk


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25.

    Args:
        text: Input string.

    Returns:
        List of lowercase string tokens.
    """
    return re.findall(r"\w+", text.lower())


class BM25SparseStore:
    """Sparse lexical store implementing BM25Okapi."""

    def __init__(
        self,
        chunks: List[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 index over chunks.

        Args:
            chunks: List of Chunk objects.
            k1: Term saturation parameter (default 1.5).
            b: Length normalization parameter (default 0.75).
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize_text(c.text) for c in chunks]
        self.bm25: Optional[BM25Okapi] = None

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Perform BM25 keyword search.

        Args:
            query: Query string.
            top_k: Number of highest-scoring passages to return.

        Returns:
            List of RetrievedChunk objects ordered by BM25 score descending.
        """
        if not self.chunks or self.bm25 is None or top_k <= 0 or not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievedChunk] = []
        for rank_idx, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank_idx,
                )
            )
        return results
