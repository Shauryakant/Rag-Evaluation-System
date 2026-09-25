"""Retrieval algorithms: Vector search, BM25, and Hybrid RRF."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking import Chunk
from src.embed_index import get_embeddings, get_or_build_faiss_index


def _tokenize(text: str) -> List[str]:
    """Simple regex word tokenizer for BM25.

    Args:
        text: Input text string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\w+", text.lower())


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


class BM25Retriever:
    """Sparse retriever implementing BM25 ranking algorithm."""

    def __init__(self, chunks: List[Chunk]) -> None:
        """Initialize BM25Retriever with tokenized corpus.

        Args:
            chunks: List of Chunk objects to index.
        """
        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve top_k highest scoring chunks for a query using BM25.

        Args:
            query: Query text string.
            top_k: Maximum number of top chunks to return.

        Returns:
            List of RetrievedChunk objects.
        """
        if not self.chunks or not self.bm25 or top_k <= 0 or not query.strip():
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
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

