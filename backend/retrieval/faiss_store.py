"""FAISS IndexFlatIP dense vector store for exact cosine similarity retrieval."""

from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np

from backend.chunking.splitter import Chunk
from backend.embeddings.embedder import BGEEmbedder


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its relevance score and rank."""

    chunk: Chunk
    score: float
    rank: int


class FAISSDenseStore:
    """Dense vector store utilizing FAISS IndexFlatIP over normalized embeddings."""

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        """Initialize and populate the FAISS index.

        Args:
            chunks: List of Chunk objects to index.
            embedder: BGEEmbedder instance.
        """
        self.chunks = chunks
        self.embedder = embedder or BGEEmbedder()
        self.index: Optional[faiss.IndexFlatIP] = None

        if self.chunks:
            self._build_index()

    def _build_index(self) -> None:
        """Compute embeddings and build FAISS IndexFlatIP index."""
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Perform dense vector search for a given query.

        Args:
            query: Query string.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievedChunk objects ordered by cosine similarity descending.
        """
        if not self.chunks or self.index is None or top_k <= 0 or not query.strip():
            return []

        q_vec = self.embedder.embed_query(query)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

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
