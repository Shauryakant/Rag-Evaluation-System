"""Embedding generation module utilizing BGE-small with deterministic disk caching."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import CACHE_DIR, EMBEDDER_MODEL_NAME

_LOADED_MODELS: Dict[str, SentenceTransformer] = {}


def load_sentence_transformer(model_name: str = EMBEDDER_MODEL_NAME) -> SentenceTransformer:
    """Lazy-load and cache SentenceTransformer models in memory.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        SentenceTransformer instance.
    """
    if model_name not in _LOADED_MODELS:
        _LOADED_MODELS[model_name] = SentenceTransformer(model_name)
    return _LOADED_MODELS[model_name]


def hash_text_payload(texts: List[str], model_name: str) -> str:
    """Compute a deterministic SHA-256 hash digest for a list of texts and model identifier.

    Args:
        texts: List of text strings.
        model_name: Name of the embedding model.

    Returns:
        16-character hex hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for text in texts:
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()[:16]


class BGEEmbedder:
    """Dense vector embedder producing L2-normalized vectors with disk caching."""

    def __init__(
        self,
        model_name: str = EMBEDDER_MODEL_NAME,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize BGEEmbedder.

        Args:
            model_name: HuggingFace sentence transformer identifier.
            cache_dir: Directory path for disk caching embeddings.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Get or lazily initialize the SentenceTransformer model."""
        if self._model is None:
            self._model = load_sentence_transformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text passages into normalized float32 vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            Numpy array of shape (N, dimension) with dtype float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        payload_hash = hash_text_payload(texts, self.model_name)
        safe_model_str = self.model_name.replace("/", "_")
        cache_path = self.cache_dir / f"emb_{safe_model_str}_{payload_hash}.npy"

        if cache_path.exists():
            try:
                cached_array = np.load(cache_path)
                if cached_array.shape[0] == len(texts):
                    return cached_array.astype(np.float32)
            except Exception:
                # Corrupted cache fallback
                pass

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        np.save(cache_path, embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a normalized 2D vector array.

        Args:
            query: Query string.

Returns:
            Numpy array of shape (1, dimension) with dtype float32.
        """
        if not query.strip():
            return np.zeros((1, 384), dtype=np.float32)
        return self.embed_texts([query])
