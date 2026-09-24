"""Embedding generation module with disk caching using sentence-transformers."""

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CACHE_DIR


def _hash_payload(texts: List[str], model_name: str) -> str:
    """Generate a deterministic SHA256 hex digest for model name and texts.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model name / identifier.

    Returns:
        Hex string hash.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8"))
    return hasher.hexdigest()[:16]


_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Lazy-load and cache SentenceTransformer models in memory.

    Args:
        model_name: HuggingFace sentence transformer model string.

    Returns:
        Loaded SentenceTransformer model instance.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def get_embeddings(
    texts: List[str],
    model_name: str,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Generate normalized sentence embeddings with disk caching.

    Args:
        texts: List of text strings to encode.
        model_name: Model name identifier.
        cache_dir: Directory path for disk caching. Defaults to CACHE_DIR from config.

    Returns:
        float32 numpy array of shape (N, dimension).
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    target_cache_dir = cache_dir or CACHE_DIR
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    payload_hash = _hash_payload(texts, model_name)
    safe_model_name = model_name.replace("/", "_")
    cache_file = target_cache_dir / f"emb_{safe_model_name}_{payload_hash}.npy"

    if cache_file.exists():
        try:
            embeddings = np.load(cache_file)
            if embeddings.shape[0] == len(texts):
                return embeddings.astype(np.float32)
        except Exception:
            # Fallback if cache file is corrupted
            pass

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    np.save(cache_file, embeddings)
    return embeddings
