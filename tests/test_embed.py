"""Unit tests for embedding wrapper and disk caching logic."""

import numpy as np
import pytest
from src.embed_index import _hash_payload, get_embeddings


def test_hash_payload_deterministic():
    texts = ["hello world", "rag evaluation"]
    model_name = "all-MiniLM-L6-v2"
    hash1 = _hash_payload(texts, model_name)
    hash2 = _hash_payload(texts, model_name)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_get_embeddings_empty():
    res = get_embeddings([], "all-MiniLM-L6-v2")
    assert res.shape == (0, 0)
