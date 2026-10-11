"""Unit tests for BGE embedder, vector normalization, and disk caching."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from backend.embeddings.embedder import BGEEmbedder, hash_text_payload


def test_hash_text_payload():
    texts1 = ["first document", "second document"]
    texts2 = ["first document", "second document"]
    texts3 = ["different document"]

    h1 = hash_text_payload(texts1, "model_a")
    h2 = hash_text_payload(texts2, "model_a")
    h3 = hash_text_payload(texts3, "model_a")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_bge_embedder_normalization_and_cache():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        embedder = BGEEmbedder(cache_dir=tmp_dir)
        texts = ["What is retrieval augmented generation?", "BM25 is a sparse lexical algorithm."]

        embeddings = embedder.embed_texts(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

        # Check L2 normalization: norm of each vector must be approximately 1.0
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            assert pytest.approx(norm, abs=1e-4) == 1.0

        # Verify disk cache file was created
        cache_files = list(tmp_dir.glob("emb_*.npy"))
        assert len(cache_files) == 1

        # Second call must load from disk cache
        embeddings_cached = embedder.embed_texts(texts)
        np.testing.assert_array_almost_equal(embeddings, embeddings_cached)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_embed_query_shape():
    embedder = BGEEmbedder()
    q_vec = embedder.embed_query("sample query")
    assert q_vec.shape == (1, 384)
    assert q_vec.dtype == np.float32
    assert pytest.approx(float(np.linalg.norm(q_vec[0])), abs=1e-4) == 1.0


def test_embed_empty():
    embedder = BGEEmbedder()
    res = embedder.embed_texts([])
    assert res.shape == (0, 384)
