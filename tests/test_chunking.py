"""Unit tests for chunking module, verifying overlap, edge cases, and character span accuracy."""

import pytest
from src.chunking import Chunk, chunk_text


def test_chunk_text_basic():
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    doc_id = "doc1.txt"
    chunk_size = 10
    overlap = 2

    chunks = chunk_text(text, doc_id, chunk_size, overlap)
    assert len(chunks) > 0

    # Verify character spans match chunk text content
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.doc_id == doc_id


def test_chunk_text_overlap_behavior():
    text = "01234567890123456789"  # 20 chars
    chunk_size = 10
    overlap = 2  # stride = 8

    chunks = chunk_text(text, "doc1", chunk_size, overlap)
    # Chunk 0: [0..10], Chunk 1: [8..18], Chunk 2: [16..20]
    assert len(chunks) == 3
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 10
    assert chunks[1].start_char == 8
    assert chunks[1].end_char == 18
    assert chunks[2].start_char == 16
    assert chunks[2].end_char == 20


def test_chunk_text_short_text():
    text = "Hello world"
    chunks = chunk_text(text, "short.txt", chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_chunk_text_empty():
    assert chunk_text("", "empty.txt", chunk_size=100, overlap=10) == []
    assert chunk_text("   ", "empty.txt", chunk_size=100, overlap=10) == []


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_text("test", "doc.txt", chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must be >= 0 and < chunk_size"):
        chunk_text("test", "doc.txt", chunk_size=10, overlap=10)
