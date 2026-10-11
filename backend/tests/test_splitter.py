"""Unit tests for recursive heading splitter and character span accuracy."""

import pytest
from backend.chunking.splitter import Chunk, RecursiveHeadingSplitter
from backend.ingestion.loader import Document


def test_splitter_span_exact_match():
    content = (
        "# Introduction\n\n"
        "Retrieval-Augmented Generation enhances language models with external context. "
        "It queries indexed document stores before generating an answer.\n\n"
        "## Subsections\n\n"
        "Chunking is the process of splitting text into manageable passages with overlap."
    )
    doc = Document("doc1.md", content)
    splitter = RecursiveHeadingSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk's text MUST match the exact character span in original document
        assert doc.content[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.doc_id == "doc1.md"


def test_splitter_small_document():
    content = "A brief document."
    doc = Document("small.md", content)
    splitter = RecursiveHeadingSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == content
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(content)


def test_splitter_empty_document():
    doc = Document("empty.md", "")
    splitter = RecursiveHeadingSplitter(chunk_size=500, chunk_overlap=50)
    assert splitter.split_document(doc) == []


def test_splitter_invalid_params():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveHeadingSplitter(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and < chunk_size"):
        RecursiveHeadingSplitter(chunk_size=100, chunk_overlap=100)


def test_splitter_overlap_behavior():
    content = "Word " * 200  # 1000 characters
    doc = Document("words.txt", content)
    splitter = RecursiveHeadingSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_document(doc)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        # Current chunk end must be strictly greater than next chunk start (overlap exists)
        assert chunks[i].end_char > chunks[i + 1].start_char
