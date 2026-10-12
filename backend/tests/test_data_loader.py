"""Tests for corpus loader and testset span validation."""

import json
from pathlib import Path

import pytest
from backend.config import CORPUS_DIR, TESTSET_PATH
from backend.ingestion.loader import load_corpus, load_single_document


def test_load_corpus():
    docs = load_corpus(CORPUS_DIR)
    assert len(docs) >= 8
    doc_ids = [d.doc_id for d in docs]
    assert "normans.md" in doc_ids
    assert "oxygen.md" in doc_ids


def test_all_testset_spans_present_in_corpus():
    assert TESTSET_PATH.exists()
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)

    assert len(testset) >= 40
    docs = {d.doc_id: d.content for d in load_corpus(CORPUS_DIR)}

    for item in testset:
        doc_name = item["document"]
        assert doc_name in docs, f"Document {doc_name} for question {item['id']} not in corpus"

        doc_text = docs[doc_name]
        for span in item["expected_answer_spans"]:
            assert len(span.strip()) > 0
            # Span must be present in document text
            assert span in doc_text, (
                f"Span for question {item['id']} not found in document {doc_name}: '{span}'"
            )
            # Span must be discriminative (appear exactly once in the document)
            occurrences = doc_text.count(span)
            assert occurrences == 1, (
                f"Span for question {item['id']} appears {occurrences} times in {doc_name}. "
                f"Spans must appear exactly once."
            )
