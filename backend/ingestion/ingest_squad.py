"""Ingestion pipeline downloading real Wikipedia articles and human-verified Q&A from SQuAD v2.0."""

import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List

from backend.chunking.splitter import RecursiveHeadingSplitter
from backend.config import CORPUS_DIR, TESTSET_PATH
from backend.evaluation.ground_truth import resolve_ground_truth
from backend.ingestion.loader import load_corpus

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

SELECTED_ARTICLES = [
    "Normans",
    "Computational_complexity_theory",
    "Southern_California",
    "Steam_engine",
    "Oxygen",
    "Victoria_(Australia)",
    "Huguenot",
    "Sky_(United_Kingdom)",
]


def extract_sentence_containing_span(paragraph_text: str, answer_text: str) -> str:
    """Find the exact verbatim sentence in paragraph_text that encompasses the answer span.

    This ensures the span is discriminative across the document rather than a short ambiguous token.
    """
    idx = paragraph_text.find(answer_text)
    if idx == -1:
        return answer_text

    # Search backwards for sentence start
    start = max(
        paragraph_text.rfind(". ", 0, idx),
        paragraph_text.rfind(".\n", 0, idx),
        paragraph_text.rfind("?\n", 0, idx),
        paragraph_text.rfind("!\n", 0, idx),
    )
    if start == -1:
        sentence_start = 0
    else:
        sentence_start = start + 2

    # Search forwards for sentence end
    end = len(paragraph_text)
    for term in [". ", ".\n", "? ", "! ", ".\t"]:
        pos = paragraph_text.find(term, idx + len(answer_text))
        if pos != -1 and pos + 1 < end:
            end = pos + 1

    sentence = paragraph_text[sentence_start:end].strip()
    # If the sentence is reasonable length (30 to 280 chars), use it; otherwise use answer_text
    if len(sentence) >= 30 and len(sentence) <= 300:
        return sentence
    return answer_text


def ingest_squad_wikipedia() -> None:
    """Download SQuAD dev data and populate real Wikipedia documents and testset."""
    print("Downloading real Wikipedia articles from SQuAD v2.0...")
    req = urllib.request.Request(SQUAD_DEV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        squad_data = json.loads(resp.read().decode("utf-8"))

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean out existing corpus files
    for old_file in CORPUS_DIR.glob("*.*"):
        old_file.unlink()
    print("Cleaned old corpus directory.")

    all_testset: List[Dict] = []
    question_count = 0

    for article in squad_data["data"]:
        title = article["title"]
        if title not in SELECTED_ARTICLES:
            continue

        clean_title = title.lower().replace("(", "").replace(")", "").replace("__", "_")
        doc_filename = f"{clean_title}.md"
        doc_path = CORPUS_DIR / doc_filename

        # Assemble full document text from paragraphs
        doc_paragraphs: List[str] = [f"# {title.replace('_', ' ')}\n"]
        article_candidates: List[Dict] = []

        for p_idx, p in enumerate(article["paragraphs"]):
            context = p["context"].strip()
            doc_paragraphs.append(context)

            # Extract answerable questions from this paragraph
            for qa in p.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue

                raw_ans = answers[0]["text"].strip()
                if not raw_ans:
                    continue

                discriminative_span = extract_sentence_containing_span(context, raw_ans)

                article_candidates.append(
                    {
                        "raw_id": qa["id"],
                        "question": qa["question"].strip(),
                        "document": doc_filename,
                        "span": discriminative_span,
                        "context": context,
                    }
                )

        # Write clean document markdown
        full_content = "\n\n".join(doc_paragraphs)
        doc_path.write_text(full_content, encoding="utf-8")
        print(f"Created real Wikipedia document: {doc_filename} ({len(full_content)} chars)")

        # Select 5 high-quality questions per article
        # Verify discriminative property within the document
        chosen_for_article = 0
        for cand in article_candidates:
            span = cand["span"]
            # Count occurrences in the document
            count_in_doc = full_content.count(span)
            if count_in_doc == 1 and len(span) >= 35:
                question_count += 1
                all_testset.append(
                    {
                        "id": f"q{question_count:02d}",
                        "question": cand["question"],
                        "document": cand["document"],
                        "expected_answer_spans": [span],
                    }
                )
                chosen_for_article += 1
                if chosen_for_article >= 5:
                    break

    # Save real testset
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(all_testset, f, indent=2)

    print(f"\nSuccessfully generated {len(all_testset)} real Wikipedia benchmark questions in {TESTSET_PATH}")


if __name__ == "__main__":
    ingest_squad_wikipedia()
