"""Test set generation script creating question-passage evaluation pairs from documents."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import DOCS_DIR, TESTSET_PATH
from src.loader import Document, load_documents
from src.llm import GeminiClient


def generate_question_for_passage(passage_text: str, llm_client: GeminiClient) -> str:
    """Prompt LLM to generate a question based on a target passage.

    Args:
        passage_text: Ground truth passage snippet.
        llm_client: GeminiClient instance.

    Returns:
        Generated question string.
    """
    prompt = (
        "You are an expert evaluator. Given the following document passage, generate ONE concise, "
        "factual question whose exact answer is contained within the passage.\n"
        "Return ONLY the plain text question string, with no quotes or extra commentary.\n\n"
        f"Passage:\n{passage_text}"
    )
    return llm_client.generate_text(prompt).strip()


def build_testset_from_docs(
    documents: List[Document],
    llm_client: Optional[GeminiClient] = None,
    output_path: Optional[Path] = None,
    passages_per_doc: int = 4,
    passage_length: int = 400,
) -> List[Dict[str, Any]]:
    """Sample passage spans across documents and generate evaluation test set.

    Args:
        documents: List of Document objects.
        llm_client: GeminiClient instance.
        output_path: Target JSON file path for test set output.
        passages_per_doc: Number of passage samples per document.
        passage_length: Target character length of each passage span.

    Returns:
        List of test set entry dictionaries.
    """
    testset: List[Dict[str, Any]] = []
    q_counter = 1

    for doc in documents:
        content_len = len(doc.content)
        if content_len == 0:
            continue

        stride = max(100, (content_len - passage_length) // max(1, passages_per_doc))
        start = 0

        while start < content_len and len([t for t in testset if t["doc_id"] == doc.doc_id]) < passages_per_doc:
            end = min(start + passage_length, content_len)
            gt_text = doc.content[start:end].strip()

            if len(gt_text) > 50:
                question_text = ""
                if llm_client is not None:
                    try:
                        question_text = generate_question_for_passage(gt_text, llm_client)
                    except Exception:
                        question_text = f"What key information is discussed in {doc.doc_id} span {start}-{end}?"
                else:
                    question_text = f"What key information is discussed in {doc.doc_id} span {start}-{end}?"

                testset.append(
                    {
                        "question_id": f"q{q_counter}",
                        "doc_id": doc.doc_id,
                        "question": question_text,
                        "gt_start_char": start,
                        "gt_end_char": end,
                        "gt_text": gt_text,
                    }
                )
                q_counter += 1

            start += stride
            if end == content_len:
                break

    target_path = output_path or TESTSET_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2)

    return testset


if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    if docs:
        client = GeminiClient()
        build_testset_from_docs(docs, llm_client=client)
        print(f"Generated test set at {TESTSET_PATH}")
