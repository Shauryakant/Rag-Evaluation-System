"""Span-based ground truth resolution module.

Anchors ground-truth relevance to exact verbatim character spans in documents
rather than volatile chunk IDs. Evaluates against the live index and fails loudly
if a span is missing (stale label) or matches multiple chunks (non-discriminative).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from backend.chunking.splitter import Chunk
from backend.config import TESTSET_PATH


class GroundTruthResolutionError(RuntimeError):
    """Raised when an expected answer span does not resolve unambiguously against the current chunk index."""
    pass


@dataclass
class ResolvedTestCase:
    """A benchmark query paired with its resolved target chunk IDs in the current index."""

    id: str
    question: str
    document: str
    expected_answer_spans: List[str]
    relevant_chunk_ids: Set[str]


def load_testset(testset_path: Path = TESTSET_PATH) -> List[Dict]:
    """Load the labelled benchmark query dataset from disk.

    Args:
        testset_path: Path to the JSON testset file.

    Returns:
        List of raw test case dictionaries.
    """
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def resolve_ground_truth(
    chunks: List[Chunk],
    testset_data: Optional[List[Dict]] = None,
    testset_path: Path = TESTSET_PATH,
    strict: bool = True,
) -> List[ResolvedTestCase]:
    """Resolve each labelled span in the testset against the LIVE index chunks.

    CRITICAL DESIGN RULE:
    1. Zero matches = stale label (fails loudly).
    2. More than one match = non-discriminative span (fails loudly).
    3. Exactly one match per span anchors relevance reliably across chunking strategies.

    Args:
        chunks: Current list of chunks produced by the splitter.
        testset_data: Optional pre-loaded list of test items. If None, loaded from testset_path.
        testset_path: Path to testset.json if testset_data is None.
        strict: If True, raises GroundTruthResolutionError on ambiguity or missing span.

    Returns:
        List of ResolvedTestCase objects containing current relevant_chunk_ids.

    Raises:
        GroundTruthResolutionError: If strict is True and any span fails resolution.
    """
    items = testset_data if testset_data is not None else load_testset(testset_path)
    resolved_cases: List[ResolvedTestCase] = []

    # Group chunks by doc_id for fast lookup
    doc_chunk_map: Dict[str, List[Chunk]] = {}
    for c in chunks:
        doc_chunk_map.setdefault(c.doc_id, []).append(c)

    for item in items:
        q_id = item.get("id", "unknown")
        question = item.get("question", "")
        doc_name = item.get("document", "")
        spans = item.get("expected_answer_spans", [])

        if not spans:
            if strict:
                raise GroundTruthResolutionError(
                    f"Test question [{q_id}] has no expected_answer_spans specified."
                )
            continue

        candidate_chunks = doc_chunk_map.get(doc_name, [])
        if not candidate_chunks:
            if strict:
                raise GroundTruthResolutionError(
                    f"Test question [{q_id}] targets document '{doc_name}', "
                    f"but no chunks were indexed for this document."
                )
            continue

        relevant_chunk_ids: Set[str] = set()

        for raw_span in spans:
            span = raw_span.strip()
            if not span:
                continue

            matching = [c for c in candidate_chunks if span in c.text]

            if len(matching) == 0:
                if strict:
                    raise GroundTruthResolutionError(
                        f"CRITICAL EVALUATION FAILURE: Zero chunks contain expected span for question [{q_id}]!\n"
                        f"Document: '{doc_name}'\n"
                        f"Span: {span!r}\n"
                        f"Cause: Stale label or chunk boundary split the text span."
                    )
            elif len(matching) > 1:
                # If chunk overlap duplicated the exact span across two adjacent chunks,
                # check if it is non-discriminative
                matching_ids = [c.chunk_id for c in matching]
                if strict:
                    raise GroundTruthResolutionError(
                        f"CRITICAL EVALUATION FAILURE: Expected span is not discriminative for question [{q_id}]!\n"
                        f"Document: '{doc_name}'\n"
                        f"Span: {span!r}\n"
                        f"Matches {len(matching)} chunks: {matching_ids}\n"
                        f"Cause: Span appears multiple times or is encompassed in overlapping chunks."
                    )
            else:
                relevant_chunk_ids.add(matching[0].chunk_id)

        resolved_cases.append(
            ResolvedTestCase(
                id=q_id,
                question=question,
                document=doc_name,
                expected_answer_spans=spans,
                relevant_chunk_ids=relevant_chunk_ids,
            )
        )

    return resolved_cases
