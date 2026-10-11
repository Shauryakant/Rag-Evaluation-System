"""Application state and shared dependency injection providers."""

from dataclasses import dataclass
from typing import List, Optional

from backend.chunking.splitter import Chunk, RecursiveHeadingSplitter
from backend.evaluation.benchmark_runner import BenchmarkRunner
from backend.evaluation.ground_truth import ResolvedTestCase, resolve_ground_truth
from backend.generation.groq_generator import GroqGenerator
from backend.ingestion.loader import Document, load_corpus
from backend.retrieval.service import RetrievalService


@dataclass
class AppState:
    """Holds long-lived application state, index caches, and pipelines."""

    documents: List[Document]
    chunks: List[Chunk]
    service: RetrievalService
    runner: BenchmarkRunner
    generator: GroqGenerator
    test_cases: List[ResolvedTestCase]


_APP_STATE: Optional[AppState] = None


def init_app_state() -> AppState:
    """Initialize corpus documents, chunks, retrieval services, and runner."""
    global _APP_STATE
    if _APP_STATE is None:
        print("[API] Initializing application state: loading corpus...")
        documents = load_corpus()
        print(f"[API] Loaded {len(documents)} documents. Splitting into chunks...")
        splitter = RecursiveHeadingSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        print(f"[API] Produced {len(chunks)} chunks. Initializing retrieval service...")
        service = RetrievalService(chunks=chunks)
        print("[API] Resolving ground truth testcases...")
        test_cases = resolve_ground_truth(chunks=chunks, strict=True)
        print(f"[API] Resolved {len(test_cases)} benchmark test cases. Setting up runner...")
        runner = BenchmarkRunner(service=service, test_cases=test_cases)
        generator = GroqGenerator()
        _APP_STATE = AppState(
            documents=documents,
            chunks=chunks,
            service=service,
            runner=runner,
            generator=generator,
            test_cases=test_cases,
        )
        print("[API] Application state successfully initialized.")
    return _APP_STATE


def get_app_state() -> AppState:
    """Retrieve singleton application state."""
    if _APP_STATE is None:
        return init_app_state()
    return _APP_STATE
