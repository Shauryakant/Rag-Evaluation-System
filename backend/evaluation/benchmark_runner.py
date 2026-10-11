"""Benchmark runner sweeping retriever configurations across the labelled benchmark dataset."""

import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.chunking.splitter import Chunk, RecursiveHeadingSplitter
from backend.config import BENCHMARK_RESULTS_CSV, RESULTS_DIR
from backend.evaluation.ground_truth import ResolvedTestCase, resolve_ground_truth
from backend.evaluation.metrics import hit_rate, mrr, precision_at_k, recall_at_k
from backend.ingestion.loader import Document, load_corpus
from backend.retrieval.service import RetrievalService, RetrieverType


@dataclass
class QueryEvalDetail:
    """Detailed evaluation result for a single test query under a specific configuration."""

    query_id: str
    question: str
    document: str
    target_chunks: List[str]
    retrieved_chunks: List[str]
    precision: float
    recall: float
    mrr: float
    hit_rate: float
    retrieval_ms: float
    reranker_ms: float
    total_ms: float
    is_hit: bool


@dataclass
class BenchmarkRunResult:
    """Aggregate benchmark metrics for a single retriever × top_k × reranker configuration."""

    retriever: str
    top_k: int
    reranker: bool
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    latency_ms: float
    total_queries: int
    failed_query_count: int
    query_details: Optional[List[QueryEvalDetail]] = None


class BenchmarkRunner:
    """Executes systematic retrieval sweeps over the benchmark testset."""

    def __init__(
        self,
        service: Optional[RetrievalService] = None,
        test_cases: Optional[List[ResolvedTestCase]] = None,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            service: Pre-configured RetrievalService. If None, corpus is loaded and indexed.
            test_cases: Pre-resolved benchmark test cases. If None, loaded and resolved from disk.
        """
        if service is None or test_cases is None:
            documents = load_corpus()
            splitter = RecursiveHeadingSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            self.service = service or RetrievalService(chunks=chunks)
            self.test_cases = test_cases or resolve_ground_truth(chunks=chunks, strict=True)
        else:
            self.service = service
            self.test_cases = test_cases

    def evaluate_configuration(
        self,
        retriever_type: RetrieverType,
        top_k: int,
        use_reranker: bool,
        include_details: bool = False,
    ) -> BenchmarkRunResult:
        """Evaluate a single configuration across all benchmark queries.

        Args:
            retriever_type: 'dense' | 'sparse' | 'hybrid'.
            top_k: Cutoff rank k.
            use_reranker: Opt-in cross-encoder reranker.
            include_details: If True, include per-query evaluation breakdown.

        Returns:
            BenchmarkRunResult containing aggregate metrics and query failure breakdown.
        """
        precisions: List[float] = []
        recalls: List[float] = []
        mrrs: List[float] = []
        hit_rates: List[float] = []
        latencies: List[float] = []
        details: List[QueryEvalDetail] = []
        failed_count = 0

        for case in self.test_cases:
            retrieved, latency_map = self.service.retrieve(
                query=case.question,
                retriever_type=retriever_type,
                top_k=top_k,
                use_reranker=use_reranker,
            )

            retrieved_ids = [item.chunk.chunk_id for item in retrieved]
            p = precision_at_k(retrieved_ids, case.relevant_chunk_ids, k=top_k)
            r = recall_at_k(retrieved_ids, case.relevant_chunk_ids, k=top_k)
            m = mrr(retrieved_ids, case.relevant_chunk_ids, k=top_k)
            h = hit_rate(retrieved_ids, case.relevant_chunk_ids, k=top_k)

            precisions.append(p)
            recalls.append(r)
            mrrs.append(m)
            hit_rates.append(h)
            latencies.append(latency_map.get("total_ms", 0.0))

            if h == 0.0:
                failed_count += 1

            if include_details:
                details.append(
                    QueryEvalDetail(
                        query_id=case.id,
                        question=case.question,
                        document=case.document,
                        target_chunks=sorted(list(case.relevant_chunk_ids)),
                        retrieved_chunks=retrieved_ids,
                        precision=p,
                        recall=r,
                        mrr=m,
                        hit_rate=h,
                        retrieval_ms=latency_map.get("retrieval_ms", 0.0),
                        reranker_ms=latency_map.get("reranker_ms", 0.0),
                        total_ms=latency_map.get("total_ms", 0.0),
                        is_hit=(h > 0.0),
                    )
                )

        n = len(self.test_cases) or 1
        return BenchmarkRunResult(
            retriever=retriever_type,
            top_k=top_k,
            reranker=use_reranker,
            precision_at_k=round(sum(precisions) / n, 4),
            recall_at_k=round(sum(recalls) / n, 4),
            mrr=round(sum(mrrs) / n, 4),
            hit_rate=round(sum(hit_rates) / n, 4),
            latency_ms=round(sum(latencies) / n, 2),
            total_queries=len(self.test_cases),
            failed_query_count=failed_count,
            query_details=details if include_details else None,
        )

    def run_full_sweep(
        self,
        retrievers: Optional[List[RetrieverType]] = None,
        top_ks: Optional[List[int]] = None,
        rerankers: Optional[List[bool]] = None,
        output_csv: Path = BENCHMARK_RESULTS_CSV,
    ) -> List[BenchmarkRunResult]:
        """Execute a full matrix sweep across retriever × top_k × reranker and write CSV.

        Args:
            retrievers: List of retrievers to evaluate (default: ['dense', 'sparse', 'hybrid']).
            top_ks: List of top_k values (default: [3, 5, 10]).
            rerankers: List of reranker options (default: [False, True]).
            output_csv: Target CSV file path.

        Returns:
            List of BenchmarkRunResult for each configuration.
        """
        eval_retrievers = retrievers or ["dense", "sparse", "hybrid"]
        eval_top_ks = top_ks or [3, 5, 10]
        eval_rerankers = rerankers or [False, True]

        results: List[BenchmarkRunResult] = []

        total_configs = len(eval_retrievers) * len(eval_top_ks) * len(eval_rerankers)
        print(f"Starting benchmark sweep over {total_configs} configurations...")

        for r_type in eval_retrievers:
            for k in eval_top_ks:
                for use_rerank in eval_rerankers:
                    cfg_label = f"retriever={r_type}, top_k={k}, reranker={'on' if use_rerank else 'off'}"
                    print(f"  Evaluating: {cfg_label}...", flush=True)
                    res = self.evaluate_configuration(
                        retriever_type=r_type,
                        top_k=k,
                        use_reranker=use_rerank,
                        include_details=False,
                    )
                    results.append(res)
                    print(
                        f"    -> Recall@{k}: {res.recall_at_k:.4f} | MRR: {res.mrr:.4f} | Latency: {res.latency_ms:.1f}ms",
                        flush=True,
                    )

        # Write results to CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "retriever",
                    "top_k",
                    "reranker",
                    "precision_at_k",
                    "recall_at_k",
                    "mrr",
                    "hit_rate",
                    "latency_ms",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.retriever,
                        r.top_k,
                        "on" if r.reranker else "off",
                        f"{r.precision_at_k:.4f}",
                        f"{r.recall_at_k:.4f}",
                        f"{r.mrr:.4f}",
                        f"{r.hit_rate:.4f}",
                        f"{r.latency_ms:.2f}",
                    ]
                )

        print(f"Benchmark results successfully written to {output_csv}")
        return results


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_full_sweep()
