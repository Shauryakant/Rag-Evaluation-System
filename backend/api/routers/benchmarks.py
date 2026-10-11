"""Benchmarks router serving precomputed matrix sweeps and CSV exports."""

import csv
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, Response

from backend.api.deps import AppState, get_app_state
from backend.api.schemas import BenchmarkMatrixResponse, BenchmarkRowDTO
from backend.config import BENCHMARK_RESULTS_CSV

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.get("", response_model=BenchmarkMatrixResponse)
def get_benchmarks_matrix(state: AppState = Depends(get_app_state)) -> BenchmarkMatrixResponse:
    """Retrieve full matrix sweep results across all 18 configurations."""
    if not BENCHMARK_RESULTS_CSV.exists():
        # If benchmark.csv doesn't exist yet, run sweep to populate it
        state.runner.run_full_sweep(output_csv=BENCHMARK_RESULTS_CSV)

    rows: List[BenchmarkRowDTO] = []
    with open(BENCHMARK_RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                BenchmarkRowDTO(
                    retriever=row["retriever"],
                    top_k=int(row["top_k"]),
                    reranker=row["reranker"],
                    precision_at_k=float(row["precision_at_k"]),
                    recall_at_k=float(row["recall_at_k"]),
                    mrr=float(row["mrr"]),
                    hit_rate=float(row["hit_rate"]),
                    latency_ms=float(row["latency_ms"]),
                )
            )

    return BenchmarkMatrixResponse(
        rows=rows,
        total_configurations=len(rows),
        dataset_size=len(state.test_cases),
    )


@router.get("/csv")
def download_benchmarks_csv(state: AppState = Depends(get_app_state)) -> Response:
    """Download benchmark results in raw CSV format."""
    if not BENCHMARK_RESULTS_CSV.exists():
        state.runner.run_full_sweep(output_csv=BENCHMARK_RESULTS_CSV)

    csv_data = BENCHMARK_RESULTS_CSV.read_text(encoding="utf-8")
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark.csv"},
    )
