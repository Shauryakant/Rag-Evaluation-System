"""Configuration settings, path constants, and evaluation grid parameters."""

import os
from pathlib import Path
from typing import List

# Base Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DOCS_DIR: Path = DATA_DIR / "docs"
TESTSET_PATH: Path = DATA_DIR / "testset.json"
RESULTS_DIR: Path = ROOT_DIR / "results"
RESULTS_CSV: Path = RESULTS_DIR / "results.csv"
CACHE_DIR: Path = ROOT_DIR / "cache"

# Ensure runtime directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# RAG Grid Parameters
CHUNK_SIZES: List[int] = [200, 500, 1000]
OVERLAP_RATIO: float = 0.10
RETRIEVERS: List[str] = ["vector", "bm25", "hybrid"]
EMBEDDERS: List[str] = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
TOP_K_LIST: List[int] = [3, 5, 10]

# Hybrid Search / Reciprocal Rank Fusion Parameter
RRF_K: int = 60

# Reproducibility
RANDOM_SEED: int = 42
