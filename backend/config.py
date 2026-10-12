"""Configuration settings and environment variable definitions for rag-eval-lab."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = BASE_DIR.parent

# Load environment variables from project root and backend directories
load_dotenv(dotenv_path=ROOT_DIR / ".env")
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()
DATA_DIR: Path = BASE_DIR / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
TESTSET_PATH: Path = DATA_DIR / "testset.json"
CACHE_DIR: Path = BASE_DIR / "cache"
RESULTS_DIR: Path = BASE_DIR / "results"
BENCHMARK_RESULTS_CSV: Path = RESULTS_DIR / "benchmark.csv"

# Runtime directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking defaults
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50

# Embedding & Reranking Models
EMBEDDER_MODEL_NAME: str = os.getenv("EMBEDDER_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reciprocal Rank Fusion constant
RRF_K: int = 60

# Generation (Groq)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

# CORS Allowed Origins
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS: List[str] = [origin.strip() for origin in _allowed_origins_raw.split(",") if origin.strip()]
