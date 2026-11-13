"""FastAPI backend application for rag-eval-lab."""

import gc
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
# Set PyTorch to 1 CPU thread to prevent memory amplification on 512MB cloud tiers
torch.set_num_threads(1)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.deps import init_app_state
from backend.api.routers import benchmarks, evaluation, health, query, session
from backend.config import ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context initializing corpus and indices asynchronously on server boot."""
    print("[rag-eval-lab] Starting backend server...")
    threading.Thread(target=init_app_state, daemon=True).start()
    yield
    print("[rag-eval-lab] Shutting down backend server.")


app = FastAPI(
    title="rag-eval-lab API",
    description="Engineering platform for building, evaluating, and benchmarking RAG retrieval and generation pipelines.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for Next.js frontend (local development & production Vercel)
origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins + ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(evaluation.router)
app.include_router(benchmarks.router)
app.include_router(session.router)


@app.get("/")
def root():
    """Root status endpoint."""
    return {
        "app": "rag-eval-lab",
        "status": "online",
        "docs": "/docs",
    }
