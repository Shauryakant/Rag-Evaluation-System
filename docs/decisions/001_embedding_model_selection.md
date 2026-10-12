# ADR-001: Embedding Model Selection

## Status
Accepted

## Context
A core goal of `rag-eval-lab` is to provide deterministic, reproducible benchmark numbers for information retrieval in RAG systems without external API dependency flakiness, rate-limiting, or recurring subscription costs. We evaluated two primary alternatives:
1. Cloud API Embeddings (e.g. OpenAI `text-embedding-3-small`, Cohere Embed v3)
2. Local Open-Source Sentence-Transformers (e.g. `BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`)

## Decision
We chose **`BAAI/bge-small-en-v1.5`** (384-dimensional dense vectors) using `sentence-transformers` with SHA-256 disk caching.

## Rationale & Consequences
- **Representation Quality:** Top tier in the MTEB (Massive Text Embedding Benchmark) leaderboard for its parameter class (~33M parameters).
- **Latency & Resource Footprint:** Produces 384-dimensional dense vectors with ~15ms inference latency per query on commodity CPU and requires <135MB memory, making it ideal for free-tier deployments (Hugging Face Spaces / Render).
- **Offline Determinism:** By pairing the model with an on-disk SHA-256 `.npy` cache, repeated runs on identical texts execute with zero compute overhead.
- **Cosine Equivalence:** When L2-normalized, Euclidean inner product reduces directly to cosine similarity, enabling exact calculation via `faiss.IndexFlatIP`.
