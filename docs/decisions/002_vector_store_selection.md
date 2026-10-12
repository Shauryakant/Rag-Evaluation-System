# ADR-002: Vector Index Choice (FAISS IndexFlatIP)

## Status
Accepted

## Context
Production vector stores range from approximate nearest neighbor (ANN) graph algorithms (HNSW, ScaNN) and managed cloud services (Pinecone, Qdrant, Weaviate) to exact exhaustive flat indices (FAISS `IndexFlatIP` / `IndexFlatL2`).

When evaluating RAG retrieval algorithms on benchmark corpora (dozens to thousands of chunks), approximation algorithms introduce hyperparameter-dependent recall errors ($ef\_search$, $M$) that confound evaluation of the underlying embedding representation.

## Decision
We selected **`faiss-cpu`** using **`IndexFlatIP`** (Exact Inner Product over L2-normalized vectors).

## Rationale & Consequences
- **Mathematical Exactness:** Because embeddings from `bge-small-en-v1.5` are L2-normalized ($\|\mathbf{u}\|_2 = 1$), the inner product is mathematically identical to cosine similarity:
  $$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta) = \cos(\theta)$$
- **Zero Quantization / Recall Degradation:** `IndexFlatIP` performs exhaustive search without approximation, ensuring that retrieval metrics reflect pure representation quality without index approximation noise.
- **Ultra-low CPU Latency:** For benchmark-scale corpora, matrix-vector multiplication via FAISS flat indices executes in sub-millisecond time (<1.0ms) on CPU.
- **Trade-off:** For scale beyond $\sim 10^5$ passages, flat indices scale as $\mathcal{O}(N)$. Production systems at scale would transition to `IndexHNSWFlat` or `IndexIVFFlat`, but `IndexFlatIP` serves as the gold-standard ground truth reference.
