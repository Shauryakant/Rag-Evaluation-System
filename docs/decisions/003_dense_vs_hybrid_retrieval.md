# ADR-003: Dense vs. Hybrid Retrieval Trade-offs

## Status
Accepted

## Context
A critical question in production RAG systems is whether to use Dense retrieval alone (bi-encoder embeddings) or Hybrid retrieval (combining Dense with BM25 lexical search via Reciprocal Rank Fusion). Rather than assuming hybrid is always better or worse, we measured both across 40 labelled technical queries over our benchmark corpus.

## Measured Empirical Results
| Configuration | Top K | Reranker | Recall@K | MRR | Hit Rate | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 5 | Off | 0.8500 | 0.6904 | 0.8500 | **1.03 ms** |
| **Dense** | 5 | On | 0.9250 | **0.8458** | 0.9250 | 281.62 ms |
| **Sparse (BM25)** | 5 | Off | 0.8000 | 0.6517 | 0.8000 | **1.52 ms** |
| **Hybrid (RRF)** | 5 | Off | **0.8750** | 0.6683 | **0.8750** | **2.68 ms** |
| **Hybrid (RRF)** | 10 | Off | **0.9500** | 0.6778 | **0.9500** | **2.67 ms** |
| **Sparse (BM25)** | 10 | On | **0.9750** | **0.8644** | **0.9750** | 391.02 ms |

## Findings & Decision
1. **Dense Retrieval Efficiency:** Dense retrieval achieves 0.8500 recall at $K=5$ in ~1.0ms on CPU, making it a great low-latency baseline.
2. **Hybrid RRF Boosts Multi-topic Recall:** At $K=10$, Hybrid retrieval achieves **0.9500 Recall (95% Hit Rate)** across all 633 chunks in only **2.67ms**, combining semantic paraphrasing with exact keyword coverage.
3. **CrossEncoder Cost-Benefit:** Adding the CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) pushes MRR to **0.846–0.864** and Recall to **0.9750**, but costs **~280–390ms** of CPU time per request.
4. **Architecture Rule:** Reranking must remain strictly **opt-in** per query, allowing high-throughput low-latency pipelines to execute pure Dense or Hybrid RRF in <3ms, while precision-critical pipelines opt into the CrossEncoder stage.
