# ADR-003: Dense vs. Hybrid Retrieval Trade-offs

## Status
Accepted

## Context
A critical question in production RAG systems is whether to use Dense retrieval alone (bi-encoder embeddings) or Hybrid retrieval (combining Dense with BM25 lexical search via Reciprocal Rank Fusion). Rather than assuming hybrid is always better or worse, we measured both across 40 labelled technical queries over our benchmark corpus.

## Measured Empirical Results
| Configuration | Top K | Reranker | Recall@K | MRR | Hit Rate | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 5 | Off | 0.9500 | 0.7733 | 0.9500 | **0.36 ms** |
| **Dense** | 5 | On | 0.9750 | **0.9375** | 0.9750 | 316.67 ms |
| **Sparse (BM25)** | 5 | Off | 0.9750 | 0.8938 | 0.9750 | **0.38 ms** |
| **Hybrid (RRF)** | 5 | Off | **0.9750** | 0.8250 | **0.9750** | **1.39 ms** |
| **Hybrid (RRF)** | 10 | Off | **1.0000** | 0.8275 | **1.0000** | **1.42 ms** |

## Findings & Decision
1. **Dense Retrieval Default:** Dense retrieval achieves strong recall (0.9500 at k=5) in sub-millisecond CPU time (0.36ms). It serves as an ultra-fast default for conversational semantic queries.
2. **Hybrid RRF Eliminates Outliers:** At $k=10$, Hybrid retrieval achieved **1.0000 Recall (100% Hit Rate)** across all 40 queries in just **1.42ms**, successfully recovering exact acronyms and code identifiers that bi-encoders occasionally miss.
3. **CrossEncoder Cost-Benefit:** Adding the CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) pushed Dense MRR from 0.7733 to 0.9375, but cost **~316-485ms** of CPU time per request.
4. **Architecture Rule:** Reranking must remain strictly **opt-in** per query, allowing high-throughput low-latency applications to run pure Dense or Hybrid RRF in <1.5ms, while precision-critical pipelines can opt into the CrossEncoder stage.
