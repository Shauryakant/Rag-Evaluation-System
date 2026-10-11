# Two-Stage Retrieval: Cross-Encoder Rerankers

Two-stage retrieval architectures decouple broad candidate recall from fine-grained semantic ranking precision.

## Bi-Encoders vs Cross-Encoders
In a bi-encoder, query and document are passed through separate transformer towers, producing independent vector embeddings compared via dot products. Bi-encoders cannot model cross-attention interactions between individual query words and document tokens.

In contrast, a cross-encoder passes the concatenated pair [CLS] Query [SEP] Passage [SEP] through a single transformer model (such as ms-marco-MiniLM-L-6-v2). Full cross-attention allows every query token to attend directly to every passage token across all transformer layers, producing significantly higher ranking fidelity.

## Operational Latency Trade-Off
Evaluating a cross-encoder on thousands of passages during a search request is computationally prohibitive. Therefore, cross-encoders operate as second-stage rerankers: a fast first-stage retriever (dense, BM25, or hybrid) fetches the top 20 candidates in under 15ms, and the cross-encoder scores and reorders only those 20 candidates, adding 50 to 150ms of CPU latency.
