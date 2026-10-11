# RAG System Architecture: Foundations and Pipelines

Retrieval-Augmented Generation (RAG) is an architectural pattern that grounds Large Language Model (LLM) responses on external knowledge retrieved dynamically from indexed document stores. Instead of relying purely on parameter weights frozen during pre-training, a RAG system fetches relevant context passages matching a user query and prepends them to the generation prompt.

## Core Pipeline Stages
A standard enterprise RAG pipeline consists of five decoupled stages:
1. Ingestion and parsing: Extracting raw text from documents (.md, .txt, .pdf) while preserving document metadata and character offsets.
2. Text segmentation (chunking): Partitioning long documents into smaller semantic units with boundary overlap.
3. Indexing: Creating representation indices—typically a dense vector index for semantic proximity and an inverted index for exact keyword matching.
4. Retrieval and candidate ranking: Querying the indices to gather candidate passages, optionally followed by a cross-encoder reranker.
5. Generation: Augmenting a system prompt with retrieved context and executing LLM completion.

## Architectural Trade-Offs
Naive RAG pipelines suffer from failure modes including semantic drift, retrieval of distracting noise passages, and lost-in-the-middle positioning effects. Advanced RAG mitigates this through two-stage retrieval where high-recall candidate generation is decoupled from precision-oriented cross-encoder reranking.
