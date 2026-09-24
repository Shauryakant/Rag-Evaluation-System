# RAG Eval Playground (`rag-eval-playground`)

A lightweight, framework-free evaluation tool that systematically compares different Retrieval-Augmented Generation (RAG) configurations across documents and scores retrieval metrics (Recall@k, MRR) and LLM faithfulness judging.

## Features
- **Chunking Sizes**: 200, 500, 1000 characters (10% sliding overlap)
- **Retrievers**: Vector Search (FAISS), BM25, and Hybrid (Reciprocal Rank Fusion)
- **Embeddings**: `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`
- **Evaluation**: Ground-truth character span overlap matching, Recall@k, MRR, Faithfulness LLM-as-judge.

*Work in progress.*
