"""Grid search execution script running all RAG configurations and writing metrics to results.csv."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.chunking import chunk_text
from src.config import (
    CHUNK_SIZES,
    DOCS_DIR,
    EMBEDDERS,
    OVERLAP_RATIO,
    RESULTS_CSV,
    RETRIEVERS,
    TESTSET_PATH,
    TOP_K_LIST,
)
from src.embed_index import get_or_build_faiss_index
from src.loader import load_documents
from src.llm import GeminiClient
from src.metrics import faithfulness_score, mrr, recall_at_k
from src.retrievers import BM25Retriever, HybridRetriever, VectorRetriever


def load_testset() -> List[Dict[str, Any]]:
    """Load test set questions and ground truth character spans."""
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_PATH}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_faithfulness_for_config(
    chunk_size: int,
    retriever_type: str,
    embedder_str: str,
    top_k: int,
    documents: List[Any],
    testset: List[Dict[str, Any]],
    llm_client: GeminiClient,
) -> float:
    """Run LLM-as-judge faithfulness scoring on a single configuration.

    Args:
        chunk_size: Chunk size.
        retriever_type: Type of retriever (vector, bm25, hybrid).
        embedder_str: Embedding model name.
        top_k: Top k retrieval parameter.
        documents: Source documents.
        testset: Evaluation testset questions.
        llm_client: GeminiClient instance.

    Returns:
        Average faithfulness score across all test set queries.
    """
    overlap = int(chunk_size * OVERLAP_RATIO)
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc.content, doc.doc_id, chunk_size, overlap))

    real_embedder = "all-MiniLM-L6-v2" if embedder_str.startswith("N/A") else embedder_str

    if retriever_type == "vector":
        retriever = VectorRetriever(chunks, real_embedder)
    elif retriever_type == "bm25":
        retriever = BM25Retriever(chunks)
    elif retriever_type == "hybrid":
        v_ret = VectorRetriever(chunks, real_embedder)
        b_ret = BM25Retriever(chunks)
        retriever = HybridRetriever(v_ret, b_ret)
    else:
        return 0.0

    scores = []
    for q in testset:
        query = q["question"]
        gt_answer = q.get("gt_text", query)
        retrieved = retriever.retrieve(query, top_k)
        retrieved_chunks = [r.chunk for r in retrieved]
        f_score = faithfulness_score(gt_answer, retrieved_chunks, llm_client=llm_client)
        scores.append(f_score)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def run_evaluation_grid() -> pd.DataFrame:
    """Execute evaluation grid search with index reuse and faithfulness scoring for top 3 configs.

    Returns:
        DataFrame containing evaluation results for all configurations.
    """
    start_time = time.time()
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise ValueError("No documents found in data/docs/")

    testset = load_testset()
    results: List[Dict[str, Any]] = []

    for chunk_size in CHUNK_SIZES:
        overlap = int(chunk_size * OVERLAP_RATIO)
        chunks = []
        for doc in documents:
            chunks.extend(chunk_text(doc.content, doc.doc_id, chunk_size, overlap))

        # Reused BM25 index across all embedder/top_k loops
        bm25_retriever = BM25Retriever(chunks)

        for embedder in EMBEDDERS:
            # Reused FAISS index across top_k loops
            faiss_index, _ = get_or_build_faiss_index(chunks, embedder)
            vector_retriever = VectorRetriever(chunks, embedder, index=faiss_index)
            hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

            for retriever_type in RETRIEVERS:
                # BM25 does not use an embedder: run it once per (chunk_size, top_k)
                if retriever_type == "bm25" and embedder != EMBEDDERS[0]:
                    continue

                for top_k in TOP_K_LIST:
                    recalls = []
                    mrrs = []

                    for q in testset:
                        query = q["question"]
                        gt_doc = q["doc_id"]
                        gt_start = q["gt_start_char"]
                        gt_end = q["gt_end_char"]

                        if retriever_type == "vector":
                            retrieved = vector_retriever.retrieve(query, top_k)
                        elif retriever_type == "bm25":
                            retrieved = bm25_retriever.retrieve(query, top_k)
                        elif retriever_type == "hybrid":
                            retrieved = hybrid_retriever.retrieve(query, top_k)
                        else:
                            continue

                        recalls.append(recall_at_k(retrieved, gt_doc, gt_start, gt_end, top_k))
                        mrrs.append(mrr(retrieved, gt_doc, gt_start, gt_end, top_k))

                    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
                    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

                    display_embedder = "N/A (Lexical)" if retriever_type == "bm25" else embedder

                    results.append(
                        {
                            "chunk_size": chunk_size,
                            "retriever": retriever_type,
                            "embedder": display_embedder,
                            "top_k": top_k,
                            "recall_at_k": round(avg_recall, 4),
                            "mrr": round(avg_mrr, 4),
                            "faithfulness": None,
                        }
                    )

    df = pd.DataFrame(results)

    # Sort configs by recall_at_k and mrr to identify top 3 configs
    df_sorted = df.sort_values(by=["recall_at_k", "mrr"], ascending=False)
    top_3_indices = df_sorted.head(3).index.tolist()

    print(f"Running faithfulness evaluation for top 3 configs (indices: {top_3_indices})...")
    llm_client = GeminiClient()

    for idx in top_3_indices:
        row = df.loc[idx]
        f_score = compute_faithfulness_for_config(
            chunk_size=int(row["chunk_size"]),
            retriever_type=str(row["retriever"]),
            embedder_str=str(row["embedder"]),
            top_k=int(row["top_k"]),
            documents=documents,
            testset=testset,
            llm_client=llm_client,
        )
        df.at[idx, "faithfulness"] = f_score

    elapsed = time.time() - start_time
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Grid search completed in {elapsed:.2f} seconds across {len(df)} configs. Saved to {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    df_results = run_evaluation_grid()
    print(f"Ran grid search across {len(df_results)} configurations. Saved to {RESULTS_CSV}")
