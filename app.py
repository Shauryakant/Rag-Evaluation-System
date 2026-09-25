"""Streamlit dashboard for rag-eval-playground."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.chunking import chunk_text
from src.config import DOCS_DIR, RESULTS_CSV, TESTSET_PATH
from src.loader import load_documents
from src.metrics import is_hit
from src.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

st.set_page_config(
    page_title="RAG Evaluation Playground",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_results_data() -> pd.DataFrame:
    """Load precomputed evaluation grid results from CSV."""
    if not RESULTS_CSV.exists():
        st.error(f"Results file not found at {RESULTS_CSV}. Please run python -m src.run_grid first.")
        return pd.DataFrame()
    return pd.read_csv(RESULTS_CSV)


@st.cache_data
def load_testset_data() -> list[dict]:
    """Load evaluation testset from JSON."""
    if not TESTSET_PATH.exists():
        return []
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.title("⚡ RAG Evaluation Playground")
    st.markdown(
        "Framework-free systematic benchmark comparing RAG chunk sizes, retrievers (Vector, BM25, Hybrid RRF), "
        "and embedding models using ground-truth character span matching and LLM-as-judge faithfulness."
    )

    df = load_results_data()
    if df.empty:
        return

    # Sidebar Filter Controls
    st.sidebar.header("🔍 Filter Configurations")

    chunk_sizes = ["All"] + [str(x) for x in sorted(df["chunk_size"].unique())]
    selected_chunk_size = st.sidebar.selectbox("Chunk Size", chunk_sizes)

    retrievers = ["All"] + sorted(df["retriever"].unique().tolist())
    selected_retriever = st.sidebar.selectbox("Retriever", retrievers)

    embedders = ["All"] + sorted(df["embedder"].unique().tolist())
    selected_embedder = st.sidebar.selectbox("Embedder", embedders)

    top_ks = ["All"] + [str(x) for x in sorted(df["top_k"].unique())]
    selected_top_k = st.sidebar.selectbox("Top K", top_ks)

    # Filter dataframe
    filtered_df = df.copy()
    if selected_chunk_size != "All":
        filtered_df = filtered_df[filtered_df["chunk_size"] == int(selected_chunk_size)]
    if selected_retriever != "All":
        filtered_df = filtered_df[filtered_df["retriever"] == selected_retriever]
    if selected_embedder != "All":
        filtered_df = filtered_df[filtered_df["embedder"] == selected_embedder]
    if selected_top_k != "All":
        filtered_df = filtered_df[filtered_df["top_k"] == int(selected_top_k)]

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Configurations", len(filtered_df))
    with col2:
        max_recall = filtered_df["recall_at_k"].max() if not filtered_df.empty else 0.0
        st.metric("Best Recall@k", f"{max_recall:.4f}")
    with col3:
        max_mrr = filtered_df["mrr"].max() if not filtered_df.empty else 0.0
        st.metric("Best MRR", f"{max_mrr:.4f}")
    with col4:
        valid_faith = filtered_df["faithfulness"].dropna()
        max_faith = valid_faith.max() if not valid_faith.empty else 0.0
        st.metric("Best Faithfulness", f"{max_faith:.4f}" if max_faith > 0 else "N/A")

    st.subheader("📊 Evaluation Grid Results")
    st.dataframe(
        filtered_df.sort_values(by=["recall_at_k", "mrr"], ascending=False),
        use_container_width=True,
    )

    # Heatmap Section: Chunk Size vs Top K
    st.subheader("🔥 Recall@k Heatmap: Chunk Size vs Top K")
    pivot_df = df.pivot_table(
        index="chunk_size",
        columns="top_k",
        values="recall_at_k",
        aggfunc="mean",
    )
    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(x="Top K", y="Chunk Size", color="Recall@k"),
        x=[str(c) for c in pivot_df.columns],
        y=[str(r) for r in pivot_df.index],
        color_continuous_scale="Blues",
        text_auto=".4f",
        title="Mean Recall@k across Chunk Sizes and Top K Values",
    )
    fig_heatmap.update_layout(xaxis_title="Top K", yaxis_title="Chunk Size")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Retriever Comparison Bar Chart Section
    st.subheader("📈 Retriever Performance Comparison (Vector vs BM25 vs Hybrid)")
    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        bar_chunk_size = st.selectbox("Select Chunk Size for Comparison", sorted(df["chunk_size"].unique()), key="bar_cs")
    with bar_col2:
        bar_top_k = st.selectbox("Select Top K for Comparison", sorted(df["top_k"].unique()), key="bar_tk")

    chart_df = df[(df["chunk_size"] == bar_chunk_size) & (df["top_k"] == bar_top_k)]
    fig_bar = px.bar(
        chart_df,
        x="retriever",
        y=["recall_at_k", "mrr"],
        barmode="group",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        labels={"value": "Score", "variable": "Metric", "retriever": "Retriever Type"},
        title=f"Retriever Performance (Chunk Size={bar_chunk_size}, Top K={bar_top_k})",
        text_auto=".4f",
    )
    fig_bar.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig_bar, use_container_width=True)

    # Question Inspector Section
    st.subheader("🔎 Single-Question Inspector")
    testset = load_testset_data()
    if testset:
        question_options = [f"{q['question_id']}: {q['question']}" for q in testset]
        selected_q_str = st.selectbox("Pick a Question to Inspect", question_options)
        selected_qid = selected_q_str.split(":")[0]
        q_data = next((q for q in testset if q["question_id"] == selected_qid), None)

        if q_data:
            st.info(f"**Ground Truth Document:** `{q_data['doc_id']}` | **Span:** `[{q_data['gt_start_char']}..{q_data['gt_end_char']}]`")
            st.markdown(f"> **Ground Truth Passage:** {q_data['gt_text']}")

            # Load docs and chunk with selected chunk size
            insp_cs = int(selected_chunk_size) if selected_chunk_size != "All" else 500
            insp_tk = int(selected_top_k) if selected_top_k != "All" else 3
            docs = load_documents(DOCS_DIR)
            chunks = []
            for d in docs:
                chunks.extend(chunk_text(d.content, d.doc_id, insp_cs, int(insp_cs * 0.10)))

            v_ret = VectorRetriever(chunks, "all-MiniLM-L6-v2")
            b_ret = BM25Retriever(chunks)
            h_ret = HybridRetriever(v_ret, b_ret)

            inspect_cols = st.columns(3)
            with inspect_cols[0]:
                st.markdown("#### 🎯 Vector Retriever")
                v_results = v_ret.retrieve(q_data["question"], top_k=insp_tk)
                for item in v_results:
                    hit = is_hit(item.chunk, q_data["doc_id"], q_data["gt_start_char"], q_data["gt_end_char"])
                    badge = "✅ HIT" if hit else "❌ MISS"
                    st.caption(f"Rank {item.rank} ({badge}) | Score: {item.score:.4f} | Span: [{item.chunk.start_char}..{item.chunk.end_char}]")
                    st.text(item.chunk.text[:200] + "...")

            with inspect_cols[1]:
                st.markdown("#### 📝 BM25 Retriever")
                b_results = b_ret.retrieve(q_data["question"], top_k=insp_tk)
                for item in b_results:
                    hit = is_hit(item.chunk, q_data["doc_id"], q_data["gt_start_char"], q_data["gt_end_char"])
                    badge = "✅ HIT" if hit else "❌ MISS"
                    st.caption(f"Rank {item.rank} ({badge}) | Score: {item.score:.4f} | Span: [{item.chunk.start_char}..{item.chunk.end_char}]")
                    st.text(item.chunk.text[:200] + "...")

            with inspect_cols[2]:
                st.markdown("#### 🔀 Hybrid RRF Retriever")
                h_results = h_ret.retrieve(q_data["question"], top_k=insp_tk)
                for item in h_results:
                    hit = is_hit(item.chunk, q_data["doc_id"], q_data["gt_start_char"], q_data["gt_end_char"])
                    badge = "✅ HIT" if hit else "❌ MISS"
                    st.caption(f"Rank {item.rank} ({badge}) | RRF Score: {item.score:.4f} | Span: [{item.chunk.start_char}..{item.chunk.end_char}]")
                    st.text(item.chunk.text[:200] + "...")


if __name__ == "__main__":
    main()
