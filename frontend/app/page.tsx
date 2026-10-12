"use client";

import React from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart2,
  CheckCircle,
  FileSearch,
  Layers,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

export default function OverviewPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10">
      {/* Hero section */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-zinc-950/60 p-8 sm:p-12">
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-400 mb-6">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Production RAG Benchmarking Platform</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-3xl leading-tight">
          Engineering platform for evaluating & benchmarking RAG systems
        </h1>

        <p className="mt-4 text-base sm:text-lg text-zinc-400 max-w-2xl leading-relaxed">
          Not a chatbot wrapper. No LangChain, no LangGraph. Every stage—from heading-aware chunking
          and FAISS/BM25 indexing to Reciprocal Rank Fusion, span-based ground truth resolution, and
          CrossEncoder reranking—is hand-written and unit-tested for full mechanical transparency.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-4">
          <Link
            href="/query"
            className="flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 shadow-md transition-all hover:bg-emerald-400"
          >
            <FileSearch className="h-4 w-4" />
            Launch Query Mode
          </Link>
          <Link
            href="/evaluation"
            className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900/90 px-5 py-2.5 text-sm font-semibold text-zinc-200 transition-all hover:bg-zinc-800 hover:text-white"
          >
            <Activity className="h-4 w-4" />
            Open Evaluation Lab
          </Link>
          <Link
            href="/benchmarks"
            className="flex items-center gap-2 rounded-lg border border-zinc-800 px-5 py-2.5 text-sm font-semibold text-zinc-400 transition-all hover:border-zinc-700 hover:text-zinc-200"
          >
            <BarChart2 className="h-4 w-4" />
            View Benchmark Matrix
          </Link>
        </div>
      </div>

      {/* Two Modes, One Pipeline architecture card */}
      <div className="mt-12">
        <div className="mb-4">
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Layers className="h-5 w-5 text-emerald-400" />
            Product Shape: Two Modes, One Pipeline
          </h2>
          <p className="text-sm text-zinc-400">
            Query mode and Evaluation Lab mode call the identical retriever function—eliminating drift between benchmarks and production.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="rounded-md bg-sky-500/10 border border-sky-500/30 px-2.5 py-1 font-mono text-xs font-semibold text-sky-400">
                  Mode 1: Ad-hoc Query
                </span>
                <span className="text-xs text-zinc-500">Live Answering</span>
              </div>
              <h3 className="text-lg font-bold text-white mb-2">Live Question Answering</h3>
              <p className="text-xs text-zinc-400 leading-relaxed">
                Ask arbitrary questions against the benchmark corpus. Inspect the exact passages retrieved,
                their similarity/RRF scores, and a granular stage-by-stage latency waterfall (retrieval vs reranking vs generation).
              </p>
              <div className="mt-4 rounded-lg bg-zinc-950 p-3 border border-zinc-800/80 text-xs text-zinc-300">
                <span className="font-semibold text-amber-400">Design Rule 4:</span> No synthetic accuracy scores are fabricated for ad-hoc queries because ground truth does not exist for arbitrary questions.
              </div>
            </div>
            <Link
              href="/query"
              className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-sky-400 hover:text-sky-300"
            >
              Try live queries <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="rounded-md bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-1 font-mono text-xs font-semibold text-emerald-400">
                  Mode 2: Evaluation Lab
                </span>
                <span className="text-xs text-zinc-500">Benchmarking</span>
              </div>
              <h3 className="text-lg font-bold text-white mb-2">Systematic Retrieval Lab</h3>
              <p className="text-xs text-zinc-400 leading-relaxed">
                Run the SAME retriever against 40 labelled technical questions with verbatim text spans.
                Measures Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Hit Rate with per-question failure breakdown.
              </p>
              <div className="mt-4 rounded-lg bg-zinc-950 p-3 border border-zinc-800/80 text-xs text-zinc-300">
                <span className="font-semibold text-emerald-400">Design Rule 1:</span> Ground truth is anchored to exact text spans, dynamically resolved against the live index so re-chunking never corrupts metrics.
              </div>
            </div>
            <Link
              href="/evaluation"
              className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300"
            >
              Run evaluation sweep <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Engineering Design Principles */}
      <div className="mt-12">
        <h2 className="text-xl font-bold tracking-tight text-white mb-4 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-emerald-400" />
          Critical Engineering Decisions
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-emerald-400 mb-2">
              <CheckCircle className="h-4 w-4" />
              <span>Span-Based Ground Truth</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Chunk IDs are fragile functions of chunk size. Ground truth is anchored to verbatim character spans in documents and resolved dynamically against the live index. 0 matches or &gt;1 matches fail loudly.
            </p>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-emerald-400 mb-2">
              <CheckCircle className="h-4 w-4" />
              <span>Precision@K Structural Cap</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              When documents have 1-2 relevant chunks, Precision@5 is mathematically capped at 0.20 (1/5). We make this structural ceiling explicit in the UI, treating Recall@K and MRR as the primary metrics.
            </p>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-emerald-400 mb-2">
              <CheckCircle className="h-4 w-4" />
              <span>Opt-In CrossEncoder</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              CrossEncoder reranking (ms-marco-MiniLM-L-6-v2) boosts MRR from 0.758 to 0.938, but adds ~450ms CPU latency. It is strictly opt-in per query, not a hidden default.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
