"use client";

import React, { useState } from "react";
import { useRunEvaluation } from "@/hooks/useRag";
import { RetrieverType } from "@/types";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Play,
  Target,
  XCircle,
} from "lucide-react";

export default function EvaluationPage() {
  const [retriever, setRetriever] = useState<RetrieverType>("hybrid");
  const [topK, setTopK] = useState(5);
  const [useReranker, setUseReranker] = useState(false);
  const [filterMode, setFilterMode] = useState<"all" | "hits" | "misses">("all");

  const evalMutation = useRunEvaluation();

  const handleRunEvaluation = () => {
    evalMutation.mutate({
      retriever,
      top_k: topK,
      use_reranker: useReranker,
      include_details: true,
    });
  };

  const evalResult = evalMutation.data;

  const queryDetails = evalResult?.query_details || [];
  const filteredQueries = queryDetails.filter((q) => {
    if (filterMode === "hits") return q.is_hit;
    if (filterMode === "misses") return !q.is_hit;
    return true;
  });

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">
          <Activity className="h-4 w-4" />
          <span>Mode 2: Evaluation Lab</span>
        </div>
        <h1 className="mt-1 text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
          Systematic Retrieval Benchmarks
        </h1>
        <p className="mt-1 text-xs sm:text-sm text-zinc-400">
          Run the production retriever against 40 labelled questions anchored to verbatim text spans.
        </p>
      </div>

      {/* Configuration bar */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div>
            <label className="text-xs font-medium text-zinc-300 block mb-1.5">
              Retriever Strategy
            </label>
            <select
              value={retriever}
              onChange={(e) => setRetriever(e.target.value as RetrieverType)}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 p-2.5 text-xs text-zinc-200 focus:border-emerald-500 focus:outline-none"
            >
              <option value="dense">Dense (FAISS IndexFlatIP)</option>
              <option value="sparse">Sparse (BM25Okapi)</option>
              <option value="hybrid">Hybrid (Reciprocal Rank Fusion)</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-zinc-300 block mb-1.5">
              Cutoff Rank (Top K)
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 p-2.5 text-xs text-zinc-200 focus:border-emerald-500 focus:outline-none"
            >
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-zinc-300 block mb-1.5">
              CrossEncoder Reranker
            </label>
            <button
              type="button"
              onClick={() => setUseReranker(!useReranker)}
              className={`w-full flex items-center justify-between rounded-lg border p-2.5 text-xs font-medium transition-colors ${
                useReranker
                  ? "border-amber-500/60 bg-amber-500/10 text-amber-300"
                  : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-700"
              }`}
            >
              <span>{useReranker ? "CrossEncoder: ON" : "CrossEncoder: OFF"}</span>
              <Cpu className="h-3.5 w-3.5" />
            </button>
          </div>

          <div>
            <button
              onClick={handleRunEvaluation}
              disabled={evalMutation.isPending}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-500 py-2.5 text-xs font-semibold text-zinc-950 shadow transition-all hover:bg-emerald-400 disabled:opacity-50"
            >
              {evalMutation.isPending ? (
                <span>Sweeping 40 queries...</span>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>Run Evaluation Sweep</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {evalMutation.isError && (
        <div className="mb-8 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
          <span className="font-bold">Evaluation Failed:</span> {evalMutation.error.message}
        </div>
      )}

      {/* Metrics Cards */}
      {evalResult && (
        <div className="space-y-8">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {/* Recall@K */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="text-xs font-medium text-zinc-400 mb-1">Recall@{evalResult.top_k}</div>
              <div className="font-mono text-2xl font-extrabold text-emerald-400">
                {(evalResult.recall_at_k * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">Relevant passages retrieved</div>
            </div>

            {/* MRR */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="text-xs font-medium text-zinc-400 mb-1">MRR</div>
              <div className="font-mono text-2xl font-extrabold text-sky-400">
                {evalResult.mrr.toFixed(4)}
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">Mean Reciprocal Rank</div>
            </div>

            {/* Hit Rate */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="text-xs font-medium text-zinc-400 mb-1">Hit Rate</div>
              <div className="font-mono text-2xl font-extrabold text-purple-400">
                {(evalResult.hit_rate * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">Queries with &ge;1 hit</div>
            </div>

            {/* Precision@K with structural ceiling notice */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 relative">
              <div className="text-xs font-medium text-zinc-400 mb-1 flex items-center justify-between">
                <span>Precision@{evalResult.top_k}</span>
                <span className="text-[10px] text-amber-400 font-mono">max ~{(1 / evalResult.top_k).toFixed(2)}</span>
              </div>
              <div className="font-mono text-2xl font-extrabold text-amber-400">
                {evalResult.precision_at_k.toFixed(4)}
              </div>
              <div className="mt-1 text-[10px] text-zinc-500">
                Structurally capped near 1/{evalResult.top_k}
              </div>
            </div>

            {/* Average Latency */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="text-xs font-medium text-zinc-400 mb-1">Avg Latency</div>
              <div className="font-mono text-2xl font-extrabold text-white">
                {evalResult.latency_ms.toFixed(1)} <span className="text-xs font-normal text-zinc-500">ms</span>
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">Per-query execution</div>
            </div>
          </div>

          {/* Precision structural cap callout (Design Rule 3) */}
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-xs text-amber-300 flex items-start gap-3">
            <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Engineering Design Rule 3 (Precision@K Structural Ceiling):</span> In technical benchmark corpora where each question targets exactly 1 or 2 relevant passages, Precision@K is structurally capped at 1/K (e.g. max 0.20 at K=5). This is a mathematical ceiling of the metric, not a system defect. Read <strong>Recall@K</strong> and <strong>MRR</strong> as your primary optimization signals.
            </div>
          </div>

          {/* Failure and query breakdown table */}
          {queryDetails.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-zinc-800">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Target className="h-4 w-4 text-emerald-400" />
                    Query-Level Inspection & Failure Analysis
                  </h3>
                  <p className="text-xs text-zinc-400">
                    Inspecting {evalResult.total_queries} queries: {evalResult.total_queries - evalResult.failed_query_count} hits, {evalResult.failed_query_count} misses
                  </p>
                </div>

                {/* Filter buttons */}
                <div className="flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950 p-1 text-xs">
                  <button
                    onClick={() => setFilterMode("all")}
                    className={`rounded px-2.5 py-1 ${
                      filterMode === "all" ? "bg-zinc-800 text-white font-semibold" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    All ({queryDetails.length})
                  </button>
                  <button
                    onClick={() => setFilterMode("hits")}
                    className={`rounded px-2.5 py-1 ${
                      filterMode === "hits" ? "bg-emerald-500/20 text-emerald-300 font-semibold" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    Hits ({queryDetails.filter((q) => q.is_hit).length})
                  </button>
                  <button
                    onClick={() => setFilterMode("misses")}
                    className={`rounded px-2.5 py-1 ${
                      filterMode === "misses" ? "bg-rose-500/20 text-rose-300 font-semibold" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    Misses ({queryDetails.filter((q) => !q.is_hit).length})
                  </button>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-zinc-300">
                  <thead className="border-b border-zinc-800 text-[11px] uppercase tracking-wider text-zinc-500 font-mono">
                    <tr>
                      <th className="py-2.5 pr-2">Status</th>
                      <th className="py-2.5 px-2">ID</th>
                      <th className="py-2.5 px-2">Question</th>
                      <th className="py-2.5 px-2">Document</th>
                      <th className="py-2.5 px-2 font-mono">Target Chunk(s)</th>
                      <th className="py-2.5 px-2 font-mono">Retrieved</th>
                      <th className="py-2.5 px-2 font-mono">MRR</th>
                      <th className="py-2.5 pl-2 font-mono">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 font-sans">
                    {filteredQueries.map((q) => (
                      <tr key={q.query_id} className="hover:bg-zinc-800/30">
                        <td className="py-2.5 pr-2">
                          {q.is_hit ? (
                            <span className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-400 border border-emerald-500/20">
                              <CheckCircle2 className="h-3 w-3" /> Hit
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded bg-rose-500/10 px-2 py-0.5 text-[10px] font-semibold text-rose-400 border border-rose-500/20">
                              <XCircle className="h-3 w-3" /> Miss
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-2 font-mono text-[11px] text-zinc-400">{q.query_id}</td>
                        <td className="py-2.5 px-2 font-medium text-white max-w-xs truncate" title={q.question}>
                          {q.question}
                        </td>
                        <td className="py-2.5 px-2 text-zinc-400 font-mono text-[11px] truncate max-w-[120px]">
                          {q.document}
                        </td>
                        <td className="py-2.5 px-2 font-mono text-[11px] text-emerald-400/90 truncate max-w-[120px]">
                          {q.target_chunks.join(", ")}
                        </td>
                        <td className="py-2.5 px-2 font-mono text-[11px] text-zinc-400 truncate max-w-[120px]">
                          {q.retrieved_chunks.join(", ")}
                        </td>
                        <td className="py-2.5 px-2 font-mono text-[11px] text-zinc-200">
                          {q.mrr.toFixed(2)}
                        </td>
                        <td className="py-2.5 pl-2 font-mono text-[11px] text-zinc-400">
                          {q.total_ms.toFixed(1)}ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
