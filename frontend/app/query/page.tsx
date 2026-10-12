"use client";

import React, { useState } from "react";
import ChunkCard from "@/components/ChunkCard";
import LatencyWaterfall from "@/components/LatencyWaterfall";
import { useSubmitQuery } from "@/hooks/useRag";
import { RetrieverType } from "@/types";
import {
  AlertCircle,
  Bot,
  Cpu,
  FileSearch,
  Send,
  Sliders,
} from "lucide-react";

const SAMPLE_QUERIES = [
  "In what country is Normandy located and who were the Normans?",
  "What is the principle of operation of a reciprocating steam engine?",
  "What role does oxygen play in cellular respiration and combustion?",
  "How is computational complexity theory distinguished from computability theory?",
];

export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [retriever, setRetriever] = useState<RetrieverType>("dense");
  const [topK, setTopK] = useState(5);
  const [useReranker, setUseReranker] = useState(false);

  const queryMutation = useSubmitQuery();

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    queryMutation.mutate({
      query: query.trim(),
      retriever,
      top_k: topK,
      use_reranker: useReranker,
    });
  };

  const handleSelectSample = (sampleText: string) => {
    setQuery(sampleText);
    queryMutation.mutate({
      query: sampleText,
      retriever,
      top_k: topK,
      use_reranker: useReranker,
    });
  };

  const result = queryMutation.data;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">
          <FileSearch className="h-4 w-4" />
          <span>Mode 1: Ad-hoc Query Pipeline</span>
        </div>
        <h1 className="mt-1 text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
          Live Corpus Question Answering
        </h1>
        <p className="mt-1 text-xs sm:text-sm text-zinc-400">
          Query the indexed benchmark corpus. Inspect retrieved passages, relevance scores, and stage-by-stage latency waterfall.
        </p>
      </div>

      {/* Critical Design Rule Banner */}
      <div className="mb-8 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200 flex items-start gap-3">
        <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold">Engineering Design Rule 4:</span> No accuracy metrics are fabricated here. Because ad-hoc queries do not have predefined ground-truth labels, accuracy metrics (Precision, Recall, MRR) appear exclusively in the{" "}
          <a href="/evaluation" className="underline font-semibold hover:text-white">
            Evaluation Lab
          </a>{" "}
          over the labelled benchmark set.
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Form & Configuration */}
        <div className="lg:col-span-5 space-y-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
              <Sliders className="h-4 w-4 text-emerald-400" />
              Retriever Configuration
            </h3>

            {/* Strategy selector */}
            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-zinc-300 block mb-2">
                  Retrieval Strategy
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(["dense", "sparse", "hybrid"] as RetrieverType[]).map((type) => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setRetriever(type)}
                      className={`rounded-lg py-2 text-xs font-medium capitalize border transition-all ${
                        retriever === type
                          ? "border-emerald-500/80 bg-emerald-500/15 text-emerald-300 font-semibold shadow-sm"
                          : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] text-zinc-500">
                  {retriever === "dense" && "FAISS IndexFlatIP (BAAI/bge-small cosine similarity)"}
                  {retriever === "sparse" && "BM25Okapi lexical term frequency & length normalization"}
                  {retriever === "hybrid" && "Reciprocal Rank Fusion (k=60) combining Dense + BM25"}
                </p>
              </div>

              {/* Top K slider */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="font-medium text-zinc-300">Cutoff Rank (Top K)</span>
                  <span className="font-mono font-bold text-emerald-400">{topK} chunks</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-emerald-500 bg-zinc-800"
                />
              </div>

              {/* Reranker toggle */}
              <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3.5 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-200">
                    <Cpu className="h-3.5 w-3.5 text-amber-400" />
                    <span>CrossEncoder Reranker</span>
                  </div>
                  <p className="text-[11px] text-zinc-400 mt-0.5">
                    ms-marco-MiniLM-L-6-v2 (~450ms latency)
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setUseReranker(!useReranker)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ${
                    useReranker ? "bg-amber-500" : "bg-zinc-800"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      useReranker ? "translate-x-4" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Sample questions */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-3">
              Benchmark Sample Queries
            </h4>
            <div className="space-y-2">
              {SAMPLE_QUERIES.map((sq, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleSelectSample(sq)}
                  className="w-full text-left rounded-lg border border-zinc-800/80 bg-zinc-950/70 p-2.5 text-xs text-zinc-300 transition-colors hover:border-zinc-700 hover:text-white"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Query Input & Results */}
        <div className="lg:col-span-7 space-y-6">
          {/* Query input card */}
          <form onSubmit={handleSubmit} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a technical question against the benchmark corpus..."
                rows={3}
                className="w-full resize-none rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-sm text-zinc-100 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
              <div className="mt-3 flex items-center justify-between">
                <span className="text-[11px] text-zinc-500">
                  Retriever: <span className="font-mono text-zinc-300">{retriever}</span> • Top K:{" "}
                  <span className="font-mono text-zinc-300">{topK}</span> • Reranker:{" "}
                  <span className="font-mono text-zinc-300">{useReranker ? "on" : "off"}</span>
                </span>
                <button
                  type="submit"
                  disabled={queryMutation.isPending || !query.trim()}
                  className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-4 py-2 text-xs font-semibold text-zinc-950 shadow transition-all hover:bg-emerald-400 disabled:opacity-50"
                >
                  {queryMutation.isPending ? (
                    <span>Executing...</span>
                  ) : (
                    <>
                      <span>Execute Query</span>
                      <Send className="h-3.5 w-3.5" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* Results section */}
          {queryMutation.isError && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
              <span className="font-bold">Error:</span> {queryMutation.error.message}
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Latency Waterfall telemetry */}
              <LatencyWaterfall latency={result.latency} />

              {/* LLM Synthesized Answer */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-5">
                <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-800">
                  <div className="flex items-center gap-2">
                    <Bot className="h-4 w-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">Synthesized Grounded Answer</h3>
                  </div>
                  <div className="font-mono text-[11px] text-zinc-400">
                    Tokens: {result.tokens.total_tokens} (Prompt: {result.tokens.prompt_tokens}, Completion: {result.tokens.completion_tokens})
                  </div>
                </div>
                <div className="text-sm leading-relaxed text-zinc-200 whitespace-pre-wrap font-sans">
                  {result.answer}
                </div>
              </div>

              {/* Retrieved Chunks */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-300">
                    Retrieved Chunks ({result.retrieved_chunks.length})
                  </h4>
                  <span className="text-[11px] text-zinc-500">
                    Ranked by {result.use_reranker ? "CrossEncoder score" : `${result.retriever} score`}
                  </span>
                </div>
                <div className="space-y-3">
                  {result.retrieved_chunks.map((chunk) => (
                    <ChunkCard key={chunk.chunk_id} chunk={chunk} />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
