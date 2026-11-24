"use client";

import React, { useEffect, useRef, useState } from "react";
import ChunkCard from "@/components/ChunkCard";
import LatencyWaterfall from "@/components/LatencyWaterfall";
import {
  useCompareSessionConfigs,
  useDeleteSession,
  useQuerySession,
  useSessionDetail,
  useUploadDocumentSession,
} from "@/hooks/useRag";
import { RetrieverType, SessionUploadResponse } from "@/types";
import {
  AlertCircle,
  Bot,
  Columns2,
  Cpu,
  FileSearch,
  FileText,
  FileUp,
  Layers,
  Play,
  Send,
  Sliders,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";

export default function PdfLabPage() {
  const [activeTab, setActiveTab] = useState<"single" | "compare">("single");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionMeta, setSessionMeta] = useState<SessionUploadResponse | null>(null);

  // Upload parameters
  const [chunkSize, setChunkSize] = useState<number>(500);
  const [chunkOverlap, setChunkOverlap] = useState<number>(50);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showChunksModal, setShowChunksModal] = useState<boolean>(false);

  // Single Playground State
  const [query, setQuery] = useState<string>("");
  const [retriever, setRetriever] = useState<RetrieverType>("hybrid");
  const [topK, setTopK] = useState<number>(5);
  const [useReranker, setUseReranker] = useState<boolean>(false);

  // Comparative Mode State
  const [compareQuery, setCompareQuery] = useState<string>("");
  const [configA, setConfigA] = useState<{
    label: string;
    retriever: RetrieverType;
    top_k: number;
    use_reranker: boolean;
  }>({
    label: "Config A (Fast Sparse)",
    retriever: "sparse",
    top_k: 3,
    use_reranker: false,
  });
  const [configB, setConfigB] = useState<{
    label: string;
    retriever: RetrieverType;
    top_k: number;
    use_reranker: boolean;
  }>({
    label: "Config B (Deep Hybrid + Reranker)",
    retriever: "hybrid",
    top_k: 5,
    use_reranker: true,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Hooks
  const uploadMutation = useUploadDocumentSession();
  const deleteMutation = useDeleteSession();
  const queryMutation = useQuerySession(sessionId || "");
  const compareMutation = useCompareSessionConfigs(sessionId || "");
  const { data: sessionDetail } = useSessionDetail(sessionId);

  // Hydrate session from localStorage on mount
  useEffect(() => {
    try {
      const savedSession = localStorage.getItem("rag_eval_pdf_session");
      if (savedSession) {
        const parsed = JSON.parse(savedSession);
        setSessionId(parsed.session_id);
        setSessionMeta(parsed);
      }
    } catch {
      localStorage.removeItem("rag_eval_pdf_session");
    }
  }, []);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;

    uploadMutation.mutate(
      {
        file: selectedFile,
        chunkSize,
        chunkOverlap,
      },
      {
        onSuccess: (data) => {
          setSessionId(data.session_id);
          setSessionMeta(data);
          localStorage.setItem("rag_eval_pdf_session", JSON.stringify(data));
          setSelectedFile(null);
          // Set initial suggestion if available
          if (data.sample_questions && data.sample_questions.length > 0) {
            setQuery(data.sample_questions[0]);
            setCompareQuery(data.sample_questions[0]);
          }
        },
      }
    );
  };

  const handleClearSession = () => {
    if (sessionId) {
      deleteMutation.mutate(sessionId);
    }
    setSessionId(null);
    setSessionMeta(null);
    localStorage.removeItem("rag_eval_pdf_session");
  };

  const handleSingleQuery = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || !sessionId) return;

    queryMutation.mutate({
      query: query.trim(),
      retriever,
      top_k: topK,
      use_reranker: useReranker,
    });
  };

  const handleCompareSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!compareQuery.trim() || !sessionId) return;

    compareMutation.mutate({
      query: compareQuery.trim(),
      config_a: configA,
      config_b: configB,
    });
  };

  const singleResult = queryMutation.data;
  const compareResult = compareMutation.data;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">
            <FileUp className="h-4 w-4" />
            <span>Document Session & Interactive Evaluation</span>
          </div>
          <h1 className="mt-1 text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
            Custom PDF Evaluation Lab
          </h1>
          <p className="mt-1 text-xs sm:text-sm text-zinc-400">
            Upload custom documents to test retrieval architectures, latency waterfalls, and comparative A/B configurations with ephemeral in-memory sessions.
          </p>
        </div>

        {/* Database Session Concept Note */}
        <div className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-2 text-xs text-zinc-300">
          <Zap className="h-4 w-4 text-amber-400 shrink-0" />
          <span>
            <strong className="text-white">Zero DB Overhead:</strong> Ephemeral in-memory FAISS & BM25 indices with automatic TTL garbage collection.
          </span>
        </div>
      </div>

      {/* Upload Screen if no active session */}
      {!sessionId ? (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 sm:p-10 mb-8 backdrop-blur-sm">
          <div className="max-w-2xl mx-auto text-center mb-8">
            <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 mb-4">
              <FileUp className="h-7 w-7" />
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white">
              Upload Document to Start Evaluation Session
            </h2>
            <p className="mt-2 text-xs sm:text-sm text-zinc-400 leading-relaxed">
              Upload a <strong>PDF, Markdown (.md), or Plaintext (.txt)</strong> document. We&apos;ll split it with heading-aware boundaries, encode it with BGE-small, and build isolated FAISS vector & BM25 search indices in memory.
            </p>
          </div>

          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-8 sm:p-12 text-center transition-all ${
              isDragging
                ? "border-emerald-500 bg-emerald-500/10"
                : selectedFile
                ? "border-emerald-500/60 bg-zinc-900/90"
                : "border-zinc-800 bg-zinc-950/60 hover:border-zinc-700 hover:bg-zinc-900/30"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.md"
              onChange={handleFileSelect}
              className="hidden"
            />
            {selectedFile ? (
              <div className="flex flex-col items-center gap-2">
                <FileText className="h-10 w-10 text-emerald-400" />
                <span className="font-semibold text-white text-sm">{selectedFile.name}</span>
                <span className="text-xs text-zinc-400 font-mono">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </span>
                <span className="mt-2 inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400 border border-emerald-500/30">
                  Ready to ingest
                </span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-zinc-400">
                <FileUp className="h-10 w-10 text-zinc-500" />
                <p className="text-sm font-medium text-zinc-300">
                  Drop your PDF here, or <span className="text-emerald-400 underline">browse files</span>
                </p>
                <p className="text-xs text-zinc-500">
                  Supports PDF (with selectable text), Markdown (.md), and TXT
                </p>
              </div>
            )}
          </div>

          {/* Advanced Chunking Configuration */}
          <div className="mt-6 rounded-xl border border-zinc-800/80 bg-zinc-950/40 p-4 max-w-2xl mx-auto">
            <h4 className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5 mb-3">
              <Sliders className="h-3.5 w-3.5 text-emerald-400" />
              Chunking Configuration (Heading-Aware Splitter)
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-zinc-400">Chunk Size (chars):</span>
                  <span className="font-mono font-bold text-white">{chunkSize}</span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="1200"
                  step="50"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-zinc-400">Chunk Overlap (chars):</span>
                  <span className="font-mono font-bold text-white">{chunkOverlap}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="200"
                  step="10"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(Number(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Upload Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleUploadSubmit}
              disabled={!selectedFile || uploadMutation.isPending}
              className={`flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-bold text-white shadow-lg transition-all ${
                !selectedFile || uploadMutation.isPending
                  ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                  : "bg-emerald-600 hover:bg-emerald-500 shadow-emerald-950/40"
              }`}
            >
              {uploadMutation.isPending ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  <span>Ingesting, Embedding & Indexing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Build In-Memory Document Session</span>
                </>
              )}
            </button>
          </div>

          {uploadMutation.isError && (
            <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300 flex items-center gap-2 max-w-xl mx-auto">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
              <span>{uploadMutation.error?.message || "Failed to process document."}</span>
            </div>
          )}
        </div>
      ) : (
        /* Active Document Session View */
        <div className="space-y-6 mb-8">
          {/* Active Session Info Bar */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                <FileText className="h-5 w-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-white">
                    {sessionMeta?.filename || sessionDetail?.filename || "Uploaded Document"}
                  </h3>
                  <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-400 border border-emerald-500/20">
                    Live Session
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400 font-mono mt-0.5">
                  <span>Pages: {sessionMeta?.page_count ?? sessionDetail?.page_count ?? 1}</span>
                  <span>•</span>
                  <span>
                    Chars:{" "}
                    {(sessionMeta?.char_count ?? sessionDetail?.char_count ?? 0).toLocaleString()}
                  </span>
                  <span>•</span>
                  <span>
                    Chunks: {sessionMeta?.chunk_count ?? sessionDetail?.chunk_count ?? 0}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowChunksModal(!showChunksModal)}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-900 hover:text-white transition-colors"
              >
                <Layers className="h-3.5 w-3.5 text-emerald-400" />
                <span>{showChunksModal ? "Hide Chunks" : "Inspect Chunks"}</span>
              </button>

              <button
                onClick={handleClearSession}
                className="flex items-center gap-1.5 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-400 hover:bg-rose-500/20 transition-colors"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear Session</span>
              </button>
            </div>
          </div>

          {/* Inspect Chunks Accordion */}
          {showChunksModal && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-white flex items-center gap-2">
                  <Layers className="h-4 w-4 text-emerald-400" />
                  <span>All Indexed Chunks ({sessionDetail?.chunks?.length ?? sessionMeta?.chunk_count})</span>
                </h4>
                <span className="text-[11px] text-zinc-400">
                  Exact character spans resolved for retrieval evaluation
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-96 overflow-y-auto pr-2">
                {(sessionDetail?.chunks || sessionMeta?.preview_chunks || []).map((chunk, idx) => (
                  <div
                    key={chunk.chunk_id || idx}
                    className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-xs"
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mb-1">
                      <span className="text-emerald-400 font-bold">{chunk.chunk_id}</span>
                      <span>[{chunk.start_char}:{chunk.end_char}]</span>
                    </div>
                    <p className="text-zinc-300 line-clamp-3 font-sans leading-relaxed">
                      {chunk.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Mode Switcher Tabs */}
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <button
              onClick={() => setActiveTab("single")}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all ${
                activeTab === "single"
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
              }`}
            >
              <FileSearch className="h-4 w-4" />
              <span>Interactive Query Mode</span>
            </button>

            <button
              onClick={() => setActiveTab("compare")}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all ${
                activeTab === "compare"
                  ? "bg-purple-500/10 text-purple-400 border border-purple-500/30"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
              }`}
            >
              <Columns2 className="h-4 w-4" />
              <span>A/B Configuration Comparison Mode</span>
            </button>
          </div>

          {/* Suggested Sample Questions */}
          {((sessionMeta?.sample_questions && sessionMeta.sample_questions.length > 0) ||
            (sessionDetail?.sample_questions && sessionDetail.sample_questions.length > 0)) && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-zinc-400 flex items-center gap-1">
                <Sparkles className="h-3 w-3 text-amber-400" />
                Suggested Queries:
              </span>
              {(sessionMeta?.sample_questions || sessionDetail?.sample_questions || []).map(
                (sq, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      if (activeTab === "single") setQuery(sq);
                      else setCompareQuery(sq);
                    }}
                    className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-[11px] text-zinc-300 hover:border-emerald-500/40 hover:text-white transition-colors"
                  >
                    {sq}
                  </button>
                )
              )}
            </div>
          )}

          {/* TAB 1: Single Config Playground */}
          {activeTab === "single" && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* Controls Column */}
              <div className="lg:col-span-5 space-y-6">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
                    <Sliders className="h-4 w-4 text-emerald-400" />
                    Retriever Parameters
                  </h3>

                  <div className="space-y-4">
                    {/* Strategy selector */}
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
                            className={`rounded-lg border px-3 py-2 text-xs font-semibold capitalize transition-all ${
                              retriever === type
                                ? "border-emerald-500 bg-emerald-500/10 text-emerald-400"
                                : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                            }`}
                          >
                            {type}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Top K */}
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-xs font-medium text-zinc-300">
                          Cutoff Rank (Top K Candidates)
                        </label>
                        <span className="font-mono text-xs font-bold text-white">{topK}</span>
                      </div>
                      <div className="grid grid-cols-5 gap-1.5">
                        {[1, 3, 5, 8, 10].map((kVal) => (
                          <button
                            key={kVal}
                            type="button"
                            onClick={() => setTopK(kVal)}
                            className={`rounded-lg border py-1.5 text-xs font-mono font-bold transition-all ${
                              topK === kVal
                                ? "border-emerald-500 bg-emerald-500/10 text-emerald-400"
                                : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:bg-zinc-900"
                            }`}
                          >
                            {kVal}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* CrossEncoder toggle */}
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
                            : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-200"
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <Cpu className="h-4 w-4" />
                          <span>ms-marco-MiniLM-L-6-v2</span>
                        </span>
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                            useReranker ? "bg-amber-500/20 text-amber-300" : "bg-zinc-800 text-zinc-500"
                          }`}
                        >
                          {useReranker ? "Enabled" : "Disabled"}
                        </span>
                      </button>
                      <p className="mt-1 text-[11px] text-zinc-500">
                        Reranks candidate chunks with full query-passage attention (~280ms CPU cost).
                      </p>
                    </div>
                  </div>
                </div>

                {/* Question Input */}
                <form onSubmit={handleSingleQuery} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
                  <label className="text-xs font-medium text-zinc-300 block mb-2">
                    Question for Uploaded Document
                  </label>
                  <textarea
                    rows={3}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask anything about the uploaded PDF..."
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-200 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={!query.trim() || queryMutation.isPending}
                    className={`mt-3 w-full flex items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold text-white transition-all ${
                      !query.trim() || queryMutation.isPending
                        ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                        : "bg-emerald-600 hover:bg-emerald-500"
                    }`}
                  >
                    {queryMutation.isPending ? (
                      <>
                        <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        <span>Running Retrieval & Generation...</span>
                      </>
                    ) : (
                      <>
                        <Send className="h-3.5 w-3.5" />
                        <span>Submit Query</span>
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* Output Results Column */}
              <div className="lg:col-span-7 space-y-6">
                {singleResult ? (
                  <>
                    {/* Latency Waterfall */}
                    <LatencyWaterfall latency={singleResult.latency} />

                    {/* Grounded Generation Answer Card */}
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
                      <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
                        <div className="flex items-center gap-2">
                          <Bot className="h-4 w-4 text-emerald-400" />
                          <h4 className="text-xs font-semibold tracking-wider uppercase text-zinc-300">
                            Grounded AI Answer
                          </h4>
                        </div>
                        <span className="font-mono text-[11px] text-zinc-400">
                          {singleResult.tokens.total_tokens} tokens
                        </span>
                      </div>
                      <div className="mt-4 text-xs leading-relaxed text-zinc-200 whitespace-pre-line font-sans">
                        {singleResult.answer}
                      </div>
                    </div>

                    {/* Retrieved Chunks */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                          Retrieved Candidate Passages ({singleResult.retrieved_chunks.length})
                        </h4>
                        <span className="text-[11px] text-zinc-500">
                          Strategy: <strong className="text-zinc-300">{singleResult.retriever}</strong> | Reranker:{" "}
                          <strong className="text-zinc-300">{singleResult.use_reranker ? "On" : "Off"}</strong>
                        </span>
                      </div>

                      {singleResult.retrieved_chunks.map((chunk) => (
                        <ChunkCard key={chunk.chunk_id} chunk={chunk} />
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-zinc-800 p-12 text-center text-zinc-500">
                    <FileSearch className="h-8 w-8 mx-auto mb-2 text-zinc-600" />
                    <p className="text-xs">Type a question or click a suggested prompt to run evaluation.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: A/B Configuration Comparison Mode */}
          {activeTab === "compare" && (
            <div className="space-y-6">
              {/* Dual Config Pickers */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Config A Box */}
                <div className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                      Configuration A
                    </span>
                    <span className="text-[10px] rounded bg-sky-500/20 text-sky-300 px-2 py-0.5 font-mono">
                      Baseline
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Strategy</label>
                      <select
                        value={configA.retriever}
                        onChange={(e) =>
                          setConfigA({ ...configA, retriever: e.target.value as RetrieverType })
                        }
                        className="w-full rounded border border-zinc-800 bg-zinc-950 p-1.5 text-xs text-white"
                      >
                        <option value="dense">Dense (FAISS)</option>
                        <option value="sparse">Sparse (BM25)</option>
                        <option value="hybrid">Hybrid (RRF)</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Top-K</label>
                      <select
                        value={configA.top_k}
                        onChange={(e) => setConfigA({ ...configA, top_k: Number(e.target.value) })}
                        className="w-full rounded border border-zinc-800 bg-zinc-950 p-1.5 text-xs text-white"
                      >
                        <option value={1}>1</option>
                        <option value={3}>3</option>
                        <option value={5}>5</option>
                        <option value={8}>8</option>
                        <option value={10}>10</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Reranker</label>
                      <button
                        type="button"
                        onClick={() =>
                          setConfigA({ ...configA, use_reranker: !configA.use_reranker })
                        }
                        className={`w-full rounded border p-1.5 text-xs font-semibold ${
                          configA.use_reranker
                            ? "border-amber-500/60 bg-amber-500/20 text-amber-300"
                            : "border-zinc-800 bg-zinc-950 text-zinc-400"
                        }`}
                      >
                        {configA.use_reranker ? "Rerank ON" : "OFF"}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Config B Box */}
                <div className="rounded-xl border border-purple-500/30 bg-purple-500/5 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                      Configuration B
                    </span>
                    <span className="text-[10px] rounded bg-purple-500/20 text-purple-300 px-2 py-0.5 font-mono">
                      Challenger
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Strategy</label>
                      <select
                        value={configB.retriever}
                        onChange={(e) =>
                          setConfigB({ ...configB, retriever: e.target.value as RetrieverType })
                        }
                        className="w-full rounded border border-zinc-800 bg-zinc-950 p-1.5 text-xs text-white"
                      >
                        <option value="dense">Dense (FAISS)</option>
                        <option value="sparse">Sparse (BM25)</option>
                        <option value="hybrid">Hybrid (RRF)</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Top-K</label>
                      <select
                        value={configB.top_k}
                        onChange={(e) => setConfigB({ ...configB, top_k: Number(e.target.value) })}
                        className="w-full rounded border border-zinc-800 bg-zinc-950 p-1.5 text-xs text-white"
                      >
                        <option value={1}>1</option>
                        <option value={3}>3</option>
                        <option value={5}>5</option>
                        <option value={8}>8</option>
                        <option value={10}>10</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] text-zinc-400 block mb-1">Reranker</label>
                      <button
                        type="button"
                        onClick={() =>
                          setConfigB({ ...configB, use_reranker: !configB.use_reranker })
                        }
                        className={`w-full rounded border p-1.5 text-xs font-semibold ${
                          configB.use_reranker
                            ? "border-amber-500/60 bg-amber-500/20 text-amber-300"
                            : "border-zinc-800 bg-zinc-950 text-zinc-400"
                        }`}
                      >
                        {configB.use_reranker ? "Rerank ON" : "OFF"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Compare Query Form */}
              <form
                onSubmit={handleCompareSubmit}
                className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 flex flex-col sm:flex-row items-center gap-3"
              >
                <input
                  type="text"
                  value={compareQuery}
                  onChange={(e) => setCompareQuery(e.target.value)}
                  placeholder="Enter evaluation query to test across Config A and Config B simultaneously..."
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-white placeholder-zinc-500 focus:border-purple-500 focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={!compareQuery.trim() || compareMutation.isPending}
                  className={`shrink-0 flex items-center gap-2 rounded-lg px-5 py-2 text-xs font-bold text-white transition-all ${
                    !compareQuery.trim() || compareMutation.isPending
                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                      : "bg-purple-600 hover:bg-purple-500 shadow-lg shadow-purple-950/40"
                  }`}
                >
                  {compareMutation.isPending ? (
                    <>
                      <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      <span>Comparing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5" />
                      <span>Run A/B Comparison</span>
                    </>
                  )}
                </button>
              </form>

              {/* Comparison Results */}
              {compareResult && (
                <div className="space-y-6">
                  {/* Top Comparison Metrics Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-center">
                      <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider block mb-1">
                        Chunk Overlap (Jaccard Index)
                      </span>
                      <span className="text-2xl font-extrabold text-emerald-400 font-mono">
                        {compareResult.chunk_overlap_percentage}%
                      </span>
                      <span className="text-[11px] text-zinc-500 block mt-1">
                        {compareResult.shared_chunk_count} shared / {compareResult.total_unique_chunks} unique chunks
                      </span>
                    </div>

                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-center">
                      <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider block mb-1">
                        Latency Delta (B vs A)
                      </span>
                      <span
                        className={`text-2xl font-extrabold font-mono ${
                          compareResult.latency_difference_ms > 0
                            ? "text-amber-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {compareResult.latency_difference_ms > 0 ? "+" : ""}
                        {compareResult.latency_difference_ms.toFixed(1)} ms
                      </span>
                      <span className="text-[11px] text-zinc-500 block mt-1">
                        {compareResult.result_a.latency.total_ms.toFixed(1)}ms (A) vs{" "}
                        {compareResult.result_b.latency.total_ms.toFixed(1)}ms (B)
                      </span>
                    </div>

                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-center">
                      <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider block mb-1">
                        Token Consumption
                      </span>
                      <span className="text-2xl font-extrabold text-purple-400 font-mono">
                        {compareResult.result_a.tokens.total_tokens} vs{" "}
                        {compareResult.result_b.tokens.total_tokens}
                      </span>
                      <span className="text-[11px] text-zinc-500 block mt-1">
                        Context prompt + generation tokens
                      </span>
                    </div>
                  </div>

                  {/* Side-by-Side Dual Columns */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Left: Config A */}
                    <div className="space-y-4">
                      <div className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-4">
                        <div className="flex items-center justify-between pb-2 border-b border-sky-500/20">
                          <h4 className="text-xs font-bold text-sky-400 uppercase">
                            Config A: {compareResult.result_a.retriever} (k={compareResult.result_a.top_k})
                          </h4>
                          <span className="font-mono text-xs text-sky-300">
                            {compareResult.result_a.latency.total_ms.toFixed(1)} ms
                          </span>
                        </div>
                        <div className="mt-3 text-xs leading-relaxed text-zinc-200 font-sans whitespace-pre-line">
                          {compareResult.result_a.answer}
                        </div>
                      </div>

                      <h5 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                        Config A Retrieved Passages ({compareResult.result_a.retrieved_chunks.length})
                      </h5>
                      <div className="space-y-2">
                        {compareResult.result_a.retrieved_chunks.map((c) => {
                          const isShared = compareResult.result_b.retrieved_chunks.some(
                            (bc) => bc.chunk_id === c.chunk_id
                          );
                          return (
                            <div
                              key={c.chunk_id}
                              className={`rounded-lg border p-3 text-xs transition-all ${
                                isShared
                                  ? "border-emerald-500/30 bg-emerald-500/5"
                                  : "border-zinc-800 bg-zinc-900/50"
                              }`}
                            >
                              <div className="flex items-center justify-between font-mono text-[11px] mb-1">
                                <span className="font-bold text-sky-400">#{c.rank} {c.chunk_id}</span>
                                <div className="flex items-center gap-2">
                                  {isShared && (
                                    <span className="rounded bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 text-[10px]">
                                      Shared
                                    </span>
                                  )}
                                  <span className="text-zinc-400">{c.score.toFixed(4)}</span>
                                </div>
                              </div>
                              <p className="text-zinc-300 line-clamp-3 leading-relaxed">{c.text}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Right: Config B */}
                    <div className="space-y-4">
                      <div className="rounded-xl border border-purple-500/30 bg-purple-500/5 p-4">
                        <div className="flex items-center justify-between pb-2 border-b border-purple-500/20">
                          <h4 className="text-xs font-bold text-purple-400 uppercase">
                            Config B: {compareResult.result_b.retriever} (k={compareResult.result_b.top_k},{" "}
                            {compareResult.result_b.use_reranker ? "Rerank ON" : "Off"})
                          </h4>
                          <span className="font-mono text-xs text-purple-300">
                            {compareResult.result_b.latency.total_ms.toFixed(1)} ms
                          </span>
                        </div>
                        <div className="mt-3 text-xs leading-relaxed text-zinc-200 font-sans whitespace-pre-line">
                          {compareResult.result_b.answer}
                        </div>
                      </div>

                      <h5 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                        Config B Retrieved Passages ({compareResult.result_b.retrieved_chunks.length})
                      </h5>
                      <div className="space-y-2">
                        {compareResult.result_b.retrieved_chunks.map((c) => {
                          const isShared = compareResult.result_a.retrieved_chunks.some(
                            (ac) => ac.chunk_id === c.chunk_id
                          );
                          return (
                            <div
                              key={c.chunk_id}
                              className={`rounded-lg border p-3 text-xs transition-all ${
                                isShared
                                  ? "border-emerald-500/30 bg-emerald-500/5"
                                  : "border-zinc-800 bg-zinc-900/50"
                              }`}
                            >
                              <div className="flex items-center justify-between font-mono text-[11px] mb-1">
                                <span className="font-bold text-purple-400">#{c.rank} {c.chunk_id}</span>
                                <div className="flex items-center gap-2">
                                  {isShared && (
                                    <span className="rounded bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 text-[10px]">
                                      Shared
                                    </span>
                                  )}
                                  <span className="text-zinc-400">{c.score.toFixed(4)}</span>
                                </div>
                              </div>
                              <p className="text-zinc-300 line-clamp-3 leading-relaxed">{c.text}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
