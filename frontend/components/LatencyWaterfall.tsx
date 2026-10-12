"use client";

import React from "react";
import { LatencyBreakdownDTO } from "@/types";
import { Clock, Cpu, Sparkles, Zap } from "lucide-react";

interface LatencyWaterfallProps {
  latency: LatencyBreakdownDTO;
}

export default function LatencyWaterfall({ latency }: LatencyWaterfallProps) {
  const total = Math.max(latency.total_ms, 0.01);
  const retrievalPct = Math.min(100, Math.round((latency.retrieval_ms / total) * 100));
  const rerankerPct = Math.min(100, Math.round((latency.reranker_ms / total) * 100));
  const generationPct = Math.min(100, Math.round((latency.generation_ms / total) * 100));

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-emerald-400" />
          <h4 className="text-xs font-semibold tracking-wider uppercase text-zinc-300">
            Pipeline Latency Waterfall
          </h4>
        </div>
        <span className="font-mono text-xs font-bold text-emerald-400">
          {latency.total_ms.toFixed(1)} ms total
        </span>
      </div>

      {/* Progress bar split */}
      <div className="mt-3 flex h-2 w-full overflow-hidden rounded-full bg-zinc-950">
        <div
          style={{ width: `${retrievalPct}%` }}
          className="bg-sky-500 transition-all duration-300"
          title={`Retrieval: ${latency.retrieval_ms}ms (${retrievalPct}%)`}
        />
        <div
          style={{ width: `${rerankerPct}%` }}
          className="bg-amber-500 transition-all duration-300"
          title={`Reranking: ${latency.reranker_ms}ms (${rerankerPct}%)`}
        />
        <div
          style={{ width: `${generationPct}%` }}
          className="bg-purple-500 transition-all duration-300"
          title={`Generation: ${latency.generation_ms}ms (${generationPct}%)`}
        />
      </div>

      {/* Legend and metrics breakdown */}
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className="rounded-lg bg-zinc-950/60 p-2.5 border border-zinc-800/50">
          <div className="flex items-center gap-1.5 text-sky-400 font-medium mb-1">
            <Zap className="h-3.5 w-3.5" />
            <span>Retrieval</span>
          </div>
          <div className="font-mono text-sm font-semibold text-white">
            {latency.retrieval_ms.toFixed(1)} <span className="text-xs text-zinc-500">ms</span>
          </div>
          <div className="text-[10px] text-zinc-500">{retrievalPct}% of pipeline</div>
        </div>

        <div className="rounded-lg bg-zinc-950/60 p-2.5 border border-zinc-800/50">
          <div className="flex items-center gap-1.5 text-amber-400 font-medium mb-1">
            <Cpu className="h-3.5 w-3.5" />
            <span>CrossEncoder</span>
          </div>
          <div className="font-mono text-sm font-semibold text-white">
            {latency.reranker_ms > 0 ? (
              <>
                {latency.reranker_ms.toFixed(1)} <span className="text-xs text-zinc-500">ms</span>
              </>
            ) : (
              <span className="text-zinc-500 italic text-xs">Bypassed (0ms)</span>
            )}
          </div>
          <div className="text-[10px] text-zinc-500">
            {latency.reranker_ms > 0 ? `${rerankerPct}% of pipeline` : "Opt-in disabled"}
          </div>
        </div>

        <div className="rounded-lg bg-zinc-950/60 p-2.5 border border-zinc-800/50">
          <div className="flex items-center gap-1.5 text-purple-400 font-medium mb-1">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Generation</span>
          </div>
          <div className="font-mono text-sm font-semibold text-white">
            {latency.generation_ms.toFixed(1)} <span className="text-xs text-zinc-500">ms</span>
          </div>
          <div className="text-[10px] text-zinc-500">{generationPct}% of pipeline</div>
        </div>
      </div>
    </div>
  );
}
