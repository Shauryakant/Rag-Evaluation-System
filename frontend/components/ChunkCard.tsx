"use client";

import React, { useState } from "react";
import { ChunkDTO } from "@/types";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";

interface ChunkCardProps {
  chunk: ChunkDTO;
}

export default function ChunkCard({ chunk }: ChunkCardProps) {
  const [expanded, setExpanded] = useState(false);
  const heading = chunk.metadata?.heading || "Section";

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 transition-all hover:border-zinc-700">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-zinc-800 font-mono text-xs font-bold text-zinc-200">
            #{chunk.rank}
          </span>
          <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-300">
            <FileText className="h-3.5 w-3.5 text-emerald-400" />
            <span className="font-mono text-emerald-400">{chunk.doc_id}</span>
            <span className="text-zinc-600">/</span>
            <span className="text-zinc-400">{heading}</span>
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-0.5 font-mono text-[11px] text-zinc-300">
            Score: <span className="font-bold text-white">{chunk.score.toFixed(4)}</span>
          </span>
          <span className="font-mono text-[10px] text-zinc-500">
            [{chunk.start_char}:{chunk.end_char}]
          </span>
        </div>
      </div>

      <div className="mt-2 text-xs leading-relaxed text-zinc-300 font-sans">
        <p className={expanded ? "" : "line-clamp-3"}>
          {chunk.text}
        </p>
      </div>

      {chunk.text.length > 200 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 flex items-center gap-1 text-[11px] font-medium text-emerald-400 hover:text-emerald-300"
        >
          {expanded ? (
            <>
              Show less <ChevronUp className="h-3 w-3" />
            </>
          ) : (
            <>
              Show full passage <ChevronDown className="h-3 w-3" />
            </>
          )}
        </button>
      )}
    </div>
  );
}
