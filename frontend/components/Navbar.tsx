"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useHealth } from "@/hooks/useRag";
import { Activity, BarChart2, Cpu, FileSearch, Layers, Sparkles } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const { data: health, isLoading, isError } = useHealth();

  const navLinks = [
    { href: "/", label: "Overview", icon: Layers },
    { href: "/query", label: "Query", icon: FileSearch },
    { href: "/evaluation", label: "Evaluation Lab", icon: Activity },
    { href: "/benchmarks", label: "Benchmarks", icon: BarChart2 },
    { href: "/about", label: "About & ADRs", icon: Cpu },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2.5 font-mono text-base font-bold tracking-tight text-white hover:opacity-90">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Sparkles className="h-4 w-4" />
            </span>
            <span>rag-eval-lab</span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-zinc-800/80 text-white font-semibold shadow-sm border border-zinc-700/60"
                      : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* System telemetry indicator */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/90 px-3 py-1 text-xs">
            {isLoading ? (
              <span className="flex items-center gap-1.5 text-zinc-500">
                <span className="h-2 w-2 animate-ping rounded-full bg-zinc-500" />
                Connecting...
              </span>
            ) : isError ? (
              <span className="flex items-center gap-1.5 text-rose-400">
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                Backend Offline
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span>{health?.indexed_documents} Docs</span>
                <span className="text-zinc-600">/</span>
                <span>{health?.total_chunks} Chunks</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
