import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Navbar from "@/components/Navbar";
import QueryProvider from "@/components/QueryProvider";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "rag-eval-lab | Engineering Platform for Evaluating RAG Systems",
  description:
    "Production engineering platform for building, evaluating, and benchmarking RAG systems with dense, sparse, and hybrid search.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen bg-zinc-950 font-sans text-zinc-100 antialiased selection:bg-emerald-500/30 selection:text-emerald-200`}
      >
        <QueryProvider>
          <div className="flex min-h-screen flex-col">
            <Navbar />
            <main className="flex-1 pb-16">{children}</main>
            <footer className="border-t border-zinc-800/80 bg-zinc-950 py-6 text-center text-xs text-zinc-500">
              <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
                <p>rag-eval-lab — Decoupled evaluation platform without framework abstractions</p>
                <p className="font-mono text-zinc-600">FastAPI • FAISS • BM25 • CrossEncoder • Next.js</p>
              </div>
            </footer>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
