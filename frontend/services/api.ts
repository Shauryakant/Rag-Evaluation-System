/**
 * Centralized API service for communicating with the rag-eval-lab FastAPI backend.
 * The ONLY place that calls the backend.
 */

import {
  BenchmarkMatrixResponse,
  EvaluationRequest,
  EvaluationResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let message = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const parsed = JSON.parse(errorText);
      if (parsed.detail) message = parsed.detail;
    } catch {
      if (errorText) message = errorText;
    }
    throw new Error(message);
  }

  return response.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function submitQuery(payload: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runEvaluation(payload: EvaluationRequest): Promise<EvaluationResponse> {
  return request<EvaluationResponse>("/api/evaluation/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchBenchmarks(): Promise<BenchmarkMatrixResponse> {
  return request<BenchmarkMatrixResponse>("/api/benchmarks");
}

export function getBenchmarkCsvUrl(): string {
  return `${API_BASE_URL}/api/benchmarks/csv`;
}
