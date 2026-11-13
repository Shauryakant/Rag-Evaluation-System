/**
 * Centralized API service for communicating with the rag-eval-lab FastAPI backend.
 * The ONLY place that calls the backend.
 */

import {
  BenchmarkMatrixResponse,
  CompareRequest,
  CompareResponse,
  EvaluationRequest,
  EvaluationResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  SessionDetailResponse,
  SessionUploadResponse,
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

export async function uploadDocumentSession(
  file: File,
  chunkSize: number = 500,
  chunkOverlap: number = 50
): Promise<SessionUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("chunk_size", chunkSize.toString());
  formData.append("chunk_overlap", chunkOverlap.toString());

  const url = `${API_BASE_URL}/api/session/upload`;
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let message = `Upload failed: ${response.status} ${response.statusText}`;
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

export async function fetchSessionDetail(sessionId: string): Promise<SessionDetailResponse> {
  return request<SessionDetailResponse>(`/api/session/${sessionId}`);
}

export async function querySession(
  sessionId: string,
  payload: QueryRequest
): Promise<QueryResponse> {
  return request<QueryResponse>(`/api/session/${sessionId}/query`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function compareSessionConfigs(
  sessionId: string,
  payload: CompareRequest
): Promise<CompareResponse> {
  return request<CompareResponse>(`/api/session/${sessionId}/compare`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteSession(sessionId: string): Promise<{ status: string; message: string }> {
  return request<{ status: string; message: string }>(`/api/session/${sessionId}`, {
    method: "DELETE",
  });
}

