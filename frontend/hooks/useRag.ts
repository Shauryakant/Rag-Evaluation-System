/**
 * TanStack Query hooks for rag-eval-lab API operations.
 */

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  compareSessionConfigs,
  deleteSession,
  fetchBenchmarks,
  fetchHealth,
  fetchSessionDetail,
  querySession,
  runEvaluation,
  submitQuery,
  uploadDocumentSession,
} from "@/services/api";
import {
  CompareRequest,
  EvaluationRequest,
  QueryRequest,
} from "@/types";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: 2,
  });
}

export function useSubmitQuery() {
  return useMutation({
    mutationFn: (payload: QueryRequest) => submitQuery(payload),
  });
}

export function useRunEvaluation() {
  return useMutation({
    mutationFn: (payload: EvaluationRequest) => runEvaluation(payload),
  });
}

export function useBenchmarks() {
  return useQuery({
    queryKey: ["benchmarks"],
    queryFn: fetchBenchmarks,
    staleTime: 60000,
  });
}

export function useUploadDocumentSession() {
  return useMutation({
    mutationFn: ({
      file,
      chunkSize,
      chunkOverlap,
    }: {
      file: File;
      chunkSize?: number;
      chunkOverlap?: number;
    }) => uploadDocumentSession(file, chunkSize, chunkOverlap),
  });
}

export function useSessionDetail(sessionId: string | null) {
  return useQuery({
    queryKey: ["sessionDetail", sessionId],
    queryFn: () => (sessionId ? fetchSessionDetail(sessionId) : null),
    enabled: !!sessionId,
    staleTime: 30000,
  });
}

export function useQuerySession(sessionId: string) {
  return useMutation({
    mutationFn: (payload: QueryRequest) => querySession(sessionId, payload),
  });
}

export function useCompareSessionConfigs(sessionId: string) {
  return useMutation({
    mutationFn: (payload: CompareRequest) => compareSessionConfigs(sessionId, payload),
  });
}

export function useDeleteSession() {
  return useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
  });
}

