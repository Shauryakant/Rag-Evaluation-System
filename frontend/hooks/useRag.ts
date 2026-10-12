/**
 * TanStack Query hooks for rag-eval-lab API operations.
 */

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  fetchBenchmarks,
  fetchHealth,
  runEvaluation,
  submitQuery,
} from "@/services/api";
import { EvaluationRequest, QueryRequest } from "@/types";

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
