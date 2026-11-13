/**
 * Hand-written TypeScript definitions matching rag-eval-lab backend schemas.
 */

export type RetrieverType = "dense" | "sparse" | "hybrid";

export interface ChunkDTO {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  rank: number;
  start_char: number;
  end_char: number;
  metadata: Record<string, string>;
}

export interface LatencyBreakdownDTO {
  retrieval_ms: number;
  reranker_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface TokenUsageDTO {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface QueryRequest {
  query: string;
  retriever: RetrieverType;
  top_k: number;
  use_reranker: boolean;
}

export interface QueryResponse {
  query: string;
  retriever: string;
  top_k: number;
  use_reranker: boolean;
  answer: string;
  retrieved_chunks: ChunkDTO[];
  latency: LatencyBreakdownDTO;
  tokens: TokenUsageDTO;
}

export interface EvaluationRequest {
  retriever: RetrieverType;
  top_k: number;
  use_reranker: boolean;
  include_details?: boolean;
}

export interface QueryEvalDetailDTO {
  query_id: string;
  question: string;
  document: string;
  target_chunks: string[];
  retrieved_chunks: string[];
  precision: number;
  recall: number;
  mrr: number;
  hit_rate: number;
  retrieval_ms: number;
  reranker_ms: number;
  total_ms: number;
  is_hit: boolean;
}

export interface EvaluationResponse {
  retriever: string;
  top_k: number;
  reranker: boolean;
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  hit_rate: number;
  latency_ms: number;
  total_queries: number;
  failed_query_count: number;
  query_details?: QueryEvalDetailDTO[];
  note: string;
}

export interface BenchmarkRowDTO {
  retriever: string;
  top_k: number;
  reranker: string;
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  hit_rate: number;
  latency_ms: number;
}

export interface BenchmarkMatrixResponse {
  rows: BenchmarkRowDTO[];
  total_configurations: number;
  dataset_size: number;
}

export interface HealthResponse {
  status: string;
  indexed_documents: number;
  total_chunks: number;
  embedder_model: string;
  reranker_model: string;
  groq_configured: boolean;
}

export interface SessionUploadResponse {
  session_id: string;
  filename: string;
  page_count: number;
  char_count: number;
  chunk_count: number;
  sample_questions: string[];
  preview_chunks: ChunkDTO[];
  message: string;
}

export interface SessionDetailResponse {
  session_id: string;
  filename: string;
  page_count: number;
  char_count: number;
  chunk_count: number;
  sample_questions: string[];
  chunks: ChunkDTO[];
  created_at: number;
  last_accessed: number;
}

export interface CompareConfigDTO {
  label: string;
  retriever: RetrieverType;
  top_k: number;
  use_reranker: boolean;
}

export interface CompareRequest {
  query: string;
  config_a: CompareConfigDTO;
  config_b: CompareConfigDTO;
}

export interface CompareResponse {
  query: string;
  result_a: QueryResponse;
  result_b: QueryResponse;
  chunk_overlap_percentage: number;
  shared_chunk_count: number;
  total_unique_chunks: number;
  latency_difference_ms: number;
}
