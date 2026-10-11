"""Pydantic data schemas for API requests, responses, and telemetry telemetry."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

RetrieverType = Literal["dense", "sparse", "hybrid"]


class ChunkDTO(BaseModel):
    """Data transfer object for a retrieved chunk."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    start_char: int
    end_char: int
    metadata: Dict[str, str] = Field(default_factory=dict)


class LatencyBreakdownDTO(BaseModel):
    """Detailed stage-by-stage latency telemetry in milliseconds."""

    retrieval_ms: float
    reranker_ms: float
    generation_ms: float
    total_ms: float


class TokenUsageDTO(BaseModel):
    """LLM token usage telemetry."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryRequest(BaseModel):
    """Request payload for live ad-hoc RAG querying."""

    query: str = Field(..., min_length=1, description="Question to answer against corpus")
    retriever: RetrieverType = Field("dense", description="Retrieval strategy")
    top_k: int = Field(5, ge=1, le=20, description="Number of candidate chunks to retrieve")
    use_reranker: bool = Field(False, description="Opt-in CrossEncoder reranking")


class QueryResponse(BaseModel):
    """Response payload for live ad-hoc RAG querying.

    CRITICAL DESIGN RULE:
    No accuracy scores fabricated here. Live ad-hoc queries have no ground truth.
    """

    query: str
    retriever: str
    top_k: int
    use_reranker: bool
    answer: str
    retrieved_chunks: List[ChunkDTO]
    latency: LatencyBreakdownDTO
    tokens: TokenUsageDTO


class EvaluationRequest(BaseModel):
    """Request payload for running Evaluation Lab benchmark."""

    retriever: RetrieverType = Field("dense", description="Retrieval strategy")
    top_k: int = Field(5, ge=1, le=20, description="Cutoff rank K")
    use_reranker: bool = Field(False, description="Opt-in CrossEncoder reranking")
    include_details: bool = Field(True, description="Include per-query evaluation breakdown")


class QueryEvalDetailDTO(BaseModel):
    """Detailed evaluation result for an individual benchmark question."""

    query_id: str
    question: str
    document: str
    target_chunks: List[str]
    retrieved_chunks: List[str]
    precision: float
    recall: float
    mrr: float
    hit_rate: float
    retrieval_ms: float
    reranker_ms: float
    total_ms: float
    is_hit: bool


class EvaluationResponse(BaseModel):
    """Response payload for Evaluation Lab benchmark run."""

    retriever: str
    top_k: int
    reranker: bool
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    latency_ms: float
    total_queries: int
    failed_query_count: int
    query_details: Optional[List[QueryEvalDetailDTO]] = None
    note: str = (
        "Precision@K is structurally capped near 1/K when questions have 1-2 relevant passages. "
        "Recall@K and MRR are the primary signals of retrieval effectiveness."
    )


class BenchmarkRowDTO(BaseModel):
    """Single configuration row in the benchmark sweep matrix."""

    retriever: str
    top_k: int
    reranker: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    latency_ms: float


class BenchmarkMatrixResponse(BaseModel):
    """Full benchmark sweep matrix response."""

    rows: List[BenchmarkRowDTO]
    total_configurations: int
    dataset_size: int


class HealthResponse(BaseModel):
    """System health and index readiness status."""

    status: str
    indexed_documents: int
    total_chunks: int
    embedder_model: str
    reranker_model: str
    groq_configured: bool
