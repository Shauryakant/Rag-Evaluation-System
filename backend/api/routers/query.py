"""Live ad-hoc RAG query router with stage-by-stage latency telemetry."""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import AppState, get_app_state
from backend.api.schemas import (
    ChunkDTO,
    LatencyBreakdownDTO,
    QueryRequest,
    QueryResponse,
    TokenUsageDTO,
)

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(request: QueryRequest, state: AppState = Depends(get_app_state)) -> QueryResponse:
    """Execute live query against benchmark corpus.

    CRITICAL DESIGN RULE:
    1. Query mode and Evaluation Lab mode call the EXACT same retriever method.
    2. No accuracy score is fabricated or displayed for live ad-hoc questions.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # 1. Retrieval Phase
    retrieved_items, latency_map = state.service.retrieve(
        query=request.query,
        retriever_type=request.retriever,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )

    # 2. Generation Phase
    gen_result = state.generator.generate(
        query=request.query,
        retrieved_chunks=retrieved_items,
    )

    # Format chunks for response
    chunk_dtos = [
        ChunkDTO(
            chunk_id=item.chunk.chunk_id,
            doc_id=item.chunk.doc_id,
            text=item.chunk.text,
            score=item.score,
            rank=item.rank,
            start_char=item.chunk.start_char,
            end_char=item.chunk.end_char,
            metadata=item.chunk.metadata,
        )
        for item in retrieved_items
    ]

    total_latency_ms = round(latency_map.get("total_ms", 0.0) + gen_result.generation_ms, 2)

    return QueryResponse(
        query=request.query,
        retriever=request.retriever,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        answer=gen_result.answer,
        retrieved_chunks=chunk_dtos,
        latency=LatencyBreakdownDTO(
            retrieval_ms=latency_map.get("retrieval_ms", 0.0),
            reranker_ms=latency_map.get("reranker_ms", 0.0),
            generation_ms=gen_result.generation_ms,
            total_ms=total_latency_ms,
        ),
        tokens=TokenUsageDTO(
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
        ),
    )
