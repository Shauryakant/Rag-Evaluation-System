"""Evaluation Lab router running systematic benchmark evaluations over labelled testset."""

from fastapi import APIRouter, Depends

from backend.api.deps import AppState, get_app_state
from backend.api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    QueryEvalDetailDTO,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest, state: AppState = Depends(get_app_state)) -> EvaluationResponse:
    """Run evaluation sweep for the requested retriever configuration over the labelled testset.

    CRITICAL DESIGN RULES:
    1. Calls the EXACT same retriever function as Query mode.
    2. Precision@K is structurally capped near 1/K when questions have 1-2 relevant passages.
    """
    result = state.runner.evaluate_configuration(
        retriever_type=request.retriever,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        include_details=request.include_details,
    )

    detail_dtos = None
    if result.query_details:
        detail_dtos = [
            QueryEvalDetailDTO(
                query_id=d.query_id,
                question=d.question,
                document=d.document,
                target_chunks=d.target_chunks,
                retrieved_chunks=d.retrieved_chunks,
                precision=d.precision,
                recall=d.recall,
                mrr=d.mrr,
                hit_rate=d.hit_rate,
                retrieval_ms=d.retrieval_ms,
                reranker_ms=d.reranker_ms,
                total_ms=d.total_ms,
                is_hit=d.is_hit,
            )
            for d in result.query_details
        ]

    return EvaluationResponse(
        retriever=result.retriever,
        top_k=result.top_k,
        reranker=result.reranker,
        precision_at_k=result.precision_at_k,
        recall_at_k=result.recall_at_k,
        mrr=result.mrr,
        hit_rate=result.hit_rate,
        latency_ms=result.latency_ms,
        total_queries=result.total_queries,
        failed_query_count=result.failed_query_count,
        query_details=detail_dtos,
    )
