"""Health check and index readiness router."""

from fastapi import APIRouter, Depends

from backend.api.deps import AppState, get_app_state
from backend.api.schemas import HealthResponse
from backend.config import EMBEDDER_MODEL_NAME, GROQ_API_KEY, RERANKER_MODEL_NAME

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def get_health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    """Check backend operational health and corpus index status."""
    return HealthResponse(
        status="ok",
        indexed_documents=len(state.documents),
        total_chunks=len(state.chunks),
        embedder_model=EMBEDDER_MODEL_NAME,
        reranker_model=RERANKER_MODEL_NAME,
        groq_configured=bool(state.generator.get_api_key()),
    )
