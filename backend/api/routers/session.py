"""Session router for uploading custom documents (PDF, TXT, MD) and running interactive evaluation."""

from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.schemas import (
    ChunkDTO,
    CompareRequest,
    CompareResponse,
    LatencyBreakdownDTO,
    QueryRequest,
    QueryResponse,
    SessionDetailResponse,
    SessionUploadResponse,
    TokenUsageDTO,
)
from backend.session.manager import DocumentSession, get_session_manager

router = APIRouter(prefix="/api/session", tags=["session"])


def _format_chunk_dto(chunk, score: float = 0.0, rank: int = 1) -> ChunkDTO:
    """Helper to convert a backend Chunk into ChunkDTO."""
    return ChunkDTO(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        score=score,
        rank=rank,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        metadata=chunk.metadata,
    )


def _run_session_query(
    session: DocumentSession,
    query: str,
    retriever_type: str,
    top_k: int,
    use_reranker: bool,
) -> QueryResponse:
    """Helper to execute retrieval and generation against an isolated document session."""
    session_mgr = get_session_manager()
    generator = session_mgr.get_shared_generator()

    # 1. Retrieval Phase on session-specific FAISS and BM25
    retrieved_items, latency_map = session.service.retrieve(
        query=query,
        retriever_type=retriever_type,
        top_k=top_k,
        use_reranker=use_reranker,
    )

    # 2. Grounded Generation Phase
    gen_result = generator.generate(
        query=query,
        retrieved_chunks=retrieved_items,
    )

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
        query=query,
        retriever=retriever_type,
        top_k=top_k,
        use_reranker=use_reranker,
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


@router.post("/upload", response_model=SessionUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
) -> SessionUploadResponse:
    """Upload a PDF, TXT, or MD document to create an isolated in-memory session.

    Extracts text, creates heading-aware chunks, builds dedicated FAISS vector
    and BM25 sparse indices, and extracts sample exploration questions.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")

    allowed_extensions = {".pdf", ".txt", ".md"}
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload a {', '.join(allowed_extensions)} file.",
        )

    try:
        content = file.file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        session_mgr = get_session_manager()
        session = session_mgr.create_session(
            file_bytes=content,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        preview_chunks = [
            _format_chunk_dto(c, score=0.0, rank=idx + 1)
            for idx, c in enumerate(session.chunks[:5])
        ]

        return SessionUploadResponse(
            session_id=session.session_id,
            filename=session.filename,
            page_count=session.page_count,
            char_count=session.char_count,
            chunk_count=session.chunk_count,
            sample_questions=session.sample_questions,
            preview_chunks=preview_chunks,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(exc)}")


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session_detail(session_id: str) -> SessionDetailResponse:
    """Retrieve metadata and all chunks for an active document session."""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Document session not found or expired.")

    chunk_dtos = [
        _format_chunk_dto(c, score=0.0, rank=idx + 1)
        for idx, c in enumerate(session.chunks)
    ]

    return SessionDetailResponse(
        session_id=session.session_id,
        filename=session.filename,
        page_count=session.page_count,
        char_count=session.char_count,
        chunk_count=session.chunk_count,
        sample_questions=session.sample_questions,
        chunks=chunk_dtos,
        created_at=session.created_at,
        last_accessed=session.last_accessed,
    )


@router.post("/{session_id}/query", response_model=QueryResponse)
def query_session_document(
    session_id: str,
    request: QueryRequest,
) -> QueryResponse:
    """Run an ad-hoc live query against the session's isolated document indices."""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Document session not found or expired.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    return _run_session_query(
        session=session,
        query=request.query.strip(),
        retriever_type=request.retriever,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )


@router.post("/{session_id}/compare", response_model=CompareResponse)
def compare_session_configurations(
    session_id: str,
    request: CompareRequest,
) -> CompareResponse:
    """Compare two retrieval configurations side-by-side on the same query and document session.

    Computes:
    - AI generated answers for both configurations
    - Jaccard similarity / chunk overlap of retrieved candidates
    - Stage-by-stage latency differences
    """
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Document session not found or expired.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Run Config A
    result_a = _run_session_query(
        session=session,
        query=query,
        retriever_type=request.config_a.retriever,
        top_k=request.config_a.top_k,
        use_reranker=request.config_a.use_reranker,
    )

    # Run Config B
    result_b = _run_session_query(
        session=session,
        query=query,
        retriever_type=request.config_b.retriever,
        top_k=request.config_b.top_k,
        use_reranker=request.config_b.use_reranker,
    )

    # Overlap analysis
    chunks_a = {c.chunk_id for c in result_a.retrieved_chunks}
    chunks_b = {c.chunk_id for c in result_b.retrieved_chunks}

    intersection = chunks_a & chunks_b
    union = chunks_a | chunks_b

    overlap_pct = round((len(intersection) / len(union) * 100.0) if union else 100.0, 1)
    latency_diff = round(result_b.latency.total_ms - result_a.latency.total_ms, 2)

    return CompareResponse(
        query=query,
        result_a=result_a,
        result_b=result_b,
        chunk_overlap_percentage=overlap_pct,
        shared_chunk_count=len(intersection),
        total_unique_chunks=len(union),
        latency_difference_ms=latency_diff,
    )


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """Explicitly delete a document session from memory."""
    session_mgr = get_session_manager()
    success = session_mgr.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document session not found.")
    return {"status": "ok", "message": f"Session {session_id} successfully purged from memory."}
