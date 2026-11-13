"""Ephemeral, zero-database session manager for custom document playground and evaluation.

Provides in-memory session isolation for uploaded documents (PDF, TXT, MD),
building dedicated FAISS vector and BM25 lexical indices per session while
reusing shared embedding and reranking neural network model weights.
"""

import io
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pypdf

from backend.chunking.splitter import Chunk, RecursiveHeadingSplitter
from backend.embeddings.embedder import BGEEmbedder
from backend.generation.groq_generator import GroqGenerator
from backend.ingestion.loader import Document
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.service import RetrievalService


@dataclass
class DocumentSession:
    """An isolated, in-memory document session with its own indices and pipeline."""

    session_id: str
    filename: str
    doc_id: str
    page_count: int
    char_count: int
    chunk_count: int
    chunks: List[Chunk]
    sample_questions: List[str]
    service: RetrievalService
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last accessed timestamp for TTL expiration tracking."""
        self.last_accessed = time.time()


class SessionManager:
    """Manages ephemeral in-memory document sessions with TTL and LRU eviction.

    Zero external database requirements. Each session receives its own FAISS
    and BM25 indices while sharing the pre-loaded neural network weights.
    """

    def __init__(
        self,
        max_sessions: int = 25,
        session_ttl_seconds: int = 7200,  # 2 hours default
    ) -> None:
        """Initialize session manager.

        Args:
            max_sessions: Maximum concurrent active sessions before LRU eviction.
            session_ttl_seconds: Inactivity period before a session is purged.
        """
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: Dict[str, DocumentSession] = {}
        self._lock = threading.Lock()

        # Shared model singletons to avoid re-allocating PyTorch weights
        self._shared_embedder: Optional[BGEEmbedder] = None
        self._shared_reranker: Optional[CrossEncoderReranker] = None
        self._shared_generator: Optional[GroqGenerator] = None

    def get_shared_embedder(self) -> BGEEmbedder:
        """Return lazily initialized shared BGE embedder."""
        if self._shared_embedder is None:
            self._shared_embedder = BGEEmbedder()
        return self._shared_embedder

    def get_shared_reranker(self) -> CrossEncoderReranker:
        """Return lazily initialized shared CrossEncoder reranker."""
        if self._shared_reranker is None:
            self._shared_reranker = CrossEncoderReranker()
        return self._shared_reranker

    def get_shared_generator(self) -> GroqGenerator:
        """Return lazily initialized shared Groq generator."""
        if self._shared_generator is None:
            self._shared_generator = GroqGenerator()
        return self._shared_generator

    def extract_text_from_bytes(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[str, int]:
        """Extract clean text content and page count from file bytes.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Name of the uploaded file.

        Returns:
            Tuple of (extracted_text, page_count).
        """
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception as exc:
                        raise ValueError("Password-protected PDFs are not supported.") from exc

                pages_text: List[str] = []
                for idx, page in enumerate(reader.pages):
                    page_content = page.extract_text() or ""
                    clean_content = page_content.strip()
                    if clean_content:
                        pages_text.append(f"## Page {idx + 1}\n\n{clean_content}")

                full_text = "\n\n".join(pages_text).strip()
                if not full_text:
                    raise ValueError(
                        "No extractable text was found in the PDF. "
                        "Please verify the PDF contains selectable text (not scanned images without OCR)."
                    )
                return full_text, len(reader.pages)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Failed to parse PDF document: {str(e)}") from e

        # Text and Markdown formats
        try:
            full_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full_text = file_bytes.decode("latin-1", errors="ignore")

        clean_text = full_text.strip()
        if not clean_text:
            raise ValueError(f"File {filename} is empty.")

        return clean_text, 1

    def generate_sample_questions(self, chunks: List[Chunk], max_questions: int = 4) -> List[str]:
        """Derive 3-4 natural questions from headings and first sentences of chunks."""
        questions: List[str] = []
        seen = set()

        for chunk in chunks:
            if len(questions) >= max_questions:
                break

            heading = chunk.metadata.get("heading")
            if heading and len(heading) > 4 and heading not in seen and not heading.startswith("Page"):
                questions.append(f"What does the document say about {heading}?")
                seen.add(heading)

        # Fallback: extract first declarative sentence from early chunks
        if len(questions) < max_questions:
            for chunk in chunks:
                if len(questions) >= max_questions:
                    break
                text = chunk.text.strip()
                # Find first sentence ending in period
                match = re.search(r"([A-Z][^\.!\?]{20,120}[\.!\?])", text)
                if match:
                    sentence = match.group(1).strip()
                    prompt_q = f"Can you summarize: '{sentence[:60]}...'?"
                    if prompt_q not in seen:
                        questions.append(prompt_q)
                        seen.add(prompt_q)

        if not questions:
            questions = [
                "What is the main topic covered in this document?",
                "What are the key conclusions or takeaways mentioned?",
                "Can you provide an overview of the main sections?",
            ]

        return questions[:max_questions]

    def create_session(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> DocumentSession:
        """Create a new isolated document session and build its search indices."""
        self._cleanup_expired()

        full_text, page_count = self.extract_text_from_bytes(file_bytes, filename)
        doc = Document(doc_id=filename, content=full_text, path=filename)

        splitter = RecursiveHeadingSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_document(doc)
        if not chunks:
            raise ValueError("Document yielded 0 chunks after splitting.")

        sample_questions = self.generate_sample_questions(chunks)

        # Build dedicated retrieval service using shared model weights
        service = RetrievalService(
            chunks=chunks,
            embedder=self.get_shared_embedder(),
            reranker=self.get_shared_reranker(),
        )

        session_id = str(uuid.uuid4())
        session = DocumentSession(
            session_id=session_id,
            filename=filename,
            doc_id=filename,
            page_count=page_count,
            char_count=len(full_text),
            chunk_count=len(chunks),
            chunks=chunks,
            sample_questions=sample_questions,
            service=service,
        )

        with self._lock:
            # Enforce LRU eviction if maximum session capacity reached
            if len(self._sessions) >= self.max_sessions:
                oldest_id = min(self._sessions.keys(), key=lambda k: self._sessions[k].last_accessed)
                del self._sessions[oldest_id]

            self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[DocumentSession]:
        """Retrieve active document session by ID, updating last accessed time."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def delete_session(self, session_id: str) -> bool:
        """Explicitly purge a session from memory."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[Dict]:
        """Return list of active sessions with high-level stats."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "filename": s.filename,
                    "page_count": s.page_count,
                    "char_count": s.char_count,
                    "chunk_count": s.chunk_count,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                }
                for s in self._sessions.values()
            ]

    def _cleanup_expired(self) -> None:
        """Evict sessions that have exceeded the TTL threshold."""
        now = time.time()
        with self._lock:
            expired_ids = [
                s_id
                for s_id, s in self._sessions.items()
                if now - s.last_accessed > self.session_ttl_seconds
            ]
            for s_id in expired_ids:
                del self._sessions[s_id]


_GLOBAL_SESSION_MANAGER: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return singleton session manager instance."""
    global _GLOBAL_SESSION_MANAGER
    if _GLOBAL_SESSION_MANAGER is None:
        _GLOBAL_SESSION_MANAGER = SessionManager()
    return _GLOBAL_SESSION_MANAGER
