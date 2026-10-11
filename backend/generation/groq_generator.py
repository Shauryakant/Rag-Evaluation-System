"""Groq API generation client for grounded RAG answer synthesis."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL_NAME
from backend.generation.prompt_builder import build_rag_prompt
from backend.retrieval.faiss_store import RetrievedChunk


@dataclass
class GenerationResult:
    """Encapsulates generated answer, latency, and token consumption metrics."""

    answer: str
    model: str
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GroqGenerator:
    """Synthesizes grounded answers using Groq's high-throughput Llama 3 models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GROQ_MODEL_NAME,
    ) -> None:
        """Initialize GroqGenerator.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY from environment).
            model_name: Groq model identifier (default: llama-3.1-8b-instant).
        """
        self.api_key = api_key if api_key is not None else GROQ_API_KEY
        self.model_name = model_name
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Optional[Groq]:
        """Lazy initialization of Groq client if API key is configured."""
        if self._client is None and self.api_key:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> GenerationResult:
        """Generate grounded answer from retrieved context chunks.

        Args:
            query: User query string.
            retrieved_chunks: List of retrieved chunks.
            temperature: Sampling temperature (low temperature = deterministic).
            max_tokens: Maximum completion tokens.

        Returns:
            GenerationResult object.
        """
        system_prompt, user_message = build_rag_prompt(query, retrieved_chunks)

        start_time = time.perf_counter()

        # If no API key is provided, gracefully return synthesized mock response for offline testing
        if not self.api_key or self.client is None:
            synth_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
            if retrieved_chunks:
                top_c = retrieved_chunks[0].chunk
                mock_answer = (
                    f"[Offline Mode / GROQ_API_KEY not configured]\n"
                    f"Based on [{top_c.doc_id}], {top_c.text[:200]}..."
                )
            else:
                mock_answer = "[Offline Mode] No passages were retrieved."

            return GenerationResult(
                answer=mock_answer,
                model=f"{self.model_name}-offline-mock",
                generation_ms=synth_latency,
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(mock_answer.split()),
                total_tokens=len(user_message.split()) + len(mock_answer.split()),
            )

        # Call live Groq API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        generation_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        answer = chat_completion.choices[0].message.content or ""

        usage = chat_completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return GenerationResult(
            answer=answer,
            model=self.model_name,
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
