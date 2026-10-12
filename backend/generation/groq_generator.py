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
        self._cached_key: Optional[str] = None

    def get_api_key(self) -> str:
        """Resolve API key dynamically from instance, environment, or .env file."""
        if self.api_key is not None:
            return self.api_key

        import os
        from dotenv import load_dotenv
        from backend.config import BASE_DIR, ROOT_DIR

        # Dynamically reload .env in case user updated it during runtime
        load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
        return os.getenv("GROQ_API_KEY", "") or ""

    @property
    def client(self) -> Optional[Groq]:
        """Lazy initialization of Groq client, updating if API key changes."""
        active_key = self.get_api_key()
        if not active_key:
            return None
        if self._client is None or self._cached_key != active_key:
            self._client = Groq(api_key=active_key)
            self._cached_key = active_key
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
        if not self.get_api_key() or self.client is None:
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

        # Call live Groq API with automatic model fallback
        candidate_models = [
            self.model_name,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
        ]
        # Deduplicate while preserving priority
        seen = set()
        model_list = [m for m in candidate_models if not (m in seen or seen.add(m))]

        chat_completion = None
        used_model = self.model_name
        last_error = None

        for model_cand in model_list:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    model=model_cand,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                used_model = model_cand
                break
            except Exception as e:
                last_error = e
                continue

        generation_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if chat_completion is None:
            # Fallback if all models returned errors
            error_msg = f"[Groq API Error]: {last_error}"
            return GenerationResult(
                answer=error_msg,
                model="groq-error-fallback",
                generation_ms=generation_ms,
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(error_msg.split()),
                total_tokens=len(user_message.split()) + len(error_msg.split()),
            )

        answer = chat_completion.choices[0].message.content or ""
        # Some reasoning models return content or reasoning fields
        if not answer.strip() and hasattr(chat_completion.choices[0].message, "reasoning"):
            answer = getattr(chat_completion.choices[0].message, "reasoning", "") or ""

        usage = chat_completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return GenerationResult(
            answer=answer,
            model=used_model,
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
