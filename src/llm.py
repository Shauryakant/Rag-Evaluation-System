"""Thin wrapper around Google Gemini API with retries and rate limiting."""

import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Wrapper client for Gemini LLM generation with rate limiting and exponential retries."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0) -> None:
        """Initialize GeminiClient.

        Args:
            api_key: Optional Gemini API key string. Defaults to GEMINI_API_KEY env var.
            rate_limit_delay: Pause duration in seconds between API requests.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time: float = 0.0
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def generate_text(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ) -> str:
        """Generate text completion using Gemini model with retry logic.

        Args:
            prompt: User/system prompt string.
            model_name: Target Gemini model identifier.
            max_retries: Number of retry attempts on failure.

        Returns:
            Generated text content string.
        """
        if not self.api_key or self._client is None:
            raise ValueError("GEMINI_API_KEY is not set or google-genai client failed to initialize")

        # Simple rate limiting enforcement
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        last_exception = None
        for attempt in range(max_retries):
            try:
                self._last_call_time = time.time()
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text or ""
            except Exception as e:
                last_exception = e
                time.sleep(2.0 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {last_exception}")
