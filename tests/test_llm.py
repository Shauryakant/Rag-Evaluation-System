"""Unit tests for Gemini LLM wrapper."""

import pytest
from src.llm import GeminiClient


def test_gemini_client_missing_key():
    client = GeminiClient(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        client.generate_text("Test prompt")
