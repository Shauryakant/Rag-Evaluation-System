"""Unit tests for prompt builder and generation components."""

from unittest.mock import MagicMock, patch
import pytest

from backend.chunking.splitter import Chunk
from backend.generation.groq_generator import GenerationResult, GroqGenerator
from backend.generation.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from backend.retrieval.faiss_store import RetrievedChunk


@pytest.fixture
def sample_retrieved_chunk():
    """Create a sample retrieved chunk fixture."""
    chunk = Chunk(
        chunk_id="doc1_c0",
        doc_id="01_dense_search.md",
        text="IndexFlatIP performs exact cosine inner product retrieval without quantization loss.",
        start_char=0,
        end_char=85,
        metadata={"heading": "FAISS IndexFlatIP"},
    )
    return RetrievedChunk(chunk=chunk, score=0.92, rank=1)


def test_build_rag_prompt_structure(sample_retrieved_chunk):
    """Test prompt building includes context passages and citations."""
    system_prompt, user_msg = build_rag_prompt(
        query="What is IndexFlatIP?",
        retrieved_chunks=[sample_retrieved_chunk],
    )

    assert "Ground your answer ONLY" in system_prompt
    assert "Context Passage [1]" in user_msg
    assert "Source: 01_dense_search.md" in user_msg
    assert "Section: FAISS IndexFlatIP" in user_msg
    assert sample_retrieved_chunk.chunk.text in user_msg
    assert "User Question: What is IndexFlatIP?" in user_msg


def test_build_rag_prompt_empty():
    """Test prompt building when no chunks are retrieved."""
    system_prompt, user_msg = build_rag_prompt(query="Unknown question", retrieved_chunks=[])
    assert "No relevant context passages were retrieved" in user_msg


def test_groq_generator_offline_fallback(sample_retrieved_chunk):
    """Test GroqGenerator offline mode when no API key is provided."""
    generator = GroqGenerator(api_key="")
    result = generator.generate(
        query="Explain IndexFlatIP",
        retrieved_chunks=[sample_retrieved_chunk],
    )

    assert isinstance(result, GenerationResult)
    assert "Offline Mode" in result.answer
    assert result.generation_ms >= 0.0
    assert result.total_tokens > 0


def test_groq_generator_mocked_api(sample_retrieved_chunk):
    """Test GroqGenerator with a mocked Groq API client."""
    generator = GroqGenerator(api_key="gsk_mock_test_key_12345")

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "IndexFlatIP computes exact inner products without approximation."
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 45
    mock_completion.usage.completion_tokens = 12
    mock_completion.usage.total_tokens = 57

    with patch.object(generator, "_client", MagicMock()) as mock_client:
        mock_client.chat.completions.create.return_value = mock_completion

        res = generator.generate(
            query="What is IndexFlatIP?",
            retrieved_chunks=[sample_retrieved_chunk],
        )

        assert res.answer == "IndexFlatIP computes exact inner products without approximation."
        assert res.prompt_tokens == 45
        assert res.completion_tokens == 12
        assert res.total_tokens == 57
        assert res.generation_ms > 0.0
