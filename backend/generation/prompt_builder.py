"""Prompt construction for LLM generation strictly grounded in retrieved chunks."""

from typing import List

from backend.retrieval.faiss_store import RetrievedChunk

SYSTEM_PROMPT = """You are an engineering RAG assistant. Your task is to answer the user's question accurately and concisely, strictly based on the provided context passages.

CRITICAL INSTRUCTIONS:
1. Ground your answer ONLY in the provided passages.
2. If the passages do not contain sufficient information to answer the question, state: "The provided context does not contain sufficient information to answer this question."
3. Do NOT extrapolate, hallucinate, or bring in unstated facts.
4. Reference the passage numbers (e.g. [1], [2]) when asserting facts."""


def build_rag_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> tuple[str, str]:
    """Construct system prompt and user context message from retrieved chunks.

    Args:
        query: User input query.
        retrieved_chunks: Ordered list of retrieved passage chunks.

    Returns:
        Tuple of (system_prompt, user_message).
    """
    if not retrieved_chunks:
        context_str = "No relevant context passages were retrieved."
    else:
        context_blocks: List[str] = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            heading = item.chunk.metadata.get("heading", "General")
            doc = item.chunk.doc_id
            block = (
                f"--- Context Passage [{idx}] ---\n"
                f"Source: {doc} | Section: {heading}\n"
                f"{item.chunk.text.strip()}\n"
            )
            context_blocks.append(block)
        context_str = "\n".join(context_blocks)

    user_message = (
        f"Context Passages:\n\n{context_str}\n\n"
        f"User Question: {query.strip()}\n\n"
        f"Answer:"
    )

    return SYSTEM_PROMPT, user_message
