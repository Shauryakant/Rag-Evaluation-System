"""Text chunking logic with character span tracking and overlapping windows."""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """Represents a text chunk with document metadata and precise character offsets."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int

    @property
    def span(self) -> tuple[int, int]:
        """Return the (start_char, end_char) tuple."""
        return (self.start_char, self.end_char)


def chunk_text(text: str, doc_id: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Split text into fixed-size character chunks with specified overlap.

    Args:
        text: Raw text content of the document.
        doc_id: Identifier of the source document.
        chunk_size: Target character size for each chunk (must be > 0).
        overlap: Character overlap between consecutive chunks (must be < chunk_size).

    Returns:
        List of Chunk objects with start_char and end_char offsets.
    """
    if not text or len(text.strip()) == 0:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    stride = chunk_size - overlap
    text_len = len(text)
    chunks: List[Chunk] = []

    start = 0
    idx = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_slice = text[start:end]

        chunk = Chunk(
            chunk_id=f"{doc_id}_c{idx}",
            doc_id=doc_id,
            text=chunk_slice,
            start_char=start,
            end_char=end,
        )
        chunks.append(chunk)

        idx += 1
        if end == text_len:
            break
        start += stride

    return chunks
