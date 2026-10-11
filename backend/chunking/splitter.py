"""Recursive and heading-aware text chunking splitter with exact character span preservation."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.ingestion.loader import Document


@dataclass
class Chunk:
    """Represents a discrete text chunk with exact document offset boundaries."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        """Return the (start_char, end_char) tuple."""
        return (self.start_char, self.end_char)


class RecursiveHeadingSplitter:
    """Splits documents into overlapping chunks with heading and paragraph awareness."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize splitter.

        Args:
            chunk_size: Target character length per chunk (must be > 0).
            chunk_overlap: Sliding window character overlap (must be < chunk_size).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Document) -> List[Chunk]:
        """Split a document into chunks while preserving character offsets.

        Args:
            doc: Document instance containing raw text and doc_id.

        Returns:
            List of Chunk objects with valid character spans.
        """
        text = doc.content
        text_len = len(text)
        if not text or not text.strip():
            return []

        # If document is smaller than chunk_size, return single chunk
        if text_len <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{doc.doc_id}_c0",
                    doc_id=doc.doc_id,
                    text=text,
                    start_char=0,
                    end_char=text_len,
                    metadata={"heading": "root"},
                )
            ]

        # Extract heading landmarks to guide splitting
        heading_matches = list(re.finditer(r"(?m)^(#{1,3})\s+(.+)$", text))

        chunks: List[Chunk] = []
        start = 0
        chunk_index = 0

        while start < text_len:
            target_end = min(start + self.chunk_size, text_len)
            end = target_end

            if target_end < text_len:
                # Search for clean paragraph, heading, sentence, or word boundaries
                search_window_start = max(start + int(self.chunk_size * 0.3), start)
                sub_text = text[search_window_start:target_end]

                para_break = sub_text.rfind("\n\n")
                if para_break != -1:
                    end = search_window_start + para_break + 2
                else:
                    # Look for sentence boundaries
                    sentence_matches = list(re.finditer(r"[.!?]\s+", sub_text))
                    if sentence_matches:
                        end = search_window_start + sentence_matches[-1].end()
                    else:
                        # Fallback to word boundary
                        space_break = sub_text.rfind(" ")
                        if space_break != -1:
                            end = search_window_start + space_break + 1

            chunk_text = text[start:end]
            if chunk_text.strip():
                # Determine active heading for metadata
                active_heading = "root"
                for h in heading_matches:
                    if h.start() <= start:
                        active_heading = h.group(2).strip()
                    else:
                        break

                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_c{chunk_index}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        metadata={"heading": active_heading},
                    )
                )
                chunk_index += 1

            if end >= text_len:
                break

            # Advance start pointer with overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def split_documents(self, docs: List[Document]) -> List[Chunk]:
        """Split a list of documents into chunks."""
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
