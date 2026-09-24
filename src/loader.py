"""Document loader utility for reading raw text files while preserving character offsets."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """Represents a source document with tracked character offsets."""

    doc_id: str
    content: str
    start_char: int = 0
    end_char: int = 0

    def __post_init__(self) -> None:
        if self.end_char == 0 and len(self.content) > 0:
            self.end_char = len(self.content)


def load_single_document(file_path: Path) -> Document:
    """Load a single text file into a Document object.

    Args:
        file_path: Absolute path to the text file.

    Returns:
        Document instance with doc_id set to filename and full content character bounds.
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    doc_id = path.name
    return Document(
        doc_id=doc_id,
        content=content,
        start_char=0,
        end_char=len(content),
    )


def load_documents(docs_dir: Path) -> List[Document]:
    """Load all .txt documents from a given directory sorted by filename.

    Args:
        docs_dir: Path to directory containing .txt documents.

    Returns:
        Sorted list of Document objects.
    """
    directory = Path(docs_dir)
    if not directory.exists() or not directory.is_dir():
        return []

    txt_files = sorted(directory.glob("*.txt"))
    documents = [load_single_document(file_path) for file_path in txt_files]
    return documents
