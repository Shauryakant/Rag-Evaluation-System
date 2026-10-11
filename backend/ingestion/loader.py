"""Document loader for the benchmark corpus, preserving raw text content and character counts."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """Represents a benchmark corpus document."""

    doc_id: str
    content: str
    path: str
    length: int

    def __init__(self, doc_id: str, content: str, path: str = "") -> None:
        self.doc_id = doc_id
        self.content = content
        self.path = path
        self.length = len(content)


def load_single_document(file_path: Path) -> Document:
    """Load a single document (.md, .txt, .pdf) from disk.

    Args:
        file_path: Path to the target document.

    Returns:
        Document instance containing full text content.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".md", ".txt"]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    elif suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            content = "\n".join([page.extract_text() or "" for page in reader.pages])
        except ImportError:
            # Fallback if pypdf is not installed
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    return Document(doc_id=path.name, content=content, path=str(path.resolve()))


from backend.config import CORPUS_DIR


def load_corpus(corpus_dir: Optional[Path] = None) -> List[Document]:
    """Load all benchmark documents from a directory.

    Args:
        corpus_dir: Path to directory containing corpus documents (defaults to CORPUS_DIR).

    Returns:
        Sorted list of Document objects.
    """
    directory = Path(corpus_dir) if corpus_dir is not None else CORPUS_DIR
    if not directory.exists() or not directory.is_dir():
        return []

    supported_extensions = ["*.md", "*.txt", "*.pdf"]
    matched_files = []
    for ext in supported_extensions:
        matched_files.extend(directory.glob(ext))

    matched_files = sorted(set(matched_files), key=lambda p: p.name)
    return [load_single_document(p) for p in matched_files]
