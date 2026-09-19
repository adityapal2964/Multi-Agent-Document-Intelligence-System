"""Document ingestion: extract -> chunk -> LangChain Documents."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class IngestionError(Exception):
    """Raised when a file cannot be turned into usable text."""


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a PDF / txt / md file."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(1, text)]

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception as exc:
        raise IngestionError(f"Could not read PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            pages.append((number, page.extract_text() or ""))
        except Exception:
            pages.append((number, ""))  # skip unreadable page, keep going
    return pages


def chunk_pages(pages: list[tuple[int, str]], *, document_id: str,
                filename: str) -> list[Document]:
    """Split page text into overlapping chunks with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for page_number, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        for chunk_index, chunk in enumerate(splitter.split_text(text)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "chunk": chunk_index,
                    },
                )
            )
    return documents


def load_and_chunk(path: Path, *, document_id: str,
                   filename: str | None = None) -> tuple[list[Document], dict]:
    """Full pipeline for one file. Raises IngestionError on empty documents.

    `filename` lets the caller keep the user's original name in the chunk
    metadata even though the file is stored with an id prefix on disk.
    """
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    pages = extract_pages(path)
    total_chars = sum(len(t or "") for _, t in pages)
    if total_chars == 0:
        raise IngestionError(
            "No extractable text found. The file may be empty or a scanned "
            "image-only PDF (OCR is not supported in this MVP)."
        )
    chunks = chunk_pages(pages, document_id=document_id,
                         filename=filename or path.name)
    if not chunks:
        raise IngestionError("Document produced no usable chunks.")
    stats = {
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "num_chars": total_chars,
    }
    return chunks, stats
