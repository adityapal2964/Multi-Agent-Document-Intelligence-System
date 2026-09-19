"""RAG tests: ingestion, chunking, vector search."""

import pytest

from app.rag.ingestion import IngestionError, load_and_chunk
from app.rag import retriever as vector_store


def test_load_and_chunk_text_file(sample_text_file):
    chunks, stats = load_and_chunk(sample_text_file, document_id="doc1")
    assert stats["num_chunks"] == len(chunks) >= 1
    assert chunks[0].metadata["document_id"] == "doc1"
    assert chunks[0].metadata["filename"] == "policy.txt"


def test_empty_document_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_and_chunk(empty, document_id="doc2")


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(IngestionError):
        load_and_chunk(path, document_id="doc3")


def test_index_and_retrieve(sample_text_file):
    chunks, _ = load_and_chunk(sample_text_file, document_id="doc4")
    assert vector_store.add_documents(chunks) == len(chunks)
    assert vector_store.stats()["indexed_chunks"] >= len(chunks)

    hits = vector_store.search("refund window for enterprise contracts", k=2)
    assert hits, "expected at least one retrieved chunk"
    assert "refund" in hits[0]["text"].lower()
    assert hits[0]["filename"] == "policy.txt"


def test_search_with_unknown_document_filter_returns_nothing(sample_text_file):
    chunks, _ = load_and_chunk(sample_text_file, document_id="doc5")
    vector_store.add_documents(chunks)
    assert vector_store.search("refund", document_id="does-not-exist") == []


def test_sample_pdf_is_extractable():
    """The helper script must produce a PDF pypdf can read."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.create_sample_pdf import build_pdf

    pdf_path = build_pdf(Path(vector_store.stats()["index_path"]).parent / "sample.pdf")
    chunks, stats = load_and_chunk(pdf_path, document_id="pdfdoc")
    assert stats["num_pages"] == 2
    assert any("Revenue" in c.page_content for c in chunks)
