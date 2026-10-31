"""RAG tests: ingestion, chunking, vector search."""

import pytest
from langchain_core.documents import Document

from app.rag.ingestion import IngestionError, load_and_chunk
from app.rag import retriever as vector_store
from app import config


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


def test_document_filter_is_applied_during_search():
    vector_store.add_documents(
        [Document(page_content="alpha " * 20, metadata={"document_id": "many"}) for _ in range(8)]
        + [Document(page_content="alpha beta", metadata={"document_id": "target"})]
    )
    hits = vector_store.search("alpha", k=4, document_id="target")
    assert len(hits) == 1
    assert hits[0]["document_id"] == "target"


def test_embedding_fingerprint_is_written_and_validated(monkeypatch):
    vector_store.add_documents(
        [Document(page_content="fingerprint test", metadata={"document_id": "fingerprint"})]
    )
    metadata_path = config.VECTORSTORE_DIR / "index_meta.json"
    assert metadata_path.exists()
    assert metadata_path.read_text(encoding="utf-8")

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "changed-model")
    with pytest.raises(vector_store.IndexMismatchError, match="Delete data/vectorstore/"):
        vector_store.search("fingerprint")


def test_embedding_dimension_mismatch_is_detected():
    import json

    vector_store.add_documents(
        [Document(page_content="dimension test", metadata={"document_id": "dimension"})]
    )
    metadata_path = config.VECTORSTORE_DIR / "index_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dimension"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    vector_store._store = None

    try:
        with pytest.raises(vector_store.IndexMismatchError, match="dimension"):
            vector_store.search("dimension")
    finally:
        vector_store.reset()


def test_legacy_index_without_metadata_still_loads():
    vector_store.add_documents(
        [Document(page_content="legacy index", metadata={"document_id": "legacy"})]
    )
    metadata_path = config.VECTORSTORE_DIR / "index_meta.json"
    metadata_path.unlink()
    vector_store._store = None
    assert vector_store.search("legacy")
    assert metadata_path.exists()


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
