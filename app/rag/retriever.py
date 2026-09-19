"""FAISS vector store wrapper (one local index for all documents)."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.documents import Document

from app import config
from app.rag.embeddings import get_embeddings

_lock = threading.Lock()
_store: Any | None = None
_INDEX_DIR = config.VECTORSTORE_DIR / "faiss_index"


def _faiss():
    from langchain_community.vectorstores import FAISS

    return FAISS


def _load() -> Any | None:
    """Load the persisted index from disk, if present."""
    if not (_INDEX_DIR / "index.faiss").exists():
        return None
    try:
        return _faiss().load_local(
            str(_INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # we wrote this file ourselves
        )
    except Exception as exc:
        print(f"[vectorstore] could not load index, starting empty: {exc}")
        return None


def get_store() -> Any | None:
    """Return the in-memory store (lazily loaded from disk)."""
    global _store
    with _lock:
        if _store is None:
            _store = _load()
        return _store


def add_documents(documents: list[Document]) -> int:
    """Embed and persist chunks. Returns the number of chunks added."""
    global _store
    if not documents:
        return 0
    with _lock:
        if _store is None:
            _store = _load()
        if _store is None:
            _store = _faiss().from_documents(documents, get_embeddings())
        else:
            _store.add_documents(documents)
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _store.save_local(str(_INDEX_DIR))
    return len(documents)


def search(query: str, k: int | None = None,
           document_id: str | None = None) -> list[dict]:
    """Similarity search. Returns [] when nothing is indexed yet."""
    store = get_store()
    if store is None or not (query or "").strip():
        return []
    k = k or config.TOP_K
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        print(f"[vectorstore] search failed: {exc}")
        return []

    hits: list[dict] = []
    for document, score in results:
        metadata = document.metadata or {}
        if document_id and metadata.get("document_id") != document_id:
            continue
        hits.append(
            {
                "text": document.page_content,
                "filename": metadata.get("filename", "unknown"),
                "page": metadata.get("page"),
                "document_id": metadata.get("document_id"),
                "score": round(float(score), 4),  # FAISS L2 distance: lower is closer
            }
        )
    return hits


def stats() -> dict:
    store = get_store()
    count = 0
    if store is not None:
        try:
            count = int(store.index.ntotal)
        except Exception:
            count = 0
    return {"indexed_chunks": count, "index_path": str(_INDEX_DIR)}


def reset() -> None:
    """Drop the index (used by tests and the /documents DELETE endpoint)."""
    global _store
    import shutil

    with _lock:
        _store = None
        if _INDEX_DIR.exists():
            shutil.rmtree(_INDEX_DIR, ignore_errors=True)
