"""FAISS vector store wrapper (one local index for all documents)."""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app import config
from app.rag.embeddings import get_embeddings

_lock = threading.Lock()
_store: Any | None = None
_INDEX_DIR = config.VECTORSTORE_DIR / "faiss_index"
_META_PATH = config.VECTORSTORE_DIR / "index_meta.json"
logger = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    """The persisted index was built with different embedding settings."""


def _faiss():
    from langchain_community.vectorstores import FAISS

    return FAISS


@lru_cache(maxsize=1)
def _embedding_dimension() -> int:
    return len(get_embeddings().embed_query("dimension probe"))


def _fingerprint(dimension: int) -> dict[str, Any]:
    from app.rag.embeddings import embeddings_backend

    return {
        "embeddings_backend": embeddings_backend(),
        "embedding_model": config.EMBEDDING_MODEL,
        "dimension": dimension,
    }


def _read_metadata() -> dict[str, Any] | None:
    if not _META_PATH.exists():
        return None
    try:
        return json.loads(_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexMismatchError(
            f"Could not read the embedding index metadata ({exc}). "
            "Delete data/vectorstore/ and re-upload your documents."
        ) from exc


def _validate_metadata(metadata: dict[str, Any]) -> None:
    expected = _fingerprint(_embedding_dimension())
    changed = [
        key for key in ("embeddings_backend", "embedding_model", "dimension")
        if metadata.get(key) != expected.get(key)
    ]
    if changed:
        details = ", ".join(
            f"{key}: stored={metadata.get(key)!r}, current={expected.get(key)!r}"
            for key in changed
        )
        raise IndexMismatchError(
            f"Embedding index mismatch ({details}). "
            "Delete data/vectorstore/ and re-upload your documents."
        )


def _write_metadata(dimension: int) -> None:
    _META_PATH.parent.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(
        json.dumps(_fingerprint(dimension), indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_metadata(store: Any) -> None:
    metadata = _read_metadata()
    if metadata is None:
        logger.warning("Legacy FAISS index has no embedding metadata; recording current settings")
        _write_metadata(_embedding_dimension())
        return
    _validate_metadata(metadata)


def _load() -> Any | None:
    """Load the persisted index from disk, if present."""
    if not (_INDEX_DIR / "index.faiss").exists():
        return None
    metadata = _read_metadata()
    if metadata is not None:
        _validate_metadata(metadata)
    try:
        store = _faiss().load_local(
            str(_INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # we wrote this file ourselves
        )
        _ensure_metadata(store)
        return store
    except Exception as exc:
        if isinstance(exc, IndexMismatchError):
            raise
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
        if _store is not None:
            _ensure_metadata(_store)
        if _store is None:
            _store = _faiss().from_documents(documents, get_embeddings())
        else:
            _store.add_documents(documents)
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _store.save_local(str(_INDEX_DIR))
        _write_metadata(int(_store.index.d))
    return len(documents)


def search(query: str, k: int | None = None,
           document_id: str | None = None) -> list[dict]:
    """Similarity search. Returns [] when nothing is indexed yet."""
    store = get_store()
    if store is None or not (query or "").strip():
        return []
    k = k or config.TOP_K
    try:
        _ensure_metadata(store)
        if document_id:
            fetch_k = int(store.index.ntotal)
            results = store.similarity_search_with_score(
                query,
                k=k,
                filter={"document_id": document_id},
                fetch_k=fetch_k,
            )
        else:
            results = store.similarity_search_with_score(query, k=k, fetch_k=k)
    except Exception as exc:
        if isinstance(exc, IndexMismatchError):
            raise
        print(f"[vectorstore] search failed: {exc}")
        return []

    hits: list[dict] = []
    for document, score in results:
        metadata = document.metadata or {}
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
    try:
        store = get_store()
    except IndexMismatchError:
        return {"indexed_chunks": 0, "index_path": str(_INDEX_DIR)}
    count = 0
    if store is not None:
        try:
            count = int(store.index.ntotal)
        except Exception:
            count = 0
    return {"indexed_chunks": count, "index_path": str(_INDEX_DIR)}


def index_status() -> str:
    """Return the index health state without failing the health endpoint."""
    if not (_INDEX_DIR / "index.faiss").exists():
        return "empty"
    try:
        store = get_store()
        return "ok" if store is not None else "empty"
    except IndexMismatchError:
        return "mismatch"


def reset() -> None:
    """Drop the index (used by tests and the /documents DELETE endpoint)."""
    global _store
    import shutil

    with _lock:
        _store = None
        if _INDEX_DIR.exists():
            shutil.rmtree(_INDEX_DIR, ignore_errors=True)
        if _META_PATH.exists():
            _META_PATH.unlink()
