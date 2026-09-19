"""Embeddings provider.

Primary:  OpenAI embeddings (when OPENAI_API_KEY is set).
Fallback: a small deterministic hashing embedding implemented here, so the
project - and its tests - work offline with no API key at all.  The fallback
is lexical, not semantic: good enough for demos, not for production.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from app import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIM = 512


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing vectoriser (no model download, no API key)."""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall((text or "").lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorise(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorise(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Return the embedding model used for ingestion and retrieval."""
    if config.llm_enabled():
        try:
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {
                "model": config.EMBEDDING_MODEL,
                "api_key": config.OPENAI_API_KEY,
            }
            if config.OPENAI_BASE_URL:
                kwargs["base_url"] = config.OPENAI_BASE_URL
            return OpenAIEmbeddings(**kwargs)
        except Exception as exc:  # pragma: no cover
            print(f"[embeddings] falling back to hashing embeddings: {exc}")
    return HashingEmbeddings()


def embeddings_backend() -> str:
    """Name of the active backend - surfaced by /health."""
    return "openai" if config.llm_enabled() else "hashing-fallback"
