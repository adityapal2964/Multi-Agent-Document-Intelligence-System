"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    document_id: str | None = Field(None, description="Restrict retrieval to one document")


class Source(BaseModel):
    filename: str | None = None
    page: int | None = None
    score: float | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str] = []
    tools_used: list[str] = []
    sources: list[Source] = []
    critique: dict[str, Any] = {}
    retries: int = 0


class DocumentInfo(BaseModel):
    id: str
    filename: str
    num_pages: int = 0
    num_chunks: int = 0
    num_chars: int = 0
    status: str = "processed"
    uploaded_at: str


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    web_search_configured: bool
    embeddings_backend: str
    indexed_chunks: int
    documents: int
    model: str
    index_status: str
