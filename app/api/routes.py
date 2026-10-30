"""FastAPI routes: health, document upload/list, chat."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config
from app.api.schemas import (ChatRequest, ChatResponse, DocumentInfo,
                             HealthResponse, UploadResponse)
from app.database import database as db
from app.graph.workflow import run_query
from app.rag import retriever as vector_store
from app.rag.embeddings import embeddings_backend
from app.rag.ingestion import SUPPORTED_SUFFIXES, IngestionError, load_and_chunk

router = APIRouter()
MAX_UPLOAD_MB = 25


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe + a quick view of what is configured."""
    return HealthResponse(
        status="ok",
        llm_configured=config.llm_configured(),
        web_search_configured=config.web_search_enabled(),
        embeddings_backend=embeddings_backend(),
        indexed_chunks=vector_store.stats()["indexed_chunks"],
        documents=len(db.list_documents()),
        model=config.LLM_MODEL,
    )


@router.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF (or .txt/.md), extract, chunk, embed and index it."""
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    document_id = uuid.uuid4().hex[:12]
    destination = config.DOCUMENTS_DIR / f"{document_id}_{filename}"
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc
    finally:
        await file.close()

    size_mb = destination.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size_mb > MAX_UPLOAD_MB:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")

    try:
        chunks, stats = load_and_chunk(destination, document_id=document_id,
                                       filename=filename)
        vector_store.add_documents(chunks)
    except IngestionError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    record = {
        "id": document_id,
        "filename": filename,
        "file_path": str(destination),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "processed",
        "metadata": {"size_mb": round(size_mb, 3), "suffix": suffix},
        **stats,
    }
    db.record_document(record)

    return UploadResponse(
        document=DocumentInfo(**{k: record[k] for k in DocumentInfo.model_fields}),
        message=f"Indexed {stats['num_chunks']} chunks from {stats['num_pages']} page(s).",
    )


@router.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents() -> list[DocumentInfo]:
    """List every ingested document."""
    return [
        DocumentInfo(**{k: row.get(k, DocumentInfo.model_fields[k].default)
                        for k in DocumentInfo.model_fields})
        for row in db.list_documents()
    ]


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Run a query through the multi-agent LangGraph workflow."""
    result = run_query(request.query, request.document_id)
    return ChatResponse(**result)
