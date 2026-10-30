"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api.routes import router
from app.database.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables + seed the sample sales data on first boot
    print(f"[docuagent] LLM configured: {config.llm_configured()} | model: {config.LLM_MODEL}")
    print(f"[docuagent] Web search configured: {config.web_search_enabled()}")
    yield


app = FastAPI(
    title="DocuAgent",
    version="0.1.0",
    description="Multi-agent document intelligence system built with LangGraph.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {"name": "DocuAgent", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
