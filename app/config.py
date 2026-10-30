"""Central configuration.

Everything is read from environment variables (loaded from a local `.env`
file when present).  Sensible defaults are provided so the project can be
started even before any API key is configured.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from the current working directory if it exists

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DOCUMENTS_DIR = DATA_DIR / "documents"
DATABASE_DIR = DATA_DIR / "database"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

for _directory in (DOCUMENTS_DIR, DATABASE_DIR, VECTORSTORE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(DATABASE_DIR / "docuagent.db")))

# --------------------------------------------------------------------------
# LLM / embeddings
# --------------------------------------------------------------------------
# Any OpenAI-compatible endpoint works (OpenAI, Groq, Together, Ollama, ...).
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", "").strip() or None
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()

# --------------------------------------------------------------------------
# RAG / graph knobs
# --------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K: int = int(os.getenv("TOP_K", "4"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "1"))      # critic -> retriever loops
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "8"))          # hard stop for the graph
MAX_TOOL_CALLS: int = int(os.getenv("MAX_TOOL_CALLS", "3"))

# --------------------------------------------------------------------------
# Servers
# --------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
BACKEND_URL: str = os.getenv("BACKEND_URL", f"http://{API_HOST}:{API_PORT}")
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")


def llm_configured() -> bool:
    """True when an LLM key or OpenAI-compatible endpoint is configured."""
    override = os.getenv("LLM_ENABLED", "").strip().lower()
    if override in {"true", "false"}:
        return override == "true"
    return bool(
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )


def cors_origins() -> list[str]:
    """Return configured CORS origins, excluding blank entries."""
    return [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]


def web_search_enabled() -> bool:
    return bool(TAVILY_API_KEY)
