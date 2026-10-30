"""Test setup: isolate all data under a temporary directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point the app at a throwaway data directory BEFORE app.config is imported.
_TMP = Path(tempfile.mkdtemp(prefix="docuagent-tests-"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["SQLITE_PATH"] = str(_TMP / "test.db")
os.environ["OPENAI_API_KEY"] = ""   # tests run in offline fallback mode
os.environ["OPENAI_BASE_URL"] = ""
os.environ["LLM_ENABLED"] = ""
os.environ["TAVILY_API_KEY"] = ""   # tests run in offline fallback mode

import pytest  # noqa: E402

from app.database.database import init_db  # noqa: E402
from app.rag import retriever as vector_store  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_database():
    init_db()
    yield
    vector_store.reset()


@pytest.fixture()
def sample_text_file(tmp_path) -> Path:
    path = tmp_path / "policy.txt"
    path.write_text(
        "Acme Corp refund policy. Customers may request a refund within 30 days "
        "of purchase. Refunds are processed within 5 business days. "
        "Enterprise contracts are excluded from the standard refund window.\n\n"
        "Shipping policy. Orders above 100 dollars ship free of charge.",
        encoding="utf-8",
    )
    return path
