"""Tiny SQLite layer.

Three tables only:
  * documents          - one row per ingested file
  * document_metadata  - key/value extras for a document
  * sales              - a small sample business dataset for the SQL tool
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    num_pages    INTEGER DEFAULT 0,
    num_chunks   INTEGER DEFAULT 0,
    num_chars    INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'processed',
    uploaded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS sales (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    region    TEXT NOT NULL,
    product   TEXT NOT NULL,
    quantity  INTEGER NOT NULL,
    amount    REAL NOT NULL,
    sale_date TEXT NOT NULL
);
"""

SAMPLE_SALES = [
    ("North", "Enterprise Plan", 12, 24000.0, 35),
    ("North", "Pro Plan", 40, 16000.0, 28),
    ("South", "Enterprise Plan", 7, 14000.0, 20),
    ("South", "Starter Plan", 90, 9000.0, 15),
    ("East", "Pro Plan", 33, 13200.0, 12),
    ("East", "Starter Plan", 51, 5100.0, 9),
    ("West", "Enterprise Plan", 15, 30000.0, 5),
    ("West", "Pro Plan", 21, 8400.0, 3),
]


def get_connection() -> sqlite3.Connection:
    """Open a connection with dict-like rows."""
    config.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(seed: bool = True) -> Path:
    """Create tables (idempotent) and seed the sample `sales` data once."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        if seed and conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 0:
            today = datetime.now(timezone.utc)
            rows = [
                (region, product, qty, amount,
                 (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"))
                for region, product, qty, amount, days_ago in SAMPLE_SALES
            ]
            conn.executemany(
                "INSERT INTO sales (region, product, quantity, amount, sale_date)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    return config.SQLITE_PATH


def record_document(doc: dict[str, Any]) -> None:
    """Insert (or replace) a document row plus its metadata rows."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, filename, file_path, num_pages, num_chunks, num_chars, status, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc["id"], doc["filename"], doc["file_path"], doc.get("num_pages", 0),
                doc.get("num_chunks", 0), doc.get("num_chars", 0),
                doc.get("status", "processed"),
                doc.get("uploaded_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ),
        )
        conn.execute("DELETE FROM document_metadata WHERE document_id = ?", (doc["id"],))
        for key, value in (doc.get("metadata") or {}).items():
            conn.execute(
                "INSERT INTO document_metadata (document_id, key, value) VALUES (?, ?, ?)",
                (doc["id"], key, str(value)),
            )


def list_documents() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_schema_description() -> str:
    """Human/LLM readable schema, used by the SQL tool prompt."""
    with get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        parts: list[str] = []
        for (name,) in [(t["name"],) for t in tables]:
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            col_text = ", ".join(f"{c['name']} {c['type']}" for c in cols)
            parts.append(f"{name}({col_text})")
    return "\n".join(parts)


if __name__ == "__main__":  # python -m app.database.database
    path = init_db()
    print(f"Database ready at: {path}")
    print(get_schema_description())
