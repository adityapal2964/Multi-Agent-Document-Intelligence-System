"""Read-only SQL tool over the local SQLite database."""

from __future__ import annotations

import sqlite3

from langchain_core.tools import tool

from app.database.database import get_connection, get_schema_description, init_db

MAX_ROWS = 50
_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
              "replace", "attach", "pragma", "vacuum")


def run_query(query: str) -> str:
    """Execute a single read-only SELECT statement and format the result."""
    sql = (query or "").strip().rstrip(";")
    if not sql:
        return "SQL error: empty query."
    lowered = sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "SQL error: only SELECT queries are allowed."
    if any(f" {word} " in f" {lowered} " for word in _FORBIDDEN):
        return "SQL error: write statements are not allowed."
    if ";" in sql:
        return "SQL error: only a single statement is allowed."

    try:
        init_db()  # make sure the tables exist
        with get_connection() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(MAX_ROWS)
            columns = [d[0] for d in cursor.description] if cursor.description else []
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"

    if not rows:
        return "Query executed successfully but returned no rows."

    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in tuple(r)) for r in rows)
    note = f"\n({len(rows)} row(s), truncated at {MAX_ROWS})" if len(rows) == MAX_ROWS else ""
    return f"{header}\n{'-' * len(header)}\n{body}{note}"


@tool("sql_query")
def sql_query(query: str) -> str:
    """Run a read-only SQL SELECT against the local SQLite analytics database.

    Tables available:
      documents(id, filename, file_path, num_pages, num_chunks, num_chars, status, uploaded_at)
      document_metadata(id, document_id, key, value)
      sales(id, region, product, quantity, amount, sale_date)

    Dates are stored as 'YYYY-MM-DD' text, so use SQLite date functions, e.g.
    WHERE sale_date >= date('now', '-1 month'). Only SELECT is permitted.
    """
    return run_query(query)


@tool("sql_schema")
def sql_schema() -> str:
    """Return the schema of the local SQLite database (tables and columns)."""
    init_db()
    return get_schema_description()
