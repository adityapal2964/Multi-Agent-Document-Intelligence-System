"""Retriever agent: vector search over the ingested documents."""

from __future__ import annotations

from app import config
from app.graph.state import AgentState
from app.rag import retriever as vector_store


def _search_query(state: AgentState) -> str:
    """On a retry, fold the critic's feedback into the search query."""
    query = state.get("query", "")
    critique = state.get("critique") or {}
    if state.get("retries", 0) > 0 and critique.get("missing"):
        return f"{query} {critique['missing']}"
    return query


def retriever_agent(state: AgentState) -> dict:
    """LangGraph node: fetch relevant chunks into `context`."""
    is_retry = bool(state.get("answer"))          # we already answered once
    top_k = config.TOP_K * 2 if is_retry else config.TOP_K   # widen on retry

    hits = vector_store.search(
        _search_query(state), k=top_k, document_id=state.get("document_id")
    )

    update: dict = {
        "context": hits,
        "sources": hits,
        "agent_trace": [f"retriever ({len(hits)} chunks, k={top_k})"],
    }
    if is_retry:
        update["retries"] = state.get("retries", 0) + 1
        update["answer"] = ""        # force the summarizer to rewrite
    if not hits:
        update["route_reason"] = "retrieval returned no chunks"
    return update
