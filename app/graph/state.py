"""Shared state passed between every node of the LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import operator


class AgentState(TypedDict, total=False):
    # input
    query: str
    document_id: str | None

    # routing
    next_agent: str                      # retriever | summarizer | critic | final
    route_reason: str
    steps: int                           # hard loop guard
    agent_trace: Annotated[list[str], operator.add]

    # retrieval
    context: list[dict[str, Any]]        # retrieved chunks
    sources: list[dict[str, Any]]        # de-duplicated chunks for the UI
    retries: int                         # critic -> retriever loops used

    # answering
    answer: str
    tools_used: Annotated[list[str], operator.add]

    # critique
    critique: dict[str, Any]             # {grounded, verdict, needs_more_context}

    # output
    final_answer: str


def new_state(query: str, document_id: str | None = None) -> AgentState:
    return AgentState(
        query=query,
        document_id=document_id,
        next_agent="",
        route_reason="",
        steps=0,
        agent_trace=[],
        context=[],
        sources=[],
        retries=0,
        answer="",
        tools_used=[],
        critique={},
        final_answer="",
    )
