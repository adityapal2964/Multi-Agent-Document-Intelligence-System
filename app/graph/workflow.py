"""LangGraph wiring.

    START -> supervisor -> (retriever | summarizer | critic | final)
                  ^              |            |          |
                  |______________|____________|__________|

Every worker returns to the supervisor, which decides what happens next.
`final` assembles the response and ends the graph.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents import critic_agent, retriever_agent, summarizer_agent, supervisor_agent
from app.graph.state import AgentState, new_state
from app.rag.retriever import IndexMismatchError

logger = logging.getLogger(__name__)


def final_agent(state: AgentState) -> dict:
    """Collect the answer that will be returned to the user."""
    answer = state.get("answer") or "I could not produce an answer for this query."
    return {"final_answer": answer, "agent_trace": ["final"]}


def _route(state: AgentState) -> str:
    """Conditional edge: read the supervisor's decision from the state."""
    return state.get("next_agent") or "final"


def build_graph():
    """Compile the multi-agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("final", final_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {
            "retriever": "retriever",
            "summarizer": "summarizer",
            "critic": "critic",
            "final": "final",
        },
    )
    # workers always report back to the supervisor
    for worker in ("retriever", "summarizer", "critic"):
        graph.add_edge(worker, "supervisor")
    graph.add_edge("final", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Compiled graph singleton."""
    return build_graph()


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen, output = set(), []
    for source in sources or []:
        key = (source.get("filename"), source.get("page"), source.get("text", "")[:60])
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "filename": source.get("filename"),
                "page": source.get("page"),
                "score": source.get("score"),
                "snippet": " ".join((source.get("text") or "").split())[:400],
            }
        )
    return output


def run_query(query: str, document_id: str | None = None) -> dict[str, Any]:
    """Run one user query through the graph and return a clean dict."""
    if not (query or "").strip():
        return {
            "answer": "Please provide a question.",
            "agents_used": [], "sources": [], "tools_used": [],
            "critique": {}, "retries": 0,
        }

    try:
        result = get_graph().invoke(new_state(query, document_id))
    except IndexMismatchError:
        raise
    except Exception:
        logger.exception("Agent workflow failed")
        raise

    return {
        "answer": result.get("final_answer") or result.get("answer", ""),
        "agents_used": result.get("agent_trace", []),
        "sources": _dedupe_sources(result.get("sources", [])),
        "tools_used": sorted(set(result.get("tools_used", []))),
        "critique": result.get("critique", {}),
        "retries": result.get("retries", 0),
    }
