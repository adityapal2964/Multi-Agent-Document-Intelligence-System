"""Supervisor / router agent.

The LLM looks at the current state of the task and picks the next worker:
`retriever`, `summarizer`, `critic` or `final`.  It is a real LLM decision -
not an if/else chain over the user's wording.  A deterministic fallback
router is used only when no API key is configured (or the LLM call fails),
and a small `_sanitize` guard keeps the graph from looping forever.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app import config
from app.graph.state import AgentState
from app.llm import llm_text
from app.rag import retriever as vector_store

VALID_ROUTES = {"retriever", "summarizer", "critic", "final"}

SYSTEM_PROMPT = """You are the supervisor for a document-intelligence workflow.
Your only job is to choose the single next route from the current state. Do not
answer the user, perform retrieval, or invent state.

Available routes:
- retriever: search the uploaded documents and add evidence to context.
- summarizer: write or rewrite the answer. It can call calculator, web_search,
  sql_query, and sql_schema.
- critic: check an existing answer against retrieved document context.
- final: stop because the answer is accepted or no useful action remains.

Use this decision policy in order:
1. If an answer exists and the latest critique says grounded=true, choose final.
2. If an answer exists and the critique says grounded=false and
   needs_more_context=true, choose retriever, unless retries are exhausted.
3. If an answer exists and the critique says grounded=false but
   needs_more_context=false, choose summarizer so it can correct the answer.
4. If an answer exists, document context exists, and no critique exists, choose
   critic.
5. If no answer exists and the question requires document facts with no
   retrieved context, choose retriever. This includes questions about uploaded
   files, even when documents are indexed.
6. If no answer exists and the question is general, mathematical, web-based, or
   database-related, choose summarizer directly. Do not retrieve just because
   documents are available.
7. If no answer exists but context is already present, choose summarizer.
8. Choose final only when an answer exists. Never choose critic before an answer.

Prefer the route that makes concrete progress. Do not repeatedly retrieve
without critic feedback, do not send a grounded answer back for rewriting, and
do not use critic for answers that were not based on document context. Respect
the retry and step counts in the state; when a limit prevents more useful work,
choose final.

The workflow phases are strict: retrieve at most once before the first answer,
then summarize, then critic. A second retrieval is allowed only after an
existing answer was explicitly rejected with needs_more_context=true.

Return exactly one JSON object and no markdown or extra text:
{"next": "retriever|summarizer|critic|final", "reason": "brief state-based reason"}"""


def _state_digest(state: AgentState) -> str:
    critique = state.get("critique") or {}
    return json.dumps(
        {
            "user_query": state.get("query", ""),
            "documents_indexed": vector_store.stats()["indexed_chunks"],
            "context_chunks_retrieved": len(state.get("context") or []),
            "answer_written": bool(state.get("answer")),
            "critic_verdict": critique.get("verdict", None),
            "critic_says_grounded": critique.get("grounded", None),
            "retries_used": state.get("retries", 0),
            "retries_allowed": config.MAX_RETRIES,
        },
        indent=2,
    )


def _fallback_route(state: AgentState) -> tuple[str, str]:
    """Deterministic router used when the LLM is unavailable."""
    has_context = bool(state.get("context"))
    has_answer = bool(state.get("answer"))
    critique = state.get("critique") or {}
    indexed = vector_store.stats()["indexed_chunks"] > 0

    if not has_answer and not has_context and indexed:
        return "retriever", "no context yet - retrieve first (fallback router)"
    if not has_answer:
        return "summarizer", "context ready (or no documents) - write an answer (fallback router)"
    if has_context and not critique:
        return "critic", "answer written - fact-check it (fallback router)"
    if critique and not critique.get("grounded", True) and state.get("retries", 0) < config.MAX_RETRIES:
        return "retriever", "critic asked for more evidence (fallback router)"
    return "final", "answer accepted (fallback router)"


def _sanitize(route: str, state: AgentState) -> tuple[str, str]:
    """Keep the LLM's choice legal and guarantee termination."""
    note = ""
    if route not in VALID_ROUTES:
        route, note = _fallback_route(state)[0], "invalid route corrected"

    if state.get("steps", 0) >= config.MAX_STEPS:
        return "final", "step limit reached"
    if route == "critic" and not state.get("answer"):
        return "summarizer", "cannot critique before an answer exists"
    if route == "retriever":
        if state.get("context") and not state.get("answer"):
            return "summarizer", "context already retrieved - write the first answer"
        if vector_store.stats()["indexed_chunks"] == 0:
            return "summarizer", "no documents indexed - answer with tools/knowledge"
        if state.get("answer") and state.get("retries", 0) >= config.MAX_RETRIES:
            return "final", "retry budget exhausted"
    if route == "summarizer" and state.get("answer"):
        # An answer already exists. Only rewrite it when the critic rejected it
        # and we still have retry budget - otherwise move the task forward.
        critique = state.get("critique") or {}
        rewrite_requested = (
            bool(critique)
            and not critique.get("grounded", True)
            and state.get("retries", 0) < config.MAX_RETRIES
        )
        if not rewrite_requested:
            if state.get("context") and not critique:
                return "critic", "answer already written - fact-check it"
            return "final", "answer already written and nothing left to improve"
    if route == "final" and not state.get("answer"):
        return "summarizer", "cannot finish without an answer"
    return route, note or ""


def supervisor_agent(state: AgentState) -> dict:
    """LangGraph node: decide the next worker."""
    raw = llm_text(
        [SystemMessage(content=SYSTEM_PROMPT),
         HumanMessage(content=f"Current task state:\n{_state_digest(state)}")],
        temperature=0,
    )

    route, reason = "", ""
    if raw:
        try:
            payload = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            route = str(payload.get("next", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
        except Exception:
            route = ""
    if route not in VALID_ROUTES:
        route, reason = _fallback_route(state)

    route, note = _sanitize(route, state)
    if note:
        reason = f"{reason} | {note}" if reason else note

    return {
        "next_agent": route,
        "route_reason": reason,
        "steps": state.get("steps", 0) + 1,
        "agent_trace": [f"supervisor -> {route}"],
    }
