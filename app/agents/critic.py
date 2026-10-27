"""Critic / fact-checking agent.

Compares the drafted answer with the retrieved context and decides whether it
is grounded.  If it is not - and the retry budget allows it - the supervisor
sends the task back to the retriever.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import AgentState
from app.llm import llm_text

SYSTEM_PROMPT = """You are the final fact-checking worker. Compare the ANSWER
only with the supplied document CONTEXT and return a routing decision. Do not
rewrite the answer and do not request retrieval merely because the answer could
be more detailed.

Decision rules:
- grounded=true when every material document claim is supported by CONTEXT.
  A concise answer that honestly says the documents do not provide the answer
  is grounded.
- grounded=false when the answer contains an unsupported, contradicted, or
  missing material claim.
- needs_more_context=true only when additional document retrieval could
  plausibly supply the missing evidence. Set it false for a wrong answer,
  unsupported general knowledge, formatting issues, or evidence already
  present in CONTEXT.
- missing must be a short search-ready phrase only when more evidence is
  needed; otherwise use "".
- verdict must be one short sentence identifying the decisive reason.

Return exactly one JSON object with these types and no markdown:
{"grounded": true, "needs_more_context": false, "missing": "", "verdict": ""}"""

_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_OVERLAP_THRESHOLD = 0.35


def _context_text(state: AgentState) -> str:
    return "\n\n".join(chunk["text"] for chunk in (state.get("context") or []))


def _fallback_critique(answer: str, context: str) -> dict:
    """Lexical overlap check used when no LLM is configured."""
    if not context:
        return {
            "grounded": True,
            "needs_more_context": False,
            "missing": "",
            "verdict": "No document context to check against - accepted as is (heuristic).",
        }
    answer_words = set(_WORD_RE.findall(answer.lower()))
    context_words = set(_WORD_RE.findall(context.lower()))
    if not answer_words:
        return {"grounded": False, "needs_more_context": True, "missing": "an answer",
                "verdict": "The answer is empty (heuristic)."}
    overlap = len(answer_words & context_words) / len(answer_words)
    grounded = overlap >= _OVERLAP_THRESHOLD
    return {
        "grounded": grounded,
        "needs_more_context": not grounded,
        "missing": "" if grounded else "supporting sentences for the answer's claims",
        "verdict": f"Lexical overlap with the context is {overlap:.0%} (heuristic check).",
        "overlap": round(overlap, 3),
    }


def critic_agent(state: AgentState) -> dict:
    """LangGraph node: judge whether the answer is grounded in the context."""
    answer = state.get("answer", "")
    context = _context_text(state)

    raw = llm_text(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CONTEXT:\n{context or '(empty)'}\n\nANSWER:\n{answer}"),
        ],
        temperature=0,
    )

    critique: dict | None = None
    if raw:
        try:
            payload = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            critique = {
                "grounded": bool(payload.get("grounded", True)),
                "needs_more_context": bool(payload.get("needs_more_context", False)),
                "missing": str(payload.get("missing", "")),
                "verdict": str(payload.get("verdict", "")),
            }
        except Exception:
            critique = None
    if critique is None:
        critique = _fallback_critique(answer, context)

    verdict = "grounded" if critique["grounded"] else "not grounded"
    return {"critique": critique, "agent_trace": [f"critic ({verdict})"]}
