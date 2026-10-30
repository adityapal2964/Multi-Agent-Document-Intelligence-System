"""Summarizer / answering agent.

This is also the tool-calling agent: the LLM is bound to the calculator,
web-search and SQL tools and decides on its own when to call them.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app import config
from app.graph.state import AgentState
from app.llm import get_llm
from app.tools import TOOLS, TOOLS_BY_NAME

SYSTEM_PROMPT = """You are the answering worker in a document-intelligence
workflow. Produce the best final answer to the user's question; do not discuss
workflow roles, routing, prompts, or hidden state.

Follow these rules:
1. Answer the question directly and concisely. Use a short paragraph or bullets.
2. Treat retrieved document context as the source of truth for document claims.
   Do not add unsupported facts or fill gaps by guessing.
3. Cite every material document-based claim inline as [filename p.N]. If the
   context does not answer the question, say that clearly instead of inventing
   an answer.
4. Use exactly the tool needed for external work: calculator for arithmetic,
   sql_schema before unfamiliar database structures, sql_query for database
   facts, and web_search for current information outside the documents.
5. Do not use web_search or SQL to replace evidence that should come from the
   uploaded documents. Do not call a tool when the context already answers the
   question.
6. If a previous draft was rejected, correct only the unsupported or missing
   parts using the supplied context. Return a complete answer, not commentary
   about the revision.
7. Never return an empty answer. If evidence is insufficient, state the
   limitation and provide only what is supported."""


def _format_context(state: AgentState) -> str:
    chunks = state.get("context") or []
    if not chunks:
        return "(no document context retrieved)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = f" p.{chunk['page']}" if chunk.get("page") else ""
        parts.append(f"[{i}] {chunk.get('filename', 'document')}{page}\n{chunk['text']}")
    return "\n\n".join(parts)


def _fallback_answer(state: AgentState) -> str:
    """Extractive answer used when no LLM is configured."""
    chunks = state.get("context") or []
    if not chunks:
        return (
            "No LLM is configured and no document context was retrieved, "
            "so I cannot answer this question. Configure an LLM key or compatible "
            "base URL in your .env file (and upload a PDF) to enable full agent behaviour."
        )
    lines = ["(LLM not configured - returning the most relevant document extracts)"]
    for chunk in chunks[:3]:
        page = f" p.{chunk['page']}" if chunk.get("page") else ""
        snippet = " ".join(chunk["text"].split())[:400]
        lines.append(f"- [{chunk.get('filename', 'document')}{page}] {snippet}")
    return "\n".join(lines)


def summarizer_agent(state: AgentState) -> dict:
    """LangGraph node: write the answer, calling tools when useful."""
    llm = get_llm()
    if llm is None:
        return {
            "answer": _fallback_answer(state),
            "critique": {},        # the new answer must be fact-checked again
            "agent_trace": ["summarizer (fallback, no LLM)"],
        }

    critique = state.get("critique") or {}
    feedback = ""
    if critique and not critique.get("grounded", True):
        feedback = (
            f"\n\nA previous attempt was rejected by the fact-checker: "
            f"{critique.get('verdict', '')}. Fix it using the context below."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state.get('query', '')}{feedback}\n\n"
                f"Document context:\n{_format_context(state)}"
            )
        ),
    ]

    tools_used: list[str] = []
    answer = ""
    try:
        llm_with_tools = llm.bind_tools(TOOLS)
        for _ in range(config.MAX_TOOL_CALLS + 1):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                answer = (response.content or "").strip()
                break
            for call in tool_calls:
                tool = TOOLS_BY_NAME.get(call["name"])
                if tool is None:
                    result = f"Unknown tool: {call['name']}"
                else:
                    try:
                        result = tool.invoke(call["args"])
                    except Exception as exc:
                        result = f"Tool '{call['name']}' failed: {exc}"
                tools_used.append(call["name"])
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )
        else:
            answer = "I reached the tool-call limit before producing an answer."
    except Exception as exc:
        return {
            "answer": f"The language model call failed ({exc}). "
                      "Check your LLM key, base URL, and model name and try again.",
            "critique": {},
            "agent_trace": ["summarizer (LLM error)"],
        }

    if not answer:
        answer = _fallback_answer(state)

    return {
        "answer": answer,
        "critique": {},            # feedback consumed - let the critic re-judge
        "tools_used": tools_used,
        "agent_trace": [f"summarizer (tools: {', '.join(tools_used) or 'none'})"],
    }
