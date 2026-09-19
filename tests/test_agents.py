"""Agent tests with a fake LLM.

No API key is available in CI, so these tests inject a stub model to prove the
LLM-driven paths really work: supervisor JSON routing, the summarizer's
tool-calling loop, and the critic's JSON verdict parsing.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from app.agents import critic, summarizer, supervisor
from app.graph.state import new_state


class FakeLLM:
    """Returns the queued responses in order; records what it was given."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list] = []
        self.bound_tools: list = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_supervisor_parses_llm_json_route(monkeypatch):
    monkeypatch.setattr(
        supervisor, "llm_text",
        lambda *_a, **_k: '{"next": "summarizer", "reason": "maths question"}',
    )
    update = supervisor.supervisor_agent(new_state("What is 25% of 840?"))
    assert update["next_agent"] == "summarizer"
    assert "maths" in update["route_reason"]


def test_supervisor_recovers_from_garbage_llm_output(monkeypatch):
    monkeypatch.setattr(supervisor, "llm_text", lambda *_a, **_k: "no json here")
    update = supervisor.supervisor_agent(new_state("hello"))
    assert update["next_agent"] in supervisor.VALID_ROUTES


def test_summarizer_executes_a_tool_call(monkeypatch):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "0.25 * 840"},
                     "id": "call_1", "type": "tool_call"}],
    )
    final = AIMessage(content="25% of 840 is 210.")
    fake = FakeLLM([tool_call, final])
    monkeypatch.setattr(summarizer, "get_llm", lambda *_a, **_k: fake)

    update = summarizer.summarizer_agent(new_state("What is 25% of 840?"))
    assert update["answer"] == "25% of 840 is 210."
    assert update["tools_used"] == ["calculator"]
    # the tool result was fed back to the model before the final answer
    assert any("210" in str(m.content) for m in fake.calls[-1])
    assert {t.name for t in fake.bound_tools} >= {"calculator", "sql_query", "web_search"}


def test_summarizer_survives_llm_failure(monkeypatch):
    class Boom:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(summarizer, "get_llm", lambda *_a, **_k: Boom())
    update = summarizer.summarizer_agent(new_state("hello"))
    assert "failed" in update["answer"].lower()


def test_critic_parses_llm_verdict(monkeypatch):
    payload = {"grounded": False, "needs_more_context": True,
               "missing": "2023 revenue figure", "verdict": "Unsupported claim."}
    monkeypatch.setattr(critic, "llm_text", lambda *_a, **_k: json.dumps(payload))

    state = new_state("revenue?")
    state["answer"] = "Revenue tripled."
    state["context"] = [{"text": "Revenue was 42.5 million.", "filename": "r.pdf", "page": 1}]

    update = critic.critic_agent(state)
    assert update["critique"]["grounded"] is False
    assert update["critique"]["missing"] == "2023 revenue figure"


def test_critic_fallback_flags_ungrounded_answer():
    state = new_state("revenue?")
    state["answer"] = "The company launched a spaceship and acquired a bakery chain."
    state["context"] = [{"text": "Revenue was 42.5 million dollars in 2024.",
                         "filename": "r.pdf", "page": 1}]
    update = critic.critic_agent(state)          # no LLM configured in tests
    assert update["critique"]["grounded"] is False
    assert update["critique"]["needs_more_context"] is True


def test_critic_fallback_accepts_grounded_answer():
    state = new_state("revenue?")
    state["answer"] = "Revenue was 42.5 million dollars in 2024."
    state["context"] = [{"text": "Revenue was 42.5 million dollars in 2024.",
                         "filename": "r.pdf", "page": 1}]
    assert critic.critic_agent(state)["critique"]["grounded"] is True
