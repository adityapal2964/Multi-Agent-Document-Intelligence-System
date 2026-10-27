"""Graph tests: routing, execution, retry guard."""

from app.graph.state import new_state
from app.graph.workflow import build_graph, run_query
from app.agents.supervisor import VALID_ROUTES, supervisor_agent
from app.rag import retriever as vector_store
from app.rag.ingestion import load_and_chunk


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for node in ("supervisor", "retriever", "summarizer", "critic", "final"):
        assert node in nodes


def test_supervisor_returns_a_valid_route():
    update = supervisor_agent(new_state("What is 25% of 840?"))
    assert update["next_agent"] in VALID_ROUTES
    assert update["steps"] == 1


def test_supervisor_never_critiques_before_an_answer_exists():
    state = new_state("anything")
    state["critique"] = {}
    update = supervisor_agent(state)
    assert update["next_agent"] != "critic"


def test_supervisor_moves_to_summarizer_after_retrieval(monkeypatch):
    """A model must not be allowed to retrieve repeatedly before answering."""
    monkeypatch.setattr(
        "app.agents.supervisor.llm_text",
        lambda *_a, **_k: '{"next": "retriever", "reason": "need evidence"}',
    )
    state = new_state("What is the refund policy?")
    state["context"] = [{"text": "Refunds are processed within 7 business days."}]

    update = supervisor_agent(state)

    assert update["next_agent"] == "summarizer"
    assert "context already retrieved" in update["route_reason"]


def test_run_query_without_documents_still_answers():
    result = run_query("What is 25% of 840?")
    assert result["answer"]
    assert "summarizer" in " ".join(result["agents_used"])


def test_full_workflow_with_a_document(sample_text_file):
    chunks, _ = load_and_chunk(sample_text_file, document_id="graphdoc")
    vector_store.add_documents(chunks)

    result = run_query("What is the refund policy?")
    trace = " ".join(result["agents_used"])
    assert "retriever" in trace and "summarizer" in trace
    assert result["sources"]
    assert result["retries"] <= 1          # retry budget respected
    assert result["answer"]


def test_empty_query_is_handled():
    assert run_query("  ")["answer"] == "Please provide a question."


def test_critic_triggers_one_retrieval_retry(monkeypatch, sample_text_file):
    """Critic rejects the first answer -> the graph retrieves and answers again."""
    from app.agents import critic

    chunks, _ = load_and_chunk(sample_text_file, document_id="retrydoc")
    vector_store.add_documents(chunks)

    verdicts = [
        '{"grounded": false, "needs_more_context": true, '
        '"missing": "refund timeframe", "verdict": "Not supported."}',
        '{"grounded": true, "needs_more_context": false, '
        '"missing": "", "verdict": "Supported by the context."}',
    ]
    monkeypatch.setattr(critic, "llm_text",
                        lambda *_a, **_k: verdicts.pop(0) if verdicts else verdicts)

    result = run_query("What is the refund policy?")
    trace = result["agents_used"]
    assert result["retries"] == 1
    assert sum(1 for step in trace if step.startswith("retriever")) == 2
    assert sum(1 for step in trace if step.startswith("critic")) == 2
    assert trace[-1] == "final"


def test_full_workflow_in_simulated_llm_mode(monkeypatch):
    """Supervisor JSON routing + real tool execution + final answer, end to end."""
    from langchain_core.messages import AIMessage

    from app.agents import critic, summarizer, supervisor as sup

    monkeypatch.setattr(sup, "llm_text",
                        lambda *_a, **_k: '{"next": "summarizer", "reason": "db question"}')
    monkeypatch.setattr(critic, "llm_text",
                        lambda *_a, **_k: '{"grounded": true, "needs_more_context": false,'
                                          ' "missing": "", "verdict": "ok"}')

    scripted = [
        AIMessage(content="", tool_calls=[{"name": "sql_query",
                                           "args": {"query": "SELECT SUM(amount) AS total FROM sales"},
                                           "id": "c1", "type": "tool_call"}]),
        AIMessage(content="Total sales are 119700 USD."),
    ]

    class Fake:
        def bind_tools(self, _tools):
            return self

        def invoke(self, messages):
            self.last = messages
            return scripted.pop(0)

    fake = Fake()
    monkeypatch.setattr(summarizer, "get_llm", lambda *_a, **_k: fake)

    result = run_query("What is the total sales amount?")
    assert result["answer"] == "Total sales are 119700 USD."
    assert result["tools_used"] == ["sql_query"]
    assert result["agents_used"][-1] == "final"
    # the real SQL tool ran and its output was fed back to the model
    assert any("119700" in str(m.content) for m in fake.last)
