"""Web search tool backed by Tavily.

Fails gracefully (and explains how to configure itself) when no API key is
present, so the rest of the system keeps working offline.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app import config

_MISSING_KEY_MESSAGE = (
    "Web search is not configured. Add TAVILY_API_KEY=<your key> to your .env "
    "file (get a free key at https://tavily.com) and restart the API."
)


def search(query: str, max_results: int = 3) -> str:
    """Run a Tavily search and return a short plain-text digest."""
    if not config.web_search_enabled():
        return _MISSING_KEY_MESSAGE
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        payload = client.search(query=query, max_results=max_results,
                                include_answer=True)
    except ImportError:
        return "Web search failed: the 'tavily-python' package is not installed."
    except Exception as exc:
        return f"Web search failed: {exc}"

    lines: list[str] = []
    if payload.get("answer"):
        lines.append(f"Summary: {payload['answer']}")
    for i, item in enumerate(payload.get("results", [])[:max_results], start=1):
        snippet = (item.get("content") or "")[:400].replace("\n", " ")
        lines.append(f"[{i}] {item.get('title', 'untitled')} - {item.get('url', '')}\n{snippet}")
    return "\n\n".join(lines) if lines else "No web results found."


@tool("web_search")
def web_search(query: str) -> str:
    """Search the public web for current or external information.

    Use this when the answer is not in the uploaded documents or in the local
    database - for example recent news, public company facts or definitions.
    """
    return search(query)
