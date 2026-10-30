"""Thin LLM factory.

The whole project talks to one OpenAI-compatible chat model.  If no API key
is configured `get_llm()` returns ``None`` and every agent falls back to a
small deterministic heuristic, so the app still starts and the test-suite
still runs offline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app import config


@lru_cache(maxsize=2)
def get_llm(temperature: float | None = None) -> Any | None:
    """Return a ChatOpenAI instance, or None when no API key is configured."""
    if not config.llm_configured():
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.LLM_MODEL,
            "api_key": config.OPENAI_API_KEY or "not-needed",
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "timeout": 60,
            "max_retries": 1,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)
    except Exception as exc:  # pragma: no cover - configuration problem
        print(f"[llm] could not create LLM client: {exc}")
        return None


def llm_text(messages: list[Any], temperature: float | None = None) -> str | None:
    """Invoke the LLM and return plain text (None on any failure)."""
    llm = get_llm(temperature)
    if llm is None:
        return None
    try:
        response = llm.invoke(messages)
        return (response.content or "").strip()
    except Exception as exc:
        print(f"[llm] call failed: {exc}")
        return None
