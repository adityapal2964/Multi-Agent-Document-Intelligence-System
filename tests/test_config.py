from app import config


def test_llm_configured_is_false_without_key_or_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_ENABLED", raising=False)

    assert config.llm_configured() is False


def test_llm_configured_is_true_with_base_url_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("LLM_ENABLED", raising=False)

    assert config.llm_configured() is True


def test_llm_enabled_false_overrides_configured_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    monkeypatch.setenv("LLM_ENABLED", "false")

    assert config.llm_configured() is False


def test_cors_origins_are_stripped_and_empty_entries_ignored(monkeypatch):
    monkeypatch.setattr(config, "CORS_ORIGINS", "http://a.com, http://b.com, ")

    assert config.cors_origins() == ["http://a.com", "http://b.com"]
