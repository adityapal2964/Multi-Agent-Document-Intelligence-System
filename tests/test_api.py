"""API-level tests (health, documents, chat)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["llm_configured"] is False          # no key in tests
    assert payload["embeddings_backend"] == "hashing-fallback"


def test_documents_endpoint_returns_list():
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_rejects_unsupported_file_type():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.exe", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_and_chat_roundtrip():
    content = (
        b"Quarterly update. Revenue grew to 42.5 million dollars in 2024, "
        b"up from 34.0 million dollars in 2023."
    )
    upload = client.post("/documents/upload",
                         files={"file": ("update.txt", content, "text/plain")})
    assert upload.status_code == 200, upload.text
    assert upload.json()["document"]["num_chunks"] >= 1

    listed = client.get("/documents").json()
    assert any(d["filename"] == "update.txt" for d in listed)

    chat = client.post("/chat", json={"query": "What was the revenue in 2024?"})
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["answer"]
    assert payload["agents_used"]          # the graph actually ran
    assert payload["sources"]              # retrieval found something


def test_chat_rejects_empty_query():
    assert client.post("/chat", json={"query": ""}).status_code == 422
