# Running DocuAgent Locally

Step-by-step, from a clean machine to a working chat. Takes about 5 minutes.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.10+** | 3.12 recommended (that's what this was built and tested on) |
| pip | ships with Python |
| ~500 MB disk | mostly for `faiss-cpu`, `streamlit` and the LangChain packages |
| An LLM API key | OpenAI, or any OpenAI-compatible provider. **Optional** — the app boots without one in a reduced fallback mode |
| A Tavily key | **Optional** — only needed for the web-search tool |

Check your Python:

```bash
python --version     # or python3 --version
```

---

## 2. Get the code

```bash
git clone <your-repo-url> docuagent
cd docuagent
```

(If you received a ZIP: `unzip docuagent.zip && cd docuagent`.)

---

## 3. Create a virtual environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Configure `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and set at least:

```env
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
TAVILY_API_KEY=             # optional, enables the web_search tool
```

Using a non-OpenAI provider (Groq, Together, Ollama, ...)? Add its base URL:

```env
OPENAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

For Ollama, use the local OpenAI-compatible endpoint:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:latest
EMBEDDING_MODEL=nomic-embed-text
```

No placeholder API key is required for local OpenAI-compatible servers. If you
change embedding provider or model, delete
`data/vectorstore/faiss_index/` and re-upload documents.

> **No key?** Everything still starts. The router falls back to a rule-based
> state machine, embeddings fall back to a local hashing vectoriser, and the
> summarizer returns document extracts instead of written answers. Good enough
> to see the plumbing work; not good enough to judge answer quality.

---

## 6. Initialize the database

```bash
python -m app.database.database
```

Expected output:

```text
Database ready at: .../data/database/docuagent.db
documents(id TEXT, filename TEXT, ...)
document_metadata(id INTEGER, document_id TEXT, key TEXT, value TEXT)
sales(id INTEGER, region TEXT, product TEXT, quantity INTEGER, amount REAL, sale_date TEXT)
```

This creates the tables and seeds 8 sample rows into `sales`. It is idempotent —
re-running it will not duplicate data. (The API also calls it on startup.)

---

## 7. Start the FastAPI backend

**Terminal 1** (keep it running):

```bash
uvicorn app.main:app --reload
```

You should see:

```text
[docuagent] LLM configured: True | model: gpt-4o-mini
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","llm_configured":true,"web_search_configured":false,
 "embeddings_backend":"openai-compatible","indexed_chunks":0,"documents":0,"model":"gpt-4o-mini"}
```

Interactive API docs: <http://127.0.0.1:8000/docs>

---

## 8. Start the Streamlit frontend

**Terminal 2** (activate the venv again first):

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

Open <http://localhost:8501>. The sidebar should say **Backend: online**.

---

## 9. Upload a document

**Option A — the UI:** sidebar → *Upload a document* → pick a PDF → **Process document**.
You'll get `Indexed N chunks from M page(s).`

**Option B — generate the bundled sample and upload it via curl:**

```bash
python scripts/create_sample_pdf.py
curl -F "file=@data/documents/sample_report.pdf" \
     http://127.0.0.1:8000/documents/upload
```

List what's indexed:

```bash
curl http://127.0.0.1:8000/documents
```

---

## 10. Test the system

In the Streamlit chat box (or via curl), try one query of each kind:

```bash
# RAG: retriever -> summarizer -> critic
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query":"What does the document say about revenue?"}'

# Tool calling: the model should call `calculator`
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query":"What is the percentage increase from 34.0 to 42.5?"}'

# Tool calling: the model should call `sql_query`
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query":"What is the total sales amount in the database?"}'

# Tool calling: needs TAVILY_API_KEY
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query":"Search the web for what LangGraph is."}'
```

Each response includes `agents_used` (the routing trace), `tools_used`,
`sources` and the critic's `critique` — that's how you confirm the multi-agent
flow actually ran.

Run the test suite (works without any API key):

```bash
pytest -v
```

Expected: **40 passed**.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | Run commands from the **project root** (where `app/` lives), with the venv active. |
| Streamlit: *Backend unreachable* | Terminal 1 isn't running, or the port differs. Start uvicorn, or set `BACKEND_URL=http://127.0.0.1:<port>` in `.env`. |
| `/health` shows `"llm_configured": false` | `.env` is missing, in the wrong directory, or both `OPENAI_API_KEY` and `OPENAI_BASE_URL` are blank. It must sit next to `app/`. Restart uvicorn after editing. |
| Answers look like raw document extracts | Same cause — you're in fallback mode. Add a key and restart. |
| `401 / invalid_api_key` in the answer text | Bad key, or the key doesn't match `OPENAI_BASE_URL`. |
| `model_not_found` | `LLM_MODEL` isn't available on your account/provider. Try `gpt-4o-mini`. |
| Upload: *No extractable text found* | Scanned/image-only PDF. There's no OCR in this MVP — try a text-based PDF. |
| Upload: *Unsupported file type* | Only `.pdf`, `.txt`, `.md` are accepted. |
| Web search: *not configured* | Add `TAVILY_API_KEY` to `.env`, `pip install tavily-python`, restart. |
| `pip install faiss-cpu` fails | Needs Python 3.10–3.12 and a 64-bit interpreter; upgrade pip first. |
| Port 8000/8501 already in use | `uvicorn app.main:app --port 8010` and `streamlit run ... --server.port 8511` (update `BACKEND_URL`). |
| Retrieval returns nothing after re-upload | Delete `data/vectorstore/` and re-upload — the index is rebuilt from scratch. |
| Stale/odd answers after switching LLM providers | Embeddings changed too. Delete `data/vectorstore/` and re-ingest your documents. |

---

## 12. Reset everything

```bash
rm -rf data/vectorstore/faiss_index data/database/docuagent.db data/documents/*
python -m app.database.database
```

Then re-upload your documents.
