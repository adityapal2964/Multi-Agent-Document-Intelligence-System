"""Minimal Streamlit UI for DocuAgent.

Run with:  streamlit run frontend/streamlit_app.py
It only talks to the FastAPI backend over HTTP - no agent logic lives here.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 180

st.set_page_config(page_title="DocuAgent", page_icon="📄", layout="wide")


def api_get(path: str):
    return requests.get(f"{BACKEND_URL}{path}", timeout=TIMEOUT)


# --------------------------------------------------------------------------
# Sidebar: upload + document list + backend status
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocuAgent")
    st.caption("Multi-agent document intelligence")

    try:
        health = api_get("/health").json()
        st.success("Backend: online")
        st.write(
            f"**LLM configured:** {'✅' if health['llm_configured'] else '❌'}  \n"
            f"**Web search:** {'✅' if health['web_search_configured'] else '❌'}  \n"
            f"**Embeddings:** `{health['embeddings_backend']}`  \n"
            f"**Indexed chunks:** {health['indexed_chunks']}"
        )
        if not health["llm_configured"]:
            st.warning("No OPENAI_API_KEY set - agents run in limited fallback mode.")
    except Exception as exc:
        st.error(f"Backend unreachable at {BACKEND_URL}\n\n{exc}")
        st.stop()

    st.divider()
    st.subheader("Upload a document")
    uploaded = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Process document", use_container_width=True):
        with st.spinner("Extracting, chunking and embedding..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=TIMEOUT,
                )
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(response.json().get("detail", response.text))
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.subheader("Indexed documents")
    try:
        documents = api_get("/documents").json()
        if not documents:
            st.info("No documents yet.")
        for document in documents:
            st.write(
                f"• **{document['filename']}** — {document['num_chunks']} chunks, "
                f"{document['num_pages']} page(s)"
            )
    except Exception as exc:
        st.error(f"Could not load documents: {exc}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Main area: chat
# --------------------------------------------------------------------------
st.header("Chat")
st.caption(
    "Try: *Summarize the uploaded document* · *What is 25% of 840?* · "
    "*What is the total sales amount in the database?*"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if meta:
            cols = st.columns(2)
            cols[0].caption("**Agents:** " + " → ".join(meta.get("agents_used", [])))
            cols[1].caption("**Tools:** " + (", ".join(meta.get("tools_used", [])) or "none"))
            if meta.get("critique"):
                with st.expander("Fact-check"):
                    st.json(meta["critique"])
            if meta.get("sources"):
                with st.expander(f"Sources ({len(meta['sources'])})"):
                    for source in meta["sources"]:
                        st.markdown(
                            f"**{source.get('filename')}** p.{source.get('page')} "
                            f"(distance {source.get('score')})\n\n{source.get('snippet')}"
                        )

if prompt := st.chat_input("Ask something about your documents, maths, the web or the database"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat", json={"query": prompt}, timeout=TIMEOUT
                )
                payload = response.json()
                answer = payload.get("answer") or payload.get("detail", "No answer returned.")
            except Exception as exc:
                payload, answer = {}, f"Request failed: {exc}"
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "meta": payload}
        )
    st.rerun()
