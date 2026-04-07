#!/usr/bin/env python3
"""
Streamlit UI for RAG Studio.

Run locally:
  streamlit run streamlit_app.py

Deploy (e.g. Streamlit Community Cloud): set secrets for OPENAI_API_KEY if using OpenAI;
place your project on GitHub and point Streamlit at this file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    import pandas as pd
    from ingest import ingest_all
    from rag_common import CATALOG_PATH, DATA_DIR, PUBLIC_DEMO_DIR, TOP_K
    from rag_core import default_backend, run_query
except ImportError as e:
    st.error(
        "Missing dependencies. From the project folder run: `pip install -r requirements.txt`"
    )
    st.caption(str(e))
    st.stop()


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return base[:180] if base else "upload"


def catalog_payload() -> dict:
    if CATALOG_PATH.is_file():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"documents": [], "updated_at": None}


st.set_page_config(page_title="RAG Studio", page_icon="📚", layout="wide")

PUBLIC_DEMO_MODE = os.environ.get("PUBLIC_DEMO_MODE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SHOW_DOC_METADATA = os.environ.get("SHOW_DOC_METADATA", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOW_USER_UPLOAD = os.environ.get("ALLOW_USER_UPLOAD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOW_USER_REINDEX = os.environ.get("ALLOW_USER_REINDEX", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTO_BOOTSTRAP_DEMO = os.environ.get("AUTO_BOOTSTRAP_DEMO", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@st.cache_resource(show_spinner=False)
def bootstrap_public_demo() -> tuple[bool, str]:
    """Index public demo files once when cloud app starts with no catalog."""
    if not AUTO_BOOTSTRAP_DEMO:
        return False, "AUTO_BOOTSTRAP_DEMO is disabled."
    if not PUBLIC_DEMO_DIR.is_dir():
        return False, f"Missing demo folder: {PUBLIC_DEMO_DIR}"
    return ingest_all(PUBLIC_DEMO_DIR)

st.title("RAG Studio")
if PUBLIC_DEMO_MODE:
    st.caption(
        "Public demo mode: private document details are hidden. "
        "Users can ask questions, but source files/chunks are not exposed."
    )
else:
    st.caption(
        "Documents live in `resources/documents/`. Titles are tracked in "
        "`resources/document_catalog.json` and updated on ingest."
    )

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Answer model",
        ("local", "openai"),
        index=0 if default_backend() == "local" else 1,
        format_func=lambda x: "Local Llama (GGUF)"
        if x == "local"
        else "OpenAI (needs OPENAI_API_KEY)",
    )
    k = st.slider("Chunks (k)", min_value=1, max_value=16, value=TOP_K)
    no_llm = st.checkbox("Retrieval only (no LLM)", value=False)
    st.divider()
    if ALLOW_USER_REINDEX and st.button(
        "Reindex from resources/documents", use_container_width=True
    ):
        with st.spinner("Indexing..."):
            ok, msg = ingest_all(DATA_DIR)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    elif not ALLOW_USER_REINDEX:
        st.caption("Reindex is disabled for end users.")

cat = catalog_payload()
docs = cat.get("documents") or []
if PUBLIC_DEMO_MODE and not docs:
    ok, msg = bootstrap_public_demo()
    cat = catalog_payload()
    docs = cat.get("documents") or []
    if ok:
        st.success("Loaded built-in public demo dataset.")
    elif "disabled" not in msg.lower():
        st.warning(msg)

if docs and SHOW_DOC_METADATA and not PUBLIC_DEMO_MODE:
    with st.expander(f"Document library ({len(docs)} files)", expanded=False):
        rows = [
            {"Title": d.get("title", ""), "File": d.get("relative_path", ""), "Kind": d.get("kind", "")}
            for d in docs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
elif not docs:
    st.info("No catalog yet. Add files under `resources/documents/` and reindex (sidebar).")
elif PUBLIC_DEMO_MODE:
    st.info("Document library is hidden in public demo mode.")

st.subheader("Ask a question")
question = st.text_area("Question", placeholder="e.g. What is time series forecasting?", height=100)

col_a, col_b = st.columns([1, 4])
with col_a:
    ask = st.button("Ask", type="primary", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Searching and generating…"):
            r = run_query(
                question.strip(),
                k=k,
                backend=backend,
                no_llm=no_llm,
            )
        if not r.ok:
            st.error(r.error or "Query failed.")
        else:
            if r.error and r.chunks:
                st.warning(r.error)
            elif r.error and not r.chunks:
                st.error(r.error)
            if r.answer:
                label = "OpenAI" if r.answer_backend == "openai" else "Local Llama"
                st.markdown(f"### Answer ({label})")
                st.markdown(r.answer)
            elif r.chunks and no_llm:
                st.info("Retrieval only — passages below; no LLM was run.")
            elif r.chunks and not r.answer and not r.error:
                st.warning("No generated answer (check local GGUF or OpenAI key).")

            if r.chunks and not PUBLIC_DEMO_MODE:
                st.subheader("Retrieved chunks")
                for i, c in enumerate(r.chunks, 1):
                    title = (c.get("title") or "").strip()
                    head = f"**[{i}]** `{c.get('source', '?')}`"
                    if title:
                        head += f" — {title}"
                    with st.expander(head, expanded=(i <= 2)):
                        st.text(c.get("text") or "")
            elif r.chunks and PUBLIC_DEMO_MODE:
                st.info(
                    "Sources/chunks are hidden in public demo mode to protect private resources."
                )
            elif r.ok:
                st.info("No matching chunks in the index.")

if ALLOW_USER_UPLOAD:
    st.divider()
    st.subheader("Upload a document")
    up = st.file_uploader(
        "Supported: .txt, .md, .pdf (max 20 MB)",
        type=("txt", "md", "markdown", "pdf"),
    )
    if up is not None:
        content = up.getvalue()
        if len(content) > 20 * 1024 * 1024:
            st.error("File too large (max 20 MB).")
        else:
            name = safe_filename(up.name)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            dest = DATA_DIR / name
            if st.button("Save and reindex", type="primary"):
                dest.write_bytes(content)
                with st.spinner("Ingesting..."):
                    ok, msg = ingest_all(DATA_DIR)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
