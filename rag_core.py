"""Shared RAG retrieval and generation for CLI and web UI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from catalog import load_catalog, rank_sources_for_question
from rag_common import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_LLAMA_GGUF,
    EMBED_MODEL,
    TOP_K,
)

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

SYSTEM_PROMPT = """You are a helpful assistant. Answer using ONLY the provided context.
If the context does not contain enough information, say you don't know based on the documents.
Keep answers concise."""


def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"""Context from documents:

{ctx}

Question: {question}

Answer:"""


def resolve_llama_path() -> Path | None:
    raw = os.environ.get("LOCAL_LLAMA_PATH")
    if raw:
        p = Path(raw).expanduser().resolve()
        return p if p.is_file() else None
    return DEFAULT_LLAMA_GGUF if DEFAULT_LLAMA_GGUF.is_file() else None


_llama = None


def get_llama():
    global _llama
    if _llama is not None:
        return _llama
    path = resolve_llama_path()
    if path is None:
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        return None
    try:
        _llama = Llama(
            model_path=str(path),
            n_ctx=int(os.environ.get("LLAMA_N_CTX", "8192")),
            n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
            verbose=False,
        )
    except Exception:
        return None
    return _llama


def answer_local_llama(prompt: str) -> str:
    llm = get_llama()
    if llm is None:
        return ""
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("LLAMA_MAX_TOKENS", "512")),
    )
    msg = (out.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def answer_openai(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
    except ImportError:
        return ""
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()


def default_backend() -> str:
    b = (os.environ.get("LLM_BACKEND") or "local").strip().lower()
    if b in {"local", "openai"}:
        return b
    return "local"


@dataclass
class QueryResult:
    ok: bool
    chunks: list[dict]
    answer: str | None
    answer_backend: str | None
    error: str | None


def run_query(
    question: str,
    k: int | None = None,
    backend: str | None = None,
    no_llm: bool = False,
) -> QueryResult:
    k = k if k is not None else TOP_K
    if not question.strip():
        return QueryResult(False, [], None, None, "Question is empty.")
    if not CHROMA_DIR.is_dir():
        return QueryResult(
            False,
            [],
            None,
            None,
            f"No index at {CHROMA_DIR}. Run ingest or use the web UI to add documents.",
        )

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    q = question.strip()
    catalog = load_catalog()
    filter_sources = rank_sources_for_question(q, catalog)

    docs: list = []
    metas: list = []
    if filter_sources:
        try:
            res = collection.query(
                query_texts=[q],
                n_results=k,
                where={"source": {"$in": filter_sources}},
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
        except Exception:
            docs, metas = [], []

    if not docs:
        res = collection.query(query_texts=[q], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

    chunks: list[dict] = []
    for doc, meta in zip(docs, metas):
        m = meta or {}
        chunks.append(
            {
                "source": m.get("source", "?"),
                "title": m.get("title") or "",
                "text": doc or "",
            }
        )

    if not chunks:
        return QueryResult(True, [], None, None, None)

    if no_llm:
        return QueryResult(True, chunks, None, None, None)

    be = (backend or default_backend()).lower()
    prompt = build_prompt(question, [c["text"] for c in chunks])

    if be == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return QueryResult(
                True,
                chunks,
                None,
                None,
                "OPENAI_API_KEY is not set. Choose local Llama or set the key.",
            )
        ans = answer_openai(prompt)
        if not ans:
            return QueryResult(True, chunks, None, None, "OpenAI request failed.")
        return QueryResult(True, chunks, ans, "openai", None)

    ans = answer_local_llama(prompt)
    if ans:
        return QueryResult(True, chunks, ans, "local", None)

    if resolve_llama_path() is None:
        return QueryResult(
            True,
            chunks,
            None,
            None,
            f"No GGUF at {DEFAULT_LLAMA_GGUF}. Run: python download_llm.py",
        )
    return QueryResult(
        True,
        chunks,
        None,
        None,
        "Local model failed to load. Install: pip install llama-cpp-python",
    )
