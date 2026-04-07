#!/usr/bin/env python3
"""Browser UI for the RAG pipeline."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from ingest import ingest_all
from rag_common import CATALOG_PATH, DATA_DIR, ROOT, TOP_K
from rag_core import default_backend, run_query

STATIC_DIR = ROOT / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return base[:180] if base else "upload"


class QueryBody(BaseModel):
    question: str = ""
    k: int = Field(default=TOP_K, ge=1, le=50)
    backend: str | None = None
    no_llm: bool = False


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/catalog")
async def api_catalog() -> dict:
    """JSON catalog + titles (same as resources/document_catalog.json)."""
    if not CATALOG_PATH.is_file():
        return {"version": 1, "updated_at": None, "documents": []}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        return HTMLResponse("<p>Missing web/static/index.html</p>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/api/query")
async def api_query(body: QueryBody) -> JSONResponse:
    be = (body.backend or default_backend()).lower()
    if be not in {"local", "openai"}:
        be = "local"
    r = run_query(
        body.question,
        k=body.k,
        backend=be,
        no_llm=body.no_llm,
    )
    return JSONResponse(
        {
            "ok": r.ok,
            "chunks": r.chunks,
            "answer": r.answer,
            "answer_backend": r.answer_backend,
            "error": r.error,
        }
    )


@app.post("/api/reindex")
async def api_reindex() -> dict:
    ok, msg = ingest_all(DATA_DIR)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".txt", ".md", ".markdown", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, .markdown, .pdf are supported",
        )
    name = safe_filename(file.filename)
    dest = DATA_DIR / name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20 MB)")
    dest.write_bytes(content)
    ok, msg = ingest_all(DATA_DIR)
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    return {"ok": True, "message": msg, "saved_as": name}


def main() -> None:
    import uvicorn

    host = os.environ.get("RAG_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_WEB_PORT", "8765"))
    uvicorn.run(
        "web_app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
