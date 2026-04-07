"""Document catalog: JSON index + markdown TOC, synced whenever files are ingested."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rag_common import CATALOG_PATH, DOCUMENTS_DIR, INDEX_MD_PATH, ROOT

# Minimal English stopwords for title/source matching
_STOP = frozenset(
    "a an the and or but in on at to for of is are was were be been being "
    "it this that these those with from as by not what which how when why "
    "who can could should would will do does did about into out up down"
    .split()
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _humanize_stem(stem: str) -> str:
    s = re.sub(r"[_\-]+", " ", stem)
    s = re.sub(r"\s+", " ", s).strip()
    return s or stem


def _pdf_title(path: Path) -> str | None:
    try:
        from pypdf import PdfReader

        r = PdfReader(str(path))
        meta = r.metadata
        if not meta:
            return None
        t = meta.get("/Title") or meta.get("Title")
        if t:
            s = str(t).strip()
            if s and s.lower() not in {"untitled", "title"}:
                return s
    except Exception:
        pass
    return None


def _document_title(path: Path, documents_root: Path) -> str:
    if path.suffix.lower() == ".pdf":
        t = _pdf_title(path)
        if t:
            return t
    return _humanize_stem(path.stem)


def migrate_legacy_data_dir() -> list[str]:
    """Copy supported files from legacy data/ into resources/documents/ if present."""
    legacy = ROOT / "data"
    if not legacy.is_dir():
        return []
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for pat in ("*.txt", "*.md", "*.markdown", "*.pdf"):
        for p in legacy.glob(pat):
            dest = DOCUMENTS_DIR / p.name
            if dest.exists():
                continue
            try:
                shutil.copy2(p, dest)
                moved.append(p.name)
            except OSError:
                pass
    return moved


def scan_documents(documents_dir: Path) -> list[dict]:
    """Build catalog entries for every supported file on disk."""
    documents_dir = documents_dir.resolve()
    if not documents_dir.is_dir():
        return []

    patterns = ("*.txt", "*.md", "*.markdown", "*.pdf")
    files: list[Path] = []
    for pat in patterns:
        files.extend(documents_dir.glob(pat))
    files = sorted({p.resolve() for p in files})

    now = _utc_now()
    entries: list[dict] = []
    for path in files:
        try:
            rel = str(path.relative_to(documents_dir))
        except ValueError:
            rel = path.name
        try:
            st = path.stat()
        except OSError:
            continue
        title = _document_title(path, documents_dir)
        entries.append(
            {
                "relative_path": rel,
                "title": title,
                "kind": path.suffix.lower().lstrip(".") or "unknown",
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "indexed_at": now,
            }
        )
    return entries


def write_catalog(entries: list[dict]) -> None:
    """Persist JSON catalog and human-readable DOCUMENT_INDEX.md."""
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _utc_now(),
        "documents_root": str(DOCUMENTS_DIR.resolve()),
        "documents": entries,
    }
    CATALOG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Document index",
        "",
        f"Auto-generated from files in `{DOCUMENTS_DIR.name}/`. "
        "Updated whenever you run ingest or upload from the web UI.",
        "",
        "| Title | File | Kind | Size (bytes) |",
        "| --- | --- | --- | ---: |",
    ]
    for e in entries:
        title = (e.get("title") or "").replace("|", "\\|")
        rp = (e.get("relative_path") or "").replace("|", "\\|")
        kind = e.get("kind", "")
        size = e.get("size_bytes", 0)
        lines.append(f"| {title} | `{rp}` | {kind} | {size} |")
    lines.append("")
    INDEX_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.is_file():
        return []
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return list(data.get("documents") or [])
    except (OSError, json.JSONDecodeError):
        return []


def _question_tokens(q: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z0-9]+", q.lower())
    return {w for w in raw if len(w) >= 2 and w not in _STOP}


def rank_sources_for_question(question: str, catalog: list[dict]) -> list[str]:
    """
    Return relative_path values most likely relevant to the question, for Chroma filtering.
    Empty list means search the full index (no title/path gate).
    """
    words = _question_tokens(question)
    if not words:
        return []

    scored: list[tuple[float, str]] = []
    for e in catalog:
        rp = e.get("relative_path") or ""
        title = (e.get("title") or "").lower()
        blob = f"{title} {rp.lower().replace('/', ' ')}"
        score = sum(1.0 for w in words if w in blob)
        # light boost for longer substring matches
        for w in words:
            if len(w) >= 4 and w in blob:
                score += 0.25
        if score > 0:
            scored.append((score, rp))

    scored.sort(key=lambda x: -x[0])
    # enough paths to not starve vector search, but still focus
    top = [p for _, p in scored[:12]]
    return top
