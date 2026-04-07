#!/usr/bin/env python3
"""Load .txt / .md / .pdf from resources/documents/, chunk, embed, store in Chroma."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from catalog import migrate_legacy_data_dir, scan_documents, write_catalog
from rag_common import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
)

from pypdf import PdfReader

try:
    from chromadb import PersistentClient
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError as e:
    print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported type: {path}")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _truncate_meta(s: str, max_len: int = 512) -> str:
    s = s.strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def ingest_all(data_dir: Path | None = None) -> tuple[bool, str]:
    """Rebuild Chroma from all supported files under data_dir. Updates catalog + DOCUMENT_INDEX.md."""
    migrated = migrate_legacy_data_dir()
    data_dir = (data_dir or DATA_DIR).resolve()
    if not data_dir.is_dir():
        return False, f"Documents directory not found: {data_dir}"

    patterns = ("*.txt", "*.md", "*.markdown", "*.pdf")
    files: list[Path] = []
    for pat in patterns:
        files.extend(data_dir.glob(pat))
    files = sorted({p.resolve() for p in files})

    if not files:
        return False, f"No supported files in {data_dir}"

    catalog_entries = scan_documents(data_dir)
    title_map = {e["relative_path"]: e["title"] for e in catalog_entries}

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    skipped: list[str] = []
    for path in files:
        try:
            raw = load_document(path)
        except Exception as e:
            skipped.append(f"{path.name}: {e}")
            continue
        chunks = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        try:
            rel = str(path.relative_to(data_dir))
        except ValueError:
            rel = path.name
        title = _truncate_meta(str(title_map.get(rel, rel)))
        for i, chunk in enumerate(chunks):
            cid = f"{rel}::{i}"
            ids.append(cid)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": rel,
                    "chunk": i,
                    "title": title,
                }
            )

    if not documents:
        return False, "Nothing to ingest after chunking."

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    write_catalog(catalog_entries)

    msg = f"Ingested {len(documents)} chunks from {len(files)} file(s) into {CHROMA_DIR}"
    if migrated:
        msg += ". Migrated from legacy data/: " + ", ".join(migrated)
    if skipped:
        msg += ". Skipped: " + "; ".join(skipped)
    return True, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder with .txt, .md, .pdf files (default: resources/documents)",
    )
    args = parser.parse_args()
    ok, msg = ingest_all(args.data_dir)
    if not ok:
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    print(msg)


if __name__ == "__main__":
    main()
