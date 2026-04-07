"""Shared settings for ingest and query."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Writable cache when ~/.cache/huggingface is not available
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))

# Canonical document store (all uploads and ingests use this folder)
RESOURCES_DIR = ROOT / "resources"
DOCUMENTS_DIR = RESOURCES_DIR / "documents"
PUBLIC_DEMO_DIR = RESOURCES_DIR / "public_demo"
CATALOG_PATH = RESOURCES_DIR / "document_catalog.json"
INDEX_MD_PATH = RESOURCES_DIR / "DOCUMENT_INDEX.md"

# Backward-compatible name used across the codebase
DATA_DIR = DOCUMENTS_DIR

CHROMA_DIR = ROOT / "chroma_db"
MODELS_DIR = ROOT / "models"
# Public GGUF (Llama 3.2 1B Instruct, ~808 MiB Q4_K_M)
HF_LLAMA_REPO = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
HF_LLAMA_FILE = "llama-3.2-1b-instruct-q4_k_m.gguf"
DEFAULT_LLAMA_GGUF = MODELS_DIR / HF_LLAMA_FILE
COLLECTION_NAME = "documents"
# Hugging Face id (sentence-transformers/all-MiniLM-L6-v2 also works)
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K = 4
