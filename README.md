# RAG Studio (Private-by-Default)

A local Retrieval-Augmented Generation (RAG) app with:
- CLI query (`rag.py`)
- Browser UI via FastAPI (`web_app.py`)
- Streamlit UI (`streamlit_app.py`)

It uses:
- ChromaDB for vector search
- SentenceTransformers embeddings
- Local Llama GGUF (`llama-cpp-python`) or OpenAI for generation

## Project layout

- `resources/documents/` - your source files (`.txt`, `.md`, `.pdf`)
- `resources/document_catalog.json` - auto-generated catalog (ignored by git)
- `resources/DOCUMENT_INDEX.md` - auto-generated markdown index (ignored by git)
- `chroma_db/` - vector index (ignored by git)

## Quick start (local)

```bash
cd RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add documents to `resources/documents/`, then index:

```bash
python ingest.py
```

Ask from CLI:

```bash
python rag.py "What is time series forecasting?"
```

## Run UIs

FastAPI web UI:

```bash
python web_app.py
```

Open `http://127.0.0.1:8765`.

Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud deployment

Use `streamlit_app.py` as the app entry point.

Recommended public-demo environment/secrets:

- `PUBLIC_DEMO_MODE=1`
- `SHOW_DOC_METADATA=0`
- `ALLOW_USER_UPLOAD=0`
- `ALLOW_USER_REINDEX=0`

These defaults prevent exposing document names, chunks, and write operations in public demos.

If using OpenAI:
- set `OPENAI_API_KEY` in Streamlit secrets.

## Privacy notes

- `.gitignore` excludes private docs and generated indexes:
  - `resources/documents/*` (except `.gitkeep`)
  - `resources/document_catalog.json`
  - `resources/DOCUMENT_INDEX.md`
  - `.env`
  - `.streamlit/secrets.toml`
- Keep private files only on your local machine or private storage.

## Optional local Llama model

Download default GGUF:

```bash
python download_llm.py
```

Then query with local backend (default) or set `LLM_BACKEND=openai` for cloud model.
