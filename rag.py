#!/usr/bin/env python3
"""Ask a question: retrieve from Chroma, then answer with a local Llama GGUF or OpenAI."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_common import TOP_K
from rag_core import default_backend, run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "-k",
        type=int,
        default=TOP_K,
        help="Number of chunks to retrieve (default: %(default)s)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only print retrieved chunks, do not run an LLM",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "openai"),
        default=None,
        help="Generation backend (default: LLM_BACKEND env or 'local')",
    )
    args = parser.parse_args()
    if not args.question:
        parser.print_help()
        print("\nExample: python rag.py 'What is RAG?'", file=sys.stderr)
        raise SystemExit(1)

    backend = args.backend or default_backend()
    r = run_query(
        args.question,
        k=args.k,
        backend=backend,
        no_llm=args.no_llm,
    )

    if not r.ok:
        print(r.error or "Query failed.", file=sys.stderr)
        raise SystemExit(1)

    if not r.chunks:
        print("No matching chunks found.")
        raise SystemExit(0)

    print("--- Retrieved chunks ---\n")
    for i, c in enumerate(r.chunks, 1):
        title = (c.get("title") or "").strip()
        head = f"[{i}] (source: {c['source']})"
        if title:
            head += f" — {title}"
        print(f"{head}\n{c['text']}\n")

    if args.no_llm:
        raise SystemExit(0)

    if r.answer:
        label = "OpenAI" if r.answer_backend == "openai" else "local Llama"
        print(f"--- Answer ({label}) ---\n")
        print(r.answer)
        raise SystemExit(0)

    if r.error:
        print(r.error, file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
