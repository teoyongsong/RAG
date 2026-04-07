#!/usr/bin/env python3
"""Download the default Llama 3.2 1B Instruct GGUF into models/."""

from __future__ import annotations

import argparse
import sys

from rag_common import DEFAULT_LLAMA_GGUF, HF_LLAMA_FILE, HF_LLAMA_REPO, MODELS_DIR

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local Llama GGUF for rag.py")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_LLAMA_GGUF.is_file() and not args.force:
        print(f"Already present: {DEFAULT_LLAMA_GGUF}")
        print("Use --force to re-download.")
        raise SystemExit(0)

    print(f"Downloading {HF_LLAMA_REPO} / {HF_LLAMA_FILE} ...")
    path = hf_hub_download(
        repo_id=HF_LLAMA_REPO,
        filename=HF_LLAMA_FILE,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
