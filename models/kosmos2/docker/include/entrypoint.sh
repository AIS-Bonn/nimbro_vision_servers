#!/bin/bash

CACHE_DIR="/cache/kosmos2"          # HF cache root
TARGET_DIR="${CACHE_DIR}/patch14-224"    # where the snapshot lives
MODEL_REPO="microsoft/kosmos-2-patch14-224"

export CACHE_DIR TARGET_DIR MODEL_REPO
mkdir -p "$CACHE_DIR"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

cache_dir  = os.environ["CACHE_DIR"]
target_dir = os.environ["TARGET_DIR"]
repo_id    = os.environ["MODEL_REPO"]

try:  # try disk-only first
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        local_files_only=True,
    )
    print(f"✔ Model already cached at {target_dir}")
except LocalEntryNotFoundError:
    print(f"⏬ Downloading model to {target_dir} …")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns="*.bin"
    )
    print(f"✔ Download complete: {target_dir}")
PY

exec "$@"