#!/bin/bash

set -euo pipefail

CACHE_DIR="/cache/florence2"          # shared cache root
MODEL_REPOS=(
  "microsoft/Florence-2-base"
  "microsoft/Florence-2-large"
  "microsoft/Florence-2-base-ft"
  "microsoft/Florence-2-large-ft"
)

export CACHE_DIR
mkdir -p "$CACHE_DIR"

for REPO_ID in "${MODEL_REPOS[@]}"; do
  export REPO_ID
  echo "──────────────────────────────────────────────"
  echo "▶ Handling $REPO_ID"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

cache_dir = os.environ["CACHE_DIR"]
repo_id   = os.environ["REPO_ID"]
target_dir = os.path.join(cache_dir, repo_id.split("/")[-1])

try:  # prefer existing local files
    snapshot_download(
        repo_id        = repo_id,
        cache_dir      = cache_dir,
        local_dir      = target_dir,
        local_dir_use_symlinks=False,
        local_files_only=True,
    )
    print(f"✔ {repo_id} already cached at {target_dir}")
except LocalEntryNotFoundError:
    print(f"⏬ Downloading {repo_id} to {target_dir} …")
    snapshot_download(
        repo_id        = repo_id,
        cache_dir      = cache_dir,
        local_dir      = target_dir,
        local_dir_use_symlinks=False
    )
    print(f"✔ Download complete: {target_dir}")
PY
done

exec "$@"