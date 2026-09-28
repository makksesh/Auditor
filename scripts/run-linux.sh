#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_dir/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing .venv; follow README.md installation steps" >&2
  exit 1
fi

# The NVIDIA pip packages store shared libraries outside the normal loader path.
library_paths=()
for package in cublas cudnn; do
  library_dir="$project_dir/.venv/lib/python3.14/site-packages/nvidia/$package/lib"
  if [[ -d "$library_dir" ]]; then
    library_paths+=("$library_dir")
  fi
done
if (( ${#library_paths[@]} )); then
  joined="$(IFS=:; echo "${library_paths[*]}")"
  export LD_LIBRARY_PATH="$joined${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$python_bin" -m auditor_server.main
