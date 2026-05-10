#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PY="$ROOT_DIR/.venv/bin/python"

bash "$ROOT_DIR/setup_env.sh"
bash "$ROOT_DIR/download_datasets.sh" "$@"

if [[ -x "$DEFAULT_VENV_PY" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_VENV_PY}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"$PYTHON_BIN" "$ROOT_DIR/generate_configs.py" --overwrite

echo "[bootstrap] ready. Generated configs live in $ROOT_DIR/configs"
