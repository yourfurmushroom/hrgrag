#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ "${PORTABLE_NO_VENV:-0}" == "1" ]]; then
  python_cmd="$PYTHON_BIN"
  "$python_cmd" -m pip install --upgrade pip setuptools wheel

  if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
    echo "[setup] Running custom torch install command"
    eval "$TORCH_INSTALL_CMD"
  fi

  "$python_cmd" -m pip install -r "$ROOT_DIR/requirements.txt"
  "$python_cmd" -m nltk.downloader punkt
  echo "[setup] Environment ready without venv"
  exit 0
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
  echo "[setup] Running custom torch install command"
  eval "$TORCH_INSTALL_CMD"
else
  python -m pip install torch
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m nltk.downloader punkt

echo "[setup] Environment ready: $VENV_DIR"
