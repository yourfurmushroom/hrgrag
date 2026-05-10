#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"

mkdir -p \
  "$CACHE_ROOT" \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/huggingface/transformers" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/nltk"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/huggingface/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT/huggingface/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export NLTK_DATA="${NLTK_DATA:-$CACHE_ROOT/nltk}"

if [[ "${PORTABLE_NO_VENV:-0}" == "1" ]]; then
  python_cmd="$PYTHON_BIN"
  "$python_cmd" -m pip install --upgrade pip setuptools wheel

  if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
    echo "[setup] Running custom torch install command"
    eval "$TORCH_INSTALL_CMD"
  else
    "$python_cmd" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
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
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m nltk.downloader punkt

echo "[setup] Environment ready: $VENV_DIR"
