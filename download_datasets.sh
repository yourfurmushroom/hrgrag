#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PY="$ROOT_DIR/.venv/bin/python"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config.env}"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

if [[ -x "$DEFAULT_VENV_PY" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_VENV_PY}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

ARGS=()

if [[ -n "${WQSP_HF_ID:-}" ]]; then
  ARGS+=(--wqsp-hf-id "$WQSP_HF_ID")
fi
if [[ -n "${CWQ_HF_ID:-}" ]]; then
  ARGS+=(--cwq-hf-id "$CWQ_HF_ID")
fi
if [[ -n "${KQAPRO_HF_ID:-}" ]]; then
  ARGS+=(--kqapro-hf-id "$KQAPRO_HF_ID")
fi
if [[ -n "${MINTAKA_HF_ID:-}" ]]; then
  ARGS+=(--mintaka-hf-id "$MINTAKA_HF_ID")
fi

"$PYTHON_BIN" "$ROOT_DIR/download_datasets.py" "${ARGS[@]}" "$@"
