#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
VENV_READY_STAMP="$VENV_DIR/.hrgrag_setup_ready"

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

run_setup_check() {
  local python_cmd="$1"

  "$python_cmd" - <<'PY'
import importlib
import sys

required_modules = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "networkx": "networkx",
    "nltk": "nltk",
    "pandas": "pandas",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "transformers": "transformers",
}

missing = []
for package_name, module_name in required_modules.items():
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{package_name}: {type(exc).__name__}: {exc}")

if missing:
    raise RuntimeError("setup dependency check failed: " + "; ".join(missing))

import huggingface_hub
import nltk
import transformers
from transformers import LlamaForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

nltk.data.find("tokenizers/punkt")

print("[setup check] python:", sys.executable)
print("[setup check] transformers:", transformers.__version__)
print("[setup check] huggingface_hub:", huggingface_hub.__version__)
print("[setup check] LlamaForCausalLM:", LlamaForCausalLM.__name__)
print("[setup check] qwen3_5_moe:", "qwen3_5_moe" in CONFIG_MAPPING)
print("[setup check] qwen3_5_moe_text:", "qwen3_5_moe_text" in CONFIG_MAPPING)

if "qwen3_5_moe" not in CONFIG_MAPPING:
    raise RuntimeError("Transformers does not support qwen3_5_moe")
PY
}

setup_fingerprint() {
  cksum "$ROOT_DIR/requirements.txt" "$ROOT_DIR/setup_env.sh"
}

stamp_ready_venv() {
  setup_fingerprint > "$VENV_READY_STAMP"
}

ready_venv_stamp_matches() {
  [[ -f "$VENV_READY_STAMP" ]] || return 1
  [[ "$(setup_fingerprint)" == "$(cat "$VENV_READY_STAMP")" ]]
}

reuse_existing_env() {
  local python_cmd="$1"
  local ready_message="$2"

  if [[ "${FORCE_ENV_SETUP:-0}" == "1" || ! -x "$python_cmd" ]]; then
    return 1
  fi

  if ready_venv_stamp_matches; then
    echo "[setup] $ready_message"
    echo "[setup] Requirements fingerprint unchanged; skipping dependency check."
    echo "[setup] Set FORCE_ENV_SETUP=1 to reinstall dependencies."
    return 0
  fi

  echo "[setup] Checking existing environment: $python_cmd"
  if run_setup_check "$python_cmd"; then
    stamp_ready_venv
    echo "[setup] $ready_message"
    echo "[setup] Set FORCE_ENV_SETUP=1 to reinstall dependencies."
    return 0
  fi

  echo "[setup] Existing environment is incomplete; installing dependencies."
  return 1
}

if [[ "${PORTABLE_NO_VENV:-0}" != "1" ]] \
  && reuse_existing_env "$VENV_DIR/bin/python" "Environment already ready: $VENV_DIR"; then
  exit 0
fi

if [[ "${PORTABLE_NO_VENV:-0}" == "1" && "${FORCE_ENV_SETUP:-0}" != "1" ]]; then
  echo "[setup] Checking existing environment: $PYTHON_BIN"
  if run_setup_check "$PYTHON_BIN"; then
    echo "[setup] Environment already ready without venv"
    echo "[setup] Set FORCE_ENV_SETUP=1 to reinstall dependencies."
    exit 0
  fi
  echo "[setup] Existing environment is incomplete; installing dependencies."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found. Install it first:"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ "${PORTABLE_NO_VENV:-0}" == "1" ]]; then
  python_cmd="$PYTHON_BIN"

  uv pip install --python "$python_cmd" --upgrade pip wheel "setuptools<82"

  if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
    echo "[setup] Running custom torch install command"
    eval "$TORCH_INSTALL_CMD"
  else
    uv pip install --python "$python_cmd" torch torchvision --index-url https://download.pytorch.org/whl/cu124
  fi

  uv pip install --python "$python_cmd" --no-cache -r "$ROOT_DIR/requirements.txt"
  "$python_cmd" -m nltk.downloader punkt

  run_setup_check "$python_cmd"

  echo "[setup] Environment ready without venv"
  exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR" --python "$PYTHON_BIN"
fi

source "$VENV_DIR/bin/activate"

uv pip install --upgrade pip wheel "setuptools<82"

if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
  echo "[setup] Running custom torch install command"
  eval "$TORCH_INSTALL_CMD"
else
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

uv pip install --no-cache -r "$ROOT_DIR/requirements.txt"
python -m nltk.downloader punkt

run_setup_check python
stamp_ready_venv

echo "[setup] Environment ready: $VENV_DIR"
