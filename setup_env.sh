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
    uv pip install --python "$python_cmd" torch torchvision --index-url https://download.pytorch.org/whl/cu126
  fi

  uv pip install --python "$python_cmd" --no-cache -r "$ROOT_DIR/requirements.txt"
  "$python_cmd" -m nltk.downloader punkt

  "$python_cmd" - <<'PY'
import sys
import transformers
import huggingface_hub
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print("[setup check] python:", sys.executable)
print("[setup check] transformers:", transformers.__version__)
print("[setup check] huggingface_hub:", huggingface_hub.__version__)
print("[setup check] qwen3_5_moe:", "qwen3_5_moe" in CONFIG_MAPPING)
print("[setup check] qwen3_5_moe_text:", "qwen3_5_moe_text" in CONFIG_MAPPING)

if "qwen3_5_moe" not in CONFIG_MAPPING:
    raise RuntimeError("Transformers does not support qwen3_5_moe")
PY

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
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

uv pip install --no-cache -r "$ROOT_DIR/requirements.txt"
python -m nltk.downloader punkt

python - <<'PY'
import sys
import transformers
import huggingface_hub
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print("[setup check] python:", sys.executable)
print("[setup check] transformers:", transformers.__version__)
print("[setup check] huggingface_hub:", huggingface_hub.__version__)
print("[setup check] qwen3_5_moe:", "qwen3_5_moe" in CONFIG_MAPPING)
print("[setup check] qwen3_5_moe_text:", "qwen3_5_moe_text" in CONFIG_MAPPING)

if "qwen3_5_moe" not in CONFIG_MAPPING:
    raise RuntimeError("Transformers does not support qwen3_5_moe")
PY

echo "[setup] Environment ready: $VENV_DIR"
