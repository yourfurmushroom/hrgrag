#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PY="$ROOT_DIR/.venv/bin/python"
DATASET="${1:-}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -n "$DATASET" ]] || fail "usage: bash auto_benchmark.sh <dataset>"

bash "$ROOT_DIR/bootstrap_all.sh" --datasets "$DATASET"

if [[ -x "$DEFAULT_VENV_PY" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_VENV_PY}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

case "$DATASET" in
  metaqa) CONFIG_NAME="metaqa" ;;
  wikimovies) CONFIG_NAME="wikimovies" ;;
  mlpq) CONFIG_NAME="mlpq_en_zh_en_ills" ;;
  wqsp|cwq|kqapro|mintaka) CONFIG_NAME="$DATASET" ;;
  *) fail "unsupported dataset: $DATASET" ;;
esac

CONFIG_PATH="$ROOT_DIR/configs/config.${CONFIG_NAME}.env"
[[ -f "$CONFIG_PATH" ]] || fail "missing generated config: $CONFIG_PATH"

DATASET_ROOT_LINE="$(grep '^DATASET_ROOT=' "$CONFIG_PATH" || true)"
DATASET_ROOT="${DATASET_ROOT_LINE#DATASET_ROOT=}"
if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT=""
fi

RESOLVE_ARGS=(--dataset "$DATASET" --root-dir "$ROOT_DIR")
if [[ -n "$DATASET_ROOT" && -d "$DATASET_ROOT" ]]; then
  RESOLVE_ARGS+=(--dataset-root "$DATASET_ROOT")
fi

RESOLVE_JSON="$("$PYTHON_BIN" "$ROOT_DIR/resolve_kb.py" "${RESOLVE_ARGS[@]}" || true)"
KB_PATH="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1]).get("kb_path",""))' "$RESOLVE_JSON")"

if [[ -z "$KB_PATH" ]]; then
  if [[ "$DATASET" =~ ^(wqsp|cwq|mintaka)$ ]]; then
    fail "dataset=$DATASET is downloaded and its dataset path is fixed, but this repo does not bundle a benchmark-ready KG for it. Put a usable KB/triples file under portable_runner/KBs and set KB_PATH manually in $CONFIG_PATH"
  fi
  if [[ "$DATASET" == "kqapro" ]]; then
    fail "dataset=kqapro is downloaded, but the converted triples file was not found at portable_runner/Datasets/KQAPro/kqapro_kb_triples.tsv. Regenerate datasets or set KB_PATH manually in $CONFIG_PATH"
  fi
  fail "could not auto-resolve KB for dataset=$DATASET. Check that the downloaded dataset layout matches the expected paths, or set KB_PATH manually in $CONFIG_PATH"
fi

TMP_CONFIG="$ROOT_DIR/configs/.autogen.${CONFIG_NAME}.env"
awk -v new_kb="$KB_PATH" '
  BEGIN {done=0}
  /^KB_PATH=/ {print "KB_PATH=" new_kb; done=1; next}
  {print}
  END {if (!done) print "KB_PATH=" new_kb}
' "$CONFIG_PATH" > "$TMP_CONFIG"

echo "[auto] dataset=$DATASET"
echo "[auto] resolved_kb=$KB_PATH"
CONFIG_FILE="$TMP_CONFIG" bash "$ROOT_DIR/run_pipeline.sh"
