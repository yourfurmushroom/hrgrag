#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/config.metaqa.env}"

[[ -f "$CONFIG_FILE" ]] || { echo "[error] config not found: $CONFIG_FILE" >&2; exit 1; }

# shellcheck disable=SC1090
source "$CONFIG_FILE"

DATASET="${DATASET:-metaqa}"
ORIGINAL_KB_PATH="${KB_PATH:-}"
[[ -n "$ORIGINAL_KB_PATH" && -f "$ORIGINAL_KB_PATH" ]] || {
  echo "[error] KB_PATH not found: ${ORIGINAL_KB_PATH:-}" >&2
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MODEL_FILTER="${MODEL_FILTER:-Spine-Correction}"
ABLATION_TYPES="${ABLATION_TYPES:-drop_nodes drop_relations}"
ABLATION_RATIOS="${ABLATION_RATIOS:-0.1 0.2 0.3}"
KB_ABLATION_SEED="${KB_ABLATION_SEED:-42}"
KB_ABLATION_SEEDS="${KB_ABLATION_SEEDS:-$KB_ABLATION_SEED}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts}"

for seed in $KB_ABLATION_SEEDS; do
  for ablation_type in $ABLATION_TYPES; do
    for ratio in $ABLATION_RATIOS; do
      pct="$($PYTHON_BIN - <<PY
ratio = float("$ratio")
print(int(round(ratio * 100)))
PY
)"
      suffix="abl-${ablation_type}-${pct}pct-seed${seed}"
      ablation_dir="$ARTIFACTS_ROOT/_ablations/$DATASET/$suffix"
      ablated_kb="$ablation_dir/kb.tsv"

      echo "[ablation] dataset=$DATASET type=$ablation_type ratio=$ratio seed=$seed"

      "$PYTHON_BIN" "$ROOT_DIR/create_ablated_kb.py" \
        --input-kb "$ORIGINAL_KB_PATH" \
        --output-kb "$ablated_kb" \
        --mode "$ablation_type" \
        --ratio "$ratio" \
        --seed "$seed"

      env \
        CONFIG_FILE="$CONFIG_FILE" \
        DATASET="$DATASET" \
        KB_PATH="$ORIGINAL_KB_PATH" \
        GRAMMAR_KB_PATH="$ablated_kb" \
        MODEL_FILTER="$MODEL_FILTER" \
        RUN_GRAMMAR="1" \
        RUN_TAG_SUFFIX="$suffix" \
        GRAMMAR_PATH="" \
        KB_ABLATION_MODE="none" \
        KB_ABLATION_RATIO="0" \
        KB_ABLATION_SEED="$seed" \
        ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
        bash "$ROOT_DIR/run_pipeline.sh"
    done
  done
done
