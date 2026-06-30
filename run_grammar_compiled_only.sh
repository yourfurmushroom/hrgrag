#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${RERUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/new_artifacts_grammarcompiled}"
export RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-grammarcompiled-${STAMP}}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"
export EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-core}"
export ENABLE_NEW_ABLATION_SPECS=1
export ENABLE_BFS_CAP_SPECS=0
export ENABLE_RELATION_NGRAM_SPECS=0
export ENABLE_LEGACY_EXPANSION_SPECS=0
export MODEL_FILTER="HRG-GrammarCompiled"
export RETRY_FAILED="${RETRY_FAILED:-1}"

CONFIG_FILES="${CONFIG_FILES:-$ROOT_DIR/configs/config.metaqa.env}"

mkdir -p "$ARTIFACTS_ROOT"

echo "[grammar-compiled] artifacts=$ARTIFACTS_ROOT"
echo "[grammar-compiled] run_tag_suffix=$RUN_TAG_SUFFIX"
echo "[grammar-compiled] sample_limit=$SAMPLE_LIMIT"
echo "[grammar-compiled] model_filter=$MODEL_FILTER"

for config in $CONFIG_FILES; do
  echo "[grammar-compiled] config=$config"
  if [[ ! -f "$config" ]]; then
    echo "[error] config not found: $config" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  set -a
  source "$config"
  set +a

  export MODEL_FILTER="HRG-GrammarCompiled"
  export ARTIFACTS_ROOT
  export RUN_TAG_SUFFIX
  export SAMPLE_LIMIT
  export EXPERIMENT_SUITE
  export ENABLE_NEW_ABLATION_SPECS
  export ENABLE_BFS_CAP_SPECS
  export ENABLE_RELATION_NGRAM_SPECS
  export ENABLE_LEGACY_EXPANSION_SPECS
  export RETRY_FAILED

  CONFIG_FILE=/dev/null bash "$ROOT_DIR/run_pipeline.sh"
done

"${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}" "$ROOT_DIR/summarize_experiment_runs.py" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --out-dir "$ARTIFACTS_ROOT/_summary"

echo "[grammar-compiled] done"
echo "[grammar-compiled] summary=$ARTIFACTS_ROOT/_summary"
