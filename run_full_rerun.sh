#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${RERUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

export EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-full}"
export ENABLE_RELATION_NGRAM_SPECS="${ENABLE_RELATION_NGRAM_SPECS:-1}"
export ENABLE_BFS_CAP_SPECS="${ENABLE_BFS_CAP_SPECS:-1}"
export ENABLE_NEW_ABLATION_SPECS="${ENABLE_NEW_ABLATION_SPECS:-1}"
export ENABLE_LEGACY_EXPANSION_SPECS="${ENABLE_LEGACY_EXPANSION_SPECS:-1}"
export BFS_DEGREE_CAPS="${BFS_DEGREE_CAPS:-50 100 200 500}"
export BFS_CONTEXT_TOKEN_BUDGETS="${BFS_CONTEXT_TOKEN_BUDGETS:-200 500 1000}"
export FIXED_ABLATION_BUDGET="${FIXED_ABLATION_BUDGET:-1}"
export RETRY_FAILED="${RETRY_FAILED:-1}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"
export RANKING_POLICY="${RANKING_POLICY:-lax-hrg-prior-v1}"
export RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-laxhrg-full-${STAMP}}"
export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts_laxhrg}"

CONFIGS=(
  "$ROOT_DIR/configs/config.metaqa.env"
  "$ROOT_DIR/configs/config.wikimovies.env"
  "$ROOT_DIR/configs/config.mlpq_en_zh_en_ills.env"
  "$ROOT_DIR/configs/config.kqapro.env"
)

echo "[full-rerun] artifacts=$ARTIFACTS_ROOT"
echo "[full-rerun] run_tag_suffix=$RUN_TAG_SUFFIX"
echo "[full-rerun] ranking_policy=$RANKING_POLICY"
echo "[full-rerun] suite=$EXPERIMENT_SUITE sample_limit=$SAMPLE_LIMIT"

for config in "${CONFIGS[@]}"; do
  echo "[full-rerun] clean dataset config=$config"
  CONFIG_FILE="$config" bash "$ROOT_DIR/run_pipeline.sh"
done

if [[ "${RUN_PERTURBATION:-1}" == "1" ]]; then
  export KB_ABLATION_SEEDS="${KB_ABLATION_SEEDS:-0 1 2 3 4}"
  export ABLATION_TYPES="${ABLATION_TYPES:-drop_nodes drop_relations}"
  export ABLATION_RATIOS="${ABLATION_RATIOS:-0.1 0.2 0.3}"
  for config in "${CONFIGS[@]}"; do
    echo "[full-rerun] perturbation config=$config seeds=$KB_ABLATION_SEEDS"
    CONFIG_FILE="$config" bash "$ROOT_DIR/run_ablation.sh"
  done
fi

"${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}" "$ROOT_DIR/summarize_experiment_runs.py" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --out-dir "$ARTIFACTS_ROOT/_summary"

echo "[full-rerun] done"
echo "[full-rerun] summary=$ARTIFACTS_ROOT/_summary"
