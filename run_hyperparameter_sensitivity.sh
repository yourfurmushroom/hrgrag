#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts_sensitivity}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"
MODEL_FILTER="${MODEL_FILTER:-HRG-Proposed-NoExpansion-gpt-oss-triple}"
DATASET_CONFIGS="${DATASET_CONFIGS:-$ROOT_DIR/configs/config.metaqa.env $ROOT_DIR/configs/config.kqapro.env}"

NUM_CANDIDATES_LIST="${NUM_CANDIDATES_LIST:-1 3 5}"
BEAM_WIDTH_LIST="${BEAM_WIDTH_LIST:-5 10 20}"
BRANCH_CAP_LIST="${BRANCH_CAP_LIST:-5 10 20}"
CONTEXT_EDGE_BUDGET_LIST="${CONTEXT_EDGE_BUDGET_LIST:-10 30 50}"
GRAMMAR_MODE_LIST="${GRAMMAR_MODE_LIST:-label ordered}"

export ARTIFACTS_ROOT
export RUN_PERTURBATION=0
export EXPERIMENT_SUITE=full
export ENABLE_NEW_ABLATION_SPECS=1
export ENABLE_LEGACY_EXPANSION_SPECS=0
export ENABLE_RELATION_NGRAM_SPECS=0
export ENABLE_BFS_CAP_SPECS=0
export FIXED_ABLATION_BUDGET=1
export RETRY_FAILED=1
export SAMPLE_LIMIT
export MODEL_FILTER

run_one() {
  local config="$1"
  local tag="$2"
  export HRG_NUM_CANDIDATES="${HRG_NUM_CANDIDATES:-}"
  export HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH="${HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH:-}"
  export HRG_VALID_CHAIN_FALLBACK_BRANCH="${HRG_VALID_CHAIN_FALLBACK_BRANCH:-}"
  export HRG_MAX_TOTAL_CONTEXT_EDGES="${HRG_MAX_TOTAL_CONTEXT_EDGES:-}"
  export HRG_REQUIRE_ORDERED_GRAMMAR_MATCH="${HRG_REQUIRE_ORDERED_GRAMMAR_MATCH:-}"
  echo "[sensitivity] config=$config tag=$tag model_filter=$MODEL_FILTER"
  RUN_TAG_SUFFIX="sensitivity-${tag}" CONFIG_FILE="$config" bash "$ROOT_DIR/run_pipeline.sh"
}

clear_hrg_overrides() {
  unset HRG_NUM_CANDIDATES
  unset HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH
  unset HRG_VALID_CHAIN_FALLBACK_BRANCH
  unset HRG_MAX_TOTAL_CONTEXT_EDGES
  unset HRG_REQUIRE_ORDERED_GRAMMAR_MATCH
}

for config in $DATASET_CONFIGS; do
  for value in $NUM_CANDIDATES_LIST; do
    clear_hrg_overrides
    HRG_NUM_CANDIDATES="$value" run_one "$config" "numcand-${value}"
  done

  for value in $BEAM_WIDTH_LIST; do
    clear_hrg_overrides
    HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH="$value" run_one "$config" "beam-${value}"
  done

  for value in $BRANCH_CAP_LIST; do
    clear_hrg_overrides
    HRG_VALID_CHAIN_FALLBACK_BRANCH="$value" run_one "$config" "branch-${value}"
  done

  for value in $CONTEXT_EDGE_BUDGET_LIST; do
    clear_hrg_overrides
    HRG_MAX_TOTAL_CONTEXT_EDGES="$value" run_one "$config" "ctxedges-${value}"
  done

  for mode in $GRAMMAR_MODE_LIST; do
    clear_hrg_overrides
    if [[ "$mode" == "ordered" ]]; then
      HRG_REQUIRE_ORDERED_GRAMMAR_MATCH=1 run_one "$config" "grammar-ordered"
    else
      HRG_REQUIRE_ORDERED_GRAMMAR_MATCH=0 run_one "$config" "grammar-label"
    fi
  done
done

"${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}" "$ROOT_DIR/summarize_experiment_runs.py" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --out-dir "$ARTIFACTS_ROOT/_summary"

echo "[sensitivity] wrote $ARTIFACTS_ROOT/_summary"
