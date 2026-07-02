#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${RERUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

export RANKING_POLICY="${RANKING_POLICY:-lax-hrg-prior-v1}"
export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts_laxhrg}"
export RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-laxhrg-clean-${STAMP}}"
export EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-full}"
export ENABLE_RELATION_NGRAM_SPECS="${ENABLE_RELATION_NGRAM_SPECS:-1}"
export ENABLE_BFS_CAP_SPECS="${ENABLE_BFS_CAP_SPECS:-1}"
export ENABLE_NEW_ABLATION_SPECS="${ENABLE_NEW_ABLATION_SPECS:-1}"
export ENABLE_LEGACY_EXPANSION_SPECS="${ENABLE_LEGACY_EXPANSION_SPECS:-1}"
export FIXED_ABLATION_BUDGET="${FIXED_ABLATION_BUDGET:-1}"
export RETRY_FAILED="${RETRY_FAILED:-1}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"

# Keep this clean by default. Set RUN_PERTURBATION=1 when you need robustness runs.
export RUN_PERTURBATION="${RUN_PERTURBATION:-0}"

echo "[laxhrg-clean-all] artifacts=$ARTIFACTS_ROOT"
echo "[laxhrg-clean-all] run_tag_suffix=$RUN_TAG_SUFFIX"
echo "[laxhrg-clean-all] ranking_policy=$RANKING_POLICY"
echo "[laxhrg-clean-all] sample_limit=$SAMPLE_LIMIT run_perturbation=$RUN_PERTURBATION"

bash "$ROOT_DIR/run_full_rerun.sh"
