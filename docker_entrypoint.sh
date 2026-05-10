#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/app"
DATASET="${DATASET:-all}"
DATASETS="${DATASETS:-}"

mkdir -p \
  "$ROOT_DIR/Datasets" \
  "$ROOT_DIR/artifacts" \
  "$ROOT_DIR/configs" \
  "$ROOT_DIR/KBs"

if [[ -n "$DATASETS" ]]; then
  echo "[docker] datasets=$DATASETS"
  echo "[docker] starting batch benchmark flow"
  # shellcheck disable=SC2206
  DATASET_LIST=($DATASETS)
  exec bash "$ROOT_DIR/run_all_benchmarks.sh" "${DATASET_LIST[@]}"
fi

if [[ "$DATASET" == "all" ]]; then
  echo "[docker] dataset=all"
  echo "[docker] starting batch benchmark flow"
  exec bash "$ROOT_DIR/run_all_benchmarks.sh"
fi

echo "[docker] dataset=$DATASET"
echo "[docker] starting single benchmark flow"

exec bash "$ROOT_DIR/auto_benchmark.sh" "$DATASET"
