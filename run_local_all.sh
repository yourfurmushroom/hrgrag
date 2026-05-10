#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATASETS=(metaqa wikimovies mlpq kqapro)

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

echo "[local] datasets: ${DATASETS[*]}"
echo "[local] bootstrapping environment and datasets"

bash "$ROOT_DIR/bootstrap_all.sh" --datasets "${DATASETS[@]}"

echo "[local] starting benchmark batch"
bash "$ROOT_DIR/run_all_benchmarks.sh" "${DATASETS[@]}"
