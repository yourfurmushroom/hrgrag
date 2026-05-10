#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATASETS=(metaqa wikimovies mlpq kqapro)
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts}"
SUMMARY_DIR="$ARTIFACTS_ROOT/_batch"
SUMMARY_FILE="$SUMMARY_DIR/run_all_summary.txt"

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

mkdir -p "$SUMMARY_DIR"
: > "$SUMMARY_FILE"

echo "[all] datasets: ${DATASETS[*]}"

failed=0
for dataset in "${DATASETS[@]}"; do
  echo
  echo "[all] running dataset=$dataset"
  if bash "$ROOT_DIR/auto_benchmark.sh" "$dataset"; then
    echo "$dataset OK" | tee -a "$SUMMARY_FILE"
  else
    echo "$dataset FAILED" | tee -a "$SUMMARY_FILE"
    failed=$((failed + 1))
  fi
done

echo
echo "[all] summary:"
cat "$SUMMARY_FILE"

if [[ "$failed" -gt 0 ]]; then
  echo "[all] completed with failures: $failed"
  exit 1
fi

echo "[all] completed successfully"
