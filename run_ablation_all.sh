#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS=(
  "$ROOT_DIR/configs/config.metaqa.env"
  "$ROOT_DIR/configs/config.wikimovies.env"
  "$ROOT_DIR/configs/config.mlpq_en_zh_en_ills.env"
  "$ROOT_DIR/configs/config.kqapro.env"
)

for config in "${CONFIGS[@]}"; do
  echo "[dataset] config=$config"
  CONFIG_FILE="$config" bash "$ROOT_DIR/run_ablation.sh"
done
