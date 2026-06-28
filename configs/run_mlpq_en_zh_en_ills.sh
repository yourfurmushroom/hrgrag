#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/configs/config.mlpq_en_zh_en_ills.env" bash "$ROOT_DIR/run_pipeline.sh"
