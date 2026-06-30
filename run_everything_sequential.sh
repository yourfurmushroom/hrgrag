#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${RERUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts_full}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-full-${STAMP}}"
PREFLIGHT_DIR="$ARTIFACTS_ROOT/_preflight/$STAMP"
SUMMARY_DIR="$ARTIFACTS_ROOT/_summary"

mkdir -p "$PREFLIGHT_DIR" "$SUMMARY_DIR"

export ARTIFACTS_ROOT
export RUN_TAG_SUFFIX
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
export RUN_PERTURBATION="${RUN_PERTURBATION:-1}"
export KB_ABLATION_SEEDS="${KB_ABLATION_SEEDS:-0 1 2 3 4}"
export ABLATION_TYPES="${ABLATION_TYPES:-drop_nodes drop_relations}"
export ABLATION_RATIOS="${ABLATION_RATIOS:-0.1 0.2 0.3}"

CONTINUE_AFTER_QWEN_FAILURE="${CONTINUE_AFTER_QWEN_FAILURE:-0}"
RUN_QWEN35_PROBE="${RUN_QWEN35_PROBE:-0}"
RUN_QWEN_LOAD_DIAGNOSIS="${RUN_QWEN_LOAD_DIAGNOSIS:-0}"
RUN_POSTHOC_DIAGNOSTICS="${RUN_POSTHOC_DIAGNOSTICS:-1}"
QWEN_PROBE_SUFFIX="${QWEN_PROBE_SUFFIX:-qwen35-preflight-${STAMP}}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[error] $label not found: $path" >&2
    exit 1
  fi
}

run_reachability_audits() {
  local out_dir="$PREFLIGHT_DIR/reachability_clean"
  mkdir -p "$out_dir"

  log "Running clean reachability audits"
  "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
    --dataset metaqa \
    --dataset-root "$ROOT_DIR/Datasets/MetaQA" \
    --kb-path "$ROOT_DIR/Datasets/MetaQA/kb.txt" \
    --split test \
    --sample-limit "$SAMPLE_LIMIT" \
    --output-json "$out_dir/metaqa.json"

  "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
    --dataset wikimovies \
    --dataset-root "$ROOT_DIR/Datasets/WikiMovies" \
    --split test \
    --sample-limit "$SAMPLE_LIMIT" \
    --output-json "$out_dir/wikimovies.json"

  "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
    --dataset mlpq \
    --dataset-root "$ROOT_DIR/Datasets/MLPQ" \
    --mlpq-pair en-zh \
    --mlpq-question-lang en \
    --mlpq-fusion ills \
    --sample-limit "$SAMPLE_LIMIT" \
    --output-json "$out_dir/mlpq.json"

  "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
    --dataset kqapro \
    --dataset-root "$ROOT_DIR/Datasets/KQAPro" \
    --split validation \
    --sample-limit "$SAMPLE_LIMIT" \
    --output-json "$out_dir/kqapro.json"
}

check_qwen_probe_result() {
  local result_json="$1"
  "$PYTHON_BIN" - "$result_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("missing qwen probe result json")

data = json.loads(path.read_text(encoding="utf-8"))
if not data:
    print(json.dumps({"failed": {"qwen3.5": "no matching model spec was executed"}}, ensure_ascii=False, indent=2))
    raise SystemExit(1)
failed = {k: v for k, v in data.items() if v == "FAILED" or not isinstance(v, dict)}
if failed:
    print(json.dumps({"failed": failed}, ensure_ascii=False, indent=2))
    raise SystemExit(1)
print(json.dumps({"status": "ok", "models": sorted(data)}, ensure_ascii=False, indent=2))
PY
}

log "Starting sequential full rerun"
log "artifacts=$ARTIFACTS_ROOT"
log "run_tag_suffix=$RUN_TAG_SUFFIX"
log "preflight=$PREFLIGHT_DIR"

require_file "$PYTHON_BIN" "PYTHON_BIN"
require_file "$ROOT_DIR/configs/config.metaqa.env" "MetaQA config"
require_file "$ROOT_DIR/configs/config.wikimovies.env" "WikiMovies config"
require_file "$ROOT_DIR/configs/config.mlpq_en_zh_en_ills.env" "MLPQ config"
require_file "$ROOT_DIR/configs/config.kqapro.env" "KQAPro config"

log "Checking shell and Python syntax"
bash -n \
  "$ROOT_DIR/run_pipeline.sh" \
  "$ROOT_DIR/run_ablation.sh" \
  "$ROOT_DIR/run_ablation_all.sh" \
  "$ROOT_DIR/run_full_rerun.sh"
"$PYTHON_BIN" -m py_compile \
  "$ROOT_DIR/LLM_inference_benchmark/benchmark.py" \
  "$ROOT_DIR/LLM_inference_benchmark/baseline.py" \
  "$ROOT_DIR/LLM_inference_benchmark/knowledgegraph_agent.py" \
  "$ROOT_DIR/summarize_experiment_runs.py" \
  "$ROOT_DIR/diagnose_qwen35.py" \
  "$ROOT_DIR/recompute_from_dumps.py" \
  "$ROOT_DIR/analyze_metaqa_bfs_sanity.py" \
  "$ROOT_DIR/analyze_symbolic_endpoints.py" \
  "$ROOT_DIR/annotate_signed_chains_from_dumps.py" \
  "$ROOT_DIR/summarize_reachability_audits.py" \
  "$ROOT_DIR/run_context_controls_from_dumps.py"

log "Running Qwen3.5 static diagnosis"
"$PYTHON_BIN" "$ROOT_DIR/diagnose_qwen35.py" \
  --output-json "$PREFLIGHT_DIR/qwen35_static_diagnosis.json"

if [[ "$RUN_QWEN_LOAD_DIAGNOSIS" == "1" ]]; then
  log "Running Qwen3.5 model-load diagnosis"
  "$PYTHON_BIN" "$ROOT_DIR/diagnose_qwen35.py" \
    --load-model \
    --output-json "$PREFLIGHT_DIR/qwen35_load_diagnosis.json"
else
  log "Skipping heavy Qwen3.5 model-load diagnosis; set RUN_QWEN_LOAD_DIAGNOSIS=1 to enable it"
fi

if [[ "$RUN_QWEN35_PROBE" == "1" ]]; then
  log "Running Qwen3.5 single-question benchmark probe"
  QWEN_PROBE_ROOT="$ARTIFACTS_ROOT/_qwen35_probe/$STAMP"
  mkdir -p "$QWEN_PROBE_ROOT/results"

  EXPERIMENT_SUITE=core \
  RUN_GRAMMAR=0 \
  "$PYTHON_BIN" "$ROOT_DIR/LLM_inference_benchmark/benchmark.py" \
    --dataset metaqa \
    --split test \
    --dataset-root "$ROOT_DIR/Datasets/MetaQA" \
    --kb-path "$ROOT_DIR/Datasets/MetaQA/kb.txt" \
    --relation-path "$ROOT_DIR/Datasets/MetaQA/relations.json" \
    --sample-limit 1 \
    --model-filter Baseline-BFS-qwen3.5 \
    --run-tag-suffix "$QWEN_PROBE_SUFFIX" \
    --artifacts-root "$QWEN_PROBE_ROOT" \
    --output-file "$QWEN_PROBE_ROOT/results/benchmark_results.json" \
    --detail-csv "$QWEN_PROBE_ROOT/results/all_models_outputs_wide.csv" \
    --retry-failed

  if check_qwen_probe_result "$QWEN_PROBE_ROOT/results/benchmark_results.json"; then
    log "Qwen3.5 benchmark probe passed"
  else
    log "Qwen3.5 benchmark probe failed; see $QWEN_PROBE_ROOT/results/failure_report.jsonl"
    if [[ "$CONTINUE_AFTER_QWEN_FAILURE" != "1" ]]; then
      echo "[stop] Set CONTINUE_AFTER_QWEN_FAILURE=1 to continue full rerun anyway." >&2
      exit 1
    fi
    log "Continuing despite Qwen3.5 failure because CONTINUE_AFTER_QWEN_FAILURE=1"
  fi
else
  log "Skipping Qwen3.5 benchmark probe; set RUN_QWEN35_PROBE=1 to enable it"
fi

run_reachability_audits

log "Running full clean + perturbation benchmark suite"
bash "$ROOT_DIR/run_full_rerun.sh"

log "Rebuilding final summary"
"$PYTHON_BIN" "$ROOT_DIR/summarize_experiment_runs.py" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --out-dir "$SUMMARY_DIR"

if [[ "$RUN_POSTHOC_DIAGNOSTICS" == "1" ]]; then
  log "Running post-hoc diagnostics from saved dumps"
  "$PYTHON_BIN" "$ROOT_DIR/analyze_metaqa_bfs_sanity.py" \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --out-dir "$SUMMARY_DIR/metaqa_bfs_sanity"
  "$PYTHON_BIN" "$ROOT_DIR/analyze_symbolic_endpoints.py" \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --out-dir "$SUMMARY_DIR/symbolic_endpoints"
  "$PYTHON_BIN" "$ROOT_DIR/annotate_signed_chains_from_dumps.py" \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --out-dir "$SUMMARY_DIR/signed_chains"
else
  log "Skipping post-hoc diagnostics; set RUN_POSTHOC_DIAGNOSTICS=1 to enable"
fi

log "Sequential run complete"
log "summary=$SUMMARY_DIR"
