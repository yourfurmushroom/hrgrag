#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config.env}"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

DEFAULT_VENV_PY="$ROOT_DIR/.venv/bin/python"
if [[ -x "$DEFAULT_VENV_PY" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_VENV_PY}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DATASET="${DATASET:-custom}"
SPLIT="${SPLIT:-test}"
RUN_GRAMMAR="${RUN_GRAMMAR:-1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
MODEL_FILTER="${MODEL_FILTER:-}"
METAQA_VARIANT="${METAQA_VARIANT:-vanilla}"
WIKIMOVIES_SUBSET="${WIKIMOVIES_SUBSET:-wiki_entities}"
MLPQ_PAIR="${MLPQ_PAIR:-en-zh}"
MLPQ_QUESTION_LANG="${MLPQ_QUESTION_LANG:-en}"
MLPQ_FUSION="${MLPQ_FUSION:-ills}"
CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-custom}"
CUSTOM_FORMAT="${CUSTOM_FORMAT:-auto}"
CUSTOM_HOP="${CUSTOM_HOP:-1}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -n "$path" ]] || fail "$label is required"
  [[ -f "$path" ]] || fail "$label not found: $path"
}

build_run_tag() {
  case "$DATASET" in
    metaqa)
      printf 'metaqa-%s-%s' "$METAQA_VARIANT" "$SPLIT"
      ;;
    wikimovies)
      printf 'wikimovies-%s-%s' "$WIKIMOVIES_SUBSET" "$SPLIT"
      ;;
    mlpq)
      printf 'mlpq-%s-%s-%s' "$MLPQ_PAIR" "$MLPQ_QUESTION_LANG" "$MLPQ_FUSION"
      ;;
    wqsp|cwq|kqapro|mintaka)
      printf '%s-%s' "$DATASET" "${SPLIT,,}"
      ;;
    custom)
      local safe_name="${CUSTOM_DATASET_NAME// /-}"
      printf 'custom-%s-%s' "${safe_name,,}" "${SPLIT,,}"
      ;;
    *)
      fail "Unsupported DATASET: $DATASET"
      ;;
  esac
}

RUN_TAG="$(build_run_tag)"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts}"
RUN_ARTIFACT_DIR="$ARTIFACTS_ROOT/$RUN_TAG"
GRAMMAR_OUT_DIR="$RUN_ARTIFACT_DIR/grammar"
GRAMMAR_PATH="${GRAMMAR_PATH:-$GRAMMAR_OUT_DIR/hrg_grammar.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_ARTIFACT_DIR/results/benchmark_results.json}"
DETAIL_CSV="${DETAIL_CSV:-$RUN_ARTIFACT_DIR/results/all_models_outputs_wide.csv}"

require_file "${KB_PATH:-}" "KB_PATH"

GRAMMAR_ARGS=(
  --kb-path "$KB_PATH"
  --dataset "$DATASET"
  --split "$SPLIT"
  --metaqa-variant "$METAQA_VARIANT"
  --wikimovies-subset "$WIKIMOVIES_SUBSET"
  --mlpq-pair "$MLPQ_PAIR"
  --mlpq-question-lang "$MLPQ_QUESTION_LANG"
  --mlpq-fusion "$MLPQ_FUSION"
  --custom-dataset-name "$CUSTOM_DATASET_NAME"
  --out-dir "$GRAMMAR_OUT_DIR"
)

BENCHMARK_ARGS=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --metaqa-variant "$METAQA_VARIANT"
  --wikimovies-subset "$WIKIMOVIES_SUBSET"
  --mlpq-pair "$MLPQ_PAIR"
  --mlpq-question-lang "$MLPQ_QUESTION_LANG"
  --mlpq-fusion "$MLPQ_FUSION"
  --custom-dataset-name "$CUSTOM_DATASET_NAME"
  --custom-format "$CUSTOM_FORMAT"
  --custom-hop "$CUSTOM_HOP"
  --kb-path "$KB_PATH"
  --grammar-path "$GRAMMAR_PATH"
  --artifacts-root "$ARTIFACTS_ROOT"
  --sample-limit "$SAMPLE_LIMIT"
  --output-file "$OUTPUT_FILE"
  --detail-csv "$DETAIL_CSV"
)

if [[ -n "${RELATION_PATH:-}" ]]; then
  require_file "$RELATION_PATH" "RELATION_PATH"
  BENCHMARK_ARGS+=(--relation-path "$RELATION_PATH")
fi

if [[ -n "$MODEL_FILTER" ]]; then
  BENCHMARK_ARGS+=(--model-filter "$MODEL_FILTER")
fi

case "$DATASET" in
  metaqa|wikimovies|mlpq)
    [[ -n "${DATASET_ROOT:-}" ]] || fail "DATASET_ROOT is required for DATASET=$DATASET"
    [[ -d "$DATASET_ROOT" ]] || fail "DATASET_ROOT not found: $DATASET_ROOT"
    BENCHMARK_ARGS+=(--dataset-root "$DATASET_ROOT")
    ;;
  wqsp|cwq|kqapro|mintaka)
    if [[ -n "${DATASET_ROOT:-}" ]]; then
      [[ -d "$DATASET_ROOT" ]] || fail "DATASET_ROOT not found: $DATASET_ROOT"
      BENCHMARK_ARGS+=(--dataset-root "$DATASET_ROOT")
    fi
    if [[ -z "${DATASET_FILE:-}" ]]; then
      DATASET_FILE="$ROOT_DIR/Datasets/$(tr '[:lower:]' '[:upper:]' <<< "${DATASET:0:1}")${DATASET:1}/normalized/${SPLIT}.jsonl"
      case "$DATASET" in
        wqsp) DATASET_FILE="$ROOT_DIR/Datasets/WQSP/normalized/${SPLIT}.jsonl" ;;
        cwq) DATASET_FILE="$ROOT_DIR/Datasets/CWQ/normalized/${SPLIT}.jsonl" ;;
        kqapro) DATASET_FILE="$ROOT_DIR/Datasets/KQAPro/normalized/${SPLIT}.jsonl" ;;
        mintaka) DATASET_FILE="$ROOT_DIR/Datasets/Mintaka/normalized/${SPLIT}.jsonl" ;;
      esac
    fi
    require_file "$DATASET_FILE" "DATASET_FILE"
    BENCHMARK_ARGS+=(--dataset-file "$DATASET_FILE")
    ;;
  custom)
    require_file "${DATASET_FILE:-}" "DATASET_FILE"
    BENCHMARK_ARGS+=(--dataset-file "$DATASET_FILE")
    ;;
esac

echo "[run] dataset=$DATASET run_tag=$RUN_TAG"
echo "[run] python=$PYTHON_BIN"
echo "[run] grammar=$GRAMMAR_PATH"
echo "[run] results=$OUTPUT_FILE"

if [[ "$RUN_GRAMMAR" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/hrg_grammar/hrg_extract.py" "${GRAMMAR_ARGS[@]}"
else
  echo "[run] skipping grammar generation"
fi

"$PYTHON_BIN" "$ROOT_DIR/LLM_inference_benchmark/benchmark.py" "${BENCHMARK_ARGS[@]}"
