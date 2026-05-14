#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config.env}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"

mkdir -p \
  "$CACHE_ROOT" \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/huggingface/transformers" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/nltk"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/huggingface/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT/huggingface/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export NLTK_DATA="${NLTK_DATA:-$CACHE_ROOT/nltk}"

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
if [[ -n "${SPLIT:-}" ]]; then
  SPLIT="$SPLIT"
elif [[ "$DATASET" == "kqapro" ]]; then
  SPLIT="validation"
else
  SPLIT="test"
fi
RUN_GRAMMAR="${RUN_GRAMMAR:-1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
MODEL_FILTER="${MODEL_FILTER:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda:0,cuda:1}"
ENABLE_MODEL_SHARDING="${ENABLE_MODEL_SHARDING:-1}"
STRICT_GPU_SHARDING="${STRICT_GPU_SHARDING:-1}"
METAQA_VARIANT="${METAQA_VARIANT:-vanilla}"
WIKIMOVIES_SUBSET="${WIKIMOVIES_SUBSET:-wiki_entities}"
MLPQ_PAIR="${MLPQ_PAIR:-en-zh}"
MLPQ_QUESTION_LANG="${MLPQ_QUESTION_LANG:-en}"
MLPQ_FUSION="${MLPQ_FUSION:-ills}"
MLPQ_KB_MODE="${MLPQ_KB_MODE:-bilingual}"
MLPQ_KB_LANG="${MLPQ_KB_LANG:-auto}"
CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-custom}"
CUSTOM_FORMAT="${CUSTOM_FORMAT:-auto}"
CUSTOM_HOP="${CUSTOM_HOP:-1}"

export CUDA_VISIBLE_DEVICES
export TARGET_DEVICE
export ENABLE_MODEL_SHARDING
export STRICT_GPU_SHARDING

DATASETS_DIR="$ROOT_DIR/Datasets"
DEFAULT_METAQA_ROOT="$DATASETS_DIR/MetaQA"
DEFAULT_WIKIMOVIES_ROOT="$DATASETS_DIR/WikiMovies"
DEFAULT_MLPQ_ROOT="$DATASETS_DIR/MLPQ"
DEFAULT_WQSP_ROOT="$DATASETS_DIR/WQSP"
DEFAULT_CWQ_ROOT="$DATASETS_DIR/CWQ"
DEFAULT_KQAPRO_ROOT="$DATASETS_DIR/KQAPro"
DEFAULT_MINTAKA_ROOT="$DATASETS_DIR/Mintaka"

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
      if [[ "${MLPQ_KB_MODE,,}" == "monolingual" ]]; then
        local mono_lang="${MLPQ_KB_LANG,,}"
        if [[ "$mono_lang" == "auto" || -z "$mono_lang" ]]; then
          mono_lang="${MLPQ_QUESTION_LANG,,}"
        fi
        printf 'mlpq-%s-%s-%s-mono-%s' "$MLPQ_PAIR" "$MLPQ_QUESTION_LANG" "$MLPQ_FUSION" "$mono_lang"
      else
        printf 'mlpq-%s-%s-%s' "$MLPQ_PAIR" "$MLPQ_QUESTION_LANG" "$MLPQ_FUSION"
      fi
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

if [[ -z "${DATASET_ROOT:-}" ]]; then
  case "$DATASET" in
    metaqa) DATASET_ROOT="$DEFAULT_METAQA_ROOT" ;;
    wikimovies) DATASET_ROOT="$DEFAULT_WIKIMOVIES_ROOT" ;;
    mlpq) DATASET_ROOT="$DEFAULT_MLPQ_ROOT" ;;
    wqsp) DATASET_ROOT="$DEFAULT_WQSP_ROOT" ;;
    cwq) DATASET_ROOT="$DEFAULT_CWQ_ROOT" ;;
    kqapro) DATASET_ROOT="$DEFAULT_KQAPRO_ROOT" ;;
    mintaka) DATASET_ROOT="$DEFAULT_MINTAKA_ROOT" ;;
  esac
fi

if [[ -z "${KB_PATH:-}" ]]; then
  case "$DATASET" in
    metaqa) KB_PATH="$DEFAULT_METAQA_ROOT/kb.txt" ;;
    wikimovies) KB_PATH="$DEFAULT_WIKIMOVIES_ROOT/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt" ;;
    mlpq)
      if [[ "${MLPQ_KB_MODE,,}" == "monolingual" ]]; then
        mono_lang="${MLPQ_KB_LANG,,}"
        if [[ "$mono_lang" == "auto" || -z "$mono_lang" ]]; then
          mono_lang="${MLPQ_QUESTION_LANG,,}"
        fi
        KB_PATH="$DEFAULT_MLPQ_ROOT/datasets/KGs/sampled_monolingual_KGs/Sampled_${mono_lang}.txt"
      else
        pair_key="${MLPQ_PAIR//-/_}"
        if [[ "${MLPQ_FUSION,,}" == "nmn" ]]; then
          KB_PATH="$DEFAULT_MLPQ_ROOT/datasets/KGs/fusion_bilingual_KGs/NMN_fusion/merged_NMN_KG_${pair_key}.txt"
        else
          KB_PATH="$DEFAULT_MLPQ_ROOT/datasets/KGs/fusion_bilingual_KGs/ILLs_fusion/merged_ILLs_KG_${pair_key}.txt"
        fi
      fi
      ;;
    kqapro) KB_PATH="$DEFAULT_KQAPRO_ROOT/kqapro_kb_triples.tsv" ;;
  esac
fi

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
  --mlpq-kb-mode "$MLPQ_KB_MODE"
  --mlpq-kb-lang "$MLPQ_KB_LANG"
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
  --mlpq-kb-mode "$MLPQ_KB_MODE"
  --mlpq-kb-lang "$MLPQ_KB_LANG"
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
      case "$DATASET" in
        wqsp) DATASET_FILE="$DEFAULT_WQSP_ROOT/normalized/${SPLIT}.jsonl" ;;
        cwq) DATASET_FILE="$DEFAULT_CWQ_ROOT/normalized/${SPLIT}.jsonl" ;;
        kqapro) DATASET_FILE="$DEFAULT_KQAPRO_ROOT/normalized/${SPLIT}.jsonl" ;;
        mintaka) DATASET_FILE="$DEFAULT_MINTAKA_ROOT/normalized/${SPLIT}.jsonl" ;;
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
