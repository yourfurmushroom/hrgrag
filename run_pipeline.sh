#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config.env}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
USER_MODEL_FILTER="${MODEL_FILTER-__HRGRAG_UNSET__}"
USER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-__HRGRAG_UNSET__}"
USER_TARGET_DEVICE="${TARGET_DEVICE-__HRGRAG_UNSET__}"
USER_ENABLE_MODEL_SHARDING="${ENABLE_MODEL_SHARDING-__HRGRAG_UNSET__}"
USER_STRICT_GPU_SHARDING="${STRICT_GPU_SHARDING-__HRGRAG_UNSET__}"
USER_SAMPLE_LIMIT="${SAMPLE_LIMIT-__HRGRAG_UNSET__}"
USER_RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX-__HRGRAG_UNSET__}"
USER_ARTIFACTS_ROOT="${ARTIFACTS_ROOT-__HRGRAG_UNSET__}"
USER_RANKING_POLICY="${RANKING_POLICY-__HRGRAG_UNSET__}"
USER_EXPERIMENT_SUITE="${EXPERIMENT_SUITE-__HRGRAG_UNSET__}"
USER_FIXED_ABLATION_BUDGET="${FIXED_ABLATION_BUDGET-__HRGRAG_UNSET__}"
USER_RETRY_FAILED="${RETRY_FAILED-__HRGRAG_UNSET__}"

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

[[ "$USER_MODEL_FILTER" != "__HRGRAG_UNSET__" ]] && MODEL_FILTER="$USER_MODEL_FILTER"
[[ "$USER_CUDA_VISIBLE_DEVICES" != "__HRGRAG_UNSET__" ]] && CUDA_VISIBLE_DEVICES="$USER_CUDA_VISIBLE_DEVICES"
[[ "$USER_TARGET_DEVICE" != "__HRGRAG_UNSET__" ]] && TARGET_DEVICE="$USER_TARGET_DEVICE"
[[ "$USER_ENABLE_MODEL_SHARDING" != "__HRGRAG_UNSET__" ]] && ENABLE_MODEL_SHARDING="$USER_ENABLE_MODEL_SHARDING"
[[ "$USER_STRICT_GPU_SHARDING" != "__HRGRAG_UNSET__" ]] && STRICT_GPU_SHARDING="$USER_STRICT_GPU_SHARDING"
[[ "$USER_SAMPLE_LIMIT" != "__HRGRAG_UNSET__" ]] && SAMPLE_LIMIT="$USER_SAMPLE_LIMIT"
[[ "$USER_RUN_TAG_SUFFIX" != "__HRGRAG_UNSET__" ]] && RUN_TAG_SUFFIX="$USER_RUN_TAG_SUFFIX"
[[ "$USER_ARTIFACTS_ROOT" != "__HRGRAG_UNSET__" ]] && ARTIFACTS_ROOT="$USER_ARTIFACTS_ROOT"
[[ "$USER_RANKING_POLICY" != "__HRGRAG_UNSET__" ]] && RANKING_POLICY="$USER_RANKING_POLICY"
[[ "$USER_EXPERIMENT_SUITE" != "__HRGRAG_UNSET__" ]] && EXPERIMENT_SUITE="$USER_EXPERIMENT_SUITE"
[[ "$USER_FIXED_ABLATION_BUDGET" != "__HRGRAG_UNSET__" ]] && FIXED_ABLATION_BUDGET="$USER_FIXED_ABLATION_BUDGET"
[[ "$USER_RETRY_FAILED" != "__HRGRAG_UNSET__" ]] && RETRY_FAILED="$USER_RETRY_FAILED"

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
SAMPLE_LIMIT="${SAMPLE_LIMIT:-50}"
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
GRAMMAR_KB_PATH="${GRAMMAR_KB_PATH:-}"
KB_ABLATION_MODE="${KB_ABLATION_MODE:-none}"
KB_ABLATION_RATIO="${KB_ABLATION_RATIO:-0}"
KB_ABLATION_SEED="${KB_ABLATION_SEED:-0}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-laxhrg}"
RANKING_POLICY="${RANKING_POLICY:-lax-hrg-prior-v1}"
FIXED_ABLATION_BUDGET="${FIXED_ABLATION_BUDGET:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"

export CUDA_VISIBLE_DEVICES
export TARGET_DEVICE
export ENABLE_MODEL_SHARDING
export STRICT_GPU_SHARDING
export RANKING_POLICY

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
  local suffix="$RUN_TAG_SUFFIX"
  local base=""
  case "$DATASET" in
    metaqa)
      base=$(printf 'metaqa-%s-%s' "$METAQA_VARIANT" "$SPLIT")
      ;;
    wikimovies)
      base=$(printf 'wikimovies-%s-%s' "$WIKIMOVIES_SUBSET" "$SPLIT")
      ;;
    mlpq)
      if [[ "${MLPQ_KB_MODE,,}" == "monolingual" ]]; then
        local mono_lang="${MLPQ_KB_LANG,,}"
        if [[ "$mono_lang" == "auto" || -z "$mono_lang" ]]; then
          mono_lang="${MLPQ_QUESTION_LANG,,}"
        fi
        base=$(printf 'mlpq-%s-%s-%s-mono-%s' "$MLPQ_PAIR" "$MLPQ_QUESTION_LANG" "$MLPQ_FUSION" "$mono_lang")
      else
        base=$(printf 'mlpq-%s-%s-%s' "$MLPQ_PAIR" "$MLPQ_QUESTION_LANG" "$MLPQ_FUSION")
      fi
      ;;
    wqsp|cwq|kqapro|mintaka)
      base=$(printf '%s-%s' "$DATASET" "${SPLIT,,}")
      ;;
    custom)
      local safe_name="${CUSTOM_DATASET_NAME// /-}"
      base=$(printf 'custom-%s-%s' "${safe_name,,}" "${SPLIT,,}")
      ;;
    *)
      fail "Unsupported DATASET: $DATASET"
      ;;
  esac
  if [[ -n "$suffix" ]]; then
    suffix="$(printf '%s' "$suffix" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g')"
    printf '%s-%s' "$base" "$suffix"
  else
    printf '%s' "$base"
  fi
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
    wikimovies)
      WIKIMOVIES_KB_DIR="$DEFAULT_WIKIMOVIES_ROOT/movieqa/knowledge_source/wiki_entities"
      if [[ -f "$WIKIMOVIES_KB_DIR/wiki_entities_kb_normalized.txt" ]]; then
        KB_PATH="$WIKIMOVIES_KB_DIR/wiki_entities_kb_normalized.txt"
      else
        KB_PATH="$WIKIMOVIES_KB_DIR/wiki_entities_kb.txt"
      fi
      ;;
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
if [[ -n "$GRAMMAR_KB_PATH" ]]; then
  require_file "$GRAMMAR_KB_PATH" "GRAMMAR_KB_PATH"
fi

EFFECTIVE_GRAMMAR_KB_PATH="${GRAMMAR_KB_PATH:-$KB_PATH}"

GRAMMAR_ARGS=(
  --kb-path "$EFFECTIVE_GRAMMAR_KB_PATH"
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
  --kb-ablation-mode "$KB_ABLATION_MODE"
  --kb-ablation-ratio "$KB_ABLATION_RATIO"
  --kb-ablation-seed "$KB_ABLATION_SEED"
  --run-tag-suffix "$RUN_TAG_SUFFIX"
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

if [[ -n "${ALIAS_PATH:-}" ]]; then
  require_file "$ALIAS_PATH" "ALIAS_PATH"
  BENCHMARK_ARGS+=(--alias-path "$ALIAS_PATH")
fi

if [[ -n "$MODEL_FILTER" ]]; then
  BENCHMARK_ARGS+=(--model-filter "$MODEL_FILTER")
fi

if [[ "$FIXED_ABLATION_BUDGET" == "1" ]]; then
  BENCHMARK_ARGS+=(--fixed-ablation-budget)
fi

if [[ "$RETRY_FAILED" == "1" ]]; then
  BENCHMARK_ARGS+=(--retry-failed)
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
echo "[run] ranking_policy=$RANKING_POLICY"
echo "[run] cuda_visible_devices=$CUDA_VISIBLE_DEVICES target_device=$TARGET_DEVICE model_sharding=$ENABLE_MODEL_SHARDING strict_sharding=$STRICT_GPU_SHARDING"
echo "[run] grammar=$GRAMMAR_PATH"
echo "[run] results=$OUTPUT_FILE"

if [[ "$RUN_GRAMMAR" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/hrg_grammar/hrg_extract.py" "${GRAMMAR_ARGS[@]}"
else
  echo "[run] skipping grammar generation"
fi

"$PYTHON_BIN" "$ROOT_DIR/LLM_inference_benchmark/benchmark.py" "${BENCHMARK_ARGS[@]}"
