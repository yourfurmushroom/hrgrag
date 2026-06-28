#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts_full}"
OUT_DIR="${OUT_DIR:-$ARTIFACTS_ROOT/_summary/perturbation_reachability}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
ABLATION_TYPES="${ABLATION_TYPES:-drop_nodes drop_relations}"
ABLATION_RATIOS="${ABLATION_RATIOS:-0.1 0.2 0.3}"
KB_ABLATION_SEEDS="${KB_ABLATION_SEEDS:-0 1 2 3 4}"

CONFIG_FILES="${CONFIG_FILES:-$ROOT_DIR/configs/config.metaqa.env $ROOT_DIR/configs/config.wikimovies.env $ROOT_DIR/configs/config.mlpq_en_zh_en_ills.env $ROOT_DIR/configs/config.kqapro.env}"

mkdir -p "$OUT_DIR/json" "$OUT_DIR/kb"

for config in $CONFIG_FILES; do
  # shellcheck disable=SC1090
  source "$config"
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  fi
  dataset="${DATASET}"
  original_kb="${KB_PATH}"
  dataset_root="${DATASET_ROOT}"
  split="${SPLIT:-test}"
  [[ -f "$original_kb" ]] || { echo "[skip] missing KB for $dataset: $original_kb" >&2; continue; }

  for seed in $KB_ABLATION_SEEDS; do
    for ablation_type in $ABLATION_TYPES; do
      for ratio in $ABLATION_RATIOS; do
        pct="$("$PYTHON_BIN" - <<PY
ratio = float("$ratio")
print(int(round(ratio * 100)))
PY
)"
        tag="${dataset}-${ablation_type}-${pct}pct-seed${seed}"
        ablated_kb="$OUT_DIR/kb/${tag}.tsv"
        out_json="$OUT_DIR/json/${tag}.json"

        echo "[reachability] dataset=$dataset type=$ablation_type ratio=$ratio seed=$seed"
        "$PYTHON_BIN" "$ROOT_DIR/create_ablated_kb.py" \
          --input-kb "$original_kb" \
          --output-kb "$ablated_kb" \
          --mode "$ablation_type" \
          --ratio "$ratio" \
          --seed "$seed" >/dev/null

        case "$dataset" in
          metaqa)
            "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
              --dataset metaqa \
              --dataset-root "$dataset_root" \
              --kb-path "$ablated_kb" \
              --split "$split" \
              --metaqa-variant "${METAQA_VARIANT:-vanilla}" \
              --sample-limit "$SAMPLE_LIMIT" \
              --output-json "$out_json" >/dev/null
            ;;
          wikimovies)
            "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
              --dataset wikimovies \
              --dataset-root "$dataset_root" \
              --kb-path "$ablated_kb" \
              --split "$split" \
              --wikimovies-subset "${WIKIMOVIES_SUBSET:-wiki_entities}" \
              --sample-limit "$SAMPLE_LIMIT" \
              --output-json "$out_json" >/dev/null
            ;;
          mlpq)
            "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
              --dataset mlpq \
              --dataset-root "$dataset_root" \
              --kb-path "$ablated_kb" \
              --mlpq-pair "${MLPQ_PAIR:-en-zh}" \
              --mlpq-question-lang "${MLPQ_QUESTION_LANG:-en}" \
              --mlpq-fusion "${MLPQ_FUSION:-ills}" \
              --sample-limit "$SAMPLE_LIMIT" \
              --output-json "$out_json" >/dev/null
            ;;
          kqapro)
            "$PYTHON_BIN" "$ROOT_DIR/audit_dataset_reachability.py" \
              --dataset kqapro \
              --dataset-root "$dataset_root" \
              --kb-path "$ablated_kb" \
              --split "$split" \
              --sample-limit "$SAMPLE_LIMIT" \
              --output-json "$out_json" >/dev/null
            ;;
        esac
      done
    done
  done
done

"$PYTHON_BIN" "$ROOT_DIR/summarize_reachability_audits.py" \
  --input-dir "$OUT_DIR/json" \
  --out-csv "$OUT_DIR/summary.csv" \
  --out-json "$OUT_DIR/summary.json"

echo "[reachability] wrote $OUT_DIR"
