# Full Rerun Guide

## What This Rerun Includes

`run_full_rerun.sh` runs:

- existing core methods:
  - `Baseline-BFS`
  - `Spine-Only`
  - `Spine-Correction`
  - `HRG-Proposed`
- existing commented ablations restored under full suite:
  - `Spine-GrammarExpansion`
  - `Spine-RandomExpansion`
  - `Spine-FrequencyExpansion`
- new professor-requested ablations:
  - `Spine-Correction-KGValidFallback`
  - `HRG-Proposed-NoExpansion`
  - `HRG-Proposed-Expansion`
  - `HRG-GrammarFirst-NoExpansion`
  - `HRG-GrammarFirst-Expansion`
  - `RelationUnigram`
  - `RelationBigram`
  - `RelationTrigram`
- fairer BFS baselines:
  - `Degree-Capped-BFS-{50,100,200,500}`
  - `Token-Budgeted-BFS-{200,500,1000}`
- clean datasets:
  - MetaQA
  - WikiMovies
  - MLPQ
  - KQAPro
- perturbation experiments:
  - `drop_nodes`: 10%, 20%, 30%
  - `drop_relations`: 10%, 20%, 30%
  - seeds: 0, 1, 2, 3, 4

## Run

Recommended sequential driver:

```bash
./run_everything_sequential.sh
```

This runs syntax checks, Qwen3.5 preflight, clean reachability audits, full clean benchmarks, perturbation benchmarks, and final summaries in order.

The current model matrix excludes `qwen3.5` because `Qwen/Qwen3.5-35B-A3B-FP8` is not FP8-compatible with A40 GPUs. The script still writes a static Qwen3.5 diagnosis, but skips the Qwen3.5 benchmark probe unless explicitly enabled.

To enable the Qwen3.5 benchmark probe:

```bash
RUN_QWEN35_PROBE=1 ./run_everything_sequential.sh
```

If Qwen3.5 fails and you still want to continue:

```bash
RUN_QWEN35_PROBE=1 CONTINUE_AFTER_QWEN_FAILURE=1 ./run_everything_sequential.sh
```

To enable the heavy Qwen3.5 model-load diagnosis:

```bash
RUN_QWEN_LOAD_DIAGNOSIS=1 ./run_everything_sequential.sh
```

Lower-level full rerun only:

```bash
./run_full_rerun.sh
```

Default outputs:

```text
artifacts_full/
artifacts_full/_summary/
```

## Important Controls

Use fewer questions for a dry run:

```bash
SAMPLE_LIMIT=5 RUN_PERTURBATION=0 ./run_full_rerun.sh
```

Run only clean experiments:

```bash
RUN_PERTURBATION=0 ./run_full_rerun.sh
```

Use a specific model subset:

```bash
MODEL_FILTER=gpt-oss ./run_full_rerun.sh
MODEL_FILTER=qwen3.5 ./run_full_rerun.sh
```

Change BFS token budgets:

```bash
BFS_CONTEXT_TOKEN_BUDGETS="100 200 400 800" ./run_full_rerun.sh
```

Change perturbation seeds:

```bash
KB_ABLATION_SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_full_rerun.sh
```

## Qwen3.5 Diagnosis

Before rerunning Qwen3.5:

```bash
.venv/bin/python diagnose_qwen35.py --output-json artifacts_full/_summary/qwen35_static_diagnosis.json
```

`Qwen/Qwen3.5-35B-A3B-FP8` requires GPUs with compute capability >= 8.9 to run as FP8 in Transformers. A40 is compute capability 8.6, so Transformers dequantizes this checkpoint to bf16 and VRAM use increases substantially.

If Qwen3.5 fails with CUDA OOM while loading, use more GPUs for the sharded backbone:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
QWEN3_5_TARGET_DEVICE=cuda:0,cuda:1,cuda:2,cuda:3 \
MAX_MEMORY_GPU0=40GiB \
MAX_MEMORY_GPU1=40GiB \
MAX_MEMORY_GPU2=40GiB \
MAX_MEMORY_GPU3=38GiB \
./run_everything_sequential.sh
```

`QWEN3_5_TARGET_DEVICE` overrides only the Qwen3.5 backbone. `SHARDED_TARGET_DEVICE` can be used when the same GPU list should apply to all sharded backbones.

If only GPU 0 and 1 are available, first try a GPU-only load with asymmetric memory caps so GPU 0 keeps headroom:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
QWEN3_5_TARGET_DEVICE=cuda:0,cuda:1 \
MAX_MEMORY_GPU0=36GiB \
MAX_MEMORY_GPU1=43GiB \
./run_everything_sequential.sh
```

If that still OOMs, allow CPU/disk offload for Qwen3.5:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
QWEN3_5_TARGET_DEVICE=cuda:0,cuda:1 \
QWEN3_5_STRICT_GPU_SHARDING=0 \
MAX_MEMORY_GPU0=36GiB \
MAX_MEMORY_GPU1=43GiB \
MAX_MEMORY_CPU=120GiB \
OFFLOAD_FOLDER=artifacts_full/_offload/qwen35 \
./run_everything_sequential.sh
```

If GPUs are visible and you want to test actual model loading:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ENABLE_MODEL_SHARDING=1 \
STRICT_GPU_SHARDING=1 \
.venv/bin/python diagnose_qwen35.py --load-model --output-json artifacts_full/_summary/qwen35_load_diagnosis.json
```

During the benchmark, model-level failures now write:

```text
<run>/results/failure_report.jsonl
```

This is the file to inspect if Qwen3.5 fails again.

## Summary Files

After the run, `summarize_experiment_runs.py` writes:

```text
artifacts_full/_summary/run_model_summary.csv
artifacts_full/_summary/method_mean_std.csv
artifacts_full/_summary/evaluation_units.csv
artifacts_full/_summary/grammar_hit_breakdown.csv
artifacts_full/_summary/perturbation_mean_std.csv
artifacts_full/_summary/failure_summary.csv
artifacts_full/_summary/metaqa_bfs_sanity/summary.csv
artifacts_full/_summary/symbolic_endpoints/summary.csv
artifacts_full/_summary/signed_chains/signed_chains.csv
```

These include the extra metrics needed for the professor-facing tables:

- candidate validity rate
- candidate grammar hit rate
- selected candidate source counts
- grammar hit breakdown: label subset, same arity, ordered path, weak-label-only
- correction salvage rate
- expanded edge count and expanded-question rate
- final context tokens
- total online token proxy (`parse1 + correction + parse2 + final context tokens`)
- model-level mean/std across active backbones
- evaluation-unit counts for question/model-pair reporting
- MetaQA BFS context coverage/truncation sanity checks
- symbolic endpoint answer diagnostics
- signed `r+` / `r-` traversal chains for case tables
- perturbation mean and standard deviation
- failure summaries

## Additional Diagnostics

Fast post-hoc diagnostics can be run on saved dumps without loading a model:

```bash
.venv/bin/python analyze_metaqa_bfs_sanity.py --artifacts-root artifacts_full
.venv/bin/python analyze_symbolic_endpoints.py --artifacts-root artifacts_full --model-filter HRG-Proposed
.venv/bin/python annotate_signed_chains_from_dumps.py --artifacts-root artifacts_full --model-filter HRG-Proposed
```

Perturbation reachability audits rebuild ablated KBs and check whether topic entities / answers remain reachable:

```bash
SAMPLE_LIMIT=50 KB_ABLATION_SEEDS="0 1 2 3 4" ./run_perturbation_reachability_audits.sh
```

For a single-dataset audit:

```bash
CONFIG_FILES=configs/config.metaqa.env SAMPLE_LIMIT=50 ./run_perturbation_reachability_audits.sh
```

Hyperparameter sensitivity runs are intentionally small by default:

```bash
SAMPLE_LIMIT=50 MODEL_FILTER=HRG-Proposed-NoExpansion-gpt-oss-triple ./run_hyperparameter_sensitivity.sh
```

Evidence-causal controls require loading a model, so run them separately:

```bash
.venv/bin/python run_context_controls_from_dumps.py \
  --artifacts-root artifacts_full \
  --dump-model-filter HRG-Proposed-NoExpansion \
  --sample-limit 20 \
  --model-id Qwen/Qwen2.5-7B-Instruct
```
