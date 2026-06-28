# Qwen3.5 Failure Diagnosis

## Current Evidence

Existing artifacts show Qwen3.5 failed across all datasets and methods:

```text
artifacts:     189 qwen3.5 entries, 189 FAILED
artifect_all:   28 qwen3.5 entries,  28 FAILED
```

Because the old benchmark stored only:

```json
"ModelName": "FAILED"
```

the original traceback is not recoverable from the existing result JSON files.

## Static Environment Check

`diagnose_qwen35.py` was added to check the Qwen3.5 environment without loading the 35B model by default.

On the current session:

```text
transformers: 5.8.0
qwen3_5_moe support: yes
qwen3_5_moe_text support: yes
local HF cache: present
local model cache size: about 75GB
CUDA available: no
CUDA device count: 0
benchmark target_device: cuda:0,cuda:1
```

Therefore, in this current environment Qwen3.5 would fail during model initialization because the benchmark requests CUDA devices that are not visible.

## Most Likely Failure Class

The old failures are unlikely to be dataset-specific because every Qwen3.5 method failed before producing per-question results.

The likely failure stage is model initialization, not KG retrieval.

Possible causes, in priority order:

1. CUDA devices not visible to the benchmark process.
2. `STRICT_GPU_SHARDING=1` rejected CPU/disk offload when the 35B FP8 model could not fit fully on `cuda:0,cuda:1`.
3. FP8 runtime/kernel support issue for the installed torch/transformers/GPU stack.
4. Generation-time CUDA failure after successful load.

The next rerun will distinguish these because `benchmark.py` now writes:

```text
results/failure_report.jsonl
```

with:

- model name
- model id
- agent kwargs
- error type
- error message
- full traceback

## How To Diagnose Before Full Rerun

Run:

```bash
.venv/bin/python diagnose_qwen35.py --output-json artifacts_full/_summary/qwen35_static_diagnosis.json
```

If GPUs are available and you want a real model-load test:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ENABLE_MODEL_SHARDING=1 \
STRICT_GPU_SHARDING=1 \
.venv/bin/python diagnose_qwen35.py --load-model --output-json artifacts_full/_summary/qwen35_load_diagnosis.json
```

If strict sharding fails because of CPU/disk offload, test a controlled non-strict run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ENABLE_MODEL_SHARDING=1 \
STRICT_GPU_SHARDING=0 \
.venv/bin/python diagnose_qwen35.py --load-model --output-json artifacts_full/_summary/qwen35_non_strict_load_diagnosis.json
```

## Practical Recommendation

Keep Qwen3.5 in the full rerun, but treat it as a model-readiness checkpoint:

- If static diagnosis fails, do not interpret Qwen3.5 as an experimental result.
- If strict load fails but non-strict load succeeds, report it as a hardware / sharding constraint.
- If load succeeds but generation fails, use the new `failure_report.jsonl` traceback to determine whether the issue is FP8 kernels, attention implementation, tokenizer/chat-template handling, or CUDA memory.

