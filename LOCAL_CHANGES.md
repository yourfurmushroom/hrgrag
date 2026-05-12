# Local Changes Needed For Benchmark Runs

This file records local changes that were added to keep the benchmark runnable after future pulls.

## Summary

- Use the latest Hugging Face `transformers` from GitHub so Qwen3.5 MoE configs are available.
- Use `uv` in `setup_env.sh` for faster and more reproducible environment setup.
- Verify `qwen3_5_moe` support immediately after setup.
- Disable Qwen3 thinking mode in chat template generation.
- Save benchmark reports after each model finishes or fails, so partial results are not lost.
- Handle CUDA cache cleanup failures without crashing the whole benchmark process.

## Files Changed

### `requirements.txt`

- Changed `transformers` to install from the Hugging Face GitHub repository:

  ```text
  transformers @ git+https://github.com/huggingface/transformers.git
  ```

- Updated `huggingface-hub` to:

  ```text
  huggingface-hub>=1.5.0,<2.0
  ```

These changes are needed because older released `transformers` versions may not recognize `qwen3_5_moe`.

### `setup_env.sh`

- Requires `uv`.
- Uses `uv venv` and `uv pip install`.
- Installs PyTorch with CUDA 12.4 by default.
- Runs a setup check that prints:
  - Python executable
  - `transformers` version
  - `huggingface_hub` version
  - Whether `qwen3_5_moe` is present in `CONFIG_MAPPING`
  - Whether `qwen3_5_moe_text` is present in `CONFIG_MAPPING`

If `qwen3_5_moe` is missing, setup exits with an error.

### `LLM_inference_benchmark/LLM_stratiges.py`

- Stores `tokenizer_name_or_path` in `GenericHFStrategy`.
- When the tokenizer path/name contains `qwen3`, passes this option to `apply_chat_template`:

  ```python
  enable_thinking=False
  ```

This avoids Qwen3 thinking-mode output during benchmark generation.

### `LLM_inference_benchmark/agent_factory.py`

- Adds debug output for:
  - Python executable
  - `transformers` version and file path
  - Qwen3 config keys in `CONFIG_MAPPING`
  - target device
  - first model parameter device
- Loads generic Hugging Face models with:

  ```python
  device_map="cuda"
  trust_remote_code=True
  ```

This was added to make Qwen-style models load correctly on CUDA.

### `LLM_inference_benchmark/benchmark.py`

- Adds `save_full_report()` and writes the JSON report after each model finishes.
- If a model fails, saves `"FAILED"` for that model before continuing.
- Adds `safe_clear_gpu_cache()` so CUDA cleanup errors are logged instead of crashing the benchmark.

This is important for long benchmark runs because completed results remain saved even if a later model fails.

## After Pulling

If a future `git pull` changes any of these files, verify the following before running benchmarks:

1. `requirements.txt` still installs a `transformers` version that supports `qwen3_5_moe`.
2. `setup_env.sh` still checks for `qwen3_5_moe` after installing dependencies.
3. Qwen3 generation still uses `enable_thinking=False`.
4. `benchmark.py` still saves partial reports after each model.
5. CUDA cleanup failures still do not stop the full run.

Recommended setup command:

```bash
./setup_env.sh
```

Recommended pull command when local commits exist:

```bash
git pull --rebase
```

## Follow-Up Optimizations After Pulling

These are not required for the current runnable state, but should be considered after future benchmark or paper-alignment changes are pulled.

### 1. Add hop-level checkpointing

Current behavior saves `benchmark_results.json` after a full model finishes. For MetaQA, one model includes `1-hop`, `2-hop`, and `3-hop`. If the run stops in the middle of `3-hop`, the model may need to rerun from `1-hop`.

Better behavior:

- Save each `model x hop` result as soon as that hop finishes.
- Resume by skipping completed hops instead of only completed models.

### 2. Add per-question resume

Current `per_model/q_xxxx.pkl` files are useful for debugging, but they are not used as checkpoints. If a run stops at question 70, the current code does not resume from question 71.

Better behavior:

- Treat completed per-question dumps as resumable records.
- Recompute aggregate metrics from completed per-question outputs.
- Only rerun missing or invalid questions.

### 3. Fix CSV resume behavior

`benchmark_results.json` can be loaded on rerun, but `ALL_LONG_ROWS` only contains rows from the current process. If completed models are skipped, `all_models_outputs_wide.csv` may not contain skipped models.

Better behavior:

- Rebuild CSV from persisted JSON and/or per-question dumps.
- Ensure CSV and JSON always describe the same completed runs.

### 4. Improve failed-run retry behavior

Current behavior records failed models as:

```json
"ModelName": "FAILED"
```

Because `"FAILED"` still appears in `benchmark_results.json`, reruns may skip that model.

Better behavior:

- Only skip models marked as completed.
- Add a `--retry-failed` option.
- Store failure metadata separately from completed results.

### 5. Make shared retrieval cache invalidation safer

The `_shared_retrieval/*.prepared.pkl` cache can save time, but it may become stale if the KB, grammar, retrieval settings, or prompt behavior changes without changing the run tag.

Better behavior:

- Include KB file hash in the shared cache key.
- Include grammar file hash in the shared cache key.
- Include relevant retrieval configuration in the cache key.

### 6. Pin `transformers` to a commit

Current dependency:

```text
transformers @ git+https://github.com/huggingface/transformers.git
```

This installs the latest main branch, which may change over time. That is risky for paper experiments.

Better behavior:

- Pin to a known working commit hash.
- Record the installed `transformers` commit/version in benchmark artifacts.

### 7. Avoid duplicate bootstrap work

`run_local_all.sh` runs `bootstrap_all.sh`, and `auto_benchmark.sh` also runs `bootstrap_all.sh`.

Better behavior:

- Let only one layer handle bootstrap.
- Add an option to skip bootstrap when datasets and configs are already ready.

### 8. Gate debug logs behind an env flag

`agent_factory.py` currently prints detailed model-loading debug information every run.

Better behavior:

- Print these logs only when an env flag is enabled, for example:

  ```bash
  DEBUG_MODEL_LOAD=1
  ```

### 9. Make GPU loading strategy configurable

Generic Hugging Face models currently load with:

```python
device_map="cuda"
```

This is okay for the current single-GPU setup, but may be too rigid for multi-GPU, CPU fallback, or different machines.

Better behavior:

- Respect `target_device` more directly.
- Allow `device_map=auto` through configuration.
- Record actual device placement in artifacts.

### 10. Review sampling strategy for paper runs

Current behavior takes the first `SAMPLE_LIMIT` questions per split/hop:

```python
dataset = dataset[:sample_limit]
```

For MetaQA with `SAMPLE_LIMIT=100`, this means 100 questions per hop, or 300 questions per model.

Better behavior:

- Use fixed-seed random sampling if the paper setup requires unbiased sampling.
- Save sampled question IDs so reruns use the exact same subset.
