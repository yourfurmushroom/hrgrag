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
