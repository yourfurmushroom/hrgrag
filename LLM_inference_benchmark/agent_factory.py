# agent_factory.py
from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM
from LLM_stratiges import HarmonyHFStrategy, GenericHFStrategy


def needs_harmony(model_id: str) -> bool:
    # ✅ 你可以依你的命名規則調整
    # 只要是 openai/gpt-oss 類就走 Harmony
    model_id_lower = model_id.lower()
    return model_id_lower.startswith("openai/") or "gpt-oss" in model_id_lower


def build_llm_strategy(
    model_id: str,
    max_new_tokens: int = 1024,
    use_model_sharding: bool | None = None,
    strict_gpu_sharding: bool | None = None,
    target_device: str | None = None,
):
    env_sharding = os.getenv("ENABLE_MODEL_SHARDING", "").strip().lower()
    env_strict = os.getenv("STRICT_GPU_SHARDING", "").strip().lower() in {"1", "true", "yes"}

    if use_model_sharding is None:
        use_sharding = env_sharding in {"1", "true", "yes"}
    else:
        use_sharding = use_model_sharding

    if strict_gpu_sharding is None:
        strict_gpu_sharding = env_strict

    if use_sharding:
        max_memory = {}
        for idx in range(torch.cuda.device_count()):
            env_key = f"MAX_MEMORY_GPU{idx}"
            if os.getenv(env_key):
                max_memory[idx] = os.getenv(env_key)
        if os.getenv("MAX_MEMORY_CPU"):
            max_memory["cpu"] = os.getenv("MAX_MEMORY_CPU")

        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if max_memory:
            load_kwargs["max_memory"] = max_memory

        offload_folder = os.getenv("OFFLOAD_FOLDER")
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if strict_gpu_sharding:
            device_map = getattr(model, "hf_device_map", {}) or {}
            bad_placements = {
                str(device).lower()
                for device in device_map.values()
                if str(device).lower() in {"cpu", "disk"}
            }
            if bad_placements:
                raise RuntimeError(
                    f"STRICT_GPU_SHARDING is enabled, but {model_id} was offloaded to "
                    f"{sorted(bad_placements)}. Adjust GPU memory limits or disable strict mode."
                )
    else:
        target_device = target_device or os.getenv("TARGET_DEVICE")
        if not target_device:
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.to(target_device)

    model.eval()

    if needs_harmony(model_id):
        return HarmonyHFStrategy(model=model, max_new_tokens=max_new_tokens)

    return GenericHFStrategy(model=model, tokenizer_name_or_path=model_id, max_new_tokens=max_new_tokens)
