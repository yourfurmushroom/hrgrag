# agent_factory.py
from __future__ import annotations

import os
import re
import torch
from transformers import AutoModelForCausalLM
from LLM_stratiges import HarmonyHFStrategy, GenericHFStrategy


def needs_harmony(model_id: str) -> bool:
    # ✅ 你可以依你的命名規則調整
    # 只要是 openai/gpt-oss 類就走 Harmony
    model_id_lower = model_id.lower()
    return model_id_lower.startswith("openai/") or "gpt-oss" in model_id_lower


def needs_trust_remote_code(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    return "qwen" in model_id_lower


def _parse_cuda_devices(target_device: str | None) -> list[int]:
    if not target_device:
        return []

    devices: list[int] = []
    has_multiple_devices = "," in target_device
    for raw_part in target_device.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue

        if part == "cuda":
            devices.append(0)
            continue

        match = re.fullmatch(r"(?:cuda:?|gpu:?)?(\d+)", part)
        if not match:
            if not has_multiple_devices:
                return []
            raise ValueError(
                f"Unsupported target_device value: {target_device!r}. "
                "Use 'cuda:0' for one GPU or 'cuda:0,cuda:1' for model sharding."
            )
        devices.append(int(match.group(1)))

    return devices


def _validate_cuda_devices(devices: list[int], target_device: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"target_device={target_device!r} requires CUDA, but CUDA is not available.")

    device_count = torch.cuda.device_count()
    invalid = [idx for idx in devices if idx < 0 or idx >= device_count]
    if invalid:
        raise RuntimeError(
            f"target_device={target_device!r} requested CUDA device(s) {invalid}, "
            f"but only {device_count} CUDA device(s) are visible."
        )


def _normalize_device_placement(device) -> str:
    device_text = str(device).lower()
    if device_text.isdigit():
        return f"cuda:{device_text}"
    if device_text.startswith("cuda") and ":" not in device_text:
        return device_text.replace("cuda", "cuda:", 1)
    return device_text


def build_llm_strategy(
    model_id: str,
    max_new_tokens: int = 1024,
    use_model_sharding: bool | None = None,
    strict_gpu_sharding: bool | None = None,
    target_device: str | None = None,
):
    trust_remote_code = needs_trust_remote_code(model_id)
    env_sharding = os.getenv("ENABLE_MODEL_SHARDING", "").strip().lower()
    env_strict = os.getenv("STRICT_GPU_SHARDING", "").strip().lower() in {"1", "true", "yes"}

    if use_model_sharding is None:
        use_sharding = env_sharding in {"1", "true", "yes"}
    else:
        use_sharding = use_model_sharding

    if strict_gpu_sharding is None:
        strict_gpu_sharding = env_strict

    target_device = target_device or os.getenv("TARGET_DEVICE")
    target_cuda_devices = _parse_cuda_devices(target_device)

    if use_sharding:
        if target_cuda_devices:
            _validate_cuda_devices(target_cuda_devices, target_device)

        max_memory = {}
        memory_device_indices = target_cuda_devices or list(range(torch.cuda.device_count()))
        for idx in memory_device_indices:
            env_key = f"MAX_MEMORY_GPU{idx}"
            if os.getenv(env_key):
                max_memory[idx] = os.getenv(env_key)
            elif target_cuda_devices:
                max_memory[idx] = torch.cuda.mem_get_info(idx)[0]
        if os.getenv("MAX_MEMORY_CPU"):
            max_memory["cpu"] = os.getenv("MAX_MEMORY_CPU")

        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        if max_memory:
            load_kwargs["max_memory"] = max_memory

        offload_folder = os.getenv("OFFLOAD_FOLDER")
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if strict_gpu_sharding:
            device_map = getattr(model, "hf_device_map", {}) or {}
            allowed_cuda_placements = {
                f"cuda:{idx}" for idx in target_cuda_devices
            } if target_cuda_devices else None
            bad_placements = {
                _normalize_device_placement(device)
                for device in device_map.values()
                if _normalize_device_placement(device) in {"cpu", "disk"}
                or (
                    allowed_cuda_placements is not None
                    and _normalize_device_placement(device).startswith("cuda:")
                    and _normalize_device_placement(device) not in allowed_cuda_placements
                )
            }
            if bad_placements:
                raise RuntimeError(
                    f"STRICT_GPU_SHARDING is enabled, but {model_id} was offloaded to "
                    f"{sorted(bad_placements)}. Adjust target_device/GPU memory limits "
                    "or disable strict mode."
                )
    else:
        if not target_device:
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif len(target_cuda_devices) > 1:
            raise ValueError(
                f"target_device={target_device!r} contains multiple GPUs, but model sharding is disabled. "
                "Set use_model_sharding=True/ENABLE_MODEL_SHARDING=1, or use a single device such as 'cuda:0'."
            )
        elif len(target_cuda_devices) == 1:
            _validate_cuda_devices(target_cuda_devices, target_device)
            target_device = f"cuda:{target_cuda_devices[0]}"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        model.to(target_device)

    model.eval()

    if needs_harmony(model_id):
        return HarmonyHFStrategy(model=model, max_new_tokens=max_new_tokens)

    return GenericHFStrategy(model=model, tokenizer_name_or_path=model_id, max_new_tokens=max_new_tokens)
