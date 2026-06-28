#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}


def parse_cuda_device_indices(target_device: str) -> list[int]:
    indices: list[int] = []
    for part in target_device.split(","):
        part = part.strip().lower()
        if part == "cuda":
            indices.append(0)
            continue
        if part.startswith("cuda:"):
            part = part.split(":", 1)[1]
        if part.isdigit():
            indices.append(int(part))
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Qwen3.5 model-load readiness for the benchmark.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--target-device", default=os.getenv("TARGET_DEVICE", "cuda:0,cuda:1"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--load-model", action="store_true", help="Actually instantiate AutoModelForCausalLM. This can consume a lot of VRAM.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "python": sys.executable,
        "model_id": args.model_id,
        "target_device": args.target_device,
        "env": {
            "HF_HOME": os.getenv("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "ENABLE_MODEL_SHARDING": os.getenv("ENABLE_MODEL_SHARDING"),
            "STRICT_GPU_SHARDING": os.getenv("STRICT_GPU_SHARDING"),
            "MAX_MEMORY_CPU": os.getenv("MAX_MEMORY_CPU"),
        },
    }

    try:
        import torch
        import transformers
        import huggingface_hub
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [],
        }
        for idx in range(torch.cuda.device_count()):
            report["torch"]["devices"].append({
                "index": idx,
                "name": safe_call(lambda idx=idx: torch.cuda.get_device_name(idx)),
                "compute_capability": safe_call(lambda idx=idx: list(torch.cuda.get_device_capability(idx))),
                "mem_get_info": safe_call(lambda idx=idx: list(torch.cuda.mem_get_info(idx))),
            })

        qwen_keys = sorted(key for key in CONFIG_MAPPING.keys() if "qwen" in key.lower())
        report["transformers"] = {
            "version": transformers.__version__,
            "file": transformers.__file__,
            "qwen_config_keys": qwen_keys,
            "has_qwen3_5_moe": "qwen3_5_moe" in CONFIG_MAPPING,
            "has_qwen3_5_moe_text": "qwen3_5_moe_text" in CONFIG_MAPPING,
        }
        report["huggingface_hub"] = {"version": huggingface_hub.__version__}

        hf_home = os.getenv("HF_HOME") or str(ROOT_DIR / ".cache" / "huggingface")
        cache_root = Path(os.getenv("HUGGINGFACE_HUB_CACHE") or str(Path(hf_home) / "hub"))
        local_model_dir = cache_root / ("models--" + args.model_id.replace("/", "--"))
        report["cache"] = {
            "cache_root": str(cache_root),
            "local_model_dir": str(local_model_dir),
            "local_model_dir_exists": local_model_dir.exists(),
        }
        if local_model_dir.exists():
            report["cache"]["size_bytes"] = sum(p.stat().st_size for p in local_model_dir.rglob("*") if p.is_file())

        local_files_only = not args.allow_download
        report["config"] = safe_call(lambda: AutoConfig.from_pretrained(
            args.model_id,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to_dict())
        report["tokenizer"] = safe_call(lambda: {
            "class": AutoTokenizer.from_pretrained(
                args.model_id,
                local_files_only=local_files_only,
                trust_remote_code=True,
                use_fast=True,
            ).__class__.__name__
        })

        requested_cuda = parse_cuda_device_indices(args.target_device)
        config_payload = report["config"] if isinstance(report["config"], dict) else {}
        quant_config = config_payload.get("quantization_config") if isinstance(config_payload, dict) else {}
        quant_method = (quant_config or {}).get("quant_method")
        fp8_unsupported_devices = []
        if quant_method == "fp8" and torch.cuda.is_available():
            for idx in requested_cuda or range(torch.cuda.device_count()):
                if idx >= torch.cuda.device_count():
                    continue
                major, minor = torch.cuda.get_device_capability(idx)
                if (major, minor) < (8, 9):
                    fp8_unsupported_devices.append({
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "compute_capability": [major, minor],
                    })
        report["fp8_runtime"] = {
            "quant_method": quant_method,
            "requires_compute_capability": [8, 9] if quant_method == "fp8" else None,
            "unsupported_target_devices": fp8_unsupported_devices,
            "will_dequantize_to_bf16": bool(fp8_unsupported_devices),
        }

        if requested_cuda and not torch.cuda.is_available():
            report["diagnosis"] = (
                "CUDA is not available in this environment, but benchmark config requests "
                f"{args.target_device}. Qwen3.5 will fail during model initialization here."
            )
        elif requested_cuda and any(idx >= torch.cuda.device_count() for idx in requested_cuda):
            report["diagnosis"] = (
                f"Only {torch.cuda.device_count()} CUDA devices are visible, but target_device="
                f"{args.target_device} requests invalid device indices."
            )
        elif report["transformers"]["has_qwen3_5_moe"] is False:
            report["diagnosis"] = "Transformers does not support qwen3_5_moe in this environment."
        elif not report["cache"]["local_model_dir_exists"] and not args.allow_download:
            report["diagnosis"] = "Qwen3.5 is not present in the local HF cache and downloads are disabled."
        elif fp8_unsupported_devices:
            report["diagnosis"] = (
                "This checkpoint is FP8, but the target GPU compute capability is below 8.9. "
                "Transformers will dequantize the model to bf16 on this hardware, which can cause "
                "Qwen3.5 to OOM on two A40 GPUs during model loading."
            )
        else:
            report["diagnosis"] = (
                "Static checks passed. If benchmark still marks qwen3.5 as FAILED, inspect "
                "results/failure_report.jsonl; the likely remaining causes are VRAM/offload under "
                "STRICT_GPU_SHARDING, FP8 kernel support, or generation-time CUDA errors."
            )

        if args.load_model:
            from transformers import AutoModelForCausalLM

            report["model_load_attempt"] = safe_call(lambda: {
                "class": AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    local_files_only=local_files_only,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).__class__.__name__
            })

    except Exception:
        report["fatal_error"] = traceback.format_exc()

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
