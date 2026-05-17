#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "LLM_inference_benchmark" / "benchmark.py"


def set_cache_defaults(cache_root: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("NLTK_DATA", str(cache_root / "nltk"))


def load_model_backbones() -> list[dict[str, Any]]:
    tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"), filename=str(BENCHMARK_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_BACKBONES":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"MODEL_BACKBONES not found in {BENCHMARK_PATH}")


def cuda_snapshot(torch_module) -> list[dict[str, Any]]:
    if not torch_module.cuda.is_available():
        return []

    rows = []
    for idx in range(torch_module.cuda.device_count()):
        free, total = torch_module.cuda.mem_get_info(idx)
        rows.append(
            {
                "device": idx,
                "name": torch_module.cuda.get_device_name(idx),
                "free_gb": round(free / 1024**3, 3),
                "total_gb": round(total / 1024**3, 3),
                "allocated_gb": round(torch_module.cuda.memory_allocated(idx) / 1024**3, 3),
                "reserved_gb": round(torch_module.cuda.memory_reserved(idx) / 1024**3, 3),
                "peak_allocated_gb": round(torch_module.cuda.max_memory_allocated(idx) / 1024**3, 3),
            }
        )
    return rows


def print_snapshot(label: str, snapshot: list[dict[str, Any]]) -> None:
    if not snapshot:
        print(f"[{label}] CUDA unavailable")
        return
    for row in snapshot:
        print(
            f"[{label}] gpu{row['device']} {row['name']} "
            f"free={row['free_gb']}GB total={row['total_gb']}GB "
            f"allocated={row['allocated_gb']}GB reserved={row['reserved_gb']}GB "
            f"peak_allocated={row['peak_allocated_gb']}GB"
        )


def selected_backbones(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.model:
        return [
            {
                "tag": f"manual-{idx + 1}",
                "model_id": model_id,
                "use_model_sharding": args.enable_sharding,
                "strict_gpu_sharding": args.strict_gpu_sharding,
                "target_device": args.target_device,
            }
            for idx, model_id in enumerate(args.model)
        ]

    backbones = load_model_backbones()
    if args.model_filter:
        needle = args.model_filter.lower()
        backbones = [
            item
            for item in backbones
            if needle in item.get("tag", "").lower() or needle in item.get("model_id", "").lower()
        ]
    return backbones


async def run_generation(strategy, prompt: str) -> str:
    return await strategy.inference(
        "You are running a short GPU memory smoke test. Keep the answer brief.",
        prompt,
    )


def test_one_model(args: argparse.Namespace, backbone: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "LLM_inference_benchmark"))
    import torch
    from agent_factory import build_llm_strategy

    tag = backbone.get("tag", "unknown")
    model_id = backbone["model_id"]
    target_device = args.target_device if args.target_device is not None else backbone.get("target_device")
    use_model_sharding = args.enable_sharding if args.enable_sharding is not None else backbone.get("use_model_sharding")
    strict_gpu_sharding = (
        args.strict_gpu_sharding
        if args.strict_gpu_sharding is not None
        else backbone.get("strict_gpu_sharding")
    )

    for idx in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        torch.cuda.reset_peak_memory_stats(idx)

    print("=" * 80)
    print(f"[model] tag={tag}")
    print(f"[model] id={model_id}")
    print(f"[model] target_device={target_device}")
    print(f"[model] use_model_sharding={use_model_sharding}")
    print(f"[model] strict_gpu_sharding={strict_gpu_sharding}")
    print_snapshot("before", cuda_snapshot(torch))

    result: dict[str, Any] = {
        "tag": tag,
        "model_id": model_id,
        "target_device": target_device,
        "use_model_sharding": use_model_sharding,
        "strict_gpu_sharding": strict_gpu_sharding,
        "ok": False,
    }

    strategy = None
    started = time.time()
    try:
        strategy = build_llm_strategy(
            model_id=model_id,
            max_new_tokens=args.max_new_tokens,
            use_model_sharding=use_model_sharding,
            strict_gpu_sharding=strict_gpu_sharding,
            target_device=target_device,
        )
        result["load_seconds"] = round(time.time() - started, 3)
        print_snapshot("after_load", cuda_snapshot(torch))

        if not args.skip_generate:
            output = asyncio.run(run_generation(strategy, args.prompt))
            result["sample_output"] = output[:500]
            print(f"[generate] output={output[:500]!r}")
            print_snapshot("after_generate", cuda_snapshot(torch))

        final_snapshot = cuda_snapshot(torch)
        result["cuda"] = final_snapshot
        low_free = [
            row for row in final_snapshot
            if row["free_gb"] < args.reserve_gb
        ]
        if low_free:
            result["warning"] = f"free VRAM below reserve_gb={args.reserve_gb}: {low_free}"
            print(f"[warning] {result['warning']}")
        result["ok"] = True
        return result
    except RuntimeError as exc:
        message = str(exc)
        result["error"] = message
        result["error_type"] = "RuntimeError"
        result["looks_like_oom"] = "out of memory" in message.lower() or "cuda" in message.lower()
        print(f"[error] {type(exc).__name__}: {message}")
        print_snapshot("after_error", cuda_snapshot(torch))
        return result
    except Exception as exc:
        result["error"] = repr(exc)
        result["error_type"] = type(exc).__name__
        print(f"[error] {type(exc).__name__}: {exc}")
        print_snapshot("after_error", cuda_snapshot(torch))
        return result
    finally:
        if strategy is not None:
            del strategy
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                print(f"[warning] torch.cuda.empty_cache failed: {exc}")
            print_snapshot("after_unload", cuda_snapshot(torch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load benchmark models with the current environment and report CUDA memory usage."
    )
    parser.add_argument("--model", action="append", help="Explicit Hugging Face model id. Can be repeated.")
    parser.add_argument("--model-filter", default=None, help="Filter MODEL_BACKBONES by tag or model id substring.")
    parser.add_argument("--target-device", default=None, help="Override target device, e.g. cuda:0 or cuda:0,cuda:1.")
    parser.add_argument("--enable-sharding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strict-gpu-sharding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Reply with exactly one word: ready")
    parser.add_argument("--skip-generate", action="store_true", help="Only load the model; do not run generation.")
    parser.add_argument("--reserve-gb", type=float, default=2.0, help="Warn if free VRAM falls below this value.")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / ".cache"))
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face files only.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_cache_defaults(Path(args.cache_root))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"[env] python={sys.executable}")
    print(f"[env] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[env] HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    print(f"[env] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    backbones = selected_backbones(args)
    if not backbones:
        print("[error] no models selected")
        return 2

    results = []
    for backbone in backbones:
        result = test_one_model(args, backbone)
        results.append(result)
        if args.stop_on_error and not result.get("ok"):
            break

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] wrote {out_path}")

    failed = [item for item in results if not item.get("ok")]
    if failed:
        print(f"[summary] failed={len(failed)} total={len(results)}")
        return 1

    print(f"[summary] ok={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
