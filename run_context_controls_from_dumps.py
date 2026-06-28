#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "LLM_inference_benchmark"))

from agent_factory import build_llm_strategy  # noqa: E402
from recompute_from_dumps import calculate_answer_metrics  # noqa: E402


DEV_INSTRUCTION = (
    "You are a strict answer formatter for KGQA.\n"
    "Use only the provided Knowledge Graph Context to answer the question.\n"
    "Return only the final answer. If the context is insufficient, output exactly: I don't know"
)


def iter_dumps(artifacts_root: Path, model_filter: str) -> Iterable[Tuple[str, Path]]:
    for model_dir in sorted(artifacts_root.glob("**/dumps/per_model/*")):
        if not model_dir.is_dir():
            continue
        if model_filter and model_filter not in model_dir.name:
            continue
        for dump_path in sorted(model_dir.rglob("q_*.pkl")):
            yield model_dir.name, dump_path


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def mask_answers(context: str, references: List[str]) -> str:
    out = context
    for ref in sorted({r for r in references if r}, key=len, reverse=True):
        out = re.sub(re.escape(ref), "[MASKED_ANSWER]", out, flags=re.IGNORECASE)
        out = re.sub(re.escape(ref.replace(" ", "_")), "[MASKED_ANSWER]", out, flags=re.IGNORECASE)
        out = re.sub(re.escape(ref.replace("_", " ")), "[MASKED_ANSWER]", out, flags=re.IGNORECASE)
    return out


def user_content(context: str, question: str) -> str:
    if context.strip():
        return f"{context.strip()}\n\nQuestion: {question}"
    return f"Question: {question}"


async def run_controls(args: argparse.Namespace) -> List[Dict[str, Any]]:
    artifacts_root = Path(args.artifacts_root).resolve()
    samples = []
    for model_name, dump_path in iter_dumps(artifacts_root, args.dump_model_filter):
        obj = load_pickle(dump_path)
        if not obj.get("final_context") or not obj.get("references"):
            continue
        samples.append((model_name, dump_path, obj))
        if len(samples) >= args.sample_limit:
            break

    if not samples:
        return []

    llm = build_llm_strategy(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        use_model_sharding=args.use_model_sharding,
        strict_gpu_sharding=args.strict_gpu_sharding,
        target_device=args.target_device,
    )

    rows: List[Dict[str, Any]] = []
    contexts = [str(obj.get("final_context", "")) for _, _, obj in samples]
    for idx, (source_model, dump_path, obj) in enumerate(samples):
        question = str(obj.get("question", ""))
        references = list(obj.get("references", []) or [])
        correct_context = str(obj.get("final_context", ""))
        wrong_context = contexts[(idx + 1) % len(contexts)] if len(contexts) > 1 else ""
        variants = {
            "correct_context": correct_context,
            "no_context": "",
            "wrong_context": wrong_context,
            "masked_answer_context": mask_answers(correct_context, references),
        }
        for variant, context in variants.items():
            raw = await llm.inference(DEV_INSTRUCTION, user_content(context, question))
            answer = str(raw or "").strip()
            metrics = calculate_answer_metrics(references, answer) if references else {}
            rows.append({
                "source_model": source_model,
                "dump_file": str(dump_path),
                "control_model_id": args.model_id,
                "variant": variant,
                "question": question,
                "references": "|".join(references),
                "answer": answer,
                "em": metrics.get("em", 0.0),
                "answer_set_f1": metrics.get("answer_set_f1", 0.0),
                "context_chars": len(context),
            })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, group in sorted(groups.items()):
        n = len(group)
        out.append({
            "variant": variant,
            "n": n,
            "avg_em": sum(float(row["em"]) for row in group) / n if n else 0.0,
            "avg_answer_set_f1": sum(float(row["answer_set_f1"]) for row in group) / n if n else 0.0,
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no/wrong/masked/correct context controls from saved dumps.")
    parser.add_argument("--artifacts-root", default="artifacts_full")
    parser.add_argument("--dump-model-filter", default="HRG-Proposed-NoExpansion")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-model-sharding", action="store_true")
    parser.add_argument("--strict-gpu-sharding", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = asyncio.run(run_controls(args))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(args.artifacts_root).resolve() / "_summary" / "context_controls"
    summary = summarize(rows)
    write_csv(out_dir / "details.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[context-controls] wrote {out_dir} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
