#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "LLM_inference_benchmark"))

from dataset_utils import apply_kb_ablation, iter_kb_triples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ablated KB file without modifying the source KB.")
    parser.add_argument("--input-kb", required=True, help="Source KB/triples file.")
    parser.add_argument("--output-kb", required=True, help="Output ablated KB path.")
    parser.add_argument("--mode", choices=["drop_nodes", "drop_relations"], required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-triples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ratio <= 1.0:
        raise ValueError(f"--ratio must be within [0, 1], got {args.ratio}")

    triples = list(iter_kb_triples(args.input_kb, max_triples=args.max_triples))
    ablated = apply_kb_ablation(
        triples,
        mode=args.mode,
        ratio=args.ratio,
        seed=args.seed,
    )

    out_path = Path(args.output_kb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h, r, t in ablated:
            f.write(f"{h}\t{r}\t{t}\n")

    original_nodes = {h for h, _, _ in triples} | {t for _, _, t in triples}
    ablated_nodes = {h for h, _, _ in ablated} | {t for _, _, t in ablated}
    original_relations = {r for _, r, _ in triples}
    ablated_relations = {r for _, r, _ in ablated}

    print(f"input_kb={args.input_kb}")
    print(f"output_kb={out_path}")
    print(f"mode={args.mode} ratio={args.ratio} seed={args.seed}")
    print(f"triples: {len(triples)} -> {len(ablated)}")
    print(f"nodes: {len(original_nodes)} -> {len(ablated_nodes)}")
    print(f"relations: {len(original_relations)} -> {len(ablated_relations)}")


if __name__ == "__main__":
    main()
