#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from hrg_grammar.hrg_extract import (
    load_labeled_kb_graph,
    learn_phrg_from_k_bfs_samples,
    canonicalize_rhs_fast,
    Rule,
)


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HRG stability across different random seeds.")
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--out-file", default=str(ROOT_DIR / "artifacts" / "_analysis" / "grammar_stability.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--max-triples", type=int, default=None)
    return parser.parse_args()


def rule_signature(rule: Rule) -> str:
    payload = {
        "lhs_name": rule.lhs.name,
        "lhs_rank": rule.lhs.rank,
        "rhs": canonicalize_rhs_fast(rule.rhs, rule.lhs.rank),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def rule_labels(rule: Rule) -> List[str]:
    return [rel for _, rel, _ in rule.rhs.terminals]


def summarize_grammar(grammar: List[Rule], top_n: int) -> Dict[str, object]:
    sorted_rules = sorted(grammar, key=lambda r: (-r.count, r.lhs.rank, len(r.rhs.terminals), len(r.rhs.nonterms)))
    selected = sorted_rules[:top_n]
    signature_set = {rule_signature(r) for r in selected}
    label_set = set()
    for rule in selected:
        label_set.update(rule_labels(rule))

    return {
        "num_rules": len(grammar),
        "top_rule_count_sum": sum(r.count for r in selected),
        "top_signatures": sorted(signature_set),
        "top_labels": sorted(label_set),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def main() -> int:
    args = parse_args()
    out_file = Path(args.out_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    graph = load_labeled_kb_graph(args.kb_path, args.max_triples)

    per_seed = []
    summaries = {}
    for seed in args.seeds:
        grammar = learn_phrg_from_k_bfs_samples(graph, args.k_samples, args.sample_size, seed)
        summary = summarize_grammar(grammar, args.top_n)
        summaries[seed] = summary
        per_seed.append(
            {
                "seed": seed,
                "num_rules": summary["num_rules"],
                "top_rule_count_sum": summary["top_rule_count_sum"],
                "top_label_count": len(summary["top_labels"]),
            }
        )

    pairwise = []
    for s1, s2 in combinations(args.seeds, 2):
        sig1 = set(summaries[s1]["top_signatures"])
        sig2 = set(summaries[s2]["top_signatures"])
        lab1 = set(summaries[s1]["top_labels"])
        lab2 = set(summaries[s2]["top_labels"])
        pairwise.append(
            {
                "seed_a": s1,
                "seed_b": s2,
                "top_rule_signature_jaccard": jaccard(sig1, sig2),
                "top_relation_label_jaccard": jaccard(lab1, lab2),
            }
        )

    payload = {
        "kb_path": str(Path(args.kb_path).resolve()),
        "k_samples": args.k_samples,
        "sample_size": args.sample_size,
        "top_n": args.top_n,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "pairwise_overlap": pairwise,
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analysis] wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
