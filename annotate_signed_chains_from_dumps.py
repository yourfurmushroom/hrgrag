#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def norm(value: str) -> str:
    return re.sub(r"[\s_]+", " ", str(value or "").strip().lower())


def edge_parts(edge: Any) -> Tuple[str, str, str]:
    if isinstance(edge, dict):
        return str(edge.get("head", "")), str(edge.get("relation", "")), str(edge.get("tail", ""))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    return "", "", ""


def infer_signed_chain(entity: str, chain: List[str], edges: Iterable[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    frontier: Set[str] = {norm(entity)} if entity else set()
    edge_list = [edge_parts(edge) for edge in edges or []]
    signed: List[str] = []
    steps: List[Dict[str, Any]] = []

    for hop, rel in enumerate(chain, 1):
        plus_next: Set[str] = set()
        minus_next: Set[str] = set()
        plus_edges = []
        minus_edges = []
        for head, edge_rel, tail in edge_list:
            if edge_rel != rel:
                continue
            h_norm = norm(head)
            t_norm = norm(tail)
            if h_norm in frontier:
                plus_next.add(t_norm)
                plus_edges.append({"head": head, "relation": edge_rel, "tail": tail})
            if t_norm in frontier:
                minus_next.add(h_norm)
                minus_edges.append({"head": head, "relation": edge_rel, "tail": tail})

        if plus_next and minus_next:
            token = f"{rel}+/-"
        elif plus_next:
            token = f"{rel}+"
        elif minus_next:
            token = f"{rel}-"
        else:
            token = f"{rel}?"

        signed.append(token)
        steps.append({
            "hop": hop,
            "relation": rel,
            "signed_relation": token,
            "frontier_size_before": len(frontier),
            "plus_edge_count": len(plus_edges),
            "minus_edge_count": len(minus_edges),
            "plus_examples": plus_edges[:3],
            "minus_examples": minus_edges[:3],
        })
        next_frontier = plus_next | minus_next
        if next_frontier:
            frontier = next_frontier

    return signed, steps


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


def analyze(artifacts_root: Path, model_filter: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, dump_path in iter_dumps(artifacts_root, model_filter):
        obj = load_pickle(dump_path)
        chain = [str(rel) for rel in (obj.get("selected_chain") or []) if str(rel)]
        entity = str(obj.get("selected_entity") or obj.get("entity") or "")
        if not chain or not entity:
            continue
        signed, steps = infer_signed_chain(entity, chain, obj.get("spine_edges") or obj.get("edges") or [])
        rows.append({
            "model": model_name,
            "dump_file": str(dump_path),
            "hop": dump_path.parent.name,
            "question": obj.get("question", ""),
            "selected_entity": entity,
            "bare_chain": " -> ".join(chain),
            "signed_chain": " -> ".join(signed),
            "answer": obj.get("answer", ""),
            "references": "|".join(obj.get("references", []) or []),
            "failure_stage": obj.get("failure_stage", ""),
            "signed_steps_json": json.dumps(steps, ensure_ascii=False),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer signed r+/r- traversal chains from saved dumps.")
    parser.add_argument("--artifacts-root", default="artifacts_full")
    parser.add_argument("--model-filter", default="HRG-Proposed")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else artifacts_root / "_summary" / "signed_chains"
    rows = analyze(artifacts_root, args.model_filter, args.limit)
    write_csv(out_dir / "signed_chains.csv", rows)
    (out_dir / "signed_chains.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[signed-chains] wrote {out_dir} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
