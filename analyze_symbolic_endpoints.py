#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from recompute_from_dumps import calculate_answer_metrics


def edge_parts(edge: Any) -> Tuple[str, str, str]:
    if isinstance(edge, dict):
        return str(edge.get("head", "")), str(edge.get("relation", "")), str(edge.get("tail", ""))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    return "", "", ""


def symbolic_endpoints(obj: Dict[str, Any]) -> List[str]:
    chain = [str(rel) for rel in (obj.get("selected_chain") or []) if str(rel)]
    edges = list(obj.get("spine_edges") or obj.get("edges") or [])
    if not edges:
        return []

    endpoints: List[str] = []
    if chain:
        last_rel = chain[-1]
        for edge in edges:
            h, rel, t = edge_parts(edge)
            if rel == last_rel:
                endpoints.extend([h, t])
    else:
        for edge in edges:
            h, _, t = edge_parts(edge)
            endpoints.extend([h, t])

    selected_entity = str(obj.get("selected_entity") or obj.get("entity") or "")
    selected_entity_norm = selected_entity.lower().replace("_", " ").strip()
    out: List[str] = []
    seen = set()
    for endpoint in endpoints:
        clean = endpoint.replace("_", " ").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        if selected_entity_norm and key == selected_entity_norm:
            continue
        seen.add(key)
        out.append(clean)
    return out


def iter_dumps(artifacts_root: Path) -> Iterable[Tuple[str, Path]]:
    for model_dir in sorted(artifacts_root.glob("**/dumps/per_model/*")):
        if not model_dir.is_dir():
            continue
        for dump_path in sorted(model_dir.rglob("q_*.pkl")):
            yield model_dir.name, dump_path


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def analyze(artifacts_root: Path, model_filter: str = "", limit: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for model_name, dump_path in iter_dumps(artifacts_root):
        if model_filter and model_filter not in model_name:
            continue
        obj = load_pickle(dump_path)
        references = list(obj.get("references", []) or [])
        symbolic_answer = " | ".join(symbolic_endpoints(obj))
        llm_answer = str(obj.get("answer", "") or "")
        symbolic_metrics = calculate_answer_metrics(references, symbolic_answer) if references else {}
        llm_metrics = calculate_answer_metrics(references, llm_answer) if references else {}
        hop = dump_path.parent.name
        row = {
            "model": model_name,
            "hop": hop,
            "dump_file": str(dump_path),
            "question": obj.get("question", ""),
            "references": "|".join(references),
            "llm_answer": llm_answer,
            "symbolic_answer": symbolic_answer,
            "symbolic_endpoint_count": len(symbolic_answer.split(" | ")) if symbolic_answer else 0,
            "llm_em": llm_metrics.get("em", 0.0),
            "llm_answer_set_f1": llm_metrics.get("answer_set_f1", 0.0),
            "symbolic_em": symbolic_metrics.get("em", 0.0),
            "symbolic_answer_set_f1": symbolic_metrics.get("answer_set_f1", 0.0),
            "symbolic_minus_llm_f1": symbolic_metrics.get("answer_set_f1", 0.0) - llm_metrics.get("answer_set_f1", 0.0),
            "failure_stage": obj.get("failure_stage", ""),
            "retrieval_policy": obj.get("retrieval_policy", ""),
        }
        rows.append(row)
        groups.setdefault((model_name, hop), []).append(row)
        if limit and len(rows) >= limit:
            break

    summary: List[Dict[str, Any]] = []
    for (model_name, hop), group_rows in sorted(groups.items()):
        n = len(group_rows)
        summary.append({
            "model": model_name,
            "hop": hop,
            "n": n,
            "avg_llm_em": sum(float(row["llm_em"]) for row in group_rows) / n,
            "avg_llm_answer_set_f1": sum(float(row["llm_answer_set_f1"]) for row in group_rows) / n,
            "avg_symbolic_em": sum(float(row["symbolic_em"]) for row in group_rows) / n,
            "avg_symbolic_answer_set_f1": sum(float(row["symbolic_answer_set_f1"]) for row in group_rows) / n,
            "avg_symbolic_minus_llm_f1": sum(float(row["symbolic_minus_llm_f1"]) for row in group_rows) / n,
            "avg_symbolic_endpoint_count": sum(int(row["symbolic_endpoint_count"]) for row in group_rows) / n,
        })
    return summary, rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate symbolic endpoint answers from saved KG-RAG dumps.")
    parser.add_argument("--artifacts-root", default="artifacts_full")
    parser.add_argument("--model-filter", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else artifacts_root / "_summary" / "symbolic_endpoints"
    summary, rows = analyze(artifacts_root, args.model_filter, args.limit)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "details.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[symbolic-endpoints] wrote {out_dir} summary_rows={len(summary)} detail_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
