#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from recompute_from_dumps import calculate_answer_metrics, normalize_answer


def edge_parts(edge: Any) -> Tuple[str, str, str]:
    if isinstance(edge, dict):
        return str(edge.get("head", "")), str(edge.get("relation", "")), str(edge.get("tail", ""))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    return "", "", ""


def answer_coverage(edges: Iterable[Any], references: List[str]) -> float:
    ref_norms = [normalize_answer(ref).replace("_", " ") for ref in references if normalize_answer(ref)]
    if not ref_norms:
        return 0.0
    nodes = set()
    for edge in edges or []:
        h, _, t = edge_parts(edge)
        for node in (h, t):
            norm = normalize_answer(node).replace("_", " ")
            if norm:
                nodes.add(norm)
    if not nodes:
        return 0.0
    hits = 0
    for ref in ref_norms:
        if any(ref in node or node in ref for node in nodes):
            hits += 1
    return hits / len(ref_norms)


def est_context_tokens(edges: Iterable[Any]) -> int:
    lines = []
    for i, edge in enumerate(edges or [], 1):
        h, r, t = edge_parts(edge)
        lines.append(f"{i}. {h} --[{r}]--> {t}")
    text = "\n".join(lines)
    return max(1, len(text) // 4) if text else 0


def iter_baseline_dumps(artifacts_root: Path) -> Iterable[Tuple[str, Path]]:
    for model_dir in sorted(artifacts_root.glob("**/dumps/per_model/Baseline-BFS-*")):
        if not model_dir.is_dir():
            continue
        if "metaqa" not in model_dir.name.lower() and "metaqa" not in str(model_dir.parent.parent.parent).lower():
            continue
        for dump_path in sorted(model_dir.rglob("q_*.pkl")):
            yield model_dir.name, dump_path


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def analyze(artifacts_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail_rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for model_name, dump_path in iter_baseline_dumps(artifacts_root):
        obj = load_pickle(dump_path)
        hop = dump_path.parent.name
        references = list(obj.get("references", []) or [])
        raw_edges = list(obj.get("raw_bfs_edges", []) or obj.get("edges", []) or [])
        budgeted_edges = list(obj.get("edges", []) or [])
        answer = str(obj.get("answer", "") or "")
        answer_metrics = calculate_answer_metrics(references, answer) if references else {}
        raw_cov = answer_coverage(raw_edges, references)
        budgeted_cov = answer_coverage(budgeted_edges, references)
        row = {
            "model": model_name,
            "hop": hop,
            "dump_file": str(dump_path),
            "question": obj.get("question", ""),
            "references": "|".join(references),
            "answer": answer,
            "raw_edge_count": len(raw_edges),
            "budgeted_edge_count": len(budgeted_edges),
            "truncated": int(len(raw_edges) > len(budgeted_edges)),
            "raw_answer_coverage": raw_cov,
            "budgeted_answer_coverage": budgeted_cov,
            "raw_context_tokens_est": est_context_tokens(raw_edges),
            "budgeted_context_tokens": (obj.get("token_usage", {}) or {}).get("context_tokens", est_context_tokens(budgeted_edges)),
            "em": answer_metrics.get("em", 0.0),
            "answer_set_f1": answer_metrics.get("answer_set_f1", 0.0),
            "conditional_error_when_answer_in_context": int(budgeted_cov > 0.0 and answer_metrics.get("answer_set_f1", 0.0) <= 0.0),
        }
        detail_rows.append(row)
        groups.setdefault((model_name, hop), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (model_name, hop), rows in sorted(groups.items()):
        n = len(rows)
        if not n:
            continue
        answer_in_context = [row for row in rows if float(row["budgeted_answer_coverage"]) > 0.0]
        summary_rows.append({
            "model": model_name,
            "hop": hop,
            "n": n,
            "raw_answer_coverage_rate": sum(float(row["raw_answer_coverage"]) for row in rows) / n,
            "budgeted_answer_coverage_rate": sum(float(row["budgeted_answer_coverage"]) for row in rows) / n,
            "truncation_rate": sum(int(row["truncated"]) for row in rows) / n,
            "avg_raw_edge_count": sum(int(row["raw_edge_count"]) for row in rows) / n,
            "avg_budgeted_edge_count": sum(int(row["budgeted_edge_count"]) for row in rows) / n,
            "avg_raw_context_tokens_est": sum(int(row["raw_context_tokens_est"]) for row in rows) / n,
            "avg_budgeted_context_tokens": sum(float(row["budgeted_context_tokens"] or 0.0) for row in rows) / n,
            "avg_em": sum(float(row["em"]) for row in rows) / n,
            "avg_answer_set_f1": sum(float(row["answer_set_f1"]) for row in rows) / n,
            "conditional_accuracy_when_answer_in_context": (
                sum(1 for row in answer_in_context if float(row["answer_set_f1"]) > 0.0) / len(answer_in_context)
                if answer_in_context else 0.0
            ),
            "conditional_error_count_when_answer_in_context": sum(int(row["conditional_error_when_answer_in_context"]) for row in rows),
        })

    return summary_rows, detail_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MetaQA BFS coverage/truncation by hop from saved baseline dumps.")
    parser.add_argument("--artifacts-root", default="artifacts_full")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else artifacts_root / "_summary" / "metaqa_bfs_sanity"
    summary_rows, detail_rows = analyze(artifacts_root)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "details.csv", detail_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(detail_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[metaqa-bfs-sanity] wrote {out_dir} summary_rows={len(summary_rows)} detail_rows={len(detail_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
