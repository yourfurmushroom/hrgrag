#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from recompute_from_dumps import summarize_model  # noqa: E402


NUMERIC_KEYS = [
    "avg_em",
    "avg_hits_at_1",
    "avg_answer_set_f1",
    "avg_retrieval_recall",
    "avg_retrieval_precision",
    "avg_retrieval_f1",
    "avg_ctx_tokens",
    "avg_parse1_tokens",
    "avg_correction_tokens",
    "avg_parse2_tokens",
    "avg_total_online_token_proxy",
    "avg_subgraph_size",
    "candidate_validity_rate",
    "candidate_grammar_hit_rate",
    "candidate_same_arity_hit_rate",
    "candidate_ordered_path_hit_rate",
    "candidate_weak_label_hit_rate",
    "candidate_weak_label_only_rate",
    "avg_matched_rule_count",
    "correction_salvage_rate",
    "coverage_from_dump",
    "avg_num_expanded_edges",
    "expanded_question_rate",
    "generation_failure_count",
]


ABLATION_RE = re.compile(
    r"^(?P<base>.+)-abl-(?P<mode>drop_nodes|drop_relations)-(?P<pct>\d+)pct-seed(?P<seed>\d+)$"
)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def flatten_counter(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        return ";".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return str(value)


def iter_run_dirs(artifacts_root: Path) -> Iterable[Path]:
    for child in sorted(artifacts_root.iterdir()):
        if child.is_dir() and (child / "results" / "benchmark_results.json").exists():
            yield child


def model_method(model_name: str) -> str:
    name = model_name.split("@", 1)[0]
    for tag in ("-gpt-oss", "-qwen3.5", "-qwen2.5", "-llama3.1", "-gemma4"):
        name = name.replace(tag, "")
    name = name.replace("-json", "").replace("-triple", "")
    return name


def model_backbone(model_name: str) -> str:
    for tag in ("gpt-oss", "qwen3.5", "qwen2.5", "llama3.1", "gemma4"):
        if tag in model_name:
            return tag
    return ""


def serialization(model_name: str) -> str:
    if "-triple@" in model_name or model_name.endswith("-triple"):
        return "triple"
    return "json"


def status_for(results: Dict[str, Any], model_name: str) -> str:
    value = results.get(model_name)
    if value == "FAILED":
        return "FAILED"
    if isinstance(value, dict):
        return "OK"
    return "UNKNOWN"


def rows_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    run_tag = run_dir.name
    results = safe_load_json(run_dir / "results" / "benchmark_results.json") or {}
    per_model_root = run_dir / "dumps" / "per_model"
    rows: List[Dict[str, Any]] = []

    dumped_models = {p.name: p for p in per_model_root.iterdir()} if per_model_root.exists() else {}
    model_names = sorted(set(results) | set(dumped_models))

    ablation = ABLATION_RE.match(run_tag)
    ablation_info = ablation.groupdict() if ablation else {}

    for model_name in model_names:
        result_entry = results.get(model_name)
        row: Dict[str, Any] = {
            "run_tag": run_tag,
            "dataset_run": ablation_info.get("base", run_tag),
            "ablation_mode": ablation_info.get("mode", "clean"),
            "ablation_pct": ablation_info.get("pct", ""),
            "ablation_seed": ablation_info.get("seed", ""),
            "model": model_name,
            "method": model_method(model_name),
            "backbone": model_backbone(model_name),
            "serialization": serialization(model_name),
            "status": status_for(results, model_name),
        }

        if isinstance(result_entry, dict):
            for key, value in result_entry.items():
                if key == "results":
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
            if "avg_total_online_token_proxy" not in row:
                row["avg_total_online_token_proxy"] = (
                    float(row.get("avg_parse1_tokens") or 0.0)
                    + float(row.get("avg_correction_tokens") or 0.0)
                    + float(row.get("avg_parse2_tokens") or 0.0)
                    + float(row.get("avg_ctx_tokens") or 0.0)
                )
            row["failure_counts"] = flatten_counter(result_entry.get("failure_counts", {}))

        model_dir = dumped_models.get(model_name)
        if model_dir:
            dump_summary = summarize_model(model_name, model_dir)["summary"]
            row.update(dump_summary)
            row["avg_total_online_token_proxy"] = (
                float(row.get("avg_parse1_tokens") or 0.0)
                + float(row.get("avg_correction_tokens") or 0.0)
                + float(row.get("avg_parse2_tokens") or 0.0)
                + float(row.get("avg_ctx_tokens") or 0.0)
            )
            dump_count = float(row.get("dump_count") or 0.0)
            expanded_total = 0.0
            expanded_questions = 0.0
            for q_path in sorted(model_dir.rglob("q_*.pkl")):
                try:
                    import pickle

                    with q_path.open("rb") as f:
                        obj = pickle.load(f)
                    expanded = len(obj.get("expanded_edges", []) or [])
                    expanded_total += expanded
                    expanded_questions += 1 if expanded > 0 else 0
                except Exception:
                    continue
            row["avg_num_expanded_edges"] = expanded_total / dump_count if dump_count else 0.0
            row["expanded_question_rate"] = expanded_questions / dump_count if dump_count else 0.0
            row["candidate_source_counts"] = flatten_counter(row.get("candidate_source_counts", {}))
            row["selected_candidate_source_counts"] = flatten_counter(row.get("selected_candidate_source_counts", {}))
            row["failure_counts_from_dump"] = flatten_counter(row.get("failure_counts_from_dump", {}))

        rows.append(row)

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_perturbations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("ablation_mode") == "clean" or row.get("status") != "OK":
            continue
        key = (
            row.get("dataset_run"),
            row.get("ablation_mode"),
            row.get("ablation_pct"),
            row.get("method"),
            row.get("backbone"),
            row.get("serialization"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        base = {
            "dataset_run": key[0],
            "ablation_mode": key[1],
            "ablation_pct": key[2],
            "method": key[3],
            "backbone": key[4],
            "serialization": key[5],
            "seed_count": len({row.get("ablation_seed") for row in group_rows}),
        }
        for metric in NUMERIC_KEYS:
            values = []
            for row in group_rows:
                try:
                    values.append(float(row.get(metric)))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            base[f"{metric}_mean"] = sum(values) / len(values)
            base[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(base)
    return out


def summarize_method_mean_std(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        key = (
            row.get("dataset_run"),
            row.get("run_tag"),
            row.get("ablation_mode"),
            row.get("ablation_pct"),
            row.get("ablation_seed"),
            row.get("method"),
            row.get("serialization"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        backbones = sorted({str(row.get("backbone", "")) for row in group_rows if row.get("backbone")})
        base = {
            "dataset_run": key[0],
            "run_tag": key[1],
            "ablation_mode": key[2],
            "ablation_pct": key[3],
            "ablation_seed": key[4],
            "method": key[5],
            "serialization": key[6],
            "backbone_count": len(backbones),
            "backbones": ";".join(backbones),
            "question_model_pair_count": sum(int(float(row.get("dump_count") or 0)) for row in group_rows),
        }
        for metric in NUMERIC_KEYS:
            values = []
            for row in group_rows:
                try:
                    values.append(float(row.get(metric)))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            base[f"{metric}_mean"] = sum(values) / len(values)
            base[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            base[f"{metric}_n_models"] = len(values)
        out.append(base)
    return out


def summarize_evaluation_units(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("dataset_run"),
            row.get("run_tag"),
            row.get("ablation_mode"),
            row.get("ablation_pct"),
            row.get("ablation_seed"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        ok_rows = [row for row in group_rows if row.get("status") == "OK"]
        backbones = sorted({str(row.get("backbone", "")) for row in group_rows if row.get("backbone")})
        methods = sorted({str(row.get("method", "")) for row in group_rows if row.get("method")})
        serializations = sorted({str(row.get("serialization", "")) for row in group_rows if row.get("serialization")})
        question_model_pairs = sum(int(float(row.get("dump_count") or 0)) for row in ok_rows)
        per_model_counts = [
            int(float(row.get("dump_count") or 0))
            for row in ok_rows
            if row.get("dump_count") not in {None, ""}
        ]
        out.append({
            "dataset_run": key[0],
            "run_tag": key[1],
            "ablation_mode": key[2],
            "ablation_pct": key[3],
            "ablation_seed": key[4],
            "row_count": len(group_rows),
            "ok_row_count": len(ok_rows),
            "failed_row_count": len(group_rows) - len(ok_rows),
            "backbone_count": len(backbones),
            "backbones": ";".join(backbones),
            "method_count": len(methods),
            "methods": ";".join(methods),
            "serializations": ";".join(serializations),
            "question_model_pair_count": question_model_pairs,
            "question_count_min_per_model": min(per_model_counts) if per_model_counts else 0,
            "question_count_max_per_model": max(per_model_counts) if per_model_counts else 0,
        })
    return out


def summarize_grammar_breakdown(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "OK":
            continue
        if not row.get("candidate_grammar_hit_rate") and not row.get("candidate_same_arity_hit_rate"):
            continue
        out.append({
            "run_tag": row.get("run_tag"),
            "dataset_run": row.get("dataset_run"),
            "ablation_mode": row.get("ablation_mode"),
            "method": row.get("method"),
            "backbone": row.get("backbone"),
            "serialization": row.get("serialization"),
            "dump_count": row.get("dump_count"),
            "label_subset_hit_rate": row.get("candidate_grammar_hit_rate"),
            "same_arity_hit_rate": row.get("candidate_same_arity_hit_rate"),
            "ordered_path_hit_rate": row.get("candidate_ordered_path_hit_rate"),
            "weak_label_hit_rate": row.get("candidate_weak_label_hit_rate"),
            "weak_label_only_rate": row.get("candidate_weak_label_only_rate"),
            "avg_matched_rule_count": row.get("avg_matched_rule_count"),
            "selected_candidate_source_counts": row.get("selected_candidate_source_counts"),
        })
    return out


def summarize_failures(rows: List[Dict[str, Any]], artifacts_root: Path) -> List[Dict[str, Any]]:
    failures = [
        {
            "run_tag": row.get("run_tag"),
            "model": row.get("model"),
            "method": row.get("method"),
            "backbone": row.get("backbone"),
            "status": row.get("status"),
        }
        for row in rows
        if row.get("status") != "OK"
    ]
    for failure_file in sorted(artifacts_root.glob("**/results/failure_report.jsonl")):
        for line in failure_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            failures.append({
                "run_tag": item.get("run_tag"),
                "model": item.get("run_model_name"),
                "method": model_method(str(item.get("run_model_name", ""))),
                "backbone": model_backbone(str(item.get("run_model_name", ""))),
                "status": "FAILED",
                "error_type": item.get("error_type"),
                "error": item.get("error"),
                "model_id": item.get("model_id"),
            })
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark result JSON and per-question dumps.")
    parser.add_argument("--artifacts-root", default=str(ROOT_DIR / "artifacts"))
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else artifacts_root / "_summary"
    rows: List[Dict[str, Any]] = []
    for run_dir in iter_run_dirs(artifacts_root):
        rows.extend(rows_for_run(run_dir))

    perturbation_rows = summarize_perturbations(rows)
    method_mean_rows = summarize_method_mean_std(rows)
    evaluation_unit_rows = summarize_evaluation_units(rows)
    grammar_breakdown_rows = summarize_grammar_breakdown(rows)
    failure_rows = summarize_failures(rows, artifacts_root)

    write_csv(out_dir / "run_model_summary.csv", rows)
    write_csv(out_dir / "perturbation_mean_std.csv", perturbation_rows)
    write_csv(out_dir / "method_mean_std.csv", method_mean_rows)
    write_csv(out_dir / "evaluation_units.csv", evaluation_unit_rows)
    write_csv(out_dir / "grammar_hit_breakdown.csv", grammar_breakdown_rows)
    write_csv(out_dir / "failure_summary.csv", failure_rows)
    write_json(out_dir / "run_model_summary.json", rows)
    write_json(out_dir / "perturbation_mean_std.json", perturbation_rows)
    write_json(out_dir / "method_mean_std.json", method_mean_rows)
    write_json(out_dir / "evaluation_units.json", evaluation_unit_rows)
    write_json(out_dir / "grammar_hit_breakdown.json", grammar_breakdown_rows)
    write_json(out_dir / "failure_summary.json", failure_rows)

    print(f"[summary] wrote {out_dir}")
    print(
        f"[summary] model rows={len(rows)} method rows={len(method_mean_rows)} "
        f"perturbation rows={len(perturbation_rows)} failures={len(failure_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
