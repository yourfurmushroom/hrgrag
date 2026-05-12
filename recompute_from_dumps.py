#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(",", "")
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in _NUMBER_WORDS) + r")\b",
        lambda m: _NUMBER_WORDS[m.group(1)],
        text,
    )
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"[^\w\s|]", "", text)
    return text.strip()


def split_candidate_answers(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(r"\s*\|\s*|\s*;\s*|\n+", raw)
    out: List[str] = []
    seen = set()
    for part in parts:
        norm = normalize_answer(part)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def calculate_answer_metrics(references: List[str], candidate: str) -> Dict[str, float]:
    candidate_lower = (candidate or "").lower()
    candidate_norm = normalize_answer(candidate)
    ref_norms = [normalize_answer(ref) for ref in references if normalize_answer(ref)]
    candidate_parts = split_candidate_answers(candidate)
    candidate_set = set(candidate_parts)
    ref_set = set(ref_norms)

    contains_hit = 0.0
    for ref in references:
        if ref.lower() in candidate_lower:
            contains_hit = 1.0
            break

    hit_at_1_any = 0.0
    if ref_set:
        if candidate_set:
            hit_at_1_any = 1.0 if bool(candidate_set & ref_set) else 0.0
        elif candidate_norm:
            hit_at_1_any = 1.0 if candidate_norm in ref_set else 0.0

    overlap = len(candidate_set & ref_set)
    if len(ref_norms) <= 1:
        answer_recall = 1.0 if ref_norms and candidate_norm and candidate_norm == ref_norms[0] else 0.0
    else:
        answer_recall = (overlap / len(ref_set)) if ref_set else 0.0

    em = 1.0 if ref_set and candidate_set == ref_set else 0.0
    if len(ref_norms) <= 1:
        em = answer_recall

    precision = (overlap / len(candidate_set)) if candidate_set else 0.0
    recall = (overlap / len(ref_set)) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "answer_recall": float(answer_recall),
        "em": float(em),
        "contains_hit": float(contains_hit),
        "hit_at_1_any": float(hit_at_1_any),
        "answer_set_precision": float(precision),
        "answer_set_recall": float(recall),
        "answer_set_f1": float(f1),
    }


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return (sum(values) / len(values)) if values else 0.0


def load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_model_dump_files(per_model_root: Path) -> Iterable[Tuple[str, Path]]:
    for model_dir in sorted(x for x in per_model_root.iterdir() if x.is_dir()):
        dump_files = sorted(model_dir.rglob("q_*.pkl"))
        if dump_files:
            yield model_dir.name, model_dir


def summarize_model(model_name: str, model_dir: Path) -> Dict[str, Any]:
    files = sorted(model_dir.rglob("q_*.pkl"))
    rows: List[Dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    total_candidates = 0
    valid_candidates = 0
    grammar_hit_candidates = 0
    same_arity_candidates = 0
    matched_rule_count_total = 0
    grammar_hit_questions = 0
    correction_salvaged = 0

    for dump_path in files:
        obj = load_pickle(dump_path)
        token_usage = obj.get("token_usage", {})
        timing = obj.get("timing", {})
        references = obj.get("references", [])
        answer_metrics = calculate_answer_metrics(references, obj.get("answer", "")) if references else {}
        candidates = obj.get("candidates", []) or []
        failure_stage = obj.get("failure_stage", "unknown") or "unknown"
        failure_counts[failure_stage] += 1
        matched_rules = obj.get("matched_grammar_rules", []) or []
        matched_rule_count_total += len(matched_rules)
        if obj.get("grammar_hit") or matched_rules:
            grammar_hit_questions += 1

        selected_candidate = obj.get("selected_candidate", {}) or {}
        if selected_candidate.get("source"):
            selected_source_counts[str(selected_candidate["source"])] += 1

        had_initial_valid = False
        had_any_valid = False
        had_correction_valid = False
        for idx, cand in enumerate(candidates):
            total_candidates += 1
            source = str(cand.get("source", "unknown"))
            source_counts[source] += 1
            kb_result = cand.get("kb_result", {}) or {}
            is_valid = bool(kb_result.get("valid"))
            if idx == 0 and is_valid:
                had_initial_valid = True
            if is_valid:
                valid_candidates += 1
                had_any_valid = True
                if source != "llm":
                    had_correction_valid = True
            if cand.get("grammar_hit"):
                grammar_hit_candidates += 1
            if cand.get("grammar_same_arity_hit"):
                same_arity_candidates += 1
        if (not had_initial_valid) and had_any_valid and had_correction_valid:
            correction_salvaged += 1

        rows.append({
            "dump_file": str(dump_path.relative_to(PROJECT_ROOT)),
            "question": obj.get("question", ""),
            "answer": obj.get("answer", ""),
            "failure_stage": failure_stage,
            "subgraph_size": obj.get("subgraph_size", 0),
            "retrieval_recall": obj.get("retrieval_recall", 0.0),
            "retrieval_precision": obj.get("retrieval_precision", 0.0),
            "retrieval_f1": obj.get("retrieval_f1", 0.0),
            "parse1_tokens": token_usage.get("parse1_tokens", 0),
            "correction_tokens": token_usage.get("correction_tokens", 0),
            "parse2_tokens": token_usage.get("parse2_tokens", 0),
            "context_tokens": token_usage.get("context_tokens", 0),
            "parse_latency": timing.get("parse_latency", 0.0),
            "retrieval_latency": timing.get("retrieval_latency", 0.0),
            "generation_latency": timing.get("generation_latency", 0.0),
            "num_edges": len(obj.get("edges", []) or []),
            "num_spine_edges": len(obj.get("spine_edges", []) or []),
            "num_expanded_edges": len(obj.get("expanded_edges", []) or []),
            "num_candidates": len(candidates),
            "num_matched_rules": len(matched_rules),
            "grammar_hit": bool(obj.get("grammar_hit") or matched_rules),
            "selected_candidate_source": selected_candidate.get("source", ""),
            **answer_metrics,
        })

    summary: Dict[str, Any] = {
        "model": model_name,
        "dump_count": len(rows),
        "avg_subgraph_size": safe_mean(r["subgraph_size"] for r in rows),
        "avg_retrieval_recall": safe_mean(r["retrieval_recall"] for r in rows),
        "avg_retrieval_precision": safe_mean(r["retrieval_precision"] for r in rows),
        "avg_retrieval_f1": safe_mean(r["retrieval_f1"] for r in rows),
        "avg_ctx_tokens": safe_mean(r["context_tokens"] for r in rows),
        "avg_parse1_tokens": safe_mean(r["parse1_tokens"] for r in rows),
        "avg_correction_tokens": safe_mean(r["correction_tokens"] for r in rows),
        "avg_parse2_tokens": safe_mean(r["parse2_tokens"] for r in rows),
        "avg_parse_latency": safe_mean(r["parse_latency"] for r in rows),
        "avg_retrieval_latency": safe_mean(r["retrieval_latency"] for r in rows),
        "avg_generation_latency": safe_mean(r["generation_latency"] for r in rows),
        "avg_num_candidates": safe_mean(r["num_candidates"] for r in rows),
        "avg_num_matched_rules": safe_mean(r["num_matched_rules"] for r in rows),
        "coverage_from_dump": (grammar_hit_questions / len(rows)) if rows else 0.0,
        "candidate_validity_rate": (valid_candidates / total_candidates) if total_candidates else 0.0,
        "candidate_grammar_hit_rate": (grammar_hit_candidates / total_candidates) if total_candidates else 0.0,
        "candidate_same_arity_hit_rate": (same_arity_candidates / total_candidates) if total_candidates else 0.0,
        "correction_salvage_rate": (correction_salvaged / len(rows)) if rows else 0.0,
        "failure_counts_from_dump": dict(failure_counts),
        "candidate_source_counts": dict(source_counts),
        "selected_candidate_source_counts": dict(selected_source_counts),
    }

    answer_rows = [r for r in rows if r.get("em") is not None and "em" in r]
    if any("em" in r for r in rows):
        summary.update({
            "avg_answer_recall": safe_mean(r.get("answer_recall", 0.0) for r in rows),
            "avg_em": safe_mean(r.get("em", 0.0) for r in rows),
            "avg_contains_hit": safe_mean(r.get("contains_hit", 0.0) for r in rows),
            "avg_hit_at_1_any": safe_mean(r.get("hit_at_1_any", 0.0) for r in rows),
            "avg_answer_set_precision": safe_mean(r.get("answer_set_precision", 0.0) for r in rows),
            "avg_answer_set_recall": safe_mean(r.get("answer_set_recall", 0.0) for r in rows),
            "avg_answer_set_f1": safe_mean(r.get("answer_set_f1", 0.0) for r in rows),
        })

    return {"summary": summary, "rows": rows}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute dump-based summaries from saved q_*.pkl files.")
    parser.add_argument("--run-tag", required=True, help="Artifact run tag, e.g. kqapro-validation")
    parser.add_argument("--artifacts-root", default=str(ARTIFACTS_ROOT))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.artifacts_root) / args.run_tag
    per_model_root = run_root / "dumps" / "per_model"
    if not per_model_root.exists():
        raise SystemExit(f"per_model dump root not found: {per_model_root}")

    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    for model_name, model_dir in iter_model_dump_files(per_model_root):
        result = summarize_model(model_name, model_dir)
        summaries[model_name] = result["summary"]
        for row in result["rows"]:
            all_rows.append({"model": model_name, **row})

    output_json = Path(args.output_json) if args.output_json else (run_root / "results" / "dump_recomputed_summary.json")
    output_csv = Path(args.output_csv) if args.output_csv else (run_root / "results" / "dump_recomputed_rows.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"run_tag": args.run_tag, "models": summaries}, f, ensure_ascii=False, indent=2)
    write_csv(output_csv, all_rows)

    print(f"[Info] Wrote dump summary JSON: {output_json}")
    print(f"[Info] Wrote dump rows CSV: {output_csv}")
    print(f"[Info] Models summarized: {len(summaries)}")


if __name__ == "__main__":
    main()
