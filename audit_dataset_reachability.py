#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "LLM_inference_benchmark"))

from dataset_utils import (  # noqa: E402
    WIKIMOVIES_COMPOSITE_TAIL_RELATIONS,
    build_node_index,
    iter_kb_triples,
    load_kb_adjacency,
    load_metaqa_dataset,
    load_mlpq_dataset,
    load_wikimovies_dataset,
    normalize_lookup_key,
    resolve_mlpq_kb_path,
)
from download_datasets import (  # noqa: E402
    classify_kqapro_program,
    estimate_kqapro_graph_hop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether benchmark answers remain reachable after dataset-to-triple conversion."
    )
    parser.add_argument("--dataset", choices=["metaqa", "wikimovies", "mlpq", "kqapro"], required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--kb-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-frontier", type=int, default=30000)
    parser.add_argument("--metaqa-variant", default="vanilla")
    parser.add_argument("--wikimovies-subset", default="wiki_entities")
    parser.add_argument("--mlpq-pair", default="en-zh")
    parser.add_argument("--mlpq-question-lang", default="en")
    parser.add_argument("--mlpq-fusion", default="ills")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def lookup_variants(value: str) -> List[str]:
    raw = str(value or "").strip()
    normalized = normalize_lookup_key(raw)
    variants = [
        raw,
        raw.lower(),
        raw.lower().replace(" ", "_"),
        raw.lower().replace("_", " "),
        re.sub(r"[\s_]+", "", raw.lower()),
        normalized,
        normalized.replace(" ", "_"),
        normalized.replace(" ", ""),
    ]
    return [variant for variant in variants if variant]


def resolve_index_value(index: Dict[str, str], value: str) -> Optional[str]:
    for variant in lookup_variants(value):
        if variant in index:
            return index[variant]
    return None


def resolve_answer_targets(index: Dict[str, str], answers: Sequence[str]) -> Set[str]:
    return {
        node
        for answer in answers
        for node in [resolve_index_value(index, answer)]
        if node
    }


def answer_matches_value(answer: str, value: str) -> bool:
    answer_key = normalize_lookup_key(answer)
    value_key = normalize_lookup_key(value)
    return bool(answer_key and value_key and (answer_key in value_key or value_key in answer_key))


def record_limit(rows: Iterable[Any], sample_limit: Optional[int]) -> Iterable[Any]:
    for idx, row in enumerate(rows):
        if sample_limit is not None and idx >= sample_limit:
            break
        yield row


def update_counter(counter: Counter, **flags: bool) -> None:
    counter["records"] += 1
    for name, enabled in flags.items():
        if enabled:
            counter[name] += 1


def summarize_counter(counter: Counter) -> Dict[str, Any]:
    records = int(counter.get("records", 0))
    summary: Dict[str, Any] = {key: int(value) for key, value in sorted(counter.items())}
    summary["rates"] = {
        key: round(value / records, 4)
        for key, value in sorted(counter.items())
        if key != "records" and records
    }
    return summary


def summarize_buckets(buckets: DefaultDict[str, Counter]) -> Dict[str, Dict[str, Any]]:
    return {name: summarize_counter(counter) for name, counter in sorted(buckets.items())}


def graph_for_kb(kb_path: Path):
    kb_out, kb_in, all_nodes, relations, alias_map = load_kb_adjacency(str(kb_path))
    return kb_out, kb_in, all_nodes, relations, alias_map, build_node_index(all_nodes, alias_map)


def bfs_answer_depth(
    kb_out,
    kb_in,
    seeds: Set[str],
    targets: Set[str],
    max_depth: int,
    max_frontier: int,
) -> Optional[int]:
    if not seeds or not targets:
        return None
    if seeds & targets:
        return 0

    frontier = set(seeds)
    visited = set(seeds)
    for depth in range(1, max_depth + 1):
        next_frontier: Set[str] = set()
        for node in sorted(frontier)[:max_frontier]:
            for tails in kb_out.get(node, {}).values():
                next_frontier.update(tails)
            for heads in kb_in.get(node, {}).values():
                next_frontier.update(heads)
        if next_frontier & targets:
            return depth
        next_frontier -= visited
        if not next_frontier:
            return None
        if len(next_frontier) > max_frontier:
            next_frontier = set(sorted(next_frontier)[:max_frontier])
        visited.update(next_frontier)
        frontier = next_frontier
    return None


def default_wikimovies_kb(root: Path) -> Path:
    kb_dir = root / "movieqa" / "knowledge_source" / "wiki_entities"
    normalized = kb_dir / "wiki_entities_kb_normalized.txt"
    raw = kb_dir / "wiki_entities_kb.txt"
    return normalized if normalized.exists() else raw


def bracket_entity(question: str) -> Optional[str]:
    match = re.search(r"\[(.*?)\]", question or "")
    return match.group(1).strip() if match else None


def audit_metaqa(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.dataset_root or ROOT_DIR / "Datasets" / "MetaQA")
    kb_path = Path(args.kb_path or root / "kb.txt")
    kb_out, kb_in, _, _, _, index = graph_for_kb(kb_path)
    split = args.split or "test"
    buckets: DefaultDict[str, Counter] = defaultdict(Counter)

    for hop in (1, 2, 3):
        qa_path = root / f"{hop}-hop" / args.metaqa_variant / f"qa_{split}.txt"
        rows = load_metaqa_dataset(str(qa_path))
        for question, answers in record_limit(rows, args.sample_limit):
            targets = resolve_answer_targets(index, answers)
            entity = bracket_entity(question)
            seed = resolve_index_value(index, entity or "")
            answer_depth = bfs_answer_depth(
                kb_out,
                kb_in,
                {seed} if seed else set(),
                targets,
                max_depth=args.max_depth,
                max_frontier=args.max_frontier,
            )
            update_counter(
                buckets[f"{hop}-hop"],
                answer_in_graph=bool(targets),
                topic_entity_resolved=bool(seed),
                answer_reachable=answer_depth is not None,
                answer_within_hop=answer_depth is not None and answer_depth <= hop,
            )

    return {
        "dataset": "metaqa",
        "dataset_root": str(root),
        "kb_path": str(kb_path),
        "split": split,
        "buckets": summarize_buckets(buckets),
    }


def composite_tail_keys(kb_path: Path) -> Set[str]:
    keys: Set[str] = set()
    for _, relation, tail in iter_kb_triples(str(kb_path)):
        if relation not in WIKIMOVIES_COMPOSITE_TAIL_RELATIONS or "," not in tail:
            continue
        for part in tail.split(","):
            normalized = normalize_lookup_key(part)
            if normalized:
                keys.add(normalized)
    return keys


def audit_wikimovies(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.dataset_root or ROOT_DIR / "Datasets" / "WikiMovies")
    kb_path = Path(args.kb_path or default_wikimovies_kb(root))
    _, _, _, _, _, index = graph_for_kb(kb_path)
    split = args.split or "test"
    question_path = (
        root
        / "movieqa"
        / "questions"
        / args.wikimovies_subset
        / f"{args.wikimovies_subset.replace('_', '-')}_qa_{split}.txt"
    )
    raw_kb = root / "movieqa" / "knowledge_source" / "wiki_entities" / "wiki_entities_kb.txt"
    composite_keys = composite_tail_keys(raw_kb) if raw_kb.exists() else set()
    bucket = Counter()

    for question, answers in record_limit(load_wikimovies_dataset(str(question_path)).get("1-hop", []), args.sample_limit):
        del question
        targets = resolve_answer_targets(index, answers)
        answer_in_composite_tail = any(normalize_lookup_key(answer) in composite_keys for answer in answers)
        update_counter(
            bucket,
            answer_in_graph=bool(targets),
            answer_in_composite_tail=answer_in_composite_tail,
            composite_only=answer_in_composite_tail and not bool(targets),
        )

    return {
        "dataset": "wikimovies",
        "dataset_root": str(root),
        "kb_path": str(kb_path),
        "question_path": str(question_path),
        "split": split,
        "buckets": {"1-hop": summarize_counter(bucket)},
    }


def mlpq_gold_path_valid(kb_out, path_parts: Sequence[str]) -> bool:
    if len(path_parts) < 3:
        return False
    for idx in range(0, len(path_parts) - 2, 2):
        head, relation, tail = path_parts[idx:idx + 3]
        if tail not in kb_out.get(head, {}).get(relation, set()):
            return False
    return True


def audit_mlpq(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.dataset_root or ROOT_DIR / "Datasets" / "MLPQ")
    kb_path = Path(args.kb_path or resolve_mlpq_kb_path(str(root), args.mlpq_pair, args.mlpq_fusion))
    kb_out, _, _, _, _, index = graph_for_kb(kb_path)
    buckets: DefaultDict[str, Counter] = defaultdict(Counter)

    grouped = load_mlpq_dataset(
        root=str(root),
        pair=args.mlpq_pair,
        question_lang=args.mlpq_question_lang,
        inject_topic_entity=True,
    )
    for bucket_name, rows in grouped.items():
        for _, answers, metadata in record_limit(rows, args.sample_limit):
            path_parts = metadata.get("gold_path_parts") or []
            path_valid = mlpq_gold_path_valid(kb_out, path_parts)
            final_node = path_parts[-1] if path_parts else ""
            final_matches_answer = any(answer_matches_value(answer, final_node) for answer in answers)
            update_counter(
                buckets[bucket_name],
                answer_in_graph=bool(resolve_answer_targets(index, answers)),
                gold_path_present=bool(path_parts),
                gold_path_executable=path_valid,
                gold_path_answer_match=path_valid and final_matches_answer,
            )

    return {
        "dataset": "mlpq",
        "dataset_root": str(root),
        "kb_path": str(kb_path),
        "pair": args.mlpq_pair,
        "question_lang": args.mlpq_question_lang,
        "buckets": summarize_buckets(buckets),
    }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def kqapro_find_inputs(program: Any) -> List[str]:
    if not isinstance(program, list):
        return []
    inputs: List[str] = []
    for step in program:
        if not isinstance(step, dict) or step.get("function") != "Find":
            continue
        values = step.get("inputs") or []
        if values:
            value = str(values[0]).strip()
            if value:
                inputs.append(value)
    return inputs


def row_answers(row: Dict[str, Any]) -> List[str]:
    answer = row.get("answer", row.get("answers", []))
    if isinstance(answer, list):
        return [str(value).strip() for value in answer if str(value).strip()]
    return [str(answer).strip()] if str(answer or "").strip() else []


def audit_kqapro(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.dataset_root or ROOT_DIR / "Datasets" / "KQAPro")
    kb_path = Path(args.kb_path or root / "kqapro_kb_triples.tsv")
    split = args.split or "validation"
    raw_path = root / "raw" / f"{split}.jsonl"
    kb_out, kb_in, _, _, _, index = graph_for_kb(kb_path)
    buckets: DefaultDict[str, Counter] = defaultdict(Counter)

    for row in record_limit(iter_jsonl(raw_path), args.sample_limit):
        program = row.get("program", [])
        answers = row_answers(row)
        targets = resolve_answer_targets(index, answers)
        seeds = {
            node
            for value in kqapro_find_inputs(program)
            for node in [resolve_index_value(index, value)]
            if node
        }
        graph_hop = estimate_kqapro_graph_hop(program)
        answer_depth = bfs_answer_depth(
            kb_out,
            kb_in,
            seeds,
            targets,
            max_depth=args.max_depth,
            max_frontier=args.max_frontier,
        )
        update_counter(
            buckets[classify_kqapro_program(program)],
            answer_in_graph=bool(targets),
            find_seed_present=bool(kqapro_find_inputs(program)),
            find_seed_resolved=bool(seeds),
            answer_reachable=answer_depth is not None,
            answer_within_graph_hop=answer_depth is not None and answer_depth <= graph_hop,
        )
        buckets[classify_kqapro_program(program)][f"estimated_hop_{graph_hop}"] += 1

    return {
        "dataset": "kqapro",
        "dataset_root": str(root),
        "kb_path": str(kb_path),
        "raw_path": str(raw_path),
        "split": split,
        "buckets": summarize_buckets(buckets),
    }


def main() -> int:
    args = parse_args()
    audits = {
        "metaqa": audit_metaqa,
        "wikimovies": audit_wikimovies,
        "mlpq": audit_mlpq,
        "kqapro": audit_kqapro,
    }
    report = audits[args.dataset](args)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
