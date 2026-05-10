#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "Datasets"

FIXED_KB_PATHS: Dict[str, Path] = {
    "metaqa": DATASETS_DIR / "MetaQA" / "kb.txt",
    "wikimovies": DATASETS_DIR / "WikiMovies" / "movieqa" / "knowledge_source" / "wiki_entities" / "wiki_entities_kb.txt",
    "mlpq": DATASETS_DIR / "MLPQ" / "datasets" / "KGs" / "fusion_bilingual_KGs" / "ILLs_fusion" / "merged_ILLs_KG_en_zh.txt",
    "kqapro": DATASETS_DIR / "KQAPro" / "kqapro_kb_triples.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve a likely KB path for a dataset.")
    parser.add_argument("--dataset", required=True, choices=["metaqa", "wikimovies", "mlpq", "wqsp", "cwq", "kqapro", "mintaka"])
    parser.add_argument("--root-dir", default=str(ROOT_DIR))
    parser.add_argument("--dataset-root", default=None)
    return parser.parse_args()


def score_path(path: Path, patterns: List[str]) -> int:
    text = str(path).lower()
    score = 0
    for idx, pattern in enumerate(patterns):
        if re.search(pattern, text):
            score += 100 - idx
    return score


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_best(root: Path, patterns: List[str], suffixes: tuple[str, ...] = (".txt", ".tsv", ".nt", ".json", ".jsonl", ".csv")) -> Optional[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if score_path(path, patterns) > 0:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (score_path(p, patterns), -len(str(p))), reverse=True)
    return candidates[0]


def resolve_known(dataset: str, base: Path, dataset_root: Optional[Path]) -> Optional[Path]:
    if dataset in FIXED_KB_PATHS:
        fixed = FIXED_KB_PATHS[dataset]
        return fixed if fixed.exists() else None
    return None


def resolve_scanned(dataset: str, base: Path, dataset_root: Optional[Path]) -> Optional[Path]:
    search_roots = [base / "KBs", base / "Datasets", base]
    if dataset_root:
        search_roots.insert(0, dataset_root)

    pattern_map: Dict[str, List[str]] = {
        "wqsp": [r"freebase", r"webqsp", r"\bfb\b", r"\bkb\b"],
        "cwq": [r"freebase", r"cwq", r"complex", r"\bfb\b", r"\bkb\b"],
        "mintaka": [r"wikidata", r"mintaka", r"triples", r"\bkb\b", r"freebase", r"dbpedia"],
        "kqapro": [r"kqapro", r"\bkb\b", r"wikidata", r"json"],
    }
    patterns = pattern_map.get(dataset, [r"\bkb\b"])
    for root in search_roots:
        if not root.exists():
            continue
        match = find_best(root, patterns)
        if match:
            return match
    return None


def main() -> int:
    args = parse_args()
    base = Path(args.root_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None

    kb_path = resolve_known(args.dataset, base, dataset_root) or resolve_scanned(args.dataset, base, dataset_root)
    payload = {
        "dataset": args.dataset,
        "kb_path": str(kb_path) if kb_path else "",
        "found": bool(kb_path),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if kb_path else 1


if __name__ == "__main__":
    raise SystemExit(main())
