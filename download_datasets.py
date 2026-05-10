#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "Datasets"

METAQA_GDRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g&usp=sharing"
)
MLPQ_GITHUB_ZIP = "https://github.com/tan92hl/Dataset-for-QA-over-Multilingual-KG/archive/refs/heads/master.zip"
WIKIMOVIES_TAR_URL = "https://thespermwhale.com/jaseweston/babi/movieqa.tar.gz"

DEFAULT_HF_DATASETS = {
    "wqsp": "ml1996/webqsp",
    "cwq": "drt/complex_web_questions",
    "kqapro": "soongfs/kqa_pro",
    "mintaka": "AmazonScience/mintaka",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download portable_runner datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["metaqa", "wqsp", "cwq", "kqapro", "mintaka", "wikimovies", "mlpq"],
        choices=["metaqa", "wqsp", "cwq", "kqapro", "mintaka", "wikimovies", "mlpq"],
        help="Datasets to download.",
    )
    parser.add_argument("--out-dir", default=str(DATASETS_DIR), help="Dataset root output directory.")
    parser.add_argument("--force", action="store_true", help="Redownload even if target directory already exists.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--wqsp-hf-id", default=DEFAULT_HF_DATASETS["wqsp"])
    parser.add_argument("--cwq-hf-id", default=DEFAULT_HF_DATASETS["cwq"])
    parser.add_argument("--kqapro-hf-id", default=DEFAULT_HF_DATASETS["kqapro"])
    parser.add_argument("--mintaka-hf-id", default=DEFAULT_HF_DATASETS["mintaka"])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_keep_order(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def download_file(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    print(f"[download] {url} -> {out_path}")
    with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
        shutil.copyfileobj(response, f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_reset_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)


def flatten_single_child_dir(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        return
    children = [p for p in root.iterdir() if p.name != "__MACOSX"]
    if len(children) != 1 or not children[0].is_dir():
        return

    wrapped = children[0]
    temp_dir = root.parent / f".tmp_flatten_{root.name}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for child in wrapped.iterdir():
        shutil.move(str(child), temp_dir / child.name)

    shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for child in temp_dir.iterdir():
        shutil.move(str(child), root / child.name)
    shutil.rmtree(temp_dir)


def normalize_metaqa_layout(dest: Path) -> None:
    flatten_single_child_dir(dest)

    nested_candidates = list(dest.rglob("kb.txt"))
    if not nested_candidates:
        return

    kb_parent = nested_candidates[0].parent
    if kb_parent == dest:
        return

    for name in ("1-hop", "2-hop", "3-hop", "kb.txt", "entity", "relations.json"):
        src = kb_parent / name
        dst = dest / name
        if not src.exists() or dst.exists():
            continue
        if src.is_dir():
            shutil.move(str(src), dst)
        else:
            ensure_dir(dst.parent)
            shutil.move(str(src), dst)


def normalize_kqapro_layout(dest: Path) -> None:
    flatten_single_child_dir(dest)
    snapshot_dir = dest / "hf_snapshot"
    if snapshot_dir.exists():
        flatten_single_child_dir(snapshot_dir)


def download_metaqa(out_root: Path, force: bool) -> Dict[str, Any]:
    dest = out_root / "MetaQA"
    maybe_reset_dir(dest, force)
    if dest.exists() and any(dest.iterdir()):
        return {"dataset": "MetaQA", "status": "skipped", "path": str(dest)}

    ensure_dir(dest.parent)
    print(f"[MetaQA] downloading official Google Drive folder to {dest}")
    import gdown

    downloaded = gdown.download_folder(
        url=METAQA_GDRIVE_URL,
        output=str(dest),
        quiet=False,
        remaining_ok=True,
        use_cookies=False,
    )
    if not downloaded:
        raise RuntimeError("MetaQA download returned no files.")

    normalize_metaqa_layout(dest)

    notes = {
        "dataset": "MetaQA",
        "source": METAQA_GDRIVE_URL,
        "expected_variant": "vanilla test",
        "path": str(dest),
    }
    write_json(dest / "_download_meta.json", notes)
    return {"dataset": "MetaQA", "status": "downloaded", "path": str(dest)}


def download_mlpq(out_root: Path, force: bool) -> Dict[str, Any]:
    dest = out_root / "MLPQ"
    maybe_reset_dir(dest, force)
    if dest.exists() and (dest / "datasets").exists():
        return {"dataset": "MLPQ", "status": "skipped", "path": str(dest)}

    ensure_dir(dest)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "mlpq.zip"
        download_file(MLPQ_GITHUB_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir_path)

        extracted_root = tmpdir_path / "Dataset-for-QA-over-Multilingual-KG-master"
        for name in ("datasets", "resources", "baselines"):
            src = extracted_root / name
            if src.exists():
                shutil.copytree(src, dest / name, dirs_exist_ok=True)
    flatten_single_child_dir(dest)

    write_json(
        dest / "_download_meta.json",
        {
            "dataset": "MLPQ",
            "source": MLPQ_GITHUB_ZIP,
            "expected_variant": "en-zh / en / ILLs",
            "path": str(dest),
        },
    )
    return {"dataset": "MLPQ", "status": "downloaded", "path": str(dest)}


def download_wikimovies(out_root: Path, force: bool) -> Dict[str, Any]:
    dest = out_root / "WikiMovies"
    maybe_reset_dir(dest, force)
    if dest.exists() and (dest / "movieqa").exists():
        return {"dataset": "WikiMovies", "status": "skipped", "path": str(dest)}

    ensure_dir(dest)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tar_path = tmpdir_path / "movieqa.tar.gz"
        download_file(WIKIMOVIES_TAR_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmpdir_path)

        candidates = list(tmpdir_path.rglob("movieqa"))
        if not candidates:
            raise RuntimeError("WikiMovies archive did not contain a movieqa directory.")
        movieqa_dir = candidates[0]
        shutil.copytree(movieqa_dir, dest / "movieqa", dirs_exist_ok=True)
    flatten_single_child_dir(dest)

    write_json(
        dest / "_download_meta.json",
        {
            "dataset": "WikiMovies",
            "source": WIKIMOVIES_TAR_URL,
            "expected_variant": "wiki_entities train",
            "path": str(dest),
        },
    )
    return {"dataset": "WikiMovies", "status": "downloaded", "path": str(dest)}


def snapshot_hf_repo(repo_id: str, local_dir: Path, force: bool) -> None:
    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        return
    maybe_reset_dir(local_dir, force)
    ensure_dir(local_dir)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    flatten_single_child_dir(local_dir)


def choose_split(available: List[str], preferred: List[str]) -> str:
    for name in preferred:
        if name in available:
            return name
    if not available:
        raise RuntimeError("No split available.")
    return available[0]


def export_hf_dataset(
    *,
    dataset_id: str,
    out_dir: Path,
    force: bool,
    hf_cache_dir: Optional[str],
    preferred_splits: List[str],
    normalizer: Callable[[Dict[str, Any]], Dict[str, Any]],
    source_label: str,
) -> Dict[str, Any]:
    maybe_reset_dir(out_dir, force)
    normalized_dir = out_dir / "normalized"
    raw_dir = out_dir / "raw"
    if normalized_dir.exists() and any(normalized_dir.iterdir()):
        return {"dataset": source_label, "status": "skipped", "path": str(out_dir)}

    ensure_dir(out_dir)
    from datasets import get_dataset_split_names, load_dataset

    split_names = list(get_dataset_split_names(dataset_id, cache_dir=hf_cache_dir))
    selected_split = choose_split(split_names, preferred_splits)
    dataset = load_dataset(dataset_id, split=selected_split, cache_dir=hf_cache_dir)

    raw_rows = [dict(row) for row in dataset]
    normalized_rows = [normalizer(row) for row in raw_rows]

    write_jsonl(raw_dir / f"{selected_split}.jsonl", raw_rows)
    write_jsonl(normalized_dir / f"{selected_split}.jsonl", normalized_rows)
    write_json(
        out_dir / "_download_meta.json",
        {
            "dataset": source_label,
            "source": dataset_id,
            "selected_split": selected_split,
            "available_splits": split_names,
            "path": str(out_dir),
        },
    )
    return {
        "dataset": source_label,
        "status": "downloaded",
        "path": str(out_dir),
        "selected_split": selected_split,
    }


def normalize_wqsp(row: Dict[str, Any]) -> Dict[str, Any]:
    answers = row.get("answer", [])
    if not isinstance(answers, list):
        answers = [answers]
    graph = row.get("graph", [])
    hop = 1
    if isinstance(graph, list) and graph:
        hop = max(1, min(3, len(graph)))
    return {
        "id": row.get("id"),
        "question": row.get("question", ""),
        "answers": unique_keep_order(answers),
        "hop": hop,
        "source_dataset": "wqsp",
    }


def normalize_cwq(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_answers = row.get("answers", [])
    answers: List[str] = []
    if isinstance(raw_answers, list):
        for item in raw_answers:
            if isinstance(item, dict):
                answers.extend(item.get("answer", []) if isinstance(item.get("answer"), list) else [item.get("answer")])
                answers.extend(item.get("aliases", []))
            else:
                answers.append(str(item))
    if not answers and row.get("composition_answer"):
        answers.append(str(row["composition_answer"]))
    return {
        "id": row.get("ID"),
        "question": row.get("question", ""),
        "answers": unique_keep_order(answers),
        "hop": 2,
        "source_dataset": "cwq",
        "compositionality_type": row.get("compositionality_type"),
    }


def normalize_kqapro(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_answer = row.get("answer")
    answers: List[Any]
    if isinstance(raw_answer, list):
        answers = raw_answer
    elif raw_answer is None:
        answers = []
    else:
        answers = [raw_answer]
    program = row.get("program", [])
    hop = 2
    if isinstance(program, list):
        hop = max(1, min(3, len(program) // 3 or 1))
    return {
        "question": row.get("question", ""),
        "answers": unique_keep_order(answers),
        "hop": hop,
        "source_dataset": "kqapro",
    }


def normalize_mintaka(row: Dict[str, Any]) -> Dict[str, Any]:
    answers = [row.get("answerText", "")]
    answer_entities = row.get("answerEntity", [])
    if isinstance(answer_entities, list):
        for item in answer_entities:
            if isinstance(item, dict):
                answers.append(item.get("label", ""))
                answers.append(item.get("name", ""))
    complexity = str(row.get("complexityType", "")).lower()
    hop = 2 if "multihop" in complexity else 1
    return {
        "id": row.get("id"),
        "question": row.get("question", ""),
        "answers": unique_keep_order(answers),
        "hop": hop,
        "source_dataset": "mintaka",
        "lang": row.get("lang", "en"),
        "complexityType": row.get("complexityType"),
    }


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    ensure_dir(out_root)

    jobs = {
        "metaqa": lambda: download_metaqa(out_root, args.force),
        "mlpq": lambda: download_mlpq(out_root, args.force),
        "wikimovies": lambda: download_wikimovies(out_root, args.force),
        "wqsp": lambda: export_hf_dataset(
            dataset_id=args.wqsp_hf_id,
            out_dir=out_root / "WQSP",
            force=args.force,
            hf_cache_dir=args.hf_cache_dir,
            preferred_splits=["test", "validation", "dev", "train"],
            normalizer=normalize_wqsp,
            source_label="WQSP",
        ),
        "cwq": lambda: export_hf_dataset(
            dataset_id=args.cwq_hf_id,
            out_dir=out_root / "CWQ",
            force=args.force,
            hf_cache_dir=args.hf_cache_dir,
            preferred_splits=["test", "validation", "dev", "train"],
            normalizer=normalize_cwq,
            source_label="CWQ",
        ),
        "kqapro": lambda: export_hf_dataset(
            dataset_id=args.kqapro_hf_id,
            out_dir=out_root / "KQAPro",
            force=args.force,
            hf_cache_dir=args.hf_cache_dir,
            preferred_splits=["test", "validation", "val", "dev", "train"],
            normalizer=normalize_kqapro,
            source_label="KQAPro",
        ),
        "mintaka": lambda: export_hf_dataset(
            dataset_id=args.mintaka_hf_id,
            out_dir=out_root / "Mintaka",
            force=args.force,
            hf_cache_dir=args.hf_cache_dir,
            preferred_splits=["test", "validation", "dev", "train"],
            normalizer=normalize_mintaka,
            source_label="Mintaka",
        ),
    }

    results = []
    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} ===")
        result = jobs[dataset_name]()
        if dataset_name == "kqapro":
            print("[KQAPro] snapshotting full dataset repo to capture kb.json")
            snapshot_hf_repo(args.kqapro_hf_id, out_root / "KQAPro" / "hf_snapshot", args.force)
            normalize_kqapro_layout(out_root / "KQAPro")
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    write_json(
        out_root / "_download_summary.json",
        {
            "requested_datasets": args.datasets,
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
