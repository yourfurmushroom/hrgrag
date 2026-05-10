#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
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
LOCAL_DATASETS_FALLBACK = ROOT_DIR.parent / "Datasets"

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
CWQ_URLS = {
    "train": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AAAIHeWX0cPpbpwK6w06BCxva/ComplexWebQuestions_train.json?dl=1",
    "validation": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
    "test": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1",
}
MINTAKA_URLS = {
    "train": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_train.json",
    "validation": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_dev.json",
    "test": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_test.json",
}
KQAPRO_TRIPLES_FILENAME = "kqapro_kb_triples.tsv"


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


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def maybe_reset_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)


def copy_local_dataset_if_available(source: Path, dest: Path, required: Optional[List[Path]] = None) -> bool:
    if not source.exists():
        return False
    if required:
        for rel in required:
            if not (source / rel).exists():
                return False
    ensure_dir(dest.parent)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return True


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


def sanitize_relation_name(name: Any) -> str:
    return str(name or "").strip().replace("\t", " ").replace("\n", " ")


def stringify_kqapro_value(value: Any) -> str:
    if isinstance(value, dict):
        value_type = value.get("type")
        raw_value = value.get("value")
        unit = str(value.get("unit", "")).strip()
        if value_type == "quantity":
            text = str(raw_value)
            if unit and unit != "1":
                text = f"{text} {unit}"
            return text
        if value_type in {"date", "year", "string"}:
            return str(raw_value)
        if raw_value is not None:
            return str(raw_value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return " | ".join(str(x) for x in value if str(x).strip())
    return str(value)


def write_kqapro_triples(kb_json_path: Path, out_path: Path) -> int:
    with open(kb_json_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    ensure_dir(out_path.parent)
    triple_count = 0

    def emit(fh, head: Any, relation: Any, tail: Any) -> None:
        nonlocal triple_count
        head_text = str(head or "").strip()
        relation_text = sanitize_relation_name(relation)
        tail_text = stringify_kqapro_value(tail).strip()
        if not head_text or not relation_text or not tail_text:
            return
        fh.write(f"{head_text}\t{relation_text}\t{tail_text}\n")
        triple_count += 1

    with open(out_path, "w", encoding="utf-8") as out_f:
        for concept_id, concept in kb.get("concepts", {}).items():
            emit(out_f, concept_id, "name", concept.get("name", ""))
            for parent_id in concept.get("instanceOf", []) or []:
                emit(out_f, concept_id, "instanceOf", parent_id)

        stmt_idx = 0
        for entity_id, entity in kb.get("entities", {}).items():
            emit(out_f, entity_id, "name", entity.get("name", ""))
            for concept_id in entity.get("instanceOf", []) or []:
                emit(out_f, entity_id, "instanceOf", concept_id)

            for attr_idx, attr in enumerate(entity.get("attributes", []) or []):
                predicate = attr.get("key", "")
                value_text = attr.get("value")
                emit(out_f, entity_id, predicate, value_text)

                qualifiers = attr.get("qualifiers", {}) or {}
                if qualifiers:
                    stmt_id = f"stmt_attr_{entity_id}_{attr_idx}_{stmt_idx}"
                    stmt_idx += 1
                    emit(out_f, stmt_id, "fact_h", entity_id)
                    emit(out_f, stmt_id, "fact_r", predicate)
                    emit(out_f, stmt_id, "fact_t", value_text)
                    for qualifier_key, qualifier_values in qualifiers.items():
                        for qualifier_value in qualifier_values or []:
                            emit(out_f, stmt_id, qualifier_key, qualifier_value)

            for rel_idx, rel in enumerate(entity.get("relations", []) or []):
                predicate = rel.get("predicate", "")
                object_id = rel.get("object", "")
                direction = rel.get("direction", "forward")
                if direction == "backward":
                    head, tail = object_id, entity_id
                else:
                    head, tail = entity_id, object_id
                emit(out_f, head, predicate, tail)

                qualifiers = rel.get("qualifiers", {}) or {}
                if qualifiers:
                    stmt_id = f"stmt_rel_{entity_id}_{rel_idx}_{stmt_idx}"
                    stmt_idx += 1
                    emit(out_f, stmt_id, "fact_h", head)
                    emit(out_f, stmt_id, "fact_r", predicate)
                    emit(out_f, stmt_id, "fact_t", tail)
                    for qualifier_key, qualifier_values in qualifiers.items():
                        for qualifier_value in qualifier_values or []:
                            emit(out_f, stmt_id, qualifier_key, qualifier_value)

    return triple_count


def download_metaqa(out_root: Path, force: bool) -> Dict[str, Any]:
    dest = out_root / "MetaQA"
    maybe_reset_dir(dest, force)
    if dest.exists() and any(dest.iterdir()):
        return {"dataset": "MetaQA", "status": "skipped", "path": str(dest)}

    local_source = LOCAL_DATASETS_FALLBACK / "MetaQA"
    if copy_local_dataset_if_available(local_source, dest, required=[Path("kb.txt"), Path("1-hop"), Path("2-hop"), Path("3-hop")]):
        normalize_metaqa_layout(dest)
        write_json(dest / "_download_meta.json", {"dataset": "MetaQA", "source": str(local_source), "path": str(dest), "mode": "local-copy"})
        return {"dataset": "MetaQA", "status": "copied_local", "path": str(dest)}

    ensure_dir(dest.parent)
    print(f"[MetaQA] downloading official Google Drive folder to {dest}")
    import gdown

    kwargs = {
        "url": METAQA_GDRIVE_URL,
        "output": str(dest),
        "quiet": False,
    }
    sig = inspect.signature(gdown.download_folder)
    if "remaining_ok" in sig.parameters:
        kwargs["remaining_ok"] = True
    if "use_cookies" in sig.parameters:
        kwargs["use_cookies"] = False

    downloaded = gdown.download_folder(**kwargs)
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

    local_source = LOCAL_DATASETS_FALLBACK / "MLPQ"
    if copy_local_dataset_if_available(local_source, dest, required=[Path("datasets")]):
        write_json(dest / "_download_meta.json", {"dataset": "MLPQ", "source": str(local_source), "path": str(dest), "mode": "local-copy"})
        return {"dataset": "MLPQ", "status": "copied_local", "path": str(dest)}

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

    local_source = LOCAL_DATASETS_FALLBACK / "WikiMovies"
    if copy_local_dataset_if_available(local_source, dest, required=[Path("movieqa")]):
        flatten_single_child_dir(dest)
        write_json(dest / "_download_meta.json", {"dataset": "WikiMovies", "source": str(local_source), "path": str(dest), "mode": "local-copy"})
        return {"dataset": "WikiMovies", "status": "copied_local", "path": str(dest)}

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


def download_json_list(url: str) -> List[Dict[str, Any]]:
    with urllib.request.urlopen(url) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list payload from {url}")
    return data


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


def copy_local_wqsp(out_root: Path, force: bool) -> Optional[Dict[str, Any]]:
    source = LOCAL_DATASETS_FALLBACK / "WebQSP"
    if not source.exists():
        return None

    out_dir = out_root / "WQSP"
    maybe_reset_dir(out_dir, force)
    normalized_dir = out_dir / "normalized"
    raw_dir = out_dir / "raw"
    if normalized_dir.exists() and any(normalized_dir.iterdir()):
        return {"dataset": "WQSP", "status": "skipped", "path": str(out_dir)}

    ensure_dir(normalized_dir)
    ensure_dir(raw_dir)

    split_map = {"train": "train.jsonl", "test": "test.jsonl", "validation": "validation.jsonl"}
    for split, filename in split_map.items():
        src = source / filename
        if not src.exists():
            continue
        rows = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)
        write_jsonl(raw_dir / filename, rows)
        write_jsonl(normalized_dir / filename, [normalize_wqsp(row) for row in rows])

    write_json(out_dir / "_download_meta.json", {"dataset": "WQSP", "source": str(source), "path": str(out_dir), "mode": "local-copy"})
    return {"dataset": "WQSP", "status": "copied_local", "path": str(out_dir)}


def download_cwq_direct(out_root: Path, force: bool) -> Dict[str, Any]:
    out_dir = out_root / "CWQ"
    maybe_reset_dir(out_dir, force)
    normalized_dir = out_dir / "normalized"
    raw_dir = out_dir / "raw"
    if normalized_dir.exists() and any(normalized_dir.iterdir()):
        return {"dataset": "CWQ", "status": "skipped", "path": str(out_dir)}

    ensure_dir(normalized_dir)
    ensure_dir(raw_dir)
    for split, url in CWQ_URLS.items():
        rows = download_json_list(url)
        write_jsonl(raw_dir / f"{split}.jsonl", rows)
        write_jsonl(normalized_dir / f"{split}.jsonl", [normalize_cwq(row) for row in rows])

    write_json(out_dir / "_download_meta.json", {"dataset": "CWQ", "source": CWQ_URLS, "path": str(out_dir), "mode": "direct-json"})
    return {"dataset": "CWQ", "status": "downloaded", "path": str(out_dir)}


def download_kqapro_direct(out_root: Path, repo_id: str, force: bool) -> Dict[str, Any]:
    out_dir = out_root / "KQAPro"
    maybe_reset_dir(out_dir, force)
    normalized_dir = out_dir / "normalized"
    raw_dir = out_dir / "raw"
    snapshot_dir = out_dir / "hf_snapshot"
    triples_path = out_dir / KQAPRO_TRIPLES_FILENAME
    if normalized_dir.exists() and any(normalized_dir.iterdir()):
        if not triples_path.exists() and (snapshot_dir / "kb.json").exists():
            write_kqapro_triples(snapshot_dir / "kb.json", triples_path)
        return {"dataset": "KQAPro", "status": "skipped", "path": str(out_dir)}

    snapshot_hf_repo(repo_id, snapshot_dir, force)
    ensure_dir(normalized_dir)
    ensure_dir(raw_dir)

    split_map = {"train": "train.json", "validation": "val.json", "test": "test.json"}
    for split, filename in split_map.items():
        src = snapshot_dir / filename
        if not src.exists():
            continue
        with open(src, "r", encoding="utf-8") as f:
            rows = json.load(f)
        write_jsonl(raw_dir / f"{split}.jsonl", rows)
        write_jsonl(normalized_dir / f"{split}.jsonl", [normalize_kqapro(row) for row in rows])

    kb_json_path = snapshot_dir / "kb.json"
    triples_path = out_dir / KQAPRO_TRIPLES_FILENAME
    triple_count = 0
    if kb_json_path.exists():
        triple_count = write_kqapro_triples(kb_json_path, triples_path)

    write_json(
        out_dir / "_download_meta.json",
        {
            "dataset": "KQAPro",
            "source": repo_id,
            "path": str(out_dir),
            "mode": "hf-snapshot",
            "kb_json_path": str(kb_json_path) if kb_json_path.exists() else "",
            "kb_triples_path": str(triples_path) if triples_path.exists() else "",
            "kb_triple_count": triple_count,
        },
    )
    return {"dataset": "KQAPro", "status": "downloaded", "path": str(out_dir)}


def download_mintaka_direct(out_root: Path, force: bool) -> Dict[str, Any]:
    out_dir = out_root / "Mintaka"
    maybe_reset_dir(out_dir, force)
    normalized_dir = out_dir / "normalized"
    raw_dir = out_dir / "raw"
    if normalized_dir.exists() and any(normalized_dir.iterdir()):
        return {"dataset": "Mintaka", "status": "skipped", "path": str(out_dir)}

    ensure_dir(normalized_dir)
    ensure_dir(raw_dir)
    for split, url in MINTAKA_URLS.items():
        rows = download_json_list(url)
        write_jsonl(raw_dir / f"{split}.jsonl", rows)
        write_jsonl(normalized_dir / f"{split}.jsonl", [normalize_mintaka(row) for row in rows])

    write_json(out_dir / "_download_meta.json", {"dataset": "Mintaka", "source": MINTAKA_URLS, "path": str(out_dir), "mode": "direct-json"})
    return {"dataset": "Mintaka", "status": "downloaded", "path": str(out_dir)}


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
        "wqsp": lambda: copy_local_wqsp(out_root, args.force) or export_hf_dataset(
            dataset_id=args.wqsp_hf_id,
            out_dir=out_root / "WQSP",
            force=args.force,
            hf_cache_dir=args.hf_cache_dir,
            preferred_splits=["test", "validation", "dev", "train"],
            normalizer=normalize_wqsp,
            source_label="WQSP",
        ),
        "cwq": lambda: download_cwq_direct(out_root, args.force),
        "kqapro": lambda: download_kqapro_direct(out_root, args.kqapro_hf_id, args.force),
        "mintaka": lambda: download_mintaka_direct(out_root, args.force),
    }

    results = []
    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} ===")
        result = jobs[dataset_name]()
        if dataset_name == "kqapro":
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
