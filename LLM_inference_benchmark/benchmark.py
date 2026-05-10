# benchmark.py
import asyncio
import argparse
import os
import json
import hashlib
import pickle
import nltk
import time
import gc
import torch
import pandas as pd
import sys
import re
from pathlib import Path
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from knowledgegraph_agent import KnowledgeGraphAgent
from baseline import BaselineKnowledgeGraphAgent as BaselineAgent
from dataset_utils import load_custom_dataset, load_metaqa_dataset, load_mlpq_dataset, load_normalized_jsonl_dataset, load_wikimovies_dataset, resolve_mlpq_kb_path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_naming import build_run_tag, with_run_tag, grammar_candidate_paths


# ==========================================
# 1. 設定與資料集
# ==========================================
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
DEFAULT_METAQA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "MetaQA")
DEFAULT_WIKIMOVIES_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "WikiMovies")
DEFAULT_MLPQ_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "MLPQ")
DEFAULT_WQSP_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "WQSP")
DEFAULT_CWQ_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "CWQ")
DEFAULT_KQAPRO_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "KQAPro")
DEFAULT_MINTAKA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "Mintaka")
ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_QA_DUMP_ROOT = os.path.join(ARTIFACTS_ROOT, "shared", "qa_dataset_dump")

# Hop count 對應 BFS 深度（用於 baseline 及 Mode-C)
HOP_TO_DEPTH = {"1-hop": 1, "2-hop": 2, "3-hop": 3}

MODEL_BACKBONES = [
    {
        "tag": "gpt-oss",
        "model_id": "openai/gpt-oss-20b",
    },
    {
        "tag": "Qwen3.5",
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    },
    {
        "tag": "llama3.1",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "tag": "qwen2.5",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
]


def paired_serialization_specs(
    base_name: str,
    model_id: str,
    shared_group: str,
    base_kwargs: dict,
):
    common = dict(base_kwargs)
    common["model_id"] = model_id
    common["shared_retrieval_group"] = shared_group
    return [
        (f"{base_name}-json", KnowledgeGraphAgent, {
            **common,
            "serialization_format": "json",
        }, False),
        (f"{base_name}-triple", KnowledgeGraphAgent, {
            **common,
            "serialization_format": "triples",
        }, False),
    ]


def single_agent_specs(
    base_name: str,
    model_id: str,
    shared_group: str,
    base_kwargs: dict,
    agent_class=KnowledgeGraphAgent,
    is_baseline: bool = False,
):
    common = dict(base_kwargs)
    common["model_id"] = model_id
    common["shared_retrieval_group"] = shared_group
    return [
        (base_name, agent_class, common, is_baseline),
    ]


def build_model_specs():
    specs = []
    for backbone in MODEL_BACKBONES:
        tag = backbone["tag"]
        model_id = backbone["model_id"]

        specs.extend(
            single_agent_specs(
                base_name=f"Baseline-BFS-{tag}",
                model_id=model_id,
                shared_group=f"Baseline-BFS-{tag}",
                base_kwargs={"bfs_depth": 3},
                agent_class=BaselineAgent,
                is_baseline=True,
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"Spine-Only-{tag}",
                model_id=model_id,
                shared_group=f"Spine-Only-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": False,
                    "use_fallback_correction": False,
                    "use_grammar_hint": False,
                    "grammar_path": None,
                },
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"Spine-Correction-{tag}",
                model_id=model_id,
                shared_group=f"Spine-Correction-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": False,
                    "use_fallback_correction": True,
                    "use_grammar_hint": False,
                    "grammar_path": None,
                },
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"Spine-GrammarExpansion-{tag}",
                model_id=model_id,
                shared_group=f"Spine-GrammarExpansion-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": True,
                    "use_fallback_correction": False,
                    "use_grammar_hint": False,
                    "expansion_strict": True,
                    "expansion_min_prob": 0.005,
                    "expansion_per_node_cap": 5,
                },
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"Spine-RandomExpansion-{tag}",
                model_id=model_id,
                shared_group=f"Spine-RandomExpansion-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": False,
                    "use_random_expansion": True,
                    "use_fallback_correction": False,
                    "use_grammar_hint": False,
                    "expansion_per_node_cap": 5,
                    "grammar_path": None,
                },
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"Spine-FrequencyExpansion-{tag}",
                model_id=model_id,
                shared_group=f"Spine-FrequencyExpansion-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": False,
                    "use_frequency_expansion": True,
                    "use_fallback_correction": False,
                    "use_grammar_hint": False,
                    "expansion_per_node_cap": 5,
                    "grammar_path": None,
                },
            )
        )

        specs.extend(
            paired_serialization_specs(
                base_name=f"HRG-Proposed-{tag}",
                model_id=model_id,
                shared_group=f"HRG-Proposed-{tag}",
                base_kwargs={
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": True,
                    "use_fallback_correction": True,
                    "use_grammar_hint": False,
                    "expansion_strict": True,
                    "expansion_min_prob": 0.005,
                    "expansion_per_node_cap": 5,
                },
            )
        )

    return specs

# ==========================================
# MODEL_SPECS
# 格式：(顯示名稱, Agent類, 初始化參數, 是_baseline)
# ==========================================
MODEL_SPECS = build_model_specs()



TEST_SAMPLE_LIMIT = 100
OUTPUT_FILE = os.path.join(ARTIFACTS_ROOT, "shared", "benchmark_results.json")

DETAIL_DIR = os.path.join(ARTIFACTS_ROOT, "shared", "benchmark_details_csv")
DETAIL_CSV = os.path.join(DETAIL_DIR, "all_models_outputs_wide.csv")

ALL_LONG_ROWS = []

# 避免同一張 GPU 上同時 generate 多題互搶
SEM = asyncio.Semaphore(1)


# ==========================================
# 2. 工具函式
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KGQA agents on portable KGQA datasets.")
    parser.add_argument("--dataset", choices=["metaqa", "wikimovies", "mlpq", "wqsp", "cwq", "kqapro", "mintaka", "custom"], default="metaqa")
    parser.add_argument("--dataset-root", default=None, help="Dataset root directory.")
    parser.add_argument("--dataset-file", default=None, help="Direct dataset file path override.")
    parser.add_argument("--split", default=None, help="Dataset split. MetaQA: train/dev/test, WikiMovies: train/dev/test.")
    parser.add_argument("--metaqa-variant", default="vanilla", help="MetaQA question variant, e.g. vanilla or ntm.")
    parser.add_argument("--wikimovies-subset", choices=["wiki_entities", "full"], default="wiki_entities")
    parser.add_argument("--mlpq-pair", choices=["en-zh", "en-fr", "zh-fr"], default="en-zh")
    parser.add_argument("--mlpq-question-lang", choices=["en", "fr", "zh"], default="en")
    parser.add_argument("--mlpq-fusion", choices=["ills", "nmn"], default="ills")
    parser.add_argument("--custom-dataset-name", default="custom", help="Portable tag used for custom dataset runs.")
    parser.add_argument("--custom-format", choices=["auto", "tsv", "jsonl"], default="auto")
    parser.add_argument("--custom-hop", type=int, default=1, help="Fallback hop value for custom datasets.")
    parser.add_argument("--model-filter", default=None, help="Only run models whose names contain this substring.")
    parser.add_argument("--sample-limit", type=int, default=TEST_SAMPLE_LIMIT)
    parser.add_argument("--kb-path", default=None)
    parser.add_argument("--relation-path", default=None)
    parser.add_argument("--grammar-path", default=None)
    parser.add_argument("--artifacts-root", default=ARTIFACTS_ROOT)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--detail-csv", default=DETAIL_CSV)
    return parser.parse_args()


def resolve_output_paths(args, run_tag: str):
    run_root = os.path.join(args.artifacts_root, run_tag)
    if args.output_file == OUTPUT_FILE:
        output_file = os.path.join(run_root, "results", "benchmark_results.json")
    else:
        output_file = args.output_file

    if args.detail_csv == DETAIL_CSV:
        detail_csv = os.path.join(run_root, "results", "all_models_outputs_wide.csv")
    else:
        detail_csv = args.detail_csv

    return output_file, detail_csv


def resolve_default_grammar_path(args, run_tag: str):
    if args.grammar_path:
        return args.grammar_path

    for candidate in grammar_candidate_paths(PROJECT_ROOT, run_tag, args.dataset):
        if os.path.exists(candidate):
            print(f"[Grammar] Auto-selected grammar: {candidate}")
            return candidate

    print("[Grammar] No auto-matched grammar found; continuing with dataset fallback path.")
    if args.dataset == "metaqa":
        return os.path.join(PROJECT_ROOT, "hrg_grammar", "metaqa_phrg_grammar.json")
    if args.dataset == "wikimovies":
        return os.path.join(PROJECT_ROOT, "hrg_grammar", "wikimovies_phrg_grammar.json")
    if args.dataset == "mlpq":
        return os.path.join(PROJECT_ROOT, "hrg_grammar", "mlpq_phrg_grammar.json")
    return os.path.join(args.artifacts_root, run_tag, "grammar", "hrg_grammar.json")


def resolve_dump_root(args, run_tag: str) -> str:
    return os.path.join(args.artifacts_root, run_tag, "dumps")


def resolve_run_tag(args):
    return build_run_tag(
        dataset=args.dataset,
        split=args.split,
        metaqa_variant=args.metaqa_variant,
        wikimovies_subset=args.wikimovies_subset,
        mlpq_pair=args.mlpq_pair,
        mlpq_question_lang=args.mlpq_question_lang,
        mlpq_fusion=args.mlpq_fusion,
        custom_dataset_name=args.custom_dataset_name,
    )


def infer_backbone(model_name: str) -> Optional[str]:
    for backbone in MODEL_BACKBONES:
        tag = backbone["tag"]
        if tag in model_name:
            return tag
    return None


def enrich_report_with_derived_metrics(full_report: dict):
    baseline_refs = {}
    for model_name, data in full_report.items():
        if not isinstance(data, dict) or data == "FAILED":
            continue
        backbone = infer_backbone(model_name)
        if backbone and f"Baseline-BFS-{backbone}" in model_name:
            baseline_refs[backbone] = data

    for model_name, data in full_report.items():
        if not isinstance(data, dict) or data == "FAILED":
            continue
        backbone = infer_backbone(model_name)
        baseline = baseline_refs.get(backbone)
        if baseline:
            base_ctx = baseline.get("avg_ctx_tokens", 0.0) or 0.0
            base_subg = baseline.get("avg_subgraph_size", 0.0) or 0.0
            data["compression_vs_bfs_ctx_ratio"] = (data.get("avg_ctx_tokens", 0.0) / base_ctx) if base_ctx else None
            data["compression_vs_bfs_subgraph_ratio"] = (data.get("avg_subgraph_size", 0.0) / base_subg) if base_subg else None
        else:
            data["compression_vs_bfs_ctx_ratio"] = None
            data["compression_vs_bfs_subgraph_ratio"] = None


def build_dataset_splits(args):
    if args.dataset == "metaqa":
        root = args.dataset_root or DEFAULT_METAQA_ROOT
        split = args.split or "dev"
        datasets = {
            "1-hop": os.path.join(root, "1-hop", args.metaqa_variant, f"qa_{split}.txt"),
            "2-hop": os.path.join(root, "2-hop", args.metaqa_variant, f"qa_{split}.txt"),
            "3-hop": os.path.join(root, "3-hop", args.metaqa_variant, f"qa_{split}.txt"),
        }
        grouped = {name: load_metaqa_dataset(path) for name, path in datasets.items()}
        hop_overrides = {name: HOP_TO_DEPTH.get(name, 2) for name in grouped}
        return grouped, hop_overrides

    if args.dataset == "wikimovies":
        root = args.dataset_root or DEFAULT_WIKIMOVIES_ROOT
        split = args.split or "test"
        subset = args.wikimovies_subset
        dataset_file = args.dataset_file or os.path.join(root, "movieqa", "questions", subset, f"{subset.replace('_', '-')}_qa_{split}.txt")
        grouped = load_wikimovies_dataset(dataset_file)
        hop_overrides = {"1-hop": 1}
        return grouped, hop_overrides

    if args.dataset == "custom":
        return build_custom_dataset_splits(args)

    if args.dataset in {"wqsp", "cwq", "kqapro", "mintaka"}:
        root_map = {
            "wqsp": DEFAULT_WQSP_ROOT,
            "cwq": DEFAULT_CWQ_ROOT,
            "kqapro": DEFAULT_KQAPRO_ROOT,
            "mintaka": DEFAULT_MINTAKA_ROOT,
        }
        root = args.dataset_root or root_map[args.dataset]
        if args.split:
            split = args.split
        elif args.dataset == "kqapro":
            split = "validation"
        else:
            split = "test"
        dataset_file = args.dataset_file or os.path.join(root, "normalized", f"{split}.jsonl")
        grouped = load_normalized_jsonl_dataset(dataset_file, default_hop=args.custom_hop)
        hop_overrides = {name: HOP_TO_DEPTH.get(name, get_hop(name)) for name in grouped}
        return grouped, hop_overrides

    root = args.dataset_root or DEFAULT_MLPQ_ROOT
    grouped = load_mlpq_dataset(
        root=root,
        pair=args.mlpq_pair,
        question_lang=args.mlpq_question_lang,
        inject_topic_entity=True,
    )
    hop_overrides = {name: HOP_TO_DEPTH.get(name, get_hop(name)) for name in grouped}
    return grouped, hop_overrides


def build_custom_dataset_splits(args):
    if not args.dataset_file:
        raise ValueError("--dataset-file is required when --dataset custom")
    grouped = load_custom_dataset(
        file_path=args.dataset_file,
        fmt=args.custom_format,
        default_hop=args.custom_hop,
    )
    hop_overrides = {name: HOP_TO_DEPTH.get(name, get_hop(name)) for name in grouped}
    return grouped, hop_overrides


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def split_candidate_answers(text: str):
    raw = (text or "").strip()
    if not raw:
        return []

    parts = re.split(r"\s*\|\s*|\s*;\s*|\n+", raw)
    normalized = []
    seen = set()
    for part in parts:
        norm = normalize_answer(part)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)
    return normalized


def calculate_metrics(references, candidate):
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
        acc = 1.0 if ref_norms and candidate_norm and candidate_norm == ref_norms[0] else 0.0
    else:
        acc = (overlap / len(ref_set)) if ref_set else 0.0

    exact_match = 1.0 if ref_set and candidate_set == ref_set else 0.0
    if len(ref_norms) <= 1:
        exact_match = acc

    answer_set_precision = (overlap / len(candidate_set)) if candidate_set else 0.0
    answer_set_recall = (overlap / len(ref_set)) if ref_set else 0.0
    if answer_set_precision + answer_set_recall > 0:
        answer_set_f1 = 2 * answer_set_precision * answer_set_recall / (answer_set_precision + answer_set_recall)
    else:
        answer_set_f1 = 0.0

    try:
        cand_tokens = nltk.word_tokenize(candidate_lower)
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in references]
        bleu = sentence_bleu(
            ref_tokens_list,
            cand_tokens,
            smoothing_function=SmoothingFunction().method1
        )
    except Exception:
        bleu = 0.0

    return {
        "bleu": float(bleu),
        "answer_recall": float(acc),
        "em": float(exact_match),
        "contains_hit": float(contains_hit),
        "hit_at_1_any": float(hit_at_1_any),
        "answer_set_precision": float(answer_set_precision),
        "answer_set_recall": float(answer_set_recall),
        "answer_set_f1": float(answer_set_f1),
    }


def get_hop(dataset_name: str) -> int:
    try:
        return int(dataset_name.split("-")[0])
    except Exception:
        return -1


def build_retrieval_share_key(model_name: str, agent_kwargs: dict, dataset_name: str, run_tag: str) -> Optional[str]:
    serialization = agent_kwargs.get("serialization_format")
    if serialization not in {"json", "triples"}:
        return None

    share_kwargs = {k: v for k, v in agent_kwargs.items() if k != "serialization_format"}
    shared_group = share_kwargs.pop("shared_retrieval_group", None) or model_name.rsplit("-", 1)[0]
    payload = {
        "base_model_name": shared_group,
        "dataset_name": dataset_name,
        "run_tag": run_tag,
        "agent_kwargs": share_kwargs,
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return f"{payload['base_model_name']}@{run_tag}@{dataset_name}@{digest}"


def export_wide_csv_from_long_rows(long_rows, out_csv_path):
    if not long_rows:
        print("[Warning] No data to export to CSV")
        return

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_long = pd.DataFrame(long_rows)

    df_wide = df_long.pivot_table(
        index=["hop", "dataset", "idx", "question", "expected_outputs"],
        columns="model",
        values="model_output",
        aggfunc="first"
    ).reset_index()

    df_wide.columns.name = None
    df_wide = df_wide.sort_values(["hop", "idx"])
    df_wide.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Info] 已輸出多模型對齊 CSV: {out_csv_path}")


async def async_evaluate_question(
    agent, question, references, pbar,
    save_dir=None, idx=0,
    hop_override=None,
    is_baseline=False,
    shared_prepare_path: Optional[str] = None,
):
    async with SEM:
        try:
            start_time = time.time()

            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"q_{idx:04d}.pkl")

            details = None
            if hasattr(agent, "ask_detailed") and not shared_prepare_path:
                details = await agent.ask_detailed(
                    question,
                    save_path=save_path,
                    references=references,
                    hop_override=hop_override if is_baseline else None,
                )
                response = details.get("answer", "")
            elif shared_prepare_path and hasattr(agent, "prepare_retrieval") and hasattr(agent, "answer_from_prepared"):
                prepared = None
                if os.path.exists(shared_prepare_path):
                    with open(shared_prepare_path, "rb") as f:
                        prepared = pickle.load(f)
                    print(f"[SharedRetrieval] Loaded cache: {shared_prepare_path}", flush=True)
                else:
                    prepared = await agent.prepare_retrieval(question, references=references)
                    os.makedirs(os.path.dirname(shared_prepare_path), exist_ok=True)
                    with open(shared_prepare_path, "wb") as f:
                        pickle.dump(prepared, f)
                    print(f"[SharedRetrieval] Saved cache: {shared_prepare_path}", flush=True)

                answer_bundle = await agent.answer_from_prepared(
                    question,
                    prepared,
                    save_path=save_path,
                    serialization_format=getattr(agent, "serialization_format", None),
                )
                response = answer_bundle["answer"]
                details = {
                    "answer": response,
                    "failure_stage": prepared.get("status", "ok"),
                    "parse_latency": prepared.get("parse_latency", 0.0),
                    "retrieval_latency": prepared.get("retrieval_latency", 0.0),
                    "generation_latency": answer_bundle.get("generation_latency", answer_bundle.get("elapsed", 0.0)),
                    "generation_failed": answer_bundle.get("generation_failed", False),
                    "answerable": bool((response or "").strip()),
                }
            elif is_baseline and hop_override is not None:
                response = await agent.ask(
                    question, save_path=save_path, references=references,
                    hop_override=hop_override,
                )
            else:
                response = await agent.ask(
                    question, save_path=save_path, references=references,
                )

            elapsed = time.time() - start_time
            metrics = calculate_metrics(references, response)
            pbar.update(1)

            return {
                "idx": idx,
                "question": question,
                "expected_outputs": "|".join(references),
                "model_output": response,
                **metrics,
                "elapsed": float(elapsed),
                "failure_stage": (details or {}).get("failure_stage", "ok"),
                "parse_latency": float((details or {}).get("parse_latency", 0.0) or 0.0),
                "retrieval_latency": float((details or {}).get("retrieval_latency", 0.0) or 0.0),
                "generation_latency": float((details or {}).get("generation_latency", 0.0) or 0.0),
                "generation_failed": bool((details or {}).get("generation_failed", False)),
                "answerable": bool((details or {}).get("answerable", bool((response or "").strip()))),
            }

        except Exception as e:
            err_str = str(e)
            failure_stage = "oom" if ("CUDA out of memory" in err_str or "out of memory" in err_str.lower()) else "runtime_error"
            tqdm.write(f"[Error q_{idx}] {e}")
            pbar.update(1)
            return {
                "idx": idx,
                "question": question,
                "expected_outputs": "|".join(references),
                "model_output": "",
                "bleu": 0.0,
                "answer_recall": 0.0,
                "em": 0.0,
                "contains_hit": 0.0,
                "hit_at_1_any": 0.0,
                "answer_set_precision": 0.0,
                "answer_set_recall": 0.0,
                "answer_set_f1": 0.0,
                "elapsed": 0.0,
                "error": err_str,
                "failure_stage": failure_stage,
                "parse_latency": 0.0,
                "retrieval_latency": 0.0,
                "generation_latency": 0.0,
                "generation_failed": True,
                "answerable": False,
            }


# ==========================================
# 3. 單模型評測
# ==========================================
async def evaluate_single_model(
    model_name,
    run_model_name,
    agent_instance,
    agent_kwargs,
    dataset_splits,
    hop_overrides,
    sample_limit,
    run_tag,
    dump_root,
    is_baseline=False,
):
    print(f"\n>>> 🚀 開始評測模型: [{run_model_name}]")
    model_results = {}
    base_save_dir = os.path.join(dump_root, "per_model", run_model_name)
    failure_counts = {}
    generation_failure_count = 0
    answerable_count = 0
    total_parse_latency = 0.0
    total_retrieval_latency = 0.0
    total_generation_latency = 0.0

    for dataset_name, dataset in dataset_splits.items():
        if not dataset:
            print(f"[Warning] Dataset {dataset_name} is empty, skipping.")
            continue

        if sample_limit:
            dataset = dataset[:sample_limit]

        hop = get_hop(dataset_name)
        hop_depth = hop_overrides.get(dataset_name)

        pbar = tqdm(
            total=len(dataset),
            desc=f"  [{run_model_name:<40}] {dataset_name:<10}",
            unit="q",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )

        current_save_dir = os.path.join(base_save_dir, dataset_name)
        shared_key = build_retrieval_share_key(model_name, agent_kwargs, dataset_name, run_tag)
        shared_prepare_dir = (
            os.path.join(dump_root, "_shared_retrieval", shared_key)
            if shared_key else None
        )

        tasks = []
        for idx, (question, references) in enumerate(dataset):
            shared_prepare_path = (
                os.path.join(shared_prepare_dir, f"q_{idx:04d}.prepared.pkl")
                if shared_prepare_dir else None
            )
            tasks.append(
                async_evaluate_question(
                    agent_instance,
                    question,
                    references,
                    pbar,
                    save_dir=current_save_dir,
                    idx=idx,
                    hop_override=hop_depth,
                    is_baseline=is_baseline,
                    shared_prepare_path=shared_prepare_path,
                )
            )

        results = await asyncio.gather(*tasks)
        pbar.close()

        for r in results:
            ALL_LONG_ROWS.append({
                "hop": hop,
                "dataset": dataset_name,
                "idx": r["idx"],
                "question": r["question"],
                "expected_outputs": r["expected_outputs"],
                "model": run_model_name,
                "model_output": r["model_output"],
            })

        total_bleu = sum(r["bleu"] for r in results)
        total_answer_recall = sum(r["answer_recall"] for r in results)
        total_em = sum(r["em"] for r in results)
        total_contains_hit = sum(r["contains_hit"] for r in results)
        total_hit_at_1_any = sum(r["hit_at_1_any"] for r in results)
        total_answer_set_precision = sum(r["answer_set_precision"] for r in results)
        total_answer_set_recall = sum(r["answer_set_recall"] for r in results)
        total_answer_set_f1 = sum(r["answer_set_f1"] for r in results)
        total_time = sum(r["elapsed"] for r in results)
        count = len(results)

        for r in results:
            stage = r.get("failure_stage", "ok") or "ok"
            failure_counts[stage] = failure_counts.get(stage, 0) + 1
            generation_failure_count += int(r.get("generation_failed", False))
            answerable_count += int(r.get("answerable", False))
            total_parse_latency += r.get("parse_latency", 0.0)
            total_retrieval_latency += r.get("retrieval_latency", 0.0)
            total_generation_latency += r.get("generation_latency", 0.0)

        avg_bleu = total_bleu / count if count > 0 else 0
        avg_answer_recall = total_answer_recall / count if count > 0 else 0
        avg_em = total_em / count if count > 0 else 0
        avg_contains_hit = total_contains_hit / count if count > 0 else 0
        avg_hit_at_1_any = total_hit_at_1_any / count if count > 0 else 0
        avg_answer_set_precision = total_answer_set_precision / count if count > 0 else 0
        avg_answer_set_recall = total_answer_set_recall / count if count > 0 else 0
        avg_answer_set_f1 = total_answer_set_f1 / count if count > 0 else 0
        avg_time = total_time / count if count > 0 else 0

        model_results[dataset_name] = {
            "bleu": round(avg_bleu, 4),
            "answer_recall": round(avg_answer_recall, 4),
            "em": round(avg_em, 4),
            "contains_hit": round(avg_contains_hit, 4),
            "hit_at_1_any": round(avg_hit_at_1_any, 4),
            "answer_set_precision": round(avg_answer_set_precision, 4),
            "answer_set_recall": round(avg_answer_set_recall, 4),
            "answer_set_f1": round(avg_answer_set_f1, 4),
            "avg_latency": round(avg_time, 2),
        }

        tqdm.write(
            f"  ✅ [{dataset_name}] BLEU: {avg_bleu:.4f} | "
            f"AnsRecall: {avg_answer_recall:.4f} | EM: {avg_em:.4f} | "
            f"AnsF1: {avg_answer_set_f1:.4f} | Avg Time: {avg_time:.2f}s"
        )

    # ==========================================
    # 整體統計數字 (供論文使用)
    # ==========================================
    total_q = getattr(agent_instance, "total_questions", 0)

    coverage = 0.0
    avg_ctx_tokens = 0.0
    avg_parse1_tokens = 0.0
    avg_correction_tokens = 0.0
    avg_parse2_tokens = 0.0
    avg_subgraph_size = 0.0

    if total_q > 0:
        coverage = getattr(agent_instance, "hit_grammar_count", 0) / total_q
        avg_ctx_tokens = getattr(agent_instance, "total_context_length", 0) / total_q
        avg_parse1_tokens = getattr(agent_instance, "total_parse1_tokens", 0) / total_q
        avg_correction_tokens = getattr(agent_instance, "total_correction_tokens", 0) / total_q
        avg_parse2_tokens = getattr(agent_instance, "total_parse2_tokens", 0) / total_q
        avg_subgraph_size = getattr(agent_instance, "total_subgraph_size", 0) / total_q

    avg_retrieval_recall = 0.0
    avg_retrieval_precision = 0.0
    avg_retrieval_f1 = 0.0
    total_eval_count = sum(len(ds[:sample_limit] if sample_limit else ds) for ds in dataset_splits.values() if ds)

    total_recall_q = getattr(agent_instance, "total_retrieval_questions", 0)
    if total_recall_q > 0:
        avg_retrieval_recall = getattr(agent_instance, "total_retrieval_recall", 0.0) / total_recall_q
        avg_retrieval_precision = getattr(agent_instance, "total_retrieval_precision", 0.0) / total_recall_q
        avg_retrieval_f1 = getattr(agent_instance, "total_retrieval_f1", 0.0) / total_recall_q

    return {
        "results": model_results,
        "coverage": coverage,
        "avg_ctx_tokens": avg_ctx_tokens,
        "avg_parse1_tokens": avg_parse1_tokens,
        "avg_correction_tokens": avg_correction_tokens,
        "avg_parse2_tokens": avg_parse2_tokens,
        "avg_subgraph_size": avg_subgraph_size,
        "avg_retrieval_recall": avg_retrieval_recall,
        "avg_retrieval_precision": avg_retrieval_precision,
        "avg_retrieval_f1": avg_retrieval_f1,
        "avg_parse_latency": (total_parse_latency / total_eval_count) if total_eval_count else 0.0,
        "avg_retrieval_latency": (total_retrieval_latency / total_eval_count) if total_eval_count else 0.0,
        "avg_generation_latency": (total_generation_latency / total_eval_count) if total_eval_count else 0.0,
        "answerable_rate": (answerable_count / total_eval_count) if total_eval_count else 0.0,
        "generation_failure_count": generation_failure_count,
        "failure_counts": failure_counts,
    }


# ==========================================
# 4. 主程式
# ==========================================
async def main():
    args = parse_args()
    dataset_splits, hop_overrides = build_dataset_splits(args)
    run_tag = resolve_run_tag(args)
    output_file, detail_csv = resolve_output_paths(args, run_tag)
    dump_root = resolve_dump_root(args, run_tag)

    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        print("[Info] Downloading NLTK punkt tokenizer.")
        nltk.download("punkt")

    print("==========================================")
    print(f"  {args.dataset.upper()} Benchmark & Data Collection")
    print(f"  Run Tag: {run_tag}")
    print("==========================================")

    full_report = {}
    dataset_labels = list(dataset_splits.keys())

    common_kb_path = args.kb_path or (
        os.path.join(PROJECT_ROOT, "Datasets", "MetaQA", "kb.txt")
        if args.dataset == "metaqa"
        else (
            (
                os.path.join(PROJECT_ROOT, "Datasets", "WikiMovies", "movieqa", "knowledge_source", "wiki_entities", "wiki_entities_kb.txt")
                if args.wikimovies_subset == "wiki_entities"
                else os.path.join(PROJECT_ROOT, "Datasets", "WikiMovies", "movieqa", "knowledge_source", "full", "full_kb.txt")
            )
            if args.dataset == "wikimovies"
            else (
                resolve_mlpq_kb_path(args.dataset_root or DEFAULT_MLPQ_ROOT, args.mlpq_pair, args.mlpq_fusion)
                if args.dataset == "mlpq"
                else None
            )
        )
    )
    common_relation_path = args.relation_path or (
        os.path.join(PROJECT_ROOT, "Datasets", "MetaQA", "relations.json")
        if args.dataset == "metaqa"
        else None
    )
    common_grammar_path = resolve_default_grammar_path(args, run_tag)

    # ==========================================
    # 載入已完成的結果（skip-if-done）
    # ==========================================
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            full_report = json.load(f)
        already_done = set(full_report.keys())
        print(f"[Resume] 已載入 {len(already_done)} 個已完成實驗: {sorted(already_done)}")
    else:
        already_done = set()

    for name, agent_class, agent_kwargs, is_baseline in MODEL_SPECS:
        run_model_name = with_run_tag(name, run_tag)
        print(f"\n\n{'=' * 60}")
        print(f"📄 準備載入並評測: {run_model_name}")
        print(f"{'=' * 60}")

        if args.model_filter and args.model_filter not in name:
            print(f"[Skip] {run_model_name} 不符合 model filter: {args.model_filter}")
            continue

        # Skip if already done
        if run_model_name in already_done:
            print(f"[Skip] {run_model_name} 已有結果，跳過。")
            continue

        agent_instance = None
        try:
            print(f"[System] Initializing {run_model_name} with class {agent_class.__name__}.")
            init_kwargs = dict(agent_kwargs)
            init_kwargs.pop("shared_retrieval_group", None)
            init_kwargs["kb_path"] = common_kb_path
            init_kwargs["relation_path"] = common_relation_path
            if agent_class is KnowledgeGraphAgent and init_kwargs.get("grammar_path", "__DEFAULT__") is not None:
                init_kwargs["grammar_path"] = common_grammar_path
            agent_instance = agent_class(**init_kwargs)

            report = await evaluate_single_model(
                name,
                run_model_name,
                agent_instance,
                agent_kwargs,
                dataset_splits=dataset_splits,
                hop_overrides=hop_overrides,
                sample_limit=args.sample_limit,
                run_tag=run_tag,
                dump_root=dump_root,
                is_baseline=is_baseline,
            )
            full_report[run_model_name] = report

        except Exception as e:
            print(f"[Error] 模型 {run_model_name} 執行失敗: {e}")
            import traceback
            traceback.print_exc()
            full_report[run_model_name] = "FAILED"

        finally:
            print(f"[System] Unloading {run_model_name} to free VRAM.")
            if agent_instance:
                del agent_instance

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[System] GPU Cache Cleared.")

            time.sleep(5)

    # ==========================================
    # 5. 輸出結果
    # ==========================================
    enrich_report_with_derived_metrics(full_report)
    SEP = "=" * 145
    LINE = "-" * 145
    print(f"\n\n{SEP}")
    label_headers = " | ".join(f"{label:<12}" for label in dataset_labels)
    print(f"{'Model Name':<30} | {'Metric':<26} | {label_headers}")
    print(LINE)

    for model_name, data in full_report.items():
        if data == "FAILED":
            print(f"{model_name:<30} | {'CRASHED':<26} | N/A          | N/A          | N/A")
            print(LINE)
            continue

        res = data.get("results", {}) if isinstance(data, dict) else data
        def v(d, k):
            return f"{d.get(k, 0):.4f}" if isinstance(d, dict) else "N/A"

        def dataset_metric_line(metric_key: str) -> str:
            return " | ".join(f"{v(res.get(label, {}), metric_key):<12}" for label in dataset_labels)

        print(f"{model_name:<30} | {'EM':<26} | {dataset_metric_line('em')}")
        print(f"{'':30} | {'Answer-Set F1':<26} | {dataset_metric_line('answer_set_f1')}")
        print(f"{'':30} | {'Answer Recall':<26} | {dataset_metric_line('answer_recall')}")
        print(f"{'':30} | {'Hit@1-any':<26} | {dataset_metric_line('hit_at_1_any')}")
        print(f"{'':30} | {'Answer-Set Precision':<26} | {dataset_metric_line('answer_set_precision')}")
        print(f"{'':30} | {'Answer-Set Recall':<26} | {dataset_metric_line('answer_set_recall')}")
        print(f"{'':30} | {'BLEU':<26} | {dataset_metric_line('bleu')}")
        print(f"{'':30} | {'Avg Latency (s)':<26} | {dataset_metric_line('avg_latency')}")

        if isinstance(data, dict) and "avg_retrieval_recall" in data:
            recall = data.get('avg_retrieval_recall', 0.0)
            prec = data.get('avg_retrieval_precision', 0.0)
            f1 = data.get('avg_retrieval_f1', 0.0)
            subg = data.get('avg_subgraph_size', 0.0)
            cov = data.get('coverage', 0.0)
            ctx = data.get('avg_ctx_tokens', 0)
            p1 = data.get('avg_parse1_tokens', 0)
            p2 = data.get('avg_parse2_tokens', 0)
            corr = data.get('avg_correction_tokens', 0)

            print(f"{'':30} | {'─'*26} | {'─'*12} | {'─'*12} | {'─'*12}")
            print(f"{'':30} | {'Subgraph Recall':<26} | {recall*100:.2f}%")
            print(f"{'':30} | {'Subgraph Precision':<26} | {prec*100:.2f}%")
            print(f"{'':30} | {'Subgraph F1':<26} | {f1*100:.2f}%")
            print(f"{'':30} | {'Avg Subgraph Size (triples)':<26} | {subg:.1f}")
            print(f"{'':30} | {'HRG Grammar Hit Rate':<26} | {cov*100:.2f}%")
            print(f"{'':30} | {'Avg Context Tokens':<26} | ~{int(ctx)}")
            print(f"{'':30} | {'Avg Parse1 Tokens':<26} | ~{int(p1)}")
            print(f"{'':30} | {'Avg Correction Tokens':<26} | ~{int(corr)}")
            print(f"{'':30} | {'Avg Parse2 Tokens':<26} | ~{int(p2)}")
            print(f"{'':30} | {'Total Avg Tokens':<26} | ~{int(p1+corr+p2)}")

        print(LINE)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)
    print(f"\n[Info] 詳細報告已儲存至 {output_file}")

    export_wide_csv_from_long_rows(ALL_LONG_ROWS, detail_csv)


if __name__ == "__main__":
    asyncio.run(main())
