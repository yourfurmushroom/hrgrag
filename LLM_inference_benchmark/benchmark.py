# benchmark.py
import asyncio
import argparse
import os
import json
import hashlib
import pickle
import time
import gc
import math
import traceback
import torch
import pandas as pd
import sys
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from huggingface_hub import login
from knowledgegraph_agent import KnowledgeGraphAgent
from baseline import BaselineKnowledgeGraphAgent as BaselineAgent
from dataset_utils import load_custom_dataset, load_metaqa_dataset, load_mlpq_dataset, load_normalized_jsonl_dataset, load_wikimovies_dataset, resolve_mlpq_kb_path, normalize_kb_token
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_naming import build_run_tag, with_run_tag, grammar_candidate_paths
# ==========================================
# 1. 設定與資料集
# ==========================================
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
DEFAULT_METAQA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "MetaQA")
DEFAULT_WIKIMOVIES_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "WikiMovies")
DEFAULT_MLPQ_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "MLPQ")
DEFAULT_MLPQ_ALIAS_PATH = os.path.join(PROJECT_ROOT, "configs", "mlpq_aliases.json")
DEFAULT_WQSP_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "WQSP")
DEFAULT_CWQ_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "CWQ")
DEFAULT_KQAPRO_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "KQAPro")
DEFAULT_MINTAKA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", "Mintaka")


def default_wikimovies_kb_path(subset: str) -> str:
    kb_root = os.path.join(PROJECT_ROOT, "Datasets", "WikiMovies", "movieqa", "knowledge_source")
    if subset != "wiki_entities":
        return os.path.join(kb_root, "full", "full_kb.txt")

    normalized = os.path.join(kb_root, "wiki_entities", "wiki_entities_kb_normalized.txt")
    raw = os.path.join(kb_root, "wiki_entities", "wiki_entities_kb.txt")
    return normalized if os.path.exists(normalized) else raw
ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_QA_DUMP_ROOT = os.path.join(ARTIFACTS_ROOT, "shared", "qa_dataset_dump")


def load_local_dotenv(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_local_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        print(f"[Warning] Hugging Face login skipped: {e}")

# Hop count 對應 BFS 深度（用於 baseline 及 Mode-C)
HOP_TO_DEPTH = {"1-hop": 1, "2-hop": 2, "3-hop": 3}
EXPERIMENT_SUITE = os.getenv("EXPERIMENT_SUITE", "core").strip().lower()
RANKING_POLICY = os.getenv("RANKING_POLICY", "lax-hrg-prior-v1").strip() or "lax-hrg-prior-v1"
CANDIDATE_RANKING_KEY = (
    "valid(c), same_arity_grammar_hit(c), ordered_path_grammar_hit(c), "
    "grammar_label_hit(c), grammar_score(c), matched_rule_count(c), "
    "llm_rerank_score(c), failed_hop_progress(c), step_survival(c), "
    "final_frontier_size(c), source_priority(c), llm_confidence(c), -original_index(c)"
)


def suite_enabled(*names: str) -> bool:
    enabled = {name.strip().lower() for name in names}
    return EXPERIMENT_SUITE in {"all", "full"} or EXPERIMENT_SUITE in enabled


def parse_int_list_env(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    out: list[int] = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        out.append(int(part))
    return out


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in {None, ""} else default


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in {None, ""} else default


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def sharded_backbone_target_device(tag: str, default: str) -> str:
    env_key = f"{re.sub(r'[^A-Za-z0-9]+', '_', tag).upper()}_TARGET_DEVICE"
    return os.getenv(env_key, os.getenv("SHARDED_TARGET_DEVICE", default))


def sharded_backbone_bool(tag: str, name: str, default: bool) -> bool:
    tag_key = f"{re.sub(r'[^A-Za-z0-9]+', '_', tag).upper()}_{name}"
    raw = os.getenv(tag_key, os.getenv(f"SHARDED_{name}"))
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODEL_BACKBONES = [
    # {
    #     "tag": "gpt-oss",
    #     "model_id": "openai/gpt-oss-20b",
    #     "use_model_sharding": True,
    #     "strict_gpu_sharding": sharded_backbone_bool("gpt-oss", "STRICT_GPU_SHARDING", True),
    #     "target_device": sharded_backbone_target_device("gpt-oss", "cuda:0,cuda:1"),
    # },
    # {
    #     "tag": "qwen3.5",
    #     "model_id": "Qwen/Qwen3.5-35B-A3B-FP8",
    #     "use_model_sharding": True,
    #     "strict_gpu_sharding": sharded_backbone_bool("qwen3.5", "STRICT_GPU_SHARDING", True),
    #     "target_device": sharded_backbone_target_device("qwen3.5", "cuda:0,cuda:1"),
    # },
    {
       "tag": "llama3.1",
       "model_id": "meta-llama/Llama-3.1-8B-Instruct",
       "use_model_sharding": False,
       "strict_gpu_sharding": False,
       "target_device": "cuda:0",
    },
    {
        "tag": "gemma4",
        "model_id": "google/gemma-4-E4B-it",
        "use_model_sharding": False,
        "strict_gpu_sharding": False,
        "target_device": "cuda:0",
    },
    {
       "tag": "qwen2.5",
       "model_id": "Qwen/Qwen2.5-7B-Instruct",
       "use_model_sharding": False,
       "strict_gpu_sharding": False,
       "target_device": "cuda:0",
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
    include_full_suite = suite_enabled("full", "all")
    include_legacy_expansion_specs = include_full_suite or os.getenv("ENABLE_LEGACY_EXPANSION_SPECS", "0") == "1"
    include_new_ablation_specs = include_full_suite or os.getenv("ENABLE_NEW_ABLATION_SPECS", "0") == "1"
    include_ngram_specs = include_full_suite or os.getenv("ENABLE_RELATION_NGRAM_SPECS", "0") == "1"
    include_bfs_cap_specs = include_full_suite or os.getenv("ENABLE_BFS_CAP_SPECS", "0") == "1"
    bfs_caps = parse_int_list_env("BFS_DEGREE_CAPS", "50 100 200 500")
    bfs_token_budgets = parse_int_list_env("BFS_CONTEXT_TOKEN_BUDGETS", "200 500 1000")
    for backbone in MODEL_BACKBONES:
        tag = backbone["tag"]
        model_id = backbone["model_id"]
        llm_device_kwargs = {
            "use_model_sharding": backbone.get("use_model_sharding", False),
            "strict_gpu_sharding": backbone.get("strict_gpu_sharding", False),
            "target_device": backbone.get("target_device"),
        }

        specs.extend(
            single_agent_specs(
                base_name=f"Baseline-BFS-{tag}",
                model_id=model_id,
                shared_group=f"Baseline-BFS-{tag}",
                base_kwargs={"bfs_depth": 3, **llm_device_kwargs},
                agent_class=BaselineAgent,
                is_baseline=True,
            )
        )

        if include_bfs_cap_specs:
            for cap in bfs_caps:
                specs.extend(
                    single_agent_specs(
                        base_name=f"Degree-Capped-BFS-{cap}-{tag}",
                        model_id=model_id,
                        shared_group=f"Degree-Capped-BFS-{cap}-{tag}",
                        base_kwargs={
                            "bfs_depth": 3,
                            "max_edges_per_hop": cap,
                            **llm_device_kwargs,
                        },
                        agent_class=BaselineAgent,
                        is_baseline=True,
                    )
                )

            for budget in bfs_token_budgets:
                specs.extend(
                    single_agent_specs(
                        base_name=f"Token-Budgeted-BFS-{budget}-{tag}",
                        model_id=model_id,
                        shared_group=f"Token-Budgeted-BFS-{budget}-{tag}",
                        base_kwargs={
                            "bfs_depth": 3,
                            "context_token_budget": budget,
                            **llm_device_kwargs,
                        },
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
                    **llm_device_kwargs,
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
                    **llm_device_kwargs,
                    "use_grammar_rerank": False,
                    "use_grammar_expansion": False,
                    "use_fallback_correction": True,
                    "use_grammar_hint": False,
                    "grammar_path": None,
                },
            )
        )

        if include_legacy_expansion_specs:
            specs.extend(
                paired_serialization_specs(
                    base_name=f"Spine-GrammarExpansion-{tag}",
                    model_id=model_id,
                    shared_group=f"Spine-GrammarExpansion-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        "use_grammar_rerank": True,
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
                        **llm_device_kwargs,
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
                        **llm_device_kwargs,
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

        if include_new_ablation_specs:
            controlled_budget = {
                "num_candidates": int_env("HRG_NUM_CANDIDATES", 8),
                "valid_chain_fallback_topk": int_env("HRG_VALID_CHAIN_FALLBACK_TOPK", 12),
                "valid_chain_fallback_beam_width": int_env("HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH", 48),
                "valid_chain_fallback_branch": int_env("HRG_VALID_CHAIN_FALLBACK_BRANCH", 16),
                "max_total_context_edges": int_env("HRG_MAX_TOTAL_CONTEXT_EDGES", 160),
                "use_low_confidence_valid_chain_fallback": bool_env("HRG_USE_LOW_CONFIDENCE_VALID_CHAIN_FALLBACK", False),
                "low_confidence_min_valid_candidates": int_env("HRG_LOW_CONFIDENCE_MIN_VALID_CANDIDATES", 2),
                "subgraph_support_saturation": int_env("HRG_SUBGRAPH_SUPPORT_SATURATION", 12),
            }

            specs.extend(
                paired_serialization_specs(
                    base_name=f"Spine-Correction-KGValidFallback-{tag}",
                    model_id=model_id,
                    shared_group=f"Spine-Correction-KGValidFallback-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        **controlled_budget,
                        "use_grammar_rerank": False,
                        "use_grammar_expansion": False,
                        "use_fallback_correction": True,
                        "use_deterministic_valid_chain_fallback": True,
                        "use_valid_chain_llm_rerank": False,
                        "use_grammar_hint": False,
                        "grammar_path": None,
                    },
                )
            )

            specs.extend(
                paired_serialization_specs(
                    base_name=f"HRG-Proposed-NoExpansion-{tag}",
                    model_id=model_id,
                    shared_group=f"HRG-Proposed-NoExpansion-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        **controlled_budget,
                        "use_grammar_rerank": True,
                        "use_grammar_expansion": False,
                        "use_fallback_correction": True,
                        "use_deterministic_valid_chain_fallback": True,
                        "use_valid_chain_llm_rerank": True,
                        "use_grammar_hint": False,
                        "require_ordered_grammar_match": bool_env("HRG_REQUIRE_ORDERED_GRAMMAR_MATCH", False),
                    },
                )
            )

            specs.extend(
                paired_serialization_specs(
                    base_name=f"HRG-Proposed-Expansion-{tag}",
                    model_id=model_id,
                    shared_group=f"HRG-Proposed-Expansion-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        **controlled_budget,
                        "use_grammar_rerank": True,
                        "use_grammar_expansion": True,
                        "use_fallback_correction": True,
                        "use_deterministic_valid_chain_fallback": True,
                        "use_valid_chain_llm_rerank": True,
                        "use_grammar_hint": False,
                        "expansion_strict": True,
                        "expansion_min_prob": 2.0,
                        "expansion_per_node_cap": 4,
                        "min_grammar_score_for_expansion": 2.0,
                        "max_spine_edges_for_expansion": 80,
                        "require_ordered_grammar_match": bool_env("HRG_REQUIRE_ORDERED_GRAMMAR_MATCH", False),
                        "require_exact_grammar_match_for_expansion": False,
                        "max_expansion_edges": 48,
                        "max_expansion_edge_ratio": 0.5,
                    },
                )
            )

            specs.extend(
                paired_serialization_specs(
                    base_name=f"HRG-GrammarFirst-NoExpansion-{tag}",
                    model_id=model_id,
                    shared_group=f"HRG-GrammarFirst-NoExpansion-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        **controlled_budget,
                        "use_grammar_first_retrieval": True,
                        "use_grammar_rerank": True,
                        "use_grammar_expansion": False,
                        "use_fallback_correction": False,
                        "use_deterministic_valid_chain_fallback": True,
                        "use_valid_chain_llm_rerank": True,
                        "use_grammar_hint": False,
                        "require_ordered_grammar_match": bool_env("HRG_REQUIRE_ORDERED_GRAMMAR_MATCH", False),
                    },
                )
            )

            specs.extend(
                paired_serialization_specs(
                    base_name=f"HRG-GrammarFirst-Expansion-{tag}",
                    model_id=model_id,
                    shared_group=f"HRG-GrammarFirst-Expansion-{tag}",
                    base_kwargs={
                        **llm_device_kwargs,
                        **controlled_budget,
                        "use_grammar_first_retrieval": True,
                        "use_grammar_rerank": True,
                        "use_grammar_expansion": True,
                        "use_fallback_correction": False,
                        "use_deterministic_valid_chain_fallback": True,
                        "use_valid_chain_llm_rerank": True,
                        "use_grammar_hint": False,
                        "expansion_strict": True,
                        "expansion_min_prob": 2.0,
                        "expansion_per_node_cap": 4,
                        "min_grammar_score_for_expansion": 2.0,
                        "max_spine_edges_for_expansion": 80,
                        "require_ordered_grammar_match": bool_env("HRG_REQUIRE_ORDERED_GRAMMAR_MATCH", False),
                        "require_exact_grammar_match_for_expansion": False,
                        "max_expansion_edges": 48,
                        "max_expansion_edge_ratio": 0.5,
                    },
                )
            )

        specs.extend(
            paired_serialization_specs(
                base_name=f"HRG-Proposed-{tag}",
                model_id=model_id,
                shared_group=f"HRG-Proposed-{tag}",
                base_kwargs={
                    **llm_device_kwargs,
                    "num_candidates": int_env("HRG_NUM_CANDIDATES", 8),
                    "use_grammar_rerank": True,
                    "use_grammar_expansion": True,
                    "use_fallback_correction": True,
                    "use_deterministic_valid_chain_fallback": True,
                    "use_valid_chain_llm_rerank": True,
                    "valid_chain_fallback_topk": int_env("HRG_VALID_CHAIN_FALLBACK_TOPK", 12),
                    "valid_chain_fallback_beam_width": int_env("HRG_VALID_CHAIN_FALLBACK_BEAM_WIDTH", 48),
                    "valid_chain_fallback_branch": int_env("HRG_VALID_CHAIN_FALLBACK_BRANCH", 16),
                    "use_grammar_hint": False,
                    "expansion_strict": True,
                    "expansion_min_prob": 2.0,
                    "expansion_per_node_cap": 4,
                    "min_grammar_score_for_expansion": 2.0,
                    "max_spine_edges_for_expansion": 80,
                    "require_ordered_grammar_match": bool_env("HRG_REQUIRE_ORDERED_GRAMMAR_MATCH", False),
                    "require_exact_grammar_match_for_expansion": False,
                    "max_expansion_edges": 48,
                    "max_expansion_edge_ratio": 0.5,
                    "max_total_context_edges": int_env("HRG_MAX_TOTAL_CONTEXT_EDGES", 160),
                    "use_low_confidence_valid_chain_fallback": bool_env("HRG_USE_LOW_CONFIDENCE_VALID_CHAIN_FALLBACK", False),
                    "low_confidence_min_valid_candidates": int_env("HRG_LOW_CONFIDENCE_MIN_VALID_CANDIDATES", 2),
                    "subgraph_support_saturation": int_env("HRG_SUBGRAPH_SUPPORT_SATURATION", 12),
                },
            )
        )

        if include_ngram_specs:
            for prior_name, prior_order in [
                ("RelationUnigram", 1),
                ("RelationBigram", 2),
                ("RelationTrigram", 3),
            ]:
                specs.extend(
                    paired_serialization_specs(
                        base_name=f"{prior_name}-{tag}",
                        model_id=model_id,
                        shared_group=f"{prior_name}-{tag}",
                        base_kwargs={
                            **llm_device_kwargs,
                            "num_candidates": 8,
                            "use_grammar_rerank": False,
                            "use_grammar_expansion": False,
                            "use_fallback_correction": True,
                            "use_deterministic_valid_chain_fallback": True,
                            "use_valid_chain_llm_rerank": False,
                            "valid_chain_fallback_topk": 12,
                            "valid_chain_fallback_beam_width": 48,
                            "valid_chain_fallback_branch": 16,
                            "use_grammar_hint": False,
                            "relation_ngram_prior_order": prior_order,
                            "relation_ngram_prior_alpha": 0.1,
                            "max_total_context_edges": 160,
                            "use_low_confidence_valid_chain_fallback": False,
                            "low_confidence_min_valid_candidates": 2,
                            "subgraph_support_saturation": 12,
                        },
                    )
                )

    return specs

# ==========================================
# MODEL_SPECS
# 格式：(顯示名稱, Agent類, 初始化參數, 是_baseline)
# ==========================================
MODEL_SPECS = build_model_specs()



TEST_SAMPLE_LIMIT = 50
OUTPUT_FILE = os.path.join(ARTIFACTS_ROOT, "shared", "benchmark_results.json")

DETAIL_DIR = os.path.join(ARTIFACTS_ROOT, "shared", "benchmark_details_csv")
DETAIL_CSV = os.path.join(DETAIL_DIR, "all_models_outputs_wide.csv")

ALL_LONG_ROWS = []

ANSWER_METRIC_KEYS = [
    "em",
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "mrr",
    "answer_set_precision",
    "answer_set_recall",
    "answer_set_f1",
]

RETRIEVAL_RANK_METRIC_KEYS = [
    "retrieval_recall_at_1",
    "retrieval_recall_at_3",
    "retrieval_recall_at_5",
    "retrieval_ndcg_at_1",
    "retrieval_ndcg_at_3",
    "retrieval_ndcg_at_5",
]

CLAIM_METRIC_KEYS = [
    "claim_faithfulness",
    "claim_hallucination",
]

EVIDENCE_METRIC_KEYS = [
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
    "citation_correctness",
]

EVIDENCE_COVERAGE_METRIC_KEYS = [
    "endpoint_coverage_final",
    "endpoint_coverage_spine",
    "endpoint_coverage_expanded",
    "answer_in_final_context",
    "answer_in_spine",
    "answer_in_expanded_edges",
    "gold_only_in_expansion",
]

CONDITIONAL_ANSWER_METRIC_KEYS = [
    "em_when_answer_in_context",
    "answer_f1_when_answer_in_context",
    "em_when_answer_not_in_context",
    "answer_f1_when_answer_not_in_context",
    "answer_f1_when_answer_in_spine",
    "answer_f1_when_answer_only_in_expansion",
    "answer_in_context_but_wrong",
    "answer_correct_without_context",
]

CONTEXT_DIAGNOSTIC_METRIC_KEYS = [
    "final_edge_count",
    "spine_edge_count",
    "expanded_edge_count",
    "raw_retrieved_edge_count",
    "context_truncated",
    "context_truncation_ratio",
]

GRAMMAR_DIAGNOSTIC_METRIC_KEYS = [
    "grammar_label_subset_hit",
    "grammar_ordered_path_hit",
    "grammar_arity_compatible_hit",
    "grammar_structural_proxy_hit",
    "grammar_full_structural_hit",
]

DIAGNOSTIC_METRIC_KEYS = (
    EVIDENCE_COVERAGE_METRIC_KEYS
    + CONDITIONAL_ANSWER_METRIC_KEYS
    + CONTEXT_DIAGNOSTIC_METRIC_KEYS
    + GRAMMAR_DIAGNOSTIC_METRIC_KEYS
)

EXTRA_METRIC_KEYS = (
    RETRIEVAL_RANK_METRIC_KEYS
    + CLAIM_METRIC_KEYS
    + EVIDENCE_METRIC_KEYS
    + DIAGNOSTIC_METRIC_KEYS
)

LEGACY_ANSWER_METRIC_KEYS = [
    "bleu",
    "answer_recall",
    "contains_hit",
    "hit_at_1_any",
]

# 避免同一張 GPU 上同時 generate 多題互搶
SEM = asyncio.Semaphore(1)


# ==========================================
# 2. 工具函式
# ==========================================

def apply_dataset_agent_overrides(args, model_name: str, agent_class, agent_kwargs: dict) -> dict:
    """
    Low-risk dataset-specific tuning.

    Goal:
    - Keep the method definition unchanged.
    - Only enlarge search budgets on datasets where current bottlenecks are
      candidate discovery and KG-valid fallback coverage.
    """
    tuned = dict(agent_kwargs)
    if agent_class is not KnowledgeGraphAgent:
        return tuned
    if getattr(args, "fixed_ablation_budget", False):
        return tuned

    if args.dataset == "mlpq":
        # Multilingual KG needs broader chain exploration and fallback search.
        tuned["num_candidates"] = max(int(tuned.get("num_candidates", 5) or 5), 8)
        tuned["valid_chain_fallback_topk"] = max(int(tuned.get("valid_chain_fallback_topk", 8) or 8), 12)
        tuned["valid_chain_fallback_beam_width"] = max(int(tuned.get("valid_chain_fallback_beam_width", 24) or 24), 40)
        tuned["valid_chain_fallback_branch"] = max(int(tuned.get("valid_chain_fallback_branch", 12) or 12), 16)
        tuned["per_entity_cap"] = max(int(tuned.get("per_entity_cap", 500) or 500), 700)
        tuned["max_total_context_edges"] = max(int(tuned.get("max_total_context_edges", 160) or 160), 220)
        tuned["max_expansion_edge_ratio"] = max(float(tuned.get("max_expansion_edge_ratio", 0.5) or 0.5), 0.75)
        tuned["low_confidence_min_valid_candidates"] = max(int(tuned.get("low_confidence_min_valid_candidates", 2) or 2), 3)

    elif args.dataset == "kqapro":
        # KQAPro currently fails mostly at no_candidates / no_valid_chain.
        tuned["num_candidates"] = max(int(tuned.get("num_candidates", 5) or 5), 10)
        tuned["valid_chain_fallback_topk"] = max(int(tuned.get("valid_chain_fallback_topk", 8) or 8), 16)
        tuned["valid_chain_fallback_beam_width"] = max(int(tuned.get("valid_chain_fallback_beam_width", 24) or 24), 64)
        tuned["valid_chain_fallback_branch"] = max(int(tuned.get("valid_chain_fallback_branch", 12) or 12), 20)
        tuned["per_entity_cap"] = max(int(tuned.get("per_entity_cap", 500) or 500), 800)
        tuned["max_frontier"] = max(int(tuned.get("max_frontier", 20000) or 20000), 30000)
        tuned["max_total_context_edges"] = max(int(tuned.get("max_total_context_edges", 160) or 160), 240)
        tuned["max_expansion_edge_ratio"] = max(float(tuned.get("max_expansion_edge_ratio", 0.5) or 0.5), 0.75)
        tuned["low_confidence_min_valid_candidates"] = max(int(tuned.get("low_confidence_min_valid_candidates", 2) or 2), 3)

    return tuned

def save_full_report(full_report: dict, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)


def append_failure_record(failure_file: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(failure_file), exist_ok=True)
    with open(failure_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def safe_clear_gpu_cache(model_name: str) -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(
            f"[Warning] GPU cache cleanup failed after {model_name}: {e}. "
            "This usually means an earlier CUDA kernel failed asynchronously; "
            "completed benchmark results were kept, but restarting the Python "
            "process may be required before more CUDA work."
        )
        return False
    print("[System] GPU Cache Cleared.")
    return True


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
    parser.add_argument("--mlpq-kb-mode", choices=["bilingual", "monolingual"], default="bilingual")
    parser.add_argument("--mlpq-kb-lang", choices=["auto", "en", "fr", "zh"], default="auto")
    parser.add_argument("--custom-dataset-name", default="custom", help="Portable tag used for custom dataset runs.")
    parser.add_argument("--custom-format", choices=["auto", "tsv", "jsonl"], default="auto")
    parser.add_argument("--custom-hop", type=int, default=1, help="Fallback hop value for custom datasets.")
    parser.add_argument("--model-filter", default=None, help="Only run models whose names contain this substring.")
    parser.add_argument("--sample-limit", type=int, default=TEST_SAMPLE_LIMIT)
    parser.add_argument("--kb-path", default=None)
    parser.add_argument("--relation-path", default=None)
    parser.add_argument("--alias-path", default=None)
    parser.add_argument("--grammar-path", default=None)
    parser.add_argument("--kb-ablation-mode", choices=["none", "drop_nodes", "drop_relations"], default="none")
    parser.add_argument("--kb-ablation-ratio", type=float, default=0.0)
    parser.add_argument("--kb-ablation-seed", type=int, default=0)
    parser.add_argument("--fixed-ablation-budget", action="store_true", help="Disable dataset-specific budget overrides for controlled ablations.")
    parser.add_argument("--retry-failed", action="store_true", help="Rerun models currently recorded as FAILED in benchmark_results.json.")
    parser.add_argument("--run-tag-suffix", default=None)
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
        return os.path.join(args.artifacts_root, run_tag, "grammar", "hrg_grammar.json")
    if args.dataset == "mlpq":
        return os.path.join(args.artifacts_root, run_tag, "grammar", "hrg_grammar.json")
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
        mlpq_kb_mode=args.mlpq_kb_mode,
        mlpq_kb_lang=args.mlpq_kb_lang,
        custom_dataset_name=args.custom_dataset_name,
        suffix=args.run_tag_suffix,
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


def prune_legacy_report_metrics(full_report: dict):
    for data in full_report.values():
        if not isinstance(data, dict) or data == "FAILED":
            continue
        for key in LEGACY_ANSWER_METRIC_KEYS:
            data.pop(key, None)
        for dataset_metrics in data.get("results", {}).values():
            if not isinstance(dataset_metrics, dict):
                continue
            for key in LEGACY_ANSWER_METRIC_KEYS:
                dataset_metrics.pop(key, None)


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


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def _normalize_numeric_token(text: str) -> str:
    token = (text or "").strip().lower()
    if not token:
        return ""
    token = token.replace(",", "")
    token = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", token)
    token = re.sub(r"\s+", " ", token).strip()
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    try:
        value = Decimal(token)
        if value == value.to_integral():
            return str(int(value))
        normalized = format(value.normalize(), "f").rstrip("0").rstrip(".")
        return normalized or "0"
    except InvalidOperation:
        return token


def normalize_answer(text: str, dataset: Optional[str] = None) -> str:
    text = (text or "").lower().strip()
    if dataset == "kqapro":
        text = text.replace(",", "")
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)
        text = re.sub(
            r"\b(" + "|".join(re.escape(k) for k in _NUMBER_WORDS) + r")\b",
            lambda m: _NUMBER_WORDS[m.group(1)],
            text,
        )
        compact_numeric = _normalize_numeric_token(text)
        if compact_numeric != text and re.fullmatch(r"[\d.\- ]+", compact_numeric):
            text = compact_numeric
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def split_candidate_answers(text: str, dataset: Optional[str] = None):
    raw = (text or "").strip()
    if not raw:
        return []

    parts = re.split(r"\s*\|\s*|\s*;\s*|\n+", raw)
    normalized = []
    seen = set()
    for part in parts:
        norm = normalize_answer(part, dataset=dataset)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)
    return normalized


def calculate_metrics(references, candidate, dataset: Optional[str] = None):
    candidate_norm = normalize_answer(candidate, dataset=dataset)
    ref_norms = [normalize_answer(ref, dataset=dataset) for ref in references if normalize_answer(ref, dataset=dataset)]
    candidate_parts = split_candidate_answers(candidate, dataset=dataset)
    candidate_set = set(candidate_parts)
    ref_set = set(ref_norms)

    overlap = len(candidate_set & ref_set)
    exact_match = 1.0 if ref_set and candidate_set == ref_set else 0.0
    if len(ref_norms) <= 1:
        exact_match = 1.0 if ref_norms and candidate_norm and candidate_norm == ref_norms[0] else 0.0

    answer_set_precision = (overlap / len(candidate_set)) if candidate_set else 0.0
    answer_set_recall = (overlap / len(ref_set)) if ref_set else 0.0
    if answer_set_precision + answer_set_recall > 0:
        answer_set_f1 = 2 * answer_set_precision * answer_set_recall / (answer_set_precision + answer_set_recall)
    else:
        answer_set_f1 = 0.0

    first_correct_rank = None
    for rank, candidate_answer in enumerate(candidate_parts, start=1):
        if candidate_answer in ref_set:
            first_correct_rank = rank
            break

    def hits_at(k: int) -> float:
        return 1.0 if first_correct_rank is not None and first_correct_rank <= k else 0.0

    return {
        "em": float(exact_match),
        "hits_at_1": hits_at(1),
        "hits_at_3": hits_at(3),
        "hits_at_5": hits_at(5),
        "mrr": float(1.0 / first_correct_rank) if first_correct_rank else 0.0,
        "answer_set_precision": float(answer_set_precision),
        "answer_set_recall": float(answer_set_recall),
        "answer_set_f1": float(answer_set_f1),
    }


def unpack_dataset_record(record):
    if isinstance(record, tuple):
        if len(record) >= 3:
            return record[0], record[1], record[2]
        if len(record) == 2:
            return record[0], record[1], {}
    raise ValueError(f"Unsupported dataset record format: {type(record)}")


def _edge_parts(edge):
    if isinstance(edge, dict):
        return edge.get("head", ""), edge.get("relation", ""), edge.get("tail", "")
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    return "", "", ""


def _compute_edge_answer_coverage(edges, references, dataset: Optional[str] = None) -> float:
    if not references or not edges:
        return 0.0

    nodes_in_subgraph = set()
    for edge in edges:
        h, _, t = _edge_parts(edge)
        for token in (h, t):
            norm = normalize_answer(token, dataset=dataset)
            if norm:
                nodes_in_subgraph.add(norm)

    if not nodes_in_subgraph:
        return 0.0

    hit_count = 0
    ref_norms = [normalize_answer(ref, dataset=dataset) for ref in references]
    for ref_norm in ref_norms:
        if not ref_norm:
            continue
        for node_norm in nodes_in_subgraph:
            if ref_norm in node_norm or node_norm in ref_norm:
                hit_count += 1
                break
    return hit_count / len(ref_norms) if ref_norms else 0.0


def calculate_retrieval_ranking_metrics(candidates, references, dataset: Optional[str] = None):
    if not references:
        return {key: None for key in RETRIEVAL_RANK_METRIC_KEYS}

    if not candidates:
        return {key: 0.0 for key in RETRIEVAL_RANK_METRIC_KEYS}

    relevances = []
    for cand in candidates:
        rel = cand.get("retrieval_recall")
        if rel is None:
            rel = _compute_edge_answer_coverage(cand.get("edges", []), references, dataset=dataset)
        relevances.append(float(rel or 0.0))

    positive_total = sum(1 for rel in relevances if rel > 0.0)

    def recall_at(k: int) -> float:
        if positive_total <= 0:
            return 0.0
        return sum(1 for rel in relevances[:k] if rel > 0.0) / positive_total

    def ndcg_at(k: int) -> float:
        topk = relevances[:k]
        if not topk:
            return 0.0
        dcg = sum(((2 ** rel) - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(topk))
        ideal = sorted(relevances, reverse=True)[:k]
        idcg = sum(((2 ** rel) - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(ideal))
        return (dcg / idcg) if idcg > 0 else 0.0

    return {
        "retrieval_recall_at_1": recall_at(1),
        "retrieval_recall_at_3": recall_at(3),
        "retrieval_recall_at_5": recall_at(5),
        "retrieval_ndcg_at_1": ndcg_at(1),
        "retrieval_ndcg_at_3": ndcg_at(3),
        "retrieval_ndcg_at_5": ndcg_at(5),
    }


def calculate_claim_metrics(candidate, edges, dataset: Optional[str] = None):
    claims = split_candidate_answers(candidate, dataset=dataset)
    if not claims or not edges:
        return {key: None for key in CLAIM_METRIC_KEYS}

    support_texts = set()
    for edge in edges:
        h, r, t = _edge_parts(edge)
        for item in (h, r, t, f"{h} {r} {t}"):
            norm = normalize_answer(item, dataset=dataset)
            if norm:
                support_texts.add(norm)

    if not support_texts:
        return {key: None for key in CLAIM_METRIC_KEYS}

    supported = 0
    for claim in claims:
        if any(claim in support or support in claim for support in support_texts):
            supported += 1

    faithfulness = supported / len(claims) if claims else None
    hallucination = (len(claims) - supported) / len(claims) if claims else None
    return {
        "claim_faithfulness": faithfulness,
        "claim_hallucination": hallucination,
    }


def _normalized_edge_tuple(head: str, relation: str, tail: str):
    return (
        normalize_kb_token(head),
        normalize_kb_token(relation),
        normalize_kb_token(tail),
    )


def calculate_evidence_metrics(details, metadata):
    gold_path_parts = (metadata or {}).get("gold_path_parts") or []
    if len(gold_path_parts) < 3:
        return {key: None for key in EVIDENCE_METRIC_KEYS}

    gold_edges = []
    gold_chain = []
    for i in range(0, len(gold_path_parts) - 2, 2):
        head = gold_path_parts[i]
        relation = gold_path_parts[i + 1]
        tail = gold_path_parts[i + 2]
        gold_edges.append(_normalized_edge_tuple(head, relation, tail))
        gold_chain.append(normalize_kb_token(relation))

    predicted_spine = details.get("spine_edges") or []
    predicted_edges = {
        _normalized_edge_tuple(*_edge_parts(edge))
        for edge in predicted_spine
    }
    gold_edge_set = set(gold_edges)
    overlap = len(predicted_edges & gold_edge_set)

    precision = (overlap / len(predicted_edges)) if predicted_edges else 0.0
    recall = (overlap / len(gold_edge_set)) if gold_edge_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    selected_chain = [normalize_kb_token(rel) for rel in (details.get("selected_chain") or [])]
    citation_correctness = 1.0 if selected_chain and selected_chain == gold_chain else 0.0

    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "citation_correctness": citation_correctness,
    }


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _binary_from_optional(value) -> Optional[float]:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return 1.0 if numeric > 0 else 0.0


def calculate_diagnostic_metrics(details, references, answer_metrics, dataset: Optional[str] = None):
    details = details or {}
    answer_metrics = answer_metrics or {}

    final_edges = details.get("edges") or []
    spine_edges = details.get("spine_edges") or []
    expanded_edges = details.get("expanded_edges") or []

    final_coverage = _compute_edge_answer_coverage(final_edges, references, dataset=dataset)
    spine_coverage = _compute_edge_answer_coverage(spine_edges, references, dataset=dataset)
    expanded_coverage = _compute_edge_answer_coverage(expanded_edges, references, dataset=dataset)

    answer_in_final = 1.0 if final_coverage > 0.0 else 0.0
    answer_in_spine = 1.0 if spine_coverage > 0.0 else 0.0
    answer_in_expanded = 1.0 if expanded_coverage > 0.0 else 0.0
    gold_only_in_expansion = 1.0 if answer_in_expanded and not answer_in_spine else 0.0

    answer_f1 = float(answer_metrics.get("answer_set_f1", 0.0) or 0.0)
    em = float(answer_metrics.get("em", 0.0) or 0.0)

    raw_count = _float_or_none(details.get("raw_final_edge_count"))
    if raw_count is None:
        raw_count = _float_or_none(details.get("raw_retrieved_edge_count"))
    if raw_count is None and details.get("raw_bfs_edges") is not None:
        raw_count = float(len(details.get("raw_bfs_edges") or []))
    if raw_count is None and details.get("raw_expanded_edge_count") is not None:
        raw_count = float(len(spine_edges) + int(details.get("raw_expanded_edge_count") or 0))
    if raw_count is None:
        raw_count = float(len(final_edges))

    final_count = float(len(final_edges))
    context_truncated = _binary_from_optional(details.get("context_truncated"))
    if context_truncated is None:
        context_truncated = 1.0 if raw_count > final_count else 0.0
    context_truncation_ratio = ((raw_count - final_count) / raw_count) if raw_count > 0 else 0.0
    context_truncation_ratio = max(0.0, context_truncation_ratio)

    selected_candidate = details.get("selected_candidate") or {}
    grammar_label = _binary_from_optional(selected_candidate.get("grammar_weak_label_match"))
    if grammar_label is None:
        grammar_label = _binary_from_optional(selected_candidate.get("grammar_hit"))
    if grammar_label is None:
        grammar_label = _binary_from_optional(details.get("grammar_hit"))

    grammar_ordered = _binary_from_optional(selected_candidate.get("grammar_ordered_path_hit"))
    grammar_arity = _binary_from_optional(selected_candidate.get("grammar_same_arity_hit"))
    grammar_structural_proxy = None
    if grammar_ordered is not None or grammar_arity is not None:
        grammar_structural_proxy = 1.0 if (grammar_ordered or 0.0) and (grammar_arity or 0.0) else 0.0
    grammar_full_structural = _binary_from_optional(selected_candidate.get("grammar_full_structural_hit"))

    return {
        "endpoint_coverage_final": final_coverage,
        "endpoint_coverage_spine": spine_coverage,
        "endpoint_coverage_expanded": expanded_coverage,
        "answer_in_final_context": answer_in_final,
        "answer_in_spine": answer_in_spine,
        "answer_in_expanded_edges": answer_in_expanded,
        "gold_only_in_expansion": gold_only_in_expansion,
        "em_when_answer_in_context": em if answer_in_final else None,
        "answer_f1_when_answer_in_context": answer_f1 if answer_in_final else None,
        "em_when_answer_not_in_context": em if not answer_in_final else None,
        "answer_f1_when_answer_not_in_context": answer_f1 if not answer_in_final else None,
        "answer_f1_when_answer_in_spine": answer_f1 if answer_in_spine else None,
        "answer_f1_when_answer_only_in_expansion": answer_f1 if gold_only_in_expansion else None,
        "answer_in_context_but_wrong": 1.0 if answer_in_final and answer_f1 <= 0.0 else 0.0,
        "answer_correct_without_context": 1.0 if (not answer_in_final and answer_f1 > 0.0) else 0.0,
        "final_edge_count": final_count,
        "spine_edge_count": float(len(spine_edges)),
        "expanded_edge_count": float(len(expanded_edges)),
        "raw_retrieved_edge_count": raw_count,
        "context_truncated": context_truncated,
        "context_truncation_ratio": context_truncation_ratio,
        "grammar_label_subset_hit": grammar_label,
        "grammar_ordered_path_hit": grammar_ordered,
        "grammar_arity_compatible_hit": grammar_arity,
        "grammar_structural_proxy_hit": grammar_structural_proxy,
        "grammar_full_structural_hit": grammar_full_structural,
    }


def average_metric(results, key):
    values = [r.get(key) for r in results if r.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


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
    dataset_key: Optional[str] = None,
    metadata: Optional[dict] = None,
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
                    "total_prepare_latency": prepared.get("total_prepare_latency", 0.0),
                    "generation_latency": answer_bundle.get("generation_latency", answer_bundle.get("elapsed", 0.0)),
                    "generation_failed": answer_bundle.get("generation_failed", False),
                    "answerable": bool((response or "").strip()),
                    "retrieval_policy": prepared.get("retrieval_policy", ""),
                    "expansion_gate": prepared.get("expansion_gate", ""),
                    "edges": prepared.get("edges", []),
                    "candidates": prepared.get("candidates", []),
                    "spine_edges": prepared.get("spine_edges", []),
                    "expanded_edges": prepared.get("expanded_edges", []),
                    "selected_candidate": prepared.get("selected_candidate", {}),
                    "selected_chain": prepared.get("selected_chain", []),
                    "selected_entity": prepared.get("selected_entity"),
                    "retrieval_recall": prepared.get("retrieval_recall", 0.0),
                    "retrieval_precision": prepared.get("retrieval_precision", 0.0),
                    "retrieval_f1": prepared.get("retrieval_f1", 0.0),
                    "subgraph_size": prepared.get("subgraph_size", 0),
                    "grammar_hit": prepared.get("grammar_hit", False),
                    "raw_final_edge_count": prepared.get("raw_final_edge_count"),
                    "raw_expanded_edge_count": prepared.get("raw_expanded_edge_count"),
                    "context_truncated": prepared.get("context_truncated"),
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
            metrics = calculate_metrics(references, response, dataset=dataset_key)
            retrieval_rank_metrics = calculate_retrieval_ranking_metrics(
                (details or {}).get("candidates", []),
                references,
                dataset=dataset_key,
            )
            claim_metrics = calculate_claim_metrics(
                response,
                (details or {}).get("edges", []),
                dataset=dataset_key,
            )
            evidence_metrics = calculate_evidence_metrics(details or {}, metadata or {})
            diagnostic_metrics = calculate_diagnostic_metrics(
                details or {},
                references,
                metrics,
                dataset=dataset_key,
            )
            pbar.update(1)

            payload = {
                "answer": response,
                **metrics,
                **retrieval_rank_metrics,
                **claim_metrics,
                **evidence_metrics,
                **diagnostic_metrics,
                "ranking_policy": RANKING_POLICY,
                "candidate_ranking_key": CANDIDATE_RANKING_KEY,
                "elapsed": float(elapsed),
                "failure_stage": (details or {}).get("failure_stage", "ok"),
                "parse_latency": float((details or {}).get("parse_latency", 0.0) or 0.0),
                "retrieval_latency": float((details or {}).get("retrieval_latency", 0.0) or 0.0),
                "total_prepare_latency": float((details or {}).get("total_prepare_latency", 0.0) or 0.0),
                "generation_latency": float((details or {}).get("generation_latency", 0.0) or 0.0),
                "generation_failed": bool((details or {}).get("generation_failed", False)),
                "answerable": bool((details or {}).get("answerable", bool((response or "").strip()))),
                "retrieval_policy": (details or {}).get("retrieval_policy", ""),
                "expansion_gate": (details or {}).get("expansion_gate", ""),
            }

            return {
                "idx": idx,
                "question": question,
                "expected_outputs": "|".join(references),
                "model_output": response,
                "model_payload": json.dumps(payload, ensure_ascii=False),
                **metrics,
                **retrieval_rank_metrics,
                **claim_metrics,
                **evidence_metrics,
                **diagnostic_metrics,
                "ranking_policy": RANKING_POLICY,
                "elapsed": float(elapsed),
                "failure_stage": payload["failure_stage"],
                "parse_latency": payload["parse_latency"],
                "retrieval_latency": payload["retrieval_latency"],
                "total_prepare_latency": payload["total_prepare_latency"],
                "generation_latency": payload["generation_latency"],
                "generation_failed": payload["generation_failed"],
                "answerable": payload["answerable"],
                "retrieval_policy": payload["retrieval_policy"],
                "expansion_gate": payload["expansion_gate"],
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
                "model_payload": json.dumps({
                    "answer": "",
                    **{key: 0.0 for key in ANSWER_METRIC_KEYS},
                    **{key: 0.0 for key in RETRIEVAL_RANK_METRIC_KEYS},
                    **{key: None for key in CLAIM_METRIC_KEYS},
                    **{key: None for key in EVIDENCE_METRIC_KEYS},
                    **{key: None for key in DIAGNOSTIC_METRIC_KEYS},
                    "ranking_policy": RANKING_POLICY,
                    "candidate_ranking_key": CANDIDATE_RANKING_KEY,
                    "elapsed": 0.0,
                    "error": err_str,
                    "failure_stage": failure_stage,
                    "parse_latency": 0.0,
                    "retrieval_latency": 0.0,
                    "total_prepare_latency": 0.0,
                    "generation_latency": 0.0,
                    "generation_failed": True,
                    "answerable": False,
                    "retrieval_policy": "",
                    "expansion_gate": "",
                }, ensure_ascii=False),
                **{key: 0.0 for key in ANSWER_METRIC_KEYS},
                **{key: 0.0 for key in RETRIEVAL_RANK_METRIC_KEYS},
                **{key: None for key in CLAIM_METRIC_KEYS},
                **{key: None for key in EVIDENCE_METRIC_KEYS},
                **{key: None for key in DIAGNOSTIC_METRIC_KEYS},
                "ranking_policy": RANKING_POLICY,
                "elapsed": 0.0,
                "error": err_str,
                "failure_stage": failure_stage,
                "parse_latency": 0.0,
                "retrieval_latency": 0.0,
                "total_prepare_latency": 0.0,
                "generation_latency": 0.0,
                "generation_failed": True,
                "answerable": False,
                "retrieval_policy": "",
                "expansion_gate": "",
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
    benchmark_dataset_name,
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
        for idx, record in enumerate(dataset):
            question, references, metadata = unpack_dataset_record(record)
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
                    dataset_key=benchmark_dataset_name,
                    metadata=metadata,
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
                "model_output": r.get("model_payload", r["model_output"]),
            })

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

        avg_metrics = {
            key: (sum(r.get(key, 0.0) for r in results) / count if count > 0 else 0.0)
            for key in ANSWER_METRIC_KEYS
        }
        avg_extra_metrics = {
            key: average_metric(results, key)
            for key in EXTRA_METRIC_KEYS
        }
        avg_time = total_time / count if count > 0 else 0

        model_results[dataset_name] = {
            **{key: round(value, 4) for key, value in avg_metrics.items()},
            **{
                key: (round(value, 4) if value is not None else None)
                for key, value in avg_extra_metrics.items()
            },
            "n_questions": count,
            "avg_latency": round(avg_time, 2),
        }

        tqdm.write(
            f"  ✅ [{dataset_name}] EM: {avg_metrics['em']:.4f} | "
            f"Hits@1: {avg_metrics['hits_at_1']:.4f} | "
            f"AnsF1: {avg_metrics['answer_set_f1']:.4f} | Avg Time: {avg_time:.2f}s"
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

    all_results = [r for ds in model_results.values() if isinstance(ds, dict) for r in [ds]]
    dataset_count = len(all_results)
    overall = {
        **{
            key: round(sum(d.get(key, 0.0) for d in all_results) / dataset_count, 4) if dataset_count else 0.0
            for key in ANSWER_METRIC_KEYS
        },
        **{
            key: (
                round(sum(d.get(key, 0.0) for d in all_results if d.get(key) is not None) / len([d for d in all_results if d.get(key) is not None]), 4)
                if any(d.get(key) is not None for d in all_results) else None
            )
            for key in EXTRA_METRIC_KEYS
        },
        "avg_latency": round(sum(d["avg_latency"] for d in all_results) / dataset_count, 2) if dataset_count else 0.0,
    }

    return {
        "results": model_results,
        **overall,
        "evaluation_unit": "question",
        "evaluated_question_count": total_eval_count,
        "sample_limit": sample_limit,
        "ranking_policy": RANKING_POLICY,
        "candidate_ranking_key": CANDIDATE_RANKING_KEY,
        "coverage": coverage,
        "avg_ctx_tokens": avg_ctx_tokens,
        "avg_parse1_tokens": avg_parse1_tokens,
        "avg_correction_tokens": avg_correction_tokens,
        "avg_parse2_tokens": avg_parse2_tokens,
        "avg_total_online_token_proxy": avg_parse1_tokens + avg_correction_tokens + avg_parse2_tokens + avg_ctx_tokens,
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
    if not 0.0 <= args.kb_ablation_ratio <= 1.0:
        raise ValueError(f"--kb-ablation-ratio must be within [0, 1], got {args.kb_ablation_ratio}")
    dataset_splits, hop_overrides = build_dataset_splits(args)
    run_tag = resolve_run_tag(args)
    output_file, detail_csv = resolve_output_paths(args, run_tag)
    failure_file = os.path.join(os.path.dirname(output_file), "failure_report.jsonl")
    dump_root = resolve_dump_root(args, run_tag)

    print("==========================================")
    print(f"  {args.dataset.upper()} Benchmark & Data Collection")
    print(f"  Run Tag: {run_tag}")
    print(f"  Experiment Suite: {EXPERIMENT_SUITE}")
    print(f"  Ranking Policy: {RANKING_POLICY}")
    if args.fixed_ablation_budget:
        print("  Budget Mode: fixed ablation budget")
    if args.kb_ablation_mode != "none" and args.kb_ablation_ratio > 0.0:
        print(f"  KG Ablation: mode={args.kb_ablation_mode} ratio={args.kb_ablation_ratio:.3f} seed={args.kb_ablation_seed}")
    print("==========================================")

    full_report = {}
    dataset_labels = list(dataset_splits.keys())

    common_kb_path = args.kb_path or (
        os.path.join(PROJECT_ROOT, "Datasets", "MetaQA", "kb.txt")
        if args.dataset == "metaqa"
        else (
            default_wikimovies_kb_path(args.wikimovies_subset)
            if args.dataset == "wikimovies"
            else (
                resolve_mlpq_kb_path(
                    args.dataset_root or DEFAULT_MLPQ_ROOT,
                    args.mlpq_pair,
                    args.mlpq_fusion,
                    kb_mode=args.mlpq_kb_mode,
                    kb_lang=None if args.mlpq_kb_lang == "auto" else args.mlpq_kb_lang,
                    question_lang=args.mlpq_question_lang,
                )
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
    common_alias_path = args.alias_path or (
        DEFAULT_MLPQ_ALIAS_PATH
        if args.dataset == "mlpq" and os.path.exists(DEFAULT_MLPQ_ALIAS_PATH)
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
        # Skip if already done. FAILED entries can be retried explicitly.
        if run_model_name in already_done:
            existing = full_report.get(run_model_name)
            if args.retry_failed and existing == "FAILED":
                print(f"[Retry] {run_model_name} 目前標記為 FAILED，依 --retry-failed 重跑。")
                full_report.pop(run_model_name, None)
                already_done.discard(run_model_name)
            else:
                print(f"[Skip] {run_model_name} 已有結果，跳過。")
                continue

        agent_instance = None
        tuned_agent_kwargs = None
        try:
            print(f"[System] Initializing {run_model_name} with class {agent_class.__name__}.")
            tuned_agent_kwargs = apply_dataset_agent_overrides(args, name, agent_class, agent_kwargs)
            if agent_class is KnowledgeGraphAgent and common_alias_path:
                tuned_agent_kwargs = dict(tuned_agent_kwargs)
                tuned_agent_kwargs["alias_path"] = common_alias_path
            init_kwargs = dict(tuned_agent_kwargs)
            init_kwargs.pop("shared_retrieval_group", None)
            init_kwargs["kb_path"] = common_kb_path
            init_kwargs["relation_path"] = common_relation_path
            init_kwargs["kb_ablation_mode"] = None if args.kb_ablation_mode == "none" else args.kb_ablation_mode
            init_kwargs["kb_ablation_ratio"] = args.kb_ablation_ratio
            init_kwargs["kb_ablation_seed"] = args.kb_ablation_seed
            if agent_class is KnowledgeGraphAgent and init_kwargs.get("grammar_path", "__DEFAULT__") is not None:
                init_kwargs["grammar_path"] = common_grammar_path
            agent_instance = agent_class(**init_kwargs)

            report = await evaluate_single_model(
                name,
                run_model_name,
                agent_instance,
                tuned_agent_kwargs,
                dataset_splits=dataset_splits,
                hop_overrides=hop_overrides,
                benchmark_dataset_name=args.dataset,
                sample_limit=args.sample_limit,
                run_tag=run_tag,
                dump_root=dump_root,
                is_baseline=is_baseline,
            )
            full_report[run_model_name] = report
            save_full_report(full_report, output_file)
            print(f"[Info] 已暫存結果至 {output_file}")

        except Exception as e:
            print(f"[Error] 模型 {run_model_name} 執行失敗: {e}")
            tb = traceback.format_exc()
            print(tb)
            full_report[run_model_name] = "FAILED"
            append_failure_record(failure_file, {
                "run_tag": run_tag,
                "dataset": args.dataset,
                "model_name": name,
                "run_model_name": run_model_name,
                "agent_class": agent_class.__name__,
                "is_baseline": is_baseline,
                "model_id": (tuned_agent_kwargs or agent_kwargs or {}).get("model_id"),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": tb,
                "experiment_suite": EXPERIMENT_SUITE,
                "ranking_policy": RANKING_POLICY,
                "candidate_ranking_key": CANDIDATE_RANKING_KEY,
                "fixed_ablation_budget": bool(args.fixed_ablation_budget),
                "kb_ablation_mode": args.kb_ablation_mode,
                "kb_ablation_ratio": args.kb_ablation_ratio,
                "kb_ablation_seed": args.kb_ablation_seed,
                "agent_kwargs": {
                    key: value
                    for key, value in dict(tuned_agent_kwargs or agent_kwargs or {}).items()
                    if key not in {"shared_retrieval_group"}
                },
            })
            save_full_report(full_report, output_file)
            print(f"[Info] 已暫存失敗狀態至 {output_file}")
            print(f"[Info] 失敗診斷已追加至 {failure_file}")

        finally:
            print(f"[System] Unloading {run_model_name} to free VRAM.")
            if agent_instance:
                del agent_instance

            gc.collect()
            safe_clear_gpu_cache(run_model_name)

            time.sleep(5)

    # ==========================================
    # 5. 輸出結果
    # ==========================================
    prune_legacy_report_metrics(full_report)
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
        print(f"{'':30} | {'Hits@1':<26} | {dataset_metric_line('hits_at_1')}")
        print(f"{'':30} | {'Hits@3':<26} | {dataset_metric_line('hits_at_3')}")
        print(f"{'':30} | {'Hits@5':<26} | {dataset_metric_line('hits_at_5')}")
        print(f"{'':30} | {'MRR':<26} | {dataset_metric_line('mrr')}")
        print(f"{'':30} | {'Answer-Set Precision':<26} | {dataset_metric_line('answer_set_precision')}")
        print(f"{'':30} | {'Answer-Set Recall':<26} | {dataset_metric_line('answer_set_recall')}")
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
            ans_ctx = data.get('answer_in_final_context')
            ans_spine = data.get('answer_in_spine')
            trunc = data.get('context_truncated')
            gold_exp = data.get('gold_only_in_expansion')

            print(f"{'':30} | {'─'*26} | {'─'*12} | {'─'*12} | {'─'*12}")
            print(f"{'':30} | {'Subgraph Recall':<26} | {recall*100:.2f}%")
            print(f"{'':30} | {'Subgraph Precision':<26} | {prec*100:.2f}%")
            print(f"{'':30} | {'Subgraph F1':<26} | {f1*100:.2f}%")
            print(f"{'':30} | {'Avg Subgraph Size (triples)':<26} | {subg:.1f}")
            if ans_ctx is not None:
                print(f"{'':30} | {'Answer in Final Context':<26} | {ans_ctx*100:.2f}%")
            if ans_spine is not None:
                print(f"{'':30} | {'Answer in Spine':<26} | {ans_spine*100:.2f}%")
            if gold_exp is not None:
                print(f"{'':30} | {'Gold Only in Expansion':<26} | {gold_exp*100:.2f}%")
            if trunc is not None:
                print(f"{'':30} | {'Context Truncated':<26} | {trunc*100:.2f}%")
            print(f"{'':30} | {'HRG Grammar Hit Rate':<26} | {cov*100:.2f}%")
            print(f"{'':30} | {'Avg Context Tokens':<26} | ~{int(ctx)}")
            print(f"{'':30} | {'Avg Parse1 Tokens':<26} | ~{int(p1)}")
            print(f"{'':30} | {'Avg Correction Tokens':<26} | ~{int(corr)}")
            print(f"{'':30} | {'Avg Parse2 Tokens':<26} | ~{int(p2)}")
            print(f"{'':30} | {'Total Avg Tokens':<26} | ~{int(p1+corr+p2)}")

        print(LINE)

    save_full_report(full_report, output_file)
    print(f"\n[Info] 詳細報告已儲存至 {output_file}")

    export_wide_csv_from_long_rows(ALL_LONG_ROWS, detail_csv)


if __name__ == "__main__":
    asyncio.run(main())
