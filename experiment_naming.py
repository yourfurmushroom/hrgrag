from __future__ import annotations

import os
import re
from typing import List


def build_run_tag(
    dataset: str,
    split: str | None = None,
    metaqa_variant: str = "vanilla",
    wikimovies_subset: str = "wiki_entities",
    mlpq_pair: str = "en-zh",
    mlpq_question_lang: str = "en",
    mlpq_fusion: str = "ills",
    mlpq_kb_mode: str = "bilingual",
    mlpq_kb_lang: str = "auto",
    custom_dataset_name: str = "custom",
    suffix: str = "",
) -> str:
    dataset = (dataset or "metaqa").lower()
    suffix = (suffix or "").strip().lower()
    suffix = re.sub(r"[^a-z0-9._-]+", "-", suffix) if suffix else ""
    if dataset == "metaqa":
        split = split or "dev"
        variant = metaqa_variant or "vanilla"
        base = f"metaqa-{variant}-{split}"
        return f"{base}-{suffix}" if suffix else base
    if dataset == "wikimovies":
        split = split or "train"
        subset = (wikimovies_subset or "wiki_entities").lower()
        base = f"wikimovies-{subset}-{split}"
        return f"{base}-{suffix}" if suffix else base
    if dataset in {"wqsp", "cwq", "kqapro", "mintaka"}:
        split = (split or "test").strip().lower()
        base = f"{dataset}-{split}"
        return f"{base}-{suffix}" if suffix else base
    if dataset == "custom":
        name = (custom_dataset_name or "custom").strip().lower()
        name = name.replace(" ", "-")
        split = (split or "test").strip().lower()
        base = f"custom-{name}-{split}"
        return f"{base}-{suffix}" if suffix else base
    pair = (mlpq_pair or "en-zh").lower()
    question_lang = (mlpq_question_lang or "en").lower()
    fusion = (mlpq_fusion or "ills").lower()
    kb_mode = (mlpq_kb_mode or "bilingual").lower()
    kb_lang = (mlpq_kb_lang or "auto").lower()
    if kb_mode == "monolingual":
        lang_suffix = question_lang if kb_lang == "auto" else kb_lang
        base = f"mlpq-{pair}-{question_lang}-{fusion}-mono-{lang_suffix}"
        return f"{base}-{suffix}" if suffix else base
    base = f"mlpq-{pair}-{question_lang}-{fusion}"
    return f"{base}-{suffix}" if suffix else base


def with_run_tag(name: str, run_tag: str) -> str:
    return f"{name}@{run_tag}"


def grammar_candidate_paths(project_root: str, run_tag: str, dataset: str) -> List[str]:
    candidates = [
        os.path.join(project_root, "artifacts", run_tag, "grammar", "hrg_grammar.json"),
        os.path.join(project_root, "hrg_grammar", "outputs", run_tag, "hrg_grammar.json"),
        os.path.join(project_root, "hrg_grammar", "outputs", run_tag, f"{run_tag}_phrg_grammar.json"),
        os.path.join(project_root, "HyperedgeReplacementGrammar", "outputs", run_tag, "hrg_grammar.json"),
        os.path.join(project_root, "HyperedgeReplacementGrammar", "outputs", run_tag, f"{run_tag}_phrg_grammar.json"),
    ]

    dataset = (dataset or "metaqa").lower()
    if dataset == "metaqa":
        candidates.extend([
            os.path.join(project_root, "hrg_grammar", "metaqa_phrg_grammar.json"),
            os.path.join(project_root, "HyperedgeReplacementGrammar", "hrg_grammar.json"),
        ])
    elif dataset == "wikimovies":
        candidates.extend([
            os.path.join(project_root, "hrg_grammar", "wikimovies_phrg_grammar.json"),
            os.path.join(project_root, "HyperedgeReplacementGrammar", "wikimovies_phrg_grammar.json"),
        ])
    elif dataset == "mlpq":
        candidates.extend([
            os.path.join(project_root, "hrg_grammar", "mlpq_phrg_grammar.json"),
            os.path.join(project_root, "HyperedgeReplacementGrammar", "mlpq_phrg_grammar.json"),
        ])

    return candidates
