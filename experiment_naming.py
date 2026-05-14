from __future__ import annotations

import os
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
) -> str:
    dataset = (dataset or "metaqa").lower()
    if dataset == "metaqa":
        split = split or "dev"
        variant = metaqa_variant or "vanilla"
        return f"metaqa-{variant}-{split}"
    if dataset == "wikimovies":
        split = split or "train"
        subset = (wikimovies_subset or "wiki_entities").lower()
        return f"wikimovies-{subset}-{split}"
    if dataset in {"wqsp", "cwq", "kqapro", "mintaka"}:
        split = (split or "test").strip().lower()
        return f"{dataset}-{split}"
    if dataset == "custom":
        name = (custom_dataset_name or "custom").strip().lower()
        name = name.replace(" ", "-")
        split = (split or "test").strip().lower()
        return f"custom-{name}-{split}"
    pair = (mlpq_pair or "en-zh").lower()
    question_lang = (mlpq_question_lang or "en").lower()
    fusion = (mlpq_fusion or "ills").lower()
    kb_mode = (mlpq_kb_mode or "bilingual").lower()
    kb_lang = (mlpq_kb_lang or "auto").lower()
    if kb_mode == "monolingual":
        lang_suffix = question_lang if kb_lang == "auto" else kb_lang
        return f"mlpq-{pair}-{question_lang}-{fusion}-mono-{lang_suffix}"
    return f"mlpq-{pair}-{question_lang}-{fusion}"


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
