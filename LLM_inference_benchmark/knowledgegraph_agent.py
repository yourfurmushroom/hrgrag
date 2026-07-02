from __future__ import annotations

import json
import re
import pickle
import time
import random
import hashlib
import math
from collections import Counter, defaultdict
from typing import Tuple, List, Optional, Dict, Any, Set, DefaultDict

from pathlib import Path

from agent_factory import build_llm_strategy
from dataset_utils import (
    build_node_index,
    load_alias_mapping,
    load_kb_adjacency,
    load_relation_list,
    normalize_lookup_key,
)


# ============================================================
# HRGMatcher
# ============================================================

class HRGMatcher:
    def __init__(self, grammar_path: str):
        print(f"[HRGMatcher] Loading grammar from: {grammar_path}")
        with open(grammar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "grammar" in data:
                self.grammar: List[Dict[str, Any]] = data["grammar"].get("production_rules", [])
            else:
                self.grammar: List[Dict[str, Any]] = data

        print(f"[HRGMatcher] Loaded {len(self.grammar)} rules.")

        self._sorted_rules: List[Dict[str, Any]] = []
        self._relation_path_prior_cache: Dict[int, Counter[Tuple[str, ...]]] = {}
        for rule in self.grammar:
            terminal_edges = self._extract_terminal_edges(rule)
            labels = [rel for _, rel, _ in terminal_edges if rel]
            rule["_cached_terminal_edges"] = terminal_edges
            rule["_cached_labels"] = set(labels)
            rule["_cached_label_counts"] = Counter(labels)
            rule["_cached_terminal_count"] = len(labels)

    def _extract_labels(self, rule: Dict[str, Any]) -> List[str]:
        return [rel for _, rel, _ in self._extract_terminal_edges(rule) if rel]

    def _extract_terminal_edges(self, rule: Dict[str, Any]) -> List[Tuple[Any, str, Any]]:
        rhs = rule.get("rhs", {})
        edges: List[Tuple[Any, str, Any]] = []
        if "terminal_edges" in rhs:
            for t in rhs["terminal_edges"]:
                rel = t.get("rel") or t.get("label")
                src = t.get("a", t.get("src", t.get("source", t.get("head"))))
                dst = t.get("b", t.get("dst", t.get("target", t.get("tail"))))
                if rel:
                    edges.append((src, rel, dst))
        elif "terminals" in rhs:
            for t in rhs["terminals"]:
                rel = t.get("rel") or t.get("label")
                src = t.get("a", t.get("src", t.get("source", t.get("head"))))
                dst = t.get("b", t.get("dst", t.get("target", t.get("tail"))))
                if rel:
                    edges.append((src, rel, dst))
        return edges

    def _chain_to_bare(self, chain: List[str]) -> List[str]:
        return list(chain)

    def _labels_contain_chain(self, rule: Dict[str, Any], chain: List[str]) -> bool:
        counts = rule.get("_cached_label_counts")
        if counts is None:
            counts = Counter(self._extract_labels(rule))
            rule["_cached_label_counts"] = counts
        chain_counts = Counter(self._chain_to_bare(chain))
        return all(counts.get(rel, 0) >= need for rel, need in chain_counts.items())

    def _rule_contains_ordered_path(self, rule: Dict[str, Any], chain: List[str]) -> bool:
        bare_chain = self._chain_to_bare(chain)
        if not bare_chain:
            return False

        terminal_edges = rule.get("_cached_terminal_edges")
        if terminal_edges is None:
            terminal_edges = self._extract_terminal_edges(rule)
            rule["_cached_terminal_edges"] = terminal_edges

        if len(bare_chain) == 1:
            return any(rel == bare_chain[0] for _, rel, _ in terminal_edges)

        adjacency: DefaultDict[Any, List[Tuple[str, Any]]] = defaultdict(list)
        for src, rel, dst in terminal_edges:
            if src is None or dst is None or not rel:
                continue
            adjacency[src].append((rel, dst))
            adjacency[dst].append((rel, src))

        if not adjacency:
            return False

        states = set(adjacency)
        for rel in bare_chain:
            next_states = set()
            for node in states:
                for edge_rel, other in adjacency.get(node, []):
                    if edge_rel == rel:
                        next_states.add(other)
            if not next_states:
                return False
            states = next_states
        return True

    def match_rules(
        self,
        chain: List[str],
        require_ordered_path: bool = False,
        require_exact_size: bool = False,
    ) -> List[Dict[str, Any]]:
        if not chain:
            return []
        matched = []

        for rule in self.grammar:
            terminal_count = int(rule.get("_cached_terminal_count", 0) or 0)
            if require_exact_size and terminal_count != len(chain):
                continue
            if require_ordered_path:
                if self._rule_contains_ordered_path(rule, chain):
                    matched.append(rule)
            elif self._labels_contain_chain(rule, chain):
                matched.append(rule)

        matched.sort(
            key=lambda r: (
                r.get("probability", r.get("count", 0)),
                -abs(int(r.get("_cached_terminal_count", 0) or 0) - len(chain)),
                -int(r.get("_cached_terminal_count", 0) or 0),
            ),
            reverse=True,
        )
        mode = "ordered" if require_ordered_path else "label"
        if require_exact_size:
            mode += "+exact"
        print(f"[HRGMatcher] chain={chain} -> mode={mode} -> matched {len(matched)} rules")
        return matched

    def summarize_matched(self, rules: List[Dict[str, Any]]) -> str:
        if not rules:
            return "No matching grammar rules found."
        lines = [f"Matched {len(rules)} grammar rule(s) (showing top 3):"]
        for i, r in enumerate(rules[:3]):
            labels = r.get("_cached_labels", set())
            prob = r.get("probability", r.get("count", 1))
            lhs = r.get("lhs", {})
            lhs_name = lhs.get("name") if isinstance(lhs, dict) else lhs
            lines.append(f"  [{i}] lhs={lhs_name} allowed_rels={list(labels)} score={prob}")
        return "\n".join(lines)

    def get_all_hints(self) -> str:
        self._sorted_rules = sorted(
            self.grammar,
            key=lambda r: r.get("probability", r.get("count", 0)),
            reverse=True
        )
        lines = ["All known semantic structures in this domain (sorted by frequency score):"]
        for i, r in enumerate(self._sorted_rules):
            labels = sorted(r.get("_cached_labels", set()))
            prob = r.get("probability", r.get("count", 1))
            lines.append(f"  pattern_id={i+1} (score={prob}): relations={labels}")
        return "\n".join(lines)

    def get_rule_by_id(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        if not hasattr(self, "_sorted_rules") or not self._sorted_rules:
            self.get_all_hints()
        idx = pattern_id - 1
        if 0 <= idx < len(self._sorted_rules):
            return self._sorted_rules[idx]
        return None

    def topk_rules(self, k: int = 10) -> List[Dict[str, Any]]:
        if not self._sorted_rules:
            self.get_all_hints()
        return self._sorted_rules[:k]

    def relation_path_prior(self, max_depth: int) -> Counter[Tuple[str, ...]]:
        """
        Build a grammar-derived path bank.

        The extracted HRG rules are local graph fragments, not gold QA paths. We
        therefore harvest relation sequences from both the terminal edge order
        and data-adaptively selected terminal-edge adjacency walks. This turns HRG into a
        candidate-space prior for grammar-first retrieval.
        """
        max_depth = max(1, int(max_depth or 1))
        if max_depth in self._relation_path_prior_cache:
            return self._relation_path_prior_cache[max_depth]

        prior: Counter[Tuple[str, ...]] = Counter()
        terminal_counts = [
            int(rule.get("_cached_terminal_count", 0) or 0)
            for rule in self.grammar
            if int(rule.get("_cached_terminal_count", 0) or 0) > 0
        ]
        structural_edge_limit = 0
        if terminal_counts:
            ordered_counts = sorted(terminal_counts)
            structural_edge_limit = ordered_counts[len(ordered_counts) // 2]

        for rule in self.grammar:
            terminal_edges = rule.get("_cached_terminal_edges")
            if terminal_edges is None:
                terminal_edges = self._extract_terminal_edges(rule)
                rule["_cached_terminal_edges"] = terminal_edges

            weight = rule.get("probability", rule.get("count", len(terminal_edges) or 1.0))
            try:
                weight = float(weight)
            except Exception:
                weight = float(len(terminal_edges) or 1.0)
            if weight <= 0:
                weight = float(len(terminal_edges) or 1.0)

            labels = [rel for _, rel, _ in terminal_edges if rel]
            for length in range(1, max_depth + 1):
                if len(labels) < length:
                    continue
                for start in range(0, len(labels) - length + 1):
                    seq = tuple(labels[start:start + length])
                    if all(seq):
                        prior[seq] += weight

            if not structural_edge_limit or len(terminal_edges) > structural_edge_limit:
                continue

            adjacency: DefaultDict[Any, List[Tuple[str, Any]]] = defaultdict(list)
            for src, rel, dst in terminal_edges:
                if src is None or dst is None or not rel:
                    continue
                adjacency[src].append((rel, dst))
                adjacency[dst].append((rel, src))
            if not adjacency:
                continue

            for node in adjacency:
                adjacency[node].sort(key=lambda x: (x[0], str(x[1])))

            structural_paths: Set[Tuple[str, ...]] = set()

            def dfs(node: Any, path: Tuple[str, ...], seen_nodes: Set[Any]) -> None:
                if path:
                    structural_paths.add(path)
                if len(path) >= max_depth:
                    return
                for rel, nxt in adjacency.get(node, []):
                    if nxt in seen_nodes:
                        continue
                    dfs(nxt, path + (rel,), seen_nodes | {nxt})

            for node in sorted(adjacency, key=str):
                dfs(node, tuple(), {node})

            for path in structural_paths:
                prior[path] += weight

        self._relation_path_prior_cache[max_depth] = prior
        print(
            f"[HRGMatcher] relation_path_prior depth<={max_depth} "
            f"contains {len(prior)} signatures."
        )
        return prior


# ============================================================
# KnowledgeGraphAgent
# ============================================================

class KnowledgeGraphAgent:
    STATEMENT_RELATION_SUFFIX = "::statement"
    LOW_INFORMATION_RELATIONS = {
        "name",
        "instanceOf",
        "fact_h",
        "fact_r",
        "fact_t",
    }
    QUESTION_ENTITY_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "by", "co", "for", "from", "got",
        "have", "has", "how", "in", "is", "it", "many", "of", "on", "or",
        "the", "to", "was", "were", "what", "when", "where", "which", "who",
        "whose", "with",
    }
    RANK_MATCHED_RULE_CAP = 20
    RANK_STEP_SURVIVAL_CAP_PER_HOP = 10
    RANK_FINAL_FRONTIER_CAP = 20
    CANDIDATE_SOURCE_DEFAULT_PRIORITY = 0
    CANDIDATE_SOURCE_PRIORITY = {
        "llm": 3,
        "llm_correction": 2,
        "kg_valid_fallback": 2,
        "grammar_first": 2,
        "grammar_fallback": 0,
    }
    CANDIDATE_FLIP_HOP_PRIORITY = 1
    GRAMMAR_FALLBACK_SAME_ARITY_BONUS = 5.0
    GRAMMAR_FALLBACK_ORDERED_PATH_BONUS = 3.0
    GRAMMAR_FALLBACK_LABEL_BONUS = 2.0
    GRAMMAR_FALLBACK_MATCHED_RULE_WEIGHT = 0.1
    GRAMMAR_FALLBACK_FRONTIER_COMPACTNESS_CAP = 100
    GRAMMAR_FALLBACK_FRONTIER_COMPACTNESS_WEIGHT = 0.01

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        kb_path: str = "../Datasets/MetaQA/kb.txt",
        relation_path: Optional[str] = "../Datasets/MetaQA/relations.json",
        grammar_path: Optional[str] = "../hrg_grammar/metaqa_phrg_grammar.json",
        max_new_tokens: int = 1024,
        use_model_sharding: bool = False,
        strict_gpu_sharding: bool = False,
        target_device: Optional[str] = None,
        max_kb_triples: Optional[int] = None,
        max_frontier: int = 20000,
        per_entity_cap: int = 500,
        serialization_format: str = "json",
        num_candidates: int = 5,
        use_grammar_rerank: bool = True,
        grammar_rerank_weight: float = 15.0,
        use_grammar_expansion: bool = True,
        use_fallback_correction: bool = True,
        topk_expansion_rules: int = 1,
        # ---- Ablation C: grammar structural hints injected into parse prompt ----
        use_grammar_hint: bool = False,
        grammar_hint_topk: int = 10,
        # ---- Ablation B: strict expansion filtering ----
        expansion_strict: bool = False,
        expansion_min_prob: float = 0.005,   # min rule probability/count to use
        expansion_per_node_cap: int = 10,    # max expansion edges added per node
        min_grammar_score_for_expansion: float = 0.0,
        max_spine_edges_for_expansion: Optional[int] = None,
        use_random_expansion: bool = False,
        use_frequency_expansion: bool = False,
        random_expansion_seed: int = 0,
        # ---- Ablation D: context top-K edge truncation ----
        top_k_edges: Optional[int] = None,
        # ---- HRG-D: grammar-guided constrained BFS retrieval (original design) ----
        use_grammar_guided_retrieval: bool = False,
        # ---- Grammar-first retrieval: enumerate KG-valid chains before LLM chain parsing ----
        use_grammar_first_retrieval: bool = False,
        # ---- Deterministic fallback: enumerate KG-valid chains before giving up ----
        use_deterministic_valid_chain_fallback: bool = False,
        valid_chain_fallback_topk: int = 8,
        valid_chain_fallback_beam_width: int = 24,
        valid_chain_fallback_branch: int = 12,
        valid_chain_fallback_max_depth: int = 3,
        use_valid_chain_llm_rerank: bool = True,
        # ---- Guide-style HRG prior: label-subset matching, not exact HRG decoding ----
        require_ordered_grammar_match: bool = False,
        require_exact_grammar_match_for_expansion: bool = False,
        max_expansion_edges: Optional[int] = None,
        max_expansion_edge_ratio: Optional[float] = None,
        max_total_context_edges: Optional[int] = None,
        use_low_confidence_valid_chain_fallback: bool = False,
        low_confidence_min_valid_candidates: int = 2,
        subgraph_support_saturation: int = 12,
        relation_ngram_prior_order: int = 0,
        relation_ngram_prior_alpha: float = 0.1,
        alias_path: Optional[str] = None,
        kb_ablation_mode: Optional[str] = None,
        kb_ablation_ratio: float = 0.0,
        kb_ablation_seed: int = 0,
    ):
        print(f"[Init] Loading LLM strategy for: {model_id} ...")
        self.llm = build_llm_strategy(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            use_model_sharding=use_model_sharding,
            strict_gpu_sharding=strict_gpu_sharding,
            target_device=target_device,
        )

        self.serialization_format = serialization_format
        self.num_candidates = num_candidates
        self.use_grammar_rerank = use_grammar_rerank
        self.grammar_rerank_weight = grammar_rerank_weight
        self.use_grammar_expansion = use_grammar_expansion
        self.use_fallback_correction = use_fallback_correction
        self.topk_expansion_rules = topk_expansion_rules

        # Ablation B/C/D flags
        self.use_grammar_hint = use_grammar_hint
        self.grammar_hint_topk = grammar_hint_topk
        self.expansion_strict = expansion_strict
        self.expansion_min_prob = expansion_min_prob
        self.expansion_per_node_cap = expansion_per_node_cap
        self.min_grammar_score_for_expansion = min_grammar_score_for_expansion
        self.max_spine_edges_for_expansion = max_spine_edges_for_expansion
        self.use_random_expansion = use_random_expansion
        self.use_frequency_expansion = use_frequency_expansion
        self.random_expansion_seed = random_expansion_seed
        self.top_k_edges = top_k_edges
        self.use_grammar_guided_retrieval = use_grammar_guided_retrieval
        self.use_grammar_first_retrieval = use_grammar_first_retrieval
        self.use_deterministic_valid_chain_fallback = use_deterministic_valid_chain_fallback
        self.valid_chain_fallback_topk = valid_chain_fallback_topk
        self.valid_chain_fallback_beam_width = valid_chain_fallback_beam_width
        self.valid_chain_fallback_branch = valid_chain_fallback_branch
        self.valid_chain_fallback_max_depth = valid_chain_fallback_max_depth
        self.use_valid_chain_llm_rerank = use_valid_chain_llm_rerank
        self.require_ordered_grammar_match = require_ordered_grammar_match
        self.require_exact_grammar_match_for_expansion = require_exact_grammar_match_for_expansion
        self.max_expansion_edges = max_expansion_edges
        self.max_expansion_edge_ratio = max_expansion_edge_ratio
        self.max_total_context_edges = max_total_context_edges
        self.use_low_confidence_valid_chain_fallback = use_low_confidence_valid_chain_fallback
        self.low_confidence_min_valid_candidates = low_confidence_min_valid_candidates
        self.subgraph_support_saturation = subgraph_support_saturation
        self.relation_ngram_prior_order = int(relation_ngram_prior_order or 0)
        self.relation_ngram_prior_alpha = float(relation_ngram_prior_alpha or 0.1)
        self._relation_ngram_prior: Optional[Dict[str, Any]] = None

        # ===== Metrics =====
        self.hit_grammar_count = 0
        self.total_questions = 0
        self.total_context_length = 0

        # token accounting
        self.total_parse1_tokens = 0   # candidate parsing
        self.total_parse2_tokens = 0   # final answer generation
        self.total_correction_tokens = 0  # fallback correction prompt cost

        # Retrieval Recall / Precision / F1 tracking
        self.total_retrieval_recall = 0.0
        self.total_retrieval_precision = 0.0
        self.total_retrieval_f1 = 0.0
        self.total_retrieval_questions = 0

        # Subgraph size (triple count)
        self.total_subgraph_size = 0

        self.kb_path = kb_path
        self.max_kb_triples = max_kb_triples
        self.max_frontier = max_frontier
        self.per_entity_cap = per_entity_cap
        self.alias_path = alias_path
        self.kb_ablation_mode = kb_ablation_mode
        self.kb_ablation_ratio = kb_ablation_ratio
        self.kb_ablation_seed = kb_ablation_seed

        print("[Init] Loading KB + building adjacency index...")
        self.kb_out, self.kb_in, self.all_nodes, derived_relations, self.alias_map = load_kb_adjacency(
            self.kb_path,
            max_triples=self.max_kb_triples,
            sanitize_entity_fn=self._sanitize_entity,
            ablation_mode=self.kb_ablation_mode,
            ablation_ratio=self.kb_ablation_ratio,
            ablation_seed=self.kb_ablation_seed,
        )
        self.extra_entity_aliases, self.relation_aliases = load_alias_mapping(alias_path)
        self._merge_entity_aliases(self.extra_entity_aliases)
        self.relations = self._load_relations(relation_path, derived_relations)
        self.allowed_rel_set = set(self.relations)
        self.allowed_rel_tokens = set(self.relations)
        self.relation_alias_index = self._build_relation_alias_index()
        self._node_index = build_node_index(self.all_nodes, self.alias_map)
        self.relation_frequency = self._compute_relation_frequency()

        if grammar_path and Path(grammar_path).exists():
            self.hrg = HRGMatcher(grammar_path)
        else:
            self.hrg = None
            print("[Init] No grammar path provided or file not found. HRG disabled.")
        self._relation_ngram_prior = self._build_relation_ngram_prior()

        print("[Init] Ready.")

    # ============================================================
    # Utils
    # ============================================================

    def _sanitize_entity(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = text.replace("[", "").replace("]", "")
        text = re.sub(r"\s+", "_", text)
        return text

    def _merge_entity_aliases(self, extra_aliases: Dict[str, Set[str]]) -> None:
        for canonical, aliases in (extra_aliases or {}).items():
            node = self._sanitize_entity(canonical)
            if not node:
                continue
            self.alias_map.setdefault(node, set()).update(
                alias for alias in aliases if alias and self._sanitize_entity(alias) != node
            )

    def _build_relation_alias_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}

        def add(alias: str, relation: str) -> None:
            normalized = self._normalize_match_text(alias)
            if normalized:
                index.setdefault(normalized, relation)
                index.setdefault(normalized.replace(" ", "_"), relation)
                index.setdefault(normalized.replace(" ", ""), relation)

        for rel in self.allowed_rel_tokens:
            add(rel, rel)
            add(rel.replace("_", " "), rel)
            for alias in self.relation_aliases.get(rel, set()):
                add(alias, rel)
        return index

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if isinstance(value, list):
                if not value:
                    return default
                value = value[0]
            return int(value)
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, list):
                if not value:
                    return default
                value = value[0]
            return float(value)
        except Exception:
            return default

    def _desanitize(self, text: str) -> str:
        return text.replace("_", " ")

    def _load_relations(self, relation_path: Optional[str], derived_relations: List[str]) -> List[str]:
        rels = load_relation_list(
            relation_path=relation_path,
            kb_path=self.kb_path,
            max_triples=self.max_kb_triples,
        )
        if rels:
            derived = set(derived_relations)
            rels = [rel for rel in rels if rel in derived]
        return rels or derived_relations

    def _compute_relation_frequency(self) -> Dict[str, int]:
        freq: Dict[str, int] = defaultdict(int)
        for head_rels in self.kb_out.values():
            for rel, tails in head_rels.items():
                freq[rel] += len(tails)
        return dict(freq)

    def _build_relation_ngram_prior(self) -> Optional[Dict[str, Any]]:
        if not self.hrg or self.relation_ngram_prior_order <= 0:
            return None

        unigrams: Counter[Tuple[str, ...]] = Counter()
        bigrams: Counter[Tuple[str, ...]] = Counter()
        trigrams: Counter[Tuple[str, ...]] = Counter()
        prefix_1: Counter[Tuple[str, ...]] = Counter()
        prefix_2: Counter[Tuple[str, ...]] = Counter()
        vocab: Set[str] = set()

        for rule in self.hrg.grammar:
            labels = [rel for _, rel, _ in self.hrg._extract_terminal_edges(rule) if rel]
            if not labels:
                continue
            weight = self._safe_float(rule.get("probability", rule.get("count", 1)), 1.0)
            for rel in labels:
                vocab.add(rel)
                unigrams[(rel,)] += weight
            for rel_a, rel_b in zip(labels, labels[1:]):
                bigrams[(rel_a, rel_b)] += weight
                prefix_1[(rel_a,)] += weight
            for rel_a, rel_b, rel_c in zip(labels, labels[1:], labels[2:]):
                trigrams[(rel_a, rel_b, rel_c)] += weight
                prefix_2[(rel_a, rel_b)] += weight

        if not unigrams:
            return None

        prior = {
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "prefix_1": prefix_1,
            "prefix_2": prefix_2,
            "vocab": vocab,
            "total_unigrams": sum(unigrams.values()),
        }
        print(
            "[RelationNgramPrior] "
            f"order={self.relation_ngram_prior_order} "
            f"vocab={len(vocab)} bigrams={len(bigrams)} trigrams={len(trigrams)}"
        )
        return prior

    def _relation_ngram_log_unigram(self, rel: str) -> float:
        prior = self._relation_ngram_prior
        if not prior:
            return 0.0
        alpha = self.relation_ngram_prior_alpha
        vocab_size = max(1, len(prior["vocab"]))
        return math.log(
            (prior["unigrams"].get((rel,), 0.0) + alpha)
            / (prior["total_unigrams"] + alpha * vocab_size)
        )

    def _relation_ngram_log_bigram(self, rel_a: str, rel_b: str) -> float:
        prior = self._relation_ngram_prior
        if not prior:
            return 0.0
        alpha = self.relation_ngram_prior_alpha
        vocab_size = max(1, len(prior["vocab"]))
        return math.log(
            (prior["bigrams"].get((rel_a, rel_b), 0.0) + alpha)
            / (prior["prefix_1"].get((rel_a,), 0.0) + alpha * vocab_size)
        )

    def _relation_ngram_log_trigram(self, rel_a: str, rel_b: str, rel_c: str) -> float:
        prior = self._relation_ngram_prior
        if not prior:
            return 0.0
        alpha = self.relation_ngram_prior_alpha
        vocab_size = max(1, len(prior["vocab"]))
        return math.log(
            (prior["trigrams"].get((rel_a, rel_b, rel_c), 0.0) + alpha)
            / (prior["prefix_2"].get((rel_a, rel_b), 0.0) + alpha * vocab_size)
        )

    def _score_relation_ngram_prior(self, chain: List[str]) -> float:
        if not self._relation_ngram_prior or self.relation_ngram_prior_order <= 0 or not chain:
            return 0.0

        order = self.relation_ngram_prior_order
        if order <= 1 or len(chain) == 1:
            vals = [self._relation_ngram_log_unigram(rel) for rel in chain]
        elif order == 2 or len(chain) == 2:
            vals = [self._relation_ngram_log_unigram(chain[0])]
            vals.extend(self._relation_ngram_log_bigram(a, b) for a, b in zip(chain, chain[1:]))
        else:
            vals = [
                self._relation_ngram_log_unigram(chain[0]),
                self._relation_ngram_log_bigram(chain[0], chain[1]),
            ]
            vals.extend(
                self._relation_ngram_log_trigram(a, b, c)
                for a, b, c in zip(chain, chain[1:], chain[2:])
            )
        return sum(vals) / len(vals) if vals else 0.0

    def _add_relation_ngram_feature(self, grammar_features: Dict[str, Any], chain: List[str]) -> Dict[str, Any]:
        if self.relation_ngram_prior_order <= 0:
            return grammar_features
        enriched = dict(grammar_features)
        enriched["relation_ngram_score"] = self._score_relation_ngram_prior(chain)
        enriched["relation_ngram_order"] = self.relation_ngram_prior_order
        return enriched

    async def _inference(self, developer_instruction: str, user_content: str) -> str:
        return await self.llm.inference(developer_instruction, user_content)

    def _extract_final_content(self, raw_text: str) -> str:
        pattern = r"<\|channel\|>final<\|message\|>\s*(.*?)(?=<\||$)"
        match = re.search(pattern, raw_text, re.DOTALL)
        text = match.group(1).strip() if match else raw_text.strip()
        return self._postprocess_answer(text)

    def _postprocess_answer(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        text = re.sub(r"<\|.*?\|>", " ", text)
        text = re.sub(r"^(System|Developer|User|Assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Final answer|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Based on .*?,\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^The answer is\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*[-*]\s*", "", text)
        text = re.sub(r"^\s*\d+\.\s*", "", text)
        text = text.strip().strip('"').strip("'").strip()

        if "\n" in text:
            lines = []
            for line in text.splitlines():
                line = re.sub(r"^\s*[-*]\s*", "", line).strip()
                line = re.sub(r"^\s*\d+\.\s*", "", line).strip()
                if line:
                    lines.append(line)
            if lines:
                text = " | ".join(lines)

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+\|\s+", " | ", text)

        if text.lower() in {
            "i dont know",
            "i do not know",
            "dont know",
            "do not know",
            "unknown",
        }:
            return "I don't know"

        if re.fullmatch(r"I don't know[.!]?", text, flags=re.IGNORECASE):
            return "I don't know"

        text = re.sub(r"[.。,，;；:：]+$", "", text).strip()
        return text

    def _estimate_tokens(self, *texts: str) -> int:
        return sum(len(t) for t in texts) // 4

    def _display_entity(self, node: str) -> str:
        aliases = sorted(self.alias_map.get(node, set()))
        if aliases:
            return aliases[0]
        return self._desanitize(node)

    def _candidate_lookup_keys(self, entity: str) -> List[str]:
        normalized = normalize_lookup_key(entity)
        return [
            entity.lower().strip(),
            entity.lower().replace(" ", "_").strip(),
            entity.lower().replace("_", " ").strip(),
            re.sub(r"[^\w\s\.]", "", entity).lower().replace(" ", "_"),
            normalized,
            normalized.replace(" ", "_"),
            normalized.replace(" ", ""),
        ]

    def _token_overlap_entity_fallback(self, entity: str, max_scan: int = 2000) -> Optional[str]:
        normalized = normalize_lookup_key(entity)
        query_tokens = [t for t in normalized.split() if len(t) >= 2]
        if not query_tokens:
            compact = normalized.replace(" ", "")
            return self._node_index.get(compact)

        candidates: List[Tuple[int, int, str, str]] = []
        scanned = 0
        seen_nodes: Set[str] = set()
        for alias_key, node in self._node_index.items():
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            scanned += 1
            if scanned > max_scan:
                break

            alias_norm = normalize_lookup_key(alias_key)
            if not alias_norm:
                continue
            alias_tokens = set(alias_norm.split())
            overlap = sum(1 for token in query_tokens if token in alias_tokens or token in alias_norm)
            if overlap <= 0:
                continue

            contains_bonus = 2 if normalized and (normalized in alias_norm or alias_norm in normalized) else 0
            exact_bonus = 4 if alias_norm == normalized else 0
            score = overlap * 10 + contains_bonus + exact_bonus
            candidates.append((score, len(alias_norm), node, alias_norm))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return candidates[0][2]

    def _relation_neighbors_for_entity(self, entity: Optional[str], max_relations: int = 48) -> List[str]:
        node = self._resolve_entity_to_kb(entity)
        if not node:
            return []

        relation_scores: Dict[str, int] = defaultdict(int)

        for rel, tails in self.kb_out.get(node, {}).items():
            if rel in self.allowed_rel_tokens:
                relation_scores[rel] += 100 + min(len(tails), 10)

        for rel, heads in self.kb_in.get(node, {}).items():
            if rel in self.allowed_rel_tokens:
                relation_scores[rel] += 100 + min(len(heads), 10)

        ranked = sorted(relation_scores.items(), key=lambda x: (-x[1], x[0]))
        return [rel for rel, _ in ranked[:max_relations]]

    def _resolve_entity_to_kb(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None

        sanitized = self._sanitize_entity(entity)
        if sanitized in self.all_nodes:
            return sanitized

        for key in self._candidate_lookup_keys(entity):
            if key in self._node_index:
                return self._node_index[key]

        return self._token_overlap_entity_fallback(entity)

    def _extract_entity_candidates_from_question(self, user_prompt: str, limit: int = 8) -> List[str]:
        """
        Best-effort entity linker for fallback-only retrieval.

        KQAPro questions are not bracketed like MetaQA, and parse failures often
        leave us with no entity at all. Build contiguous n-grams from the
        question and keep only phrases that resolve through the existing alias
        index, preferring longer mentions.
        """
        normalized = self._normalize_match_text(user_prompt)
        tokens = normalized.split()
        candidates: List[Tuple[int, int, str]] = []
        seen_nodes: Set[str] = set()

        max_n = min(8, len(tokens))
        for n in range(max_n, 0, -1):
            for start in range(0, len(tokens) - n + 1):
                phrase_tokens = tokens[start:start + n]
                if all(tok in self.QUESTION_ENTITY_STOPWORDS for tok in phrase_tokens):
                    continue
                if n == 1 and len(phrase_tokens[0]) < 4:
                    continue

                phrase = " ".join(phrase_tokens)
                lookup_keys = [
                    phrase,
                    phrase.replace(" ", "_"),
                    phrase.replace(" ", ""),
                ]
                node = None
                for key in lookup_keys:
                    if key in self._node_index:
                        node = self._node_index[key]
                        break
                if not node or node in seen_nodes:
                    continue

                seen_nodes.add(node)
                candidates.append((n, len(phrase), node))
                if len(candidates) >= limit * 2:
                    break
            if len(candidates) >= limit * 2:
                break

        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [node for _, _, node in candidates[:limit]]

    def _fuzzy_match_relation(self, raw: str) -> Optional[str]:
        if raw in self.allowed_rel_tokens:
            return raw
        norm = raw.strip().lower().replace(" ", "_")
        for tok in self.allowed_rel_tokens:
            tnorm = tok.lower().replace(" ", "_")
            if tnorm == norm:
                return tok
        alias_key = self._normalize_match_text(raw)
        if alias_key in self.relation_alias_index:
            return self.relation_alias_index[alias_key]
        compact_alias_key = alias_key.replace(" ", "")
        if compact_alias_key in self.relation_alias_index:
            return self.relation_alias_index[compact_alias_key]
        return None

    def _normalize_entity_from_question(self, user_prompt: str, entity: Optional[str]) -> Optional[str]:
        if entity and entity.strip():
            return entity.strip()
        m = re.search(r"\[(.*?)\]", user_prompt)
        if m:
            return m.group(1).strip()
        return entity

    def _chain_to_bare(self, chain: List[str]) -> List[str]:
        return list(chain)

    def _dedup_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for c in candidates:
            entity = (c.get("entity") or "").strip()
            chain = c.get("chain") or []
            key = (entity, tuple(chain))
            if key not in seen and entity and chain:
                seen.add(key)
                row = dict(c)
                row["entity"] = entity
                row["chain"] = chain
                row["source"] = c.get("source", "llm")
                row["confidence"] = float(c.get("confidence", 0.0) or 0.0)
                deduped.append(row)
        return deduped

    def _normalize_match_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("_", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _relation_match_text(self, rel_token: str) -> str:
        aliases = sorted(self.relation_aliases.get(rel_token, set()))
        return " ".join([rel_token, rel_token.replace("_", " "), *aliases])

    def _relation_alias_guide(self, relations: List[str], max_aliases: int = 5) -> str:
        guide: Dict[str, List[str]] = {}
        for rel in relations:
            aliases = [
                alias for alias in sorted(self.relation_aliases.get(rel, set()))
                if self._normalize_match_text(alias) != self._normalize_match_text(rel)
            ]
            if aliases:
                guide[rel] = aliases[:max_aliases]
        if not guide:
            return ""
        return (
            "\n[Relation Alias Guide]\n"
            "When a question uses an alias or translation, output the KG relation token on the left.\n"
            f"{json.dumps(guide, ensure_ascii=False)}\n"
        )

    def _select_relation_prompt_candidates(self, user_prompt: str, limit: int = 64) -> List[str]:
        if len(self.allowed_rel_tokens) <= limit:
            return sorted(self.allowed_rel_tokens)

        question_terms = set(self._normalize_match_text(user_prompt).split())
        scored: List[Tuple[int, str]] = []
        prioritized_entity = self._normalize_entity_from_question(user_prompt, None)

        for rel in self._relation_neighbors_for_entity(prioritized_entity, max_relations=min(limit, 48)):
            scored.append((40, rel))

        for rel in sorted(self.allowed_rel_tokens):
            rel_terms = set(self._normalize_match_text(self._relation_match_text(rel)).split())
            overlap = len(question_terms & rel_terms)
            score = overlap * 10
            rel_text = self._normalize_match_text(self._relation_match_text(rel))
            if any(len(term) >= 3 and term in rel_text for term in question_terms):
                score += 6
            if score > 0:
                scored.append((score, rel))

        if self.hrg:
            for rule in self.hrg.topk_rules(k=min(self.grammar_hint_topk, 10)):
                for label in sorted(rule.get("_cached_labels", set())):
                    if label in self.allowed_rel_tokens:
                        scored.append((5, label))

        shortlisted: List[str] = []
        seen = set()
        for _, rel in sorted(scored, key=lambda x: (-x[0], x[1])):
            if rel in seen:
                continue
            shortlisted.append(rel)
            seen.add(rel)
            if len(shortlisted) >= limit:
                return shortlisted

        for rel in sorted(self.allowed_rel_tokens):
            if rel in seen:
                continue
            shortlisted.append(rel)
            if len(shortlisted) >= limit:
                break
        return shortlisted

    def _extract_balanced_json_segments(self, text: str, opener: str, closer: str) -> List[str]:
        segments: List[str] = []
        depth = 0
        start = None
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == opener:
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    segments.append(text[start:idx + 1])
                    start = None
        return segments

    # ============================================================
    # Candidate Parsing
    # ============================================================

    def _extract_candidate_json_list(self, text: str) -> List[Dict[str, Any]]:
        """
        Try to extract a JSON list like:
        [
          {"entity": "...", "chain": [...]},
          ...
        ]
        Fallback: extract individual JSON objects.
        """
        # 1) Try balanced JSON arrays first
        for s in self._extract_balanced_json_segments(text, "[", "]"):
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict) and "chain" in d]
            except Exception:
                pass

        # 2) Fallback: extract balanced JSON objects one by one
        out = []
        for s in self._extract_balanced_json_segments(text, "{", "}"):
            try:
                d = json.loads(s)
                if isinstance(d, dict) and "chain" in d:
                    out.append(d)
            except Exception:
                continue
        return out

    async def _parse_intent_candidates(
        self,
        user_prompt: str,
        num_candidates: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Stage 1: Let LLM propose top-k candidate chains freely.
        Ablation C: if use_grammar_hint=True, inject top-k grammar patterns into the prompt.
        Returns (candidates, estimated_tokens)
        """
        if num_candidates is None:
            num_candidates = self.num_candidates

        allowed_rel_candidates = self._select_relation_prompt_candidates(user_prompt)
        allowed_rels_str = json.dumps(allowed_rel_candidates, ensure_ascii=False)
        relation_alias_guide = self._relation_alias_guide(allowed_rel_candidates)

        rel_semantics = (
            "\n[Relation Direction Guide]\n"
            "Each relation token names a KG relation type and may be traversed through actual KG adjacency.\n"
            "Use only bare relation names; do not add direction suffixes.\n"
        )

        few_shot = (
            "\n[Examples]\n"
            'Q: "What genre is [The Matrix]?"\n'
            '[{"entity": "The Matrix", "chain": ["has_genre"]}]\n\n'
            'Q: "What movies did [Tom Hanks] star in?"\n'
            '[{"entity": "Tom Hanks", "chain": ["starred_actors"]}]\n\n'
            'Q: "Who directed the films that [Tom Hanks] starred in?"\n'
            '[{"entity": "Tom Hanks", "chain": ["starred_actors", "directed_by"]}]\n\n'
            'Q: "What genres do the films written by the writer of [Cast Away] belong to?"\n'
            '[{"entity": "Cast Away", "chain": ["written_by", "written_by", "has_genre"]}]\n'
        )

        # ---- Ablation C: grammar structural hints ----
        grammar_hint_section = ""
        use_structural_hints = self.hrg and (self.use_grammar_hint or self.use_grammar_rerank or self.use_grammar_expansion)
        if use_structural_hints:
            top_rules = self.hrg.topk_rules(k=self.grammar_hint_topk)
            hint_lines = ["\n[Grammar Structural Hints]",
                          "Top frequent relation co-occurrence patterns learned from the KG:"]
            for i, rule in enumerate(top_rules, 1):
                labels = sorted(rule.get("_cached_labels", set()))
                score = rule.get("probability", rule.get("count", 0))
                hint_lines.append(f"  pattern_{i} (score={score:.4f}): {labels}")
            hint_lines.append("Use these patterns as structural priors when proposing chains.")
            grammar_hint_section = "\n".join(hint_lines) + "\n"

        developer_instruction = (
            "You are a Knowledge Graph query parser.\n"
            "Your job is to propose the TOP-K most plausible candidate relation chains for the question.\n"
            "Extract:\n"
            "  1. entity: the topic entity (usually inside [...])\n"
            "  2. chain: an ordered relation chain for multi-hop KG traversal\n"
            "  3. confidence: a float in [0, 1] for how plausible the chain is\n"
            "Rules:\n"
            f"  - Prefer relations from this candidate set: {allowed_rels_str}\n"
            "  - If the alias guide maps an English/French/Chinese phrase to a KG token, output the KG token.\n"
            "  - If needed, you may use another KG relation token only when clearly more accurate.\n"
            "  - Use only bare relation names.\n"
            "  - 1-hop question -> 1 relation, 2-hop question -> 2 relations, 3-hop question -> 3 relations.\n"
            "  - Prefer structurally coherent chains that are likely to recur in the KG.\n"
            "  - Return multiple diverse candidates ranked from most likely to less likely.\n"
            "  - Do NOT use markdown fences or explanations.\n"
            f"{rel_semantics}"
            f"{relation_alias_guide}"
            f"{grammar_hint_section}"
            f"{few_shot}\n"
            "[Output]\n"
            f"Return ONLY a valid JSON array of up to {num_candidates} objects:\n"
            '[{"entity": "<topic entity>", "chain": ["<rel1>", "<rel2>", ...], "confidence": 0.0}, ...]'
        )

        parse1_input_tokens = self._estimate_tokens(developer_instruction, user_prompt)
        raw = await self._inference(developer_instruction, user_prompt)
        clean = self._extract_final_content(raw)
        parse1_total_tokens = parse1_input_tokens + self._estimate_tokens(clean)

        raw_candidates = self._extract_candidate_json_list(clean)
        candidates = []

        for item in raw_candidates[:num_candidates]:
            entity = self._normalize_entity_from_question(user_prompt, item.get("entity"))
            raw_chain = item.get("chain", [])
            chain = []
            for c in raw_chain:
                if not isinstance(c, str):
                    continue
                matched = self._fuzzy_match_relation(c)
                if matched:
                    chain.append(matched)
                else:
                    print(f"[Parse1] Dropping invalid relation: {c}")

            if entity and chain:
                confidence = max(0.0, min(1.0, self._safe_float(item.get("confidence", 0.0), 0.0)))
                candidates.append({
                    "entity": entity,
                    "chain": chain,
                    "source": item.get("source", "llm"),
                    "confidence": confidence,
                })

        # fallback if parser fails
        if not candidates:
            entity = self._normalize_entity_from_question(user_prompt, None)
            if entity:
                candidates = [{"entity": entity, "chain": [], "source": "llm", "confidence": 0.0}]

        candidates = self._dedup_candidates(candidates)
        return candidates, parse1_total_tokens

    # ============================================================
    # KB Traversal / Validity
    # ============================================================

    @staticmethod
    def _other_endpoint(current: str, head: str, tail: str) -> str:
        return tail if head == current else head

    def _neighbors_for_token(self, ent: str, rel_token: str) -> List[Tuple[str, str, str]]:
        rel = rel_token
        edges: List[Tuple[str, str, str]] = []

        tails = list(self.kb_out.get(ent, {}).get(rel, []))
        heads = list(self.kb_in.get(ent, {}).get(rel, []))

        for tail in tails:
            edges.append((ent, rel, tail))
            if self.per_entity_cap and len(edges) >= self.per_entity_cap:
                return edges

        for head in heads:
            edges.append((head, rel, ent))
            if self.per_entity_cap and len(edges) >= self.per_entity_cap:
                return edges

        return edges

    def _statement_aware_chain_variants(self, chain: List[str]) -> List[List[str]]:
        """
        Qualifier-bearing KQAPro facts are reached through statement nodes.

        The parser will often emit the ordinary fact predicate followed by the
        qualifier predicate. Try the predicate-specific statement anchor before
        rejecting that otherwise meaningful chain.
        """
        variants = [list(chain)]
        seen = {tuple(chain)}

        for idx, rel in enumerate(chain[:-1]):
            if rel.endswith(self.STATEMENT_RELATION_SUFFIX):
                continue
            statement_rel = f"{rel}{self.STATEMENT_RELATION_SUFFIX}"
            if statement_rel not in self.allowed_rel_tokens:
                continue
            variant = list(chain)
            variant[idx] = statement_rel
            key = tuple(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)

        return variants

    def _check_chain_validity_exact(self, entity: str, chain: List[str]) -> Dict[str, Any]:
        start_entity = self._resolve_entity_to_kb(entity)
        if not start_entity or not chain:
            return {
                "valid": False,
                "step_sizes": [],
                "final_size": 0,
                "failed_hop": 0,
            }

        frontier: Set[str] = {start_entity}
        step_sizes = []

        for hop_idx, rel in enumerate(chain, start=1):
            next_frontier: Set[str] = set()
            if len(frontier) > self.max_frontier:
                frontier = set(list(frontier)[: self.max_frontier])

            for ent in frontier:
                edges = self._neighbors_for_token(ent, rel)
                for h, r, t in edges:
                    next_frontier.add(self._other_endpoint(ent, h, t))

            step_sizes.append(len(next_frontier))
            if not next_frontier:
                return {
                    "valid": False,
                    "step_sizes": step_sizes,
                    "final_size": 0,
                    "failed_hop": hop_idx,
                }
            frontier = next_frontier

        return {
            "valid": True,
            "step_sizes": step_sizes,
            "final_size": len(frontier),
            "failed_hop": None,
        }

    def _check_chain_validity(self, entity: str, chain: List[str]) -> Dict[str, Any]:
        """
        Returns whether the chain is executable from the entity in the KB.
        """
        best_result = None
        best_progress = (-1, -1, -1)

        for variant in self._statement_aware_chain_variants(chain):
            result = self._check_chain_validity_exact(entity, variant)
            result["resolved_chain"] = variant
            if result["valid"]:
                return result

            step_sizes = result.get("step_sizes", [])
            progress = (
                self._safe_int(result.get("failed_hop"), 0),
                len(step_sizes),
                sum(self._safe_int(size, 0) for size in step_sizes),
            )
            if progress > best_progress:
                best_result = result
                best_progress = progress

        return best_result or {
            "valid": False,
            "step_sizes": [],
            "final_size": 0,
            "failed_hop": 0,
            "resolved_chain": list(chain),
        }

    def _find_subgraph_multi_hop_kb_strict(self, start_entity: str, relation_chain: List[str]) -> List[Dict[str, Any]]:
        frontier: Set[str] = {start_entity}
        edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

        for hop, rel_token in enumerate(relation_chain, 1):
            next_frontier: Set[str] = set()
            if len(frontier) > self.max_frontier:
                frontier = set(list(frontier)[: self.max_frontier])

            for ent in frontier:
                edges = self._neighbors_for_token(ent, rel_token)
                for h, r, t in edges:
                    edge_counts[(h, r, t)] += 1
                    next_frontier.add(self._other_endpoint(ent, h, t))

            if not next_frontier:
                break
            frontier = next_frontier

        return [
            {"head": self._display_entity(h), "relation": r, "tail": self._display_entity(t), "count": c}
            for (h, r, t), c in edge_counts.items()
        ]

    # ============================================================
    # Grammar Scoring / Expansion / Correction
    # ============================================================

    def _find_subgraph_grammar_guided(
        self,
        start_entity: str,
        chain: List[str],
        matched_rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        HRG-D: Grammar-Guided Constrained BFS Retrieval.

        Instead of following exactly one LLM-predicted chain, the grammar rule
        defines the semantic scope of relevant relations. We do a BFS constrained
        to grammar-allowed relations up to depth = len(chain).

        This realises the original design intent:
          LLM chain → select grammar rule → grammar defines retrieval space
        """
        if not matched_rules:
            # No grammar match → fallback to strict chain traversal
            print("[HRG-D] No grammar match, falling back to strict chain traversal.", flush=True)
            return self._find_subgraph_multi_hop_kb_strict(start_entity, chain)

        # Collect all allowed relations from top matched rules
        allowed_rels: Set[str] = set()
        for rule in matched_rules[:self.topk_expansion_rules]:
            allowed_rels |= rule.get("_cached_labels", set())

        if not allowed_rels:
            print("[HRG-D] Grammar rule has no labels, falling back to strict chain traversal.", flush=True)
            return self._find_subgraph_multi_hop_kb_strict(start_entity, chain)

        print(f"[HRG-D] Grammar allowed_rels={sorted(allowed_rels)}, depth={len(chain)}", flush=True)

        max_depth = len(chain)
        frontier: Set[str] = {start_entity}
        edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        visited_edges: Set[Tuple[str, str, str]] = set()
        visited_nodes: Set[str] = {start_entity}

        for depth in range(max_depth):
            next_frontier: Set[str] = set()
            if len(frontier) > self.max_frontier:
                frontier = set(list(frontier)[:self.max_frontier])

            for ent in frontier:
                node_edge_count = 0
                # Use expansion_per_node_cap if strict, else per_entity_cap
                cap = self.expansion_per_node_cap if self.expansion_strict else self.per_entity_cap

                # Forward edges
                for rel, tails in self.kb_out.get(ent, {}).items():
                    if rel in allowed_rels:
                        for tail in tails:
                            if cap and node_edge_count >= cap:
                                break
                            edge = (ent, rel, tail)
                            if edge not in visited_edges:
                                visited_edges.add(edge)
                                edge_counts[edge] += 1
                                if tail not in visited_nodes:
                                    next_frontier.add(tail)
                                node_edge_count += 1
                        if cap and node_edge_count >= cap:
                            break

                # Backward edges (inverse traversal)
                if not (cap and node_edge_count >= cap):
                    for rel, heads in self.kb_in.get(ent, {}).items():
                        if rel in allowed_rels:
                            for head in heads:
                                if cap and node_edge_count >= cap:
                                    break
                                edge = (head, rel, ent)
                                if edge not in visited_edges:
                                    visited_edges.add(edge)
                                    edge_counts[edge] += 1
                                    if head not in visited_nodes:
                                        next_frontier.add(head)
                                    node_edge_count += 1
                            if cap and node_edge_count >= cap:
                                break

            visited_nodes.update(next_frontier)
            if not next_frontier:
                break
            frontier = next_frontier

        result = [
            {"head": self._desanitize(h), "relation": r, "tail": self._desanitize(t), "count": c}
            for (h, r, t), c in edge_counts.items()
        ]
        print(f"[HRG-D] Retrieved {len(result)} edges via grammar-guided BFS.", flush=True)
        return result

    def _empty_grammar_features(self) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "hit": 0,
            "same_arity_hit": 0,
            "ordered_path_hit": 0,
            "weak_label_match": 0,
            "matched_count": 0,
            "relation_ngram_score": 0.0,
            "relation_ngram_order": 0,
        }

    def _rule_terminal_count(self, rule: Dict[str, Any]) -> int:
        return self._safe_int(rule.get("_cached_terminal_count", 0), 0)

    def _get_grammar_match_features(self, chain: List[str]) -> Dict[str, Any]:
        if not self.hrg or not chain:
            return self._empty_grammar_features()

        label_rules = self.hrg.match_rules(chain)
        exact_label_rules = self.hrg.match_rules(
            chain,
            require_exact_size=True,
        )
        exact_ordered_rules = self.hrg.match_rules(
            chain,
            require_ordered_path=True,
            require_exact_size=True,
        )
        ordered_rules = exact_ordered_rules or self.hrg.match_rules(
            chain,
            require_ordered_path=True,
        )

        if not label_rules:
            return self._empty_grammar_features()

        if self.require_ordered_grammar_match and not ordered_rules:
            features = self._empty_grammar_features()
            features["weak_label_match"] = 1
            features["matched_count"] = len(label_rules)
            return features

        if self.require_ordered_grammar_match:
            preferred_rules = exact_ordered_rules or ordered_rules
            same_arity_hit = 1 if exact_ordered_rules else 0
        else:
            preferred_rules = exact_label_rules or label_rules
            same_arity_hit = 1 if exact_label_rules else 0

        preferred = preferred_rules[0]
        return {
            "score": float(preferred.get("probability", preferred.get("count", 0))),
            "hit": 1,
            "same_arity_hit": same_arity_hit,
            "ordered_path_hit": 1 if ordered_rules else 0,
            "weak_label_match": 1,
            "matched_count": len(preferred_rules),
        }

    def _select_matched_rules(
        self,
        chain: List[str],
        top_k: Optional[int] = None,
        require_same_arity: bool = True,
        require_ordered_path: Optional[bool] = None,
        require_exact_size: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Select the most relevant HRG rules for a predicted relation chain.

        We prefer rules whose arity matches the chain length so that grammar is
        used as a retrieval constraint around the predicted path, rather than as
        a broad high-frequency neighborhood prior.
        """
        if not self.hrg or not chain:
            return []

        if require_ordered_path is None:
            require_ordered_path = self.require_ordered_grammar_match

        matched_rules = self.hrg.match_rules(
            chain,
            require_ordered_path=require_ordered_path,
            require_exact_size=require_exact_size,
        )
        if not matched_rules:
            return []

        if top_k is None:
            top_k = self.topk_expansion_rules

        if require_same_arity:
            exact_rules = [
                rule for rule in matched_rules
                if self._rule_terminal_count(rule) == len(chain)
            ]
            if exact_rules:
                matched_rules = exact_rules

        return matched_rules[:top_k]

    def _apply_expansion_budget(
        self,
        spine_edges: List[Dict[str, Any]],
        expanded_edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not expanded_edges:
            return []

        budget = len(expanded_edges)
        if self.max_expansion_edges is not None:
            budget = min(budget, max(0, int(self.max_expansion_edges)))
        if self.max_expansion_edge_ratio is not None:
            ratio = max(0.0, float(self.max_expansion_edge_ratio))
            ratio_budget = int(len(spine_edges) * ratio)
            if ratio > 0 and spine_edges:
                ratio_budget = max(1, ratio_budget)
            budget = min(budget, ratio_budget)
        if self.max_total_context_edges is not None:
            budget = min(budget, max(0, int(self.max_total_context_edges) - len(spine_edges)))

        if budget <= 0:
            return []
        return expanded_edges[:budget]

    def _expand_subgraph_by_grammar(
        self,
        spine_edges: List[Dict[str, Any]],
        matched_rules: List[Dict[str, Any]],
        chain: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Grammar-guided expansion of the spine subgraph.

        Ablation B (expansion_strict=True):
          - Only use rules whose arity matches the chain length
          - Only use rules above expansion_min_prob threshold
          - Cap expansion edges per node at expansion_per_node_cap
        """
        if not matched_rules:
            return []

        best_rules = matched_rules[: self.topk_expansion_rules]
        # ---- Ablation B: filter rules by arity and probability ----
        if self.expansion_strict and chain is not None:
            chain_len = len(chain)
            filtered = []
            for r in best_rules:
                arity = self._rule_terminal_count(r)
                prob = r.get("probability", r.get("count", 0))
                if arity == chain_len and prob >= self.expansion_min_prob:
                    filtered.append(r)
            if filtered:
                best_rules = filtered

        allowed_rels = set()
        for r in best_rules:
            allowed_rels |= set(r.get("_cached_labels", set()))

        spine_nodes = set()
        for e in spine_edges:
            spine_nodes.add(e["head"])
            spine_nodes.add(e["tail"])

        if not spine_nodes:
            return []

        expanded_edges = []
        visited = set((e["head"], e["relation"], e["tail"]) for e in spine_edges)

        for ent in spine_nodes:
            sanitized_ent = self._resolve_entity_to_kb(ent)
            if not sanitized_ent:
                continue
            node_edge_count = 0  # for expansion_per_node_cap (Ablation B)

            for rel in self.kb_out.get(sanitized_ent, {}).keys():
                if rel in allowed_rels:
                    for t in self.kb_out[sanitized_ent][rel]:
                        edge_tuple = (ent, rel, self._desanitize(t))
                        if edge_tuple not in visited:
                            visited.add(edge_tuple)
                            expanded_edges.append({
                                "head": ent,
                                "relation": rel,
                                "tail": self._display_entity(t),
                                "count": 1
                            })
                            node_edge_count += 1
                            if self.expansion_strict and node_edge_count >= self.expansion_per_node_cap:
                                break
                if self.expansion_strict and node_edge_count >= self.expansion_per_node_cap:
                    break

            if not (self.expansion_strict and node_edge_count >= self.expansion_per_node_cap):
                for rel in self.kb_in.get(sanitized_ent, {}).keys():
                    if rel in allowed_rels:
                        for h in self.kb_in[sanitized_ent][rel]:
                            edge_tuple = (self._display_entity(h), rel, ent)
                            if edge_tuple not in visited:
                                visited.add(edge_tuple)
                                expanded_edges.append({
                                    "head": self._display_entity(h),
                                    "relation": rel,
                                    "tail": ent,
                                    "count": 1
                                })
                                node_edge_count += 1
                                if self.expansion_strict and node_edge_count >= self.expansion_per_node_cap:
                                    break
                        if self.expansion_strict and node_edge_count >= self.expansion_per_node_cap:
                            break

        if len(expanded_edges) > self.max_frontier:
            expanded_edges = expanded_edges[: self.max_frontier]
        return expanded_edges

    def _expand_subgraph_random(
        self,
        spine_edges: List[Dict[str, Any]],
        chain: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        spine_nodes = set()
        for e in spine_edges:
            spine_nodes.add(e["head"])
            spine_nodes.add(e["tail"])

        if not spine_nodes:
            return []

        chain_key = " | ".join(chain or [])
        seed_text = f"{self.random_expansion_seed}::{chain_key}::{sorted(spine_nodes)}"
        seed_int = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)

        expanded_edges = []
        visited = set((e["head"], e["relation"], e["tail"]) for e in spine_edges)

        for ent in sorted(spine_nodes):
            sanitized_ent = self._resolve_entity_to_kb(ent)
            if not sanitized_ent:
                continue

            candidate_edges = []
            for rel, tails in self.kb_out.get(sanitized_ent, {}).items():
                for t in tails:
                    edge = {
                        "head": ent,
                        "relation": rel,
                        "tail": self._display_entity(t),
                        "count": 1,
                    }
                    edge_tuple = (edge["head"], edge["relation"], edge["tail"])
                    if edge_tuple not in visited:
                        candidate_edges.append((edge_tuple, edge))

            for rel, heads in self.kb_in.get(sanitized_ent, {}).items():
                for h in heads:
                    edge = {
                        "head": self._display_entity(h),
                        "relation": rel,
                        "tail": ent,
                        "count": 1,
                    }
                    edge_tuple = (edge["head"], edge["relation"], edge["tail"])
                    if edge_tuple not in visited:
                        candidate_edges.append((edge_tuple, edge))

            if not candidate_edges:
                continue

            rng.shuffle(candidate_edges)
            selected = candidate_edges[: self.expansion_per_node_cap]
            for edge_tuple, edge in selected:
                visited.add(edge_tuple)
                expanded_edges.append(edge)

        if len(expanded_edges) > self.max_frontier:
            expanded_edges = expanded_edges[: self.max_frontier]
        return expanded_edges

    def _expand_subgraph_by_relation_frequency(
        self,
        spine_edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        spine_nodes = set()
        for e in spine_edges:
            spine_nodes.add(e["head"])
            spine_nodes.add(e["tail"])

        if not spine_nodes:
            return []

        expanded_edges = []
        visited = set((e["head"], e["relation"], e["tail"]) for e in spine_edges)

        for ent in sorted(spine_nodes):
            sanitized_ent = self._resolve_entity_to_kb(ent)
            if not sanitized_ent:
                continue

            candidate_edges = []
            for rel, tails in self.kb_out.get(sanitized_ent, {}).items():
                rel_freq = self.relation_frequency.get(rel, 0)
                for t in tails:
                    edge = {
                        "head": ent,
                        "relation": rel,
                        "tail": self._display_entity(t),
                        "count": 1,
                    }
                    edge_tuple = (edge["head"], edge["relation"], edge["tail"])
                    if edge_tuple not in visited:
                        candidate_edges.append((rel_freq, edge_tuple, edge))

            for rel, heads in self.kb_in.get(sanitized_ent, {}).items():
                rel_freq = self.relation_frequency.get(rel, 0)
                for h in heads:
                    edge = {
                        "head": self._display_entity(h),
                        "relation": rel,
                        "tail": ent,
                        "count": 1,
                    }
                    edge_tuple = (edge["head"], edge["relation"], edge["tail"])
                    if edge_tuple not in visited:
                        candidate_edges.append((rel_freq, edge_tuple, edge))

            if not candidate_edges:
                continue

            candidate_edges.sort(key=lambda x: (-x[0], x[1]))
            selected = candidate_edges[: self.expansion_per_node_cap]
            for _, edge_tuple, edge in selected:
                visited.add(edge_tuple)
                expanded_edges.append(edge)

        if len(expanded_edges) > self.max_frontier:
            expanded_edges = expanded_edges[: self.max_frontier]
        return expanded_edges

    def _make_direction_flip_candidates(self, entity: str, chain: List[str]) -> List[Dict[str, Any]]:
        """
        Direction suffixes are disabled; bare relations are traversed bidirectionally.
        """
        return []

    def _make_grammar_fallback_candidates(self, entity: str, chain: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback candidates from grammar rules by matching relation labels.
        """
        if not self.hrg:
            return []

        matched_rules = self.hrg.match_rules(chain)

        # if no subset match, use top frequent rules as very weak fallback
        if not matched_rules:
            matched_rules = self.hrg.topk_rules(k=top_k)

        out = []
        seen = set()

        for rule in matched_rules[:top_k]:
            rels = sorted(list(rule.get("_cached_labels", set())))
            if not rels:
                continue

            # Prefer rules with same number of hops as original chain
            # If lengths differ, still allow as fallback
            if len(rels) < 1:
                continue

            reconstructed = []
            for rel in rels[:max(1, len(chain))]:
                found = None
                for old_rel in chain:
                    if old_rel == rel:
                        found = rel
                        break
                reconstructed.append(found if found else rel)

            # also trim / extend to plausible length
            if len(chain) > 0:
                reconstructed = reconstructed[:len(chain)]

            key = (entity, tuple(reconstructed))
            if reconstructed and key not in seen:
                seen.add(key)
                out.append({
                    "entity": entity,
                    "chain": reconstructed,
                    "source": "grammar_fallback",
                    "rule_labels": rels
                })

        return out

    def _score_relation_for_question(
        self,
        rel_token: str,
        question_terms: Set[str],
        preferred_bare_rels: Set[str],
        edge_count: int,
    ) -> float:
        rel_match_text = self._normalize_match_text(self._relation_match_text(rel_token))
        rel_terms = set(rel_match_text.split())
        overlap = len(question_terms & rel_terms)
        contains = sum(1 for term in question_terms if len(term) >= 3 and term in rel_match_text)
        preferred_bonus = 4.0 if rel_token in preferred_bare_rels else 0.0
        low_info_penalty = 0.0
        if rel_token in self.LOW_INFORMATION_RELATIONS and overlap == 0 and preferred_bonus == 0.0:
            low_info_penalty = 10.0
        if rel_token.endswith(self.STATEMENT_RELATION_SUFFIX) and overlap == 0 and preferred_bonus == 0.0:
            low_info_penalty = max(low_info_penalty, 4.0)
        return overlap * 8.0 + contains * 2.0 + preferred_bonus + min(edge_count, 10) * 0.1 - low_info_penalty

    def _actual_relation_options_from_frontier(
        self,
        frontier: Set[str],
        question_terms: Set[str],
        preferred_bare_rels: Set[str],
        branch_limit: int,
    ) -> List[Tuple[float, str, Set[str]]]:
        relation_next: Dict[str, Set[str]] = defaultdict(set)
        relation_edge_counts: Dict[str, int] = defaultdict(int)

        for ent in sorted(frontier)[: self.max_frontier]:
            rels = set(self.kb_out.get(ent, {})) | set(self.kb_in.get(ent, {}))
            for rel in sorted(rels):
                if rel not in self.allowed_rel_tokens:
                    continue
                for h, _, t in self._neighbors_for_token(ent, rel):
                    relation_next[rel].add(self._other_endpoint(ent, h, t))
                    relation_edge_counts[rel] += 1

        options: List[Tuple[float, str, Set[str]]] = []
        for rel_token, next_frontier in relation_next.items():
            if not next_frontier:
                continue
            score = self._score_relation_for_question(
                rel_token,
                question_terms,
                preferred_bare_rels,
                relation_edge_counts.get(rel_token, 0),
            )
            options.append((score, rel_token, next_frontier))

        options.sort(key=lambda x: (-x[0], x[1]))
        return options[:branch_limit]

    def _make_deterministic_valid_chain_candidates(
        self,
        user_prompt: str,
        seed_candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enumerate relation chains that are guaranteed executable in the KB.

        This is a fallback for cases where LLM-proposed/corrected chains all fail.
        It uses KG adjacency to keep only valid paths, then lets the existing
        candidate ranking prefer grammar-compatible and question-relevant chains.
        """
        top_k = top_k or self.valid_chain_fallback_topk
        entity = None
        for cand in seed_candidates:
            entity = self._normalize_entity_from_question(user_prompt, cand.get("entity"))
            if self._resolve_entity_to_kb(entity):
                break
        if not self._resolve_entity_to_kb(entity):
            entity = self._normalize_entity_from_question(user_prompt, None)

        start_entity = self._resolve_entity_to_kb(entity)
        if not entity or not start_entity:
            return []

        target_depths = {
            len(cand.get("chain") or [])
            for cand in seed_candidates
            if cand.get("chain")
        }
        target_depths = {
            depth for depth in target_depths
            if 1 <= depth <= self.valid_chain_fallback_max_depth
        }
        if not target_depths:
            target_depths = set(range(1, self.valid_chain_fallback_max_depth + 1))

        preferred_bare_rels: Set[str] = set()
        for cand in seed_candidates:
            preferred_bare_rels.update(self._chain_to_bare(cand.get("chain") or []))
        for rel in self._select_relation_prompt_candidates(user_prompt, limit=32):
            preferred_bare_rels.add(rel)

        question_terms = set(self._normalize_match_text(user_prompt).split())
        max_depth = max(target_depths)
        branch_limit = max(1, self.valid_chain_fallback_branch)
        beam_width = max(1, self.valid_chain_fallback_beam_width)

        beams: List[Tuple[float, List[str], Set[str]]] = [(0.0, [], {start_entity})]
        completed: List[Tuple[float, List[str], Set[str]]] = []

        for depth in range(1, max_depth + 1):
            next_beams: List[Tuple[float, List[str], Set[str]]] = []
            for score, chain, frontier in beams:
                options = self._actual_relation_options_from_frontier(
                    frontier,
                    question_terms,
                    preferred_bare_rels,
                    branch_limit,
                )
                for rel_score, rel_token, next_frontier in options:
                    new_chain = chain + [rel_token]
                    grammar_features = (
                        self._get_grammar_match_features(new_chain)
                        if self.use_grammar_rerank
                        else self._empty_grammar_features()
                    )
                    grammar_features = self._add_relation_ngram_feature(grammar_features, new_chain)
                    grammar_bonus = (
                        self._safe_int(grammar_features.get("same_arity_hit", 0), 0) * self.GRAMMAR_FALLBACK_SAME_ARITY_BONUS
                        + self._safe_int(grammar_features.get("ordered_path_hit", 0), 0) * self.GRAMMAR_FALLBACK_ORDERED_PATH_BONUS
                        + self._safe_int(grammar_features.get("hit", 0), 0) * self.GRAMMAR_FALLBACK_LABEL_BONUS
                        + min(
                            self._safe_int(grammar_features.get("matched_count", 0), 0),
                            self.RANK_MATCHED_RULE_CAP,
                        ) * self.GRAMMAR_FALLBACK_MATCHED_RULE_WEIGHT
                        + self._safe_float(grammar_features.get("score", 0.0), 0.0)
                        + self._safe_float(grammar_features.get("relation_ngram_score", 0.0), 0.0)
                    )
                    compactness_penalty = (
                        min(len(next_frontier), self.GRAMMAR_FALLBACK_FRONTIER_COMPACTNESS_CAP)
                        * self.GRAMMAR_FALLBACK_FRONTIER_COMPACTNESS_WEIGHT
                    )
                    new_score = score + rel_score + grammar_bonus - compactness_penalty
                    next_beams.append((new_score, new_chain, next_frontier))

            next_beams.sort(key=lambda x: (-x[0], tuple(x[1])))
            beams = next_beams[:beam_width]
            if depth in target_depths:
                completed.extend(beams)

        completed.sort(key=lambda x: (-x[0], tuple(x[1])))

        out: List[Dict[str, Any]] = []
        seen = set()
        for score, chain, _ in completed:
            key = (entity, tuple(chain))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "entity": entity,
                "chain": chain,
                "source": "kg_valid_fallback",
                "confidence": max(0.0, min(1.0, score / 50.0)),
            })
            if len(out) >= top_k:
                break

        return out

    def _preferred_relations_for_question(
        self,
        user_prompt: str,
    ) -> Set[str]:
        question_terms = {
            term
            for term in self._normalize_match_text(user_prompt).split()
            if term not in self.QUESTION_ENTITY_STOPWORDS
        }
        preferred_rels = set()
        for rel in self.allowed_rel_tokens:
            rel_text = self._normalize_match_text(self._relation_match_text(rel))
            rel_terms = {
                term
                for term in rel_text.split()
                if term not in self.QUESTION_ENTITY_STOPWORDS
            }
            if question_terms & rel_terms:
                preferred_rels.add(rel)
        return preferred_rels

    def _score_relation_path_for_question(
        self,
        path: Tuple[str, ...],
        user_prompt: str,
        preferred_rels: Optional[Set[str]] = None,
    ) -> float:
        question_terms = {
            term
            for term in self._normalize_match_text(user_prompt).split()
            if term not in self.QUESTION_ENTITY_STOPWORDS
        }
        preferred_rels = preferred_rels if preferred_rels is not None else self._preferred_relations_for_question(user_prompt)

        if not path:
            return 0.0

        scores = [
            self._score_relation_for_question(
                rel,
                question_terms,
                preferred_rels,
                edge_count=0,
            )
            for rel in path
        ]
        return sum(scores) / len(scores)

    def _make_grammar_path_bank_candidates(
        self,
        user_prompt: str,
        entity: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate KG-valid candidates from HRG relation-path signatures.

        Unlike the deterministic fallback, this does not discover chains by
        frontier branching first. HRG provides the relation-path bank, and the
        KG only validates whether each path can execute from the topic entity.
        """
        if not self.hrg:
            return []

        resolved_entity = self._resolve_entity_to_kb(entity)
        if not resolved_entity:
            return []

        path_prior = self.hrg.relation_path_prior(self.valid_chain_fallback_max_depth)
        scored: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        preferred_rels = self._preferred_relations_for_question(user_prompt)

        for path, prior_weight in path_prior.items():
            chain = list(path)
            kb_result = self._check_chain_validity(entity, chain)
            if not kb_result.get("valid"):
                continue

            grammar_features = (
                self._get_grammar_match_features(chain)
                if self.use_grammar_rerank
                else self._empty_grammar_features()
            )
            grammar_features = self._add_relation_ngram_feature(grammar_features, chain)
            preferred_coverage = len(set(path) & preferred_rels)
            question_score = self._score_relation_path_for_question(path, user_prompt, preferred_rels)
            final_size = self._safe_int(kb_result.get("final_size", 0), 0)
            ranking_key = (
                preferred_coverage,
                question_score,
                self._safe_int(grammar_features.get("same_arity_hit", 0), 0),
                self._safe_int(grammar_features.get("ordered_path_hit", 0), 0),
                self._safe_int(grammar_features.get("hit", 0), 0),
                self._safe_float(grammar_features.get("score", 0.0), 0.0),
                self._safe_float(prior_weight, 0.0),
                -final_size,
                tuple(chain),
            )
            scored.append((
                ranking_key,
                {
                    "entity": entity,
                    "chain": chain,
                    "source": "grammar_first",
                    "confidence": 0.0,
                    "kb_result": kb_result,
                },
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored]

    def _make_grammar_first_candidates(
        self,
        user_prompt: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Grammar-first candidate generation.

        This skips LLM-proposed relation chains. It only seeds topic entities, then
        enumerates executable chains up to valid_chain_fallback_max_depth and lets
        grammar/question relevance rank them.
        """
        top_k = top_k or self.valid_chain_fallback_topk
        entity_seeds: List[str] = []
        seen_entities: Set[str] = set()

        bracket_entity = self._normalize_entity_from_question(user_prompt, None)
        if bracket_entity and self._resolve_entity_to_kb(bracket_entity):
            entity_seeds.append(bracket_entity)
            seen_entities.add(self._resolve_entity_to_kb(bracket_entity) or bracket_entity)

        for entity in self._extract_entity_candidates_from_question(user_prompt, limit=max(1, top_k)):
            resolved = self._resolve_entity_to_kb(entity)
            if not resolved or resolved in seen_entities:
                continue
            entity_seeds.append(entity)
            seen_entities.add(resolved)
            if len(entity_seeds) >= max(1, min(8, top_k)):
                break

        if not entity_seeds:
            return [], 0

        pool: List[Dict[str, Any]] = []
        for entity in entity_seeds:
            generated = self._make_grammar_path_bank_candidates(user_prompt, entity)
            if not generated:
                generated = self._make_deterministic_valid_chain_candidates(
                    user_prompt,
                    [{
                        "entity": entity,
                        "chain": [],
                        "source": "grammar_first_seed",
                        "confidence": 0.0,
                    }],
                    top_k=top_k,
                )
            for cand in generated:
                cand["source"] = "grammar_first"
                pool.append(cand)

        return self._dedup_candidates(pool), len(entity_seeds)

    # ============================================================
    # Serialization / Answer Generation
    # ============================================================

    def _serialize_edges(self, edges: List[Dict[str, Any]]) -> str:
        if self.serialization_format == "triples":
            lines = [f"{e['head']} {e['relation']} {e['tail']}." for e in edges]
            return " ".join(lines)
        return json.dumps(edges, ensure_ascii=False)

    async def _generate_rag_response(self, user_prompt: str, context_json: str) -> Tuple[str, int]:
        dev = (
            "You are a strict answer formatter for KGQA.\n"
            "Use only the provided Context to answer the question.\n"
            "Output rules:\n"
            "1. Return only the final answer.\n"
            "2. Do not output reasoning, explanations, analysis, intermediate steps, or chain-of-thought.\n"
            "3. Do not output <think> tags or any hidden-thinking markers.\n"
            "4. Do not output full sentences, prefixes, markdown, bullets, numbering, quotes, or the question.\n"
            "5. Copy answer strings verbatim from Context; do not paraphrase, translate, or add outside knowledge.\n"
            "6. For a single answer, output exactly that answer string.\n"
            "7. For multiple answers, output only the answer strings joined by ' | ' in concise surface form.\n"
            "8. Remove duplicates and keep answer strings concise.\n"
            "9. If Context is insufficient, output exactly: I don't know"
        )
        user = f"Context: {context_json}\n\nQuestion: {user_prompt}"
        parse2_tokens = self._estimate_tokens(dev, user)
        raw = await self._inference(dev, user)
        clean = self._extract_final_content(raw)
        parse2_tokens += self._estimate_tokens(clean)
        return clean, parse2_tokens

    def _parse_llm_index_ranking(self, text: str, candidate_count: int) -> List[int]:
        payloads = []
        try:
            payloads.append(json.loads(text))
        except Exception:
            pass

        for s in self._extract_balanced_json_segments(text, "[", "]"):
            try:
                payloads.append(json.loads(s))
            except Exception:
                continue
        for s in self._extract_balanced_json_segments(text, "{", "}"):
            try:
                payloads.append(json.loads(s))
            except Exception:
                continue

        for payload in payloads:
            if isinstance(payload, dict):
                for key in ("ranking", "indices", "ranked_indices", "order"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            if not isinstance(payload, list):
                continue

            out = []
            seen = set()
            for item in payload:
                if isinstance(item, dict):
                    item = item.get("idx", item.get("index"))
                try:
                    idx = int(item)
                except Exception:
                    continue
                if 0 <= idx < candidate_count and idx not in seen:
                    out.append(idx)
                    seen.add(idx)
            if out:
                out.extend(i for i in range(candidate_count) if i not in seen)
                return out

        return list(range(candidate_count))

    async def _rerank_valid_chain_candidates_with_llm(
        self,
        user_prompt: str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[int], int]:
        compact_candidates = []
        for idx, cand in enumerate(candidates):
            kb_result = cand.get("kb_result") or self._check_chain_validity(cand.get("entity", ""), cand.get("chain", []))
            compact_candidates.append({
                "idx": idx,
                "entity": cand.get("entity", ""),
                "chain": cand.get("chain", []),
                "step_sizes": kb_result.get("step_sizes", []),
                "final_size": kb_result.get("final_size", 0),
            })

        developer_instruction = (
            "You are a KG relation-chain reranker.\n"
            "All candidate chains are executable in the KG. Rank them by which chain best answers the question.\n"
            "Rules:\n"
            "  - Prefer exact semantic match to the question.\n"
            "  - Prefer the required hop count; do not reward extra hops.\n"
            "  - Do not invent or edit chains.\n"
            "  - Return only a JSON array of candidate idx values from best to worst.\n"
            "Example: [2, 0, 1]"
        )
        user_content = (
            f"Question: {user_prompt}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}"
        )
        input_tokens = self._estimate_tokens(developer_instruction, user_content)
        raw = await self._inference(developer_instruction, user_content)
        clean = self._extract_final_content(raw)
        total_tokens = input_tokens + self._estimate_tokens(clean)
        return self._parse_llm_index_ranking(clean, len(candidates)), total_tokens

    # ============================================================
    # Candidate Ranking
    # ============================================================

    def _score_candidate(
        self,
        entity: str,
        chain: List[str],
        kb_result: Dict[str, Any],
        grammar_features: Dict[str, Any],
        llm_rank_index: int,
        llm_confidence: float = 0.0,
        llm_rerank_score: float = 0.0,
        source: str = "llm",
    ) -> Tuple[Any, ...]:
        """
        Interpretable lax HRG-prior lexicographic ranking key.

        Higher tuple values are preferred.
        Priority order:
        1. KB executability
        2. Same-arity HRG compatibility
        3. Ordered-path HRG compatibility
        4. Grammar label compatibility
        5. Grammar score and matched-rule support
        6. LLM rerank score when an explicit rerank pass is used
        7. Later failure hop (or fully executable)
        8. Retrieval survivability across hops
        9. Final frontier size
        10. Candidate source prior
        11. LLM confidence / original rank as tie-breakers
        """
        step_sizes = kb_result.get("step_sizes", [])
        valid = 1 if kb_result.get("valid") else 0
        failed_hop = kb_result.get("failed_hop")
        if failed_hop is None:
            failure_progress = len(chain) + 1
        else:
            failure_progress = self._safe_int(failed_hop, 0)

        grammar_hit = self._safe_int(grammar_features.get("hit", 0), 0)
        same_arity_hit = self._safe_int(grammar_features.get("same_arity_hit", 0), 0)
        ordered_path_hit = self._safe_int(grammar_features.get("ordered_path_hit", 0), 0)
        grammar_score = self._safe_float(grammar_features.get("score", 0.0), 0.0)
        grammar_matched_count = min(
            self._safe_int(grammar_features.get("matched_count", 0), 0),
            self.RANK_MATCHED_RULE_CAP,
        )
        step_survival = sum(
            min(self._safe_int(s, 0), self.RANK_STEP_SURVIVAL_CAP_PER_HOP)
            for s in step_sizes
        )
        final_size = min(
            self._safe_int(kb_result.get("final_size", 0), 0),
            self.RANK_FINAL_FRONTIER_CAP,
        )

        source_priority = self.CANDIDATE_SOURCE_PRIORITY.get(
            source,
            self.CANDIDATE_SOURCE_DEFAULT_PRIORITY,
        )
        if source.startswith("flip_hop_"):
            source_priority = self.CANDIDATE_FLIP_HOP_PRIORITY

        llm_prior = -self._safe_int(llm_rank_index, 0)

        return (
            valid,
            same_arity_hit,
            ordered_path_hit,
            grammar_hit,
            grammar_score,
            grammar_matched_count,
            self._safe_float(llm_rerank_score, 0.0),
            failure_progress,
            step_survival,
            final_size,
            source_priority,
            llm_confidence,
            llm_prior,
        )

    def _should_add_low_confidence_fallback(self, ranked_candidates: List[Dict[str, Any]]) -> bool:
        if not self.use_low_confidence_valid_chain_fallback:
            return False

        valid_candidates = [c for c in ranked_candidates if c.get("kb_result", {}).get("valid")]
        if not valid_candidates:
            return True

        top_valid = max(valid_candidates, key=lambda c: c.get("ranking_key", ()))
        has_strong_grammar = bool(
            self._safe_int(top_valid.get("grammar_same_arity_hit", 0), 0)
            or self._safe_int(top_valid.get("grammar_ordered_path_hit", 0), 0)
        )
        if has_strong_grammar:
            return False

        return len(valid_candidates) < max(1, int(self.low_confidence_min_valid_candidates))

    def _build_candidate_subgraph(
        self,
        entity: str,
        chain: List[str],
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        start_entity = self._resolve_entity_to_kb(entity)
        selected_rules = self._select_matched_rules(chain)
        grammar_hit = bool(selected_rules)

        if self.use_grammar_guided_retrieval and self.hrg:
            spine_edges = self._find_subgraph_grammar_guided(start_entity, chain, selected_rules)
        else:
            spine_edges = self._find_subgraph_multi_hop_kb_strict(start_entity, chain)

        if not spine_edges:
            return {
                "matched_rules": selected_rules[: self.topk_expansion_rules],
                "grammar_hit": grammar_hit,
                "spine_edges": [],
                "expanded_edges": [],
                "final_edges": [],
                "retrieval_policy": "none",
                "expansion_gate": "no_spine_edges",
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_f1": 0.0,
                "subgraph_size": 0,
                "spine_edge_count": 0,
                "expanded_edge_count": 0,
                "raw_expanded_edge_count": 0,
                "raw_final_edge_count": 0,
                "context_truncated": False,
            }

        matched_rules = []
        expanded_edges = []
        raw_expanded_edge_count = 0
        context_truncated = False
        is_single_hop = len(chain) == 1
        top_rule_score = 0.0
        if selected_rules:
            top_rule_score = self._safe_float(
                selected_rules[0].get("probability", selected_rules[0].get("count", 0)),
                0.0,
            )
        expansion_gate = "disabled"
        allow_grammar_expansion = (
            bool(selected_rules)
            and top_rule_score >= self.min_grammar_score_for_expansion
            and (
                self.max_spine_edges_for_expansion is None
                or len(spine_edges) <= self.max_spine_edges_for_expansion
            )
        )
        if not selected_rules:
            expansion_gate = (
                "no_ordered_grammar_match"
                if self.require_ordered_grammar_match
                else "no_grammar_label_subset_match"
            )
        elif self.require_exact_grammar_match_for_expansion and self._rule_terminal_count(selected_rules[0]) != len(chain):
            expansion_gate = "no_exact_grammar_match"
            allow_grammar_expansion = False
        elif top_rule_score < self.min_grammar_score_for_expansion:
            expansion_gate = "grammar_score_below_min"
        elif self.max_spine_edges_for_expansion is not None and len(spine_edges) > self.max_spine_edges_for_expansion:
            expansion_gate = "spine_too_large"
        elif allow_grammar_expansion:
            expansion_gate = "allowed"

        if not self.use_grammar_guided_retrieval and self.hrg and self.use_grammar_expansion and not is_single_hop:
            matched_rules = selected_rules
            if allow_grammar_expansion:
                expanded_edges = self._expand_subgraph_by_grammar(spine_edges, matched_rules, chain=chain)
                raw_expanded_edge_count = len(expanded_edges)
                expanded_edges = self._apply_expansion_budget(spine_edges, expanded_edges)
                context_truncated = context_truncated or len(expanded_edges) < raw_expanded_edge_count
                if not expanded_edges:
                    expansion_gate = "budget_zero_or_no_new_edges"
        elif self.use_frequency_expansion and not is_single_hop:
            expanded_edges = self._expand_subgraph_by_relation_frequency(spine_edges)
            raw_expanded_edge_count = len(expanded_edges)
            expanded_edges = self._apply_expansion_budget(spine_edges, expanded_edges)
            context_truncated = context_truncated or len(expanded_edges) < raw_expanded_edge_count
        elif self.use_random_expansion and not is_single_hop:
            expanded_edges = self._expand_subgraph_random(spine_edges, chain=chain)
            raw_expanded_edge_count = len(expanded_edges)
            expanded_edges = self._apply_expansion_budget(spine_edges, expanded_edges)
            context_truncated = context_truncated or len(expanded_edges) < raw_expanded_edge_count
        elif is_single_hop:
            matched_rules = selected_rules
            expansion_gate = "single_hop_spine_floor"

        final_edges = spine_edges + expanded_edges
        raw_final_edge_count = len(final_edges)
        if self.top_k_edges is not None and len(final_edges) > self.top_k_edges:
            context_truncated = True
            spine_budget = max(0, int(self.top_k_edges))
            kept_spine_edges = spine_edges[:spine_budget]
            remaining_budget = max(0, spine_budget - len(kept_spine_edges))
            kept_expanded_edges = sorted(
                expanded_edges,
                key=lambda e: e.get("count", 0),
                reverse=True,
            )[:remaining_budget]
            spine_edges = kept_spine_edges
            expanded_edges = kept_expanded_edges
            final_edges = spine_edges + expanded_edges

        retrieval_policy = "hrg_expanded" if expanded_edges else "spine_floor"

        retrieval_recall = 0.0
        retrieval_precision = 0.0
        retrieval_f1 = 0.0
        if references:
            retrieval_recall, retrieval_precision, retrieval_f1 = self._compute_retrieval_metrics(final_edges, references)

        return {
            "matched_rules": matched_rules[: self.topk_expansion_rules],
            "grammar_hit": grammar_hit,
            "spine_edges": spine_edges,
            "expanded_edges": expanded_edges,
            "final_edges": final_edges,
            "retrieval_policy": retrieval_policy,
            "expansion_gate": expansion_gate,
            "retrieval_recall": retrieval_recall,
            "retrieval_precision": retrieval_precision,
            "retrieval_f1": retrieval_f1,
            "subgraph_size": len(final_edges),
            "spine_edge_count": len(spine_edges),
            "expanded_edge_count": len(expanded_edges),
            "raw_expanded_edge_count": raw_expanded_edge_count,
            "raw_final_edge_count": raw_final_edge_count,
            "context_truncated": context_truncated,
        }

    def _score_subgraph_candidate(
        self,
        chain_row: Dict[str, Any],
        subgraph: Dict[str, Any],
        has_references: bool,
    ) -> Tuple[Any, ...]:
        has_edges = 1 if subgraph.get("final_edges") else 0
        grammar_hit = 1 if subgraph.get("grammar_hit") else 0
        grammar_score = self._safe_float(chain_row.get("grammar_score", 0.0), 0.0)
        same_arity_hit = self._safe_int(chain_row.get("grammar_same_arity_hit", 0), 0)
        ordered_path_hit = self._safe_int(chain_row.get("grammar_ordered_path_hit", 0), 0)
        expanded_size = min(len(subgraph.get("expanded_edges", [])), 10000)
        support = min(
            self._safe_int(subgraph.get("subgraph_size", 0), 0),
            max(1, self._safe_int(self.subgraph_support_saturation, 12)),
        )
        # Smaller subgraphs are preferred once support quality is comparable.
        compactness = -min(subgraph.get("subgraph_size", 0), 10000)

        return (
            has_edges,
            same_arity_hit,
            ordered_path_hit,
            grammar_hit,
            grammar_score,
            chain_row.get("ranking_key", ()),
            support,
            compactness,
            -expanded_size,
        )

    # ============================================================
    # Dump
    # ============================================================

    def _dump(
        self,
        save_path: str,
        question: str,
        selected_entity: Optional[str],
        selected_chain: List[str],
        edges: List[Dict[str, Any]],
        answer: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        matched_rules: Optional[List[Dict[str, Any]]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        retrieval_recall: float = 0.0,
        retrieval_precision: float = 0.0,
        retrieval_f1: float = 0.0,
        subgraph_size: int = 0,
        failure_stage: str = "ok",
        grammar_hit: bool = False,
        serialization_format: Optional[str] = None,
        references: Optional[List[str]] = None,
        spine_edges: Optional[List[Dict[str, Any]]] = None,
        expanded_edges: Optional[List[Dict[str, Any]]] = None,
        selected_candidate: Optional[Dict[str, Any]] = None,
        final_context: Optional[str] = None,
        timing: Optional[Dict[str, Any]] = None,
        retrieval_policy: Optional[str] = None,
        expansion_gate: Optional[str] = None,
        retrieval_diagnostics: Optional[Dict[str, Any]] = None,
    ):
        payload = {
            "question": question,
            "selected_entity": selected_entity,
            "selected_chain": selected_chain,
            "edges": edges,
            "answer": answer,
            "candidates": candidates or [],
            "matched_grammar_rules": matched_rules or [],
            "token_usage": token_usage or {},
            "retrieval_recall": retrieval_recall,
            "retrieval_precision": retrieval_precision,
            "retrieval_f1": retrieval_f1,
            "subgraph_size": subgraph_size,
            "failure_stage": failure_stage,
            "grammar_hit": grammar_hit,
            "serialization_format": serialization_format,
            "references": references or [],
            "spine_edges": spine_edges or [],
            "expanded_edges": expanded_edges or [],
            "selected_candidate": selected_candidate or {},
            "final_context": final_context or "",
            "timing": timing or {},
            "retrieval_policy": retrieval_policy or "",
            "expansion_gate": expansion_gate or "",
        }
        payload.update(retrieval_diagnostics or {})
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    def _compute_retrieval_recall(
        self,
        edges: List[Dict[str, Any]],
        references: List[str],
    ) -> float:
        """
        Check whether retrieved subgraph contains any gold answer entity.
        Uses substring match (consistent with Acc metric in benchmark).
        """
        recall, _, _ = self._compute_retrieval_metrics(edges, references)
        return recall

    def _compute_retrieval_metrics(
        self,
        edges: List[Dict[str, Any]],
        references: List[str],
    ) -> Tuple[float, float, float]:
        """
        Compute retrieval recall, precision, F1 for the subgraph.

        - Recall:    fraction of gold answers covered by the subgraph
        - Precision: fraction of subgraph nodes that are gold answers
        - F1:        harmonic mean of recall and precision
        """
        if not references or not edges:
            return 0.0, 0.0, 0.0

        nodes_in_subgraph = set()
        for e in edges:
            nodes_in_subgraph.add(e.get("head", "").lower().replace("_", " "))
            nodes_in_subgraph.add(e.get("tail", "").lower().replace("_", " "))

        # Recall: how many gold answers appear in the subgraph
        hit_count = 0
        for ref in references:
            ref_lower = ref.lower().replace("_", " ")
            for n in nodes_in_subgraph:
                if ref_lower in n or n in ref_lower:
                    hit_count += 1
                    break
        recall = hit_count / len(references) if references else 0.0

        # Precision: how many subgraph nodes are gold answers
        refs_lower = [r.lower().replace("_", " ") for r in references]
        gold_nodes = 0
        for n in nodes_in_subgraph:
            for ref_lower in refs_lower:
                if ref_lower in n or n in ref_lower:
                    gold_nodes += 1
                    break
        precision = gold_nodes / len(nodes_in_subgraph) if nodes_in_subgraph else 0.0

        # F1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return recall, precision, f1

    # ============================================================
    # Main Ask
    # ============================================================

    def _accumulate_prepared_stats(self, prepared: Dict[str, Any], parse2_tokens: int, context_tokens: int):
        self.total_questions += 1
        token_usage = prepared.get("token_usage", {})
        self.total_parse1_tokens += token_usage.get("parse1_tokens", 0)
        self.total_correction_tokens += token_usage.get("correction_tokens", 0)
        self.total_parse2_tokens += parse2_tokens
        self.total_context_length += context_tokens

        if prepared.get("grammar_hit", False):
            self.hit_grammar_count += 1

        self.total_subgraph_size += prepared.get("subgraph_size", 0)

        if prepared.get("has_references", False):
            self.total_retrieval_recall += prepared.get("retrieval_recall", 0.0)
            self.total_retrieval_precision += prepared.get("retrieval_precision", 0.0)
            self.total_retrieval_f1 += prepared.get("retrieval_f1", 0.0)
            self.total_retrieval_questions += 1

    async def prepare_retrieval(
        self,
        user_prompt: str,
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        print(f"\n[Question] {user_prompt}\n", flush=True)
        total_t0 = time.perf_counter()
        parse_latency = 0.0
        retrieval_latency = 0.0
        correction_tokens = 0

        parserless_retrieval = self.use_grammar_first_retrieval

        if self.use_grammar_first_retrieval:
            p1_tokens = 0
            gen_t0 = time.perf_counter()
            candidates, entity_seed_count = self._make_grammar_first_candidates(
                user_prompt,
                top_k=self.valid_chain_fallback_topk,
            )
            retrieval_latency += time.perf_counter() - gen_t0
            print(
                f"[GrammarFirst] entity_seeds={entity_seed_count} "
                f"generated {len(candidates)} KG-valid candidates | tokens~0",
                flush=True,
            )
            if self.use_valid_chain_llm_rerank and len(candidates) > 1:
                rerank_t0 = time.perf_counter()
                for idx, cand in enumerate(candidates):
                    cand["_grammar_first_idx"] = idx
                    cand["kb_result"] = self._check_chain_validity(cand["entity"], cand["chain"])
                rerank_order, rerank_tokens = await self._rerank_valid_chain_candidates_with_llm(
                    user_prompt,
                    candidates,
                )
                parse_latency += time.perf_counter() - rerank_t0
                correction_tokens += rerank_tokens
                order_pos = {idx: pos for pos, idx in enumerate(rerank_order)}
                denom = max(len(candidates) - 1, 1)
                for cand in candidates:
                    idx = int(cand.get("_grammar_first_idx", 0))
                    cand["llm_rerank_score"] = 1.0 - (order_pos.get(idx, len(candidates)) / denom)
                candidates.sort(
                    key=lambda cand: (
                        order_pos.get(cand.get("_grammar_first_idx", 0), len(candidates)),
                        cand.get("_grammar_first_idx", 0),
                    )
                )
                candidates = candidates[: self.valid_chain_fallback_topk]
                print(f"[GrammarFirst] LLM reranked KG-valid candidates | tokens~{rerank_tokens}", flush=True)
            else:
                candidates = candidates[: self.valid_chain_fallback_topk]
        else:
            parse_t0 = time.perf_counter()
            candidates, p1_tokens = await self._parse_intent_candidates(user_prompt, self.num_candidates)
            parse_latency += time.perf_counter() - parse_t0
            print(f"[Parse 1] got {len(candidates)} candidates | tokens~{p1_tokens}", flush=True)

        if not candidates and not parserless_retrieval:
            fallback_entities = self._extract_entity_candidates_from_question(user_prompt)
            fallback_pool = []
            if self.use_deterministic_valid_chain_fallback and fallback_entities:
                seed_candidates = [
                    {
                        "entity": entity,
                        "chain": [],
                        "source": "question_entity_fallback",
                        "confidence": 0.0,
                    }
                    for entity in fallback_entities
                ]
                for seed in seed_candidates:
                    fallback_pool.extend(
                        self._make_deterministic_valid_chain_candidates(
                            user_prompt,
                            [seed],
                            top_k=max(1, self.valid_chain_fallback_topk // max(1, len(seed_candidates))),
                        )
                    )
                candidates = self._dedup_candidates(fallback_pool)[: self.valid_chain_fallback_topk]
                print(
                    f"[Fallback] Parse produced no candidates; entity fallback found "
                    f"{len(fallback_entities)} entities and {len(candidates)} KG-valid candidates.",
                    flush=True,
                )

            if not candidates:
                return {
                    "status": "no_candidates",
                    "answer": "No valid candidate relation chains parsed.",
                    "selected_entity": None,
                    "selected_chain": [],
                    "edges": [],
                    "candidates": [],
                    "matched_rules": [],
                    "token_usage": {
                        "parse1_tokens": p1_tokens,
                        "correction_tokens": 0,
                    },
                    "retrieval_recall": 0.0,
                    "retrieval_precision": 0.0,
                    "retrieval_f1": 0.0,
                    "subgraph_size": 0,
                    "grammar_hit": False,
                    "has_references": bool(references),
                    "references": references or [],
                    "spine_edges": [],
                    "expanded_edges": [],
                    "retrieval_policy": "none",
                    "expansion_gate": "no_candidates",
                    "selected_candidate": {},
                    "parse_latency": parse_latency,
                    "retrieval_latency": 0.0,
                    "total_prepare_latency": time.perf_counter() - total_t0,
                }

        if not candidates:
            return {
                "status": "no_candidates",
                "answer": "No valid candidate relation chains generated.",
                "selected_entity": None,
                "selected_chain": [],
                "edges": [],
                "candidates": [],
                "matched_rules": [],
                "token_usage": {
                    "parse1_tokens": p1_tokens,
                    "correction_tokens": correction_tokens,
                },
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_f1": 0.0,
                "subgraph_size": 0,
                "grammar_hit": False,
                "has_references": bool(references),
                "references": references or [],
                "spine_edges": [],
                "expanded_edges": [],
                "retrieval_policy": "none",
                "expansion_gate": "no_candidates",
                "selected_candidate": {},
                "parse_latency": parse_latency,
                "retrieval_latency": retrieval_latency,
                "total_prepare_latency": time.perf_counter() - total_t0,
            }

        rank_t0 = time.perf_counter()
        ranked_candidates = []
        failed_candidates = []

        for rank_idx, cand in enumerate(candidates):
            entity = cand["entity"]
            requested_chain = cand["chain"]
            kb_result = self._check_chain_validity(entity, requested_chain)
            chain = kb_result.get("resolved_chain", requested_chain)
            grammar_features = (
                self._get_grammar_match_features(chain)
                if self.use_grammar_rerank
                else self._empty_grammar_features()
            )
            grammar_features = self._add_relation_ngram_feature(grammar_features, chain)
            grammar_score = grammar_features["score"]
            source = cand.get("source", "llm")
            llm_rerank_score = float(cand.get("llm_rerank_score", 0.0) or 0.0)
            confidence = max(float(cand.get("confidence", 0.0) or 0.0), llm_rerank_score)
            ranking_key = self._score_candidate(
                entity, chain, kb_result, grammar_features, rank_idx,
                llm_confidence=confidence,
                llm_rerank_score=llm_rerank_score,
                source=source,
            )
            row = {
                "entity": entity,
                "chain": chain,
                "requested_chain": requested_chain,
                "source": source,
                "confidence": confidence,
                "llm_rank_index": rank_idx,
                "kb_result": kb_result,
                "grammar_score": grammar_score,
                "grammar_hit": grammar_features["hit"],
                "grammar_same_arity_hit": grammar_features["same_arity_hit"],
                "grammar_ordered_path_hit": grammar_features["ordered_path_hit"],
                "grammar_weak_label_match": grammar_features["weak_label_match"],
                "grammar_matched_count": grammar_features["matched_count"],
                "relation_ngram_score": grammar_features.get("relation_ngram_score", 0.0),
                "relation_ngram_order": grammar_features.get("relation_ngram_order", 0),
                "llm_rerank_score": llm_rerank_score,
                "ranking_key": ranking_key,
            }
            ranked_candidates.append(row)
            if not kb_result["valid"]:
                failed_candidates.append(row)

        if self.use_fallback_correction and ranked_candidates and not any(c["kb_result"]["valid"] for c in ranked_candidates):
            print("[Fallback] All initial candidates failed. Trying correction...", flush=True)
            correction_pool = []
            for cand in candidates:
                correction_pool.extend(self._make_direction_flip_candidates(cand["entity"], cand["chain"]))
            if self.hrg:
                for cand in candidates:
                    correction_pool.extend(self._make_grammar_fallback_candidates(cand["entity"], cand["chain"], top_k=5))
            corr_t0 = time.perf_counter()
            llm_corr, corr_tokens = await self._correct_candidates_with_llm(user_prompt, failed_candidates, max_new_candidates=5)
            parse_latency += time.perf_counter() - corr_t0
            correction_tokens += corr_tokens
            correction_pool.extend(llm_corr)
            correction_pool = self._dedup_candidates(correction_pool)

            print(f"[Fallback] Generated {len(correction_pool)} correction candidates | tokens~{correction_tokens}", flush=True)

            base_len = len(ranked_candidates)
            for idx, cand in enumerate(correction_pool):
                entity = cand["entity"]
                requested_chain = cand["chain"]
                kb_result = self._check_chain_validity(entity, requested_chain)
                chain = kb_result.get("resolved_chain", requested_chain)
                grammar_features = (
                    self._get_grammar_match_features(chain)
                    if self.use_grammar_rerank
                    else self._empty_grammar_features()
                )
                grammar_features = self._add_relation_ngram_feature(grammar_features, chain)
                grammar_score = grammar_features["score"]
                source = cand.get("source", "correction")
                confidence = float(cand.get("confidence", 0.0) or 0.0)
                ranking_key = self._score_candidate(
                    entity, chain, kb_result, grammar_features, base_len + idx,
                    llm_confidence=confidence, source=source
                )
                ranked_candidates.append({
                    "entity": entity,
                    "chain": chain,
                    "requested_chain": requested_chain,
                    "source": source,
                    "confidence": confidence,
                    "llm_rank_index": base_len + idx,
                    "kb_result": kb_result,
                    "grammar_score": grammar_score,
                    "grammar_hit": grammar_features["hit"],
                    "grammar_same_arity_hit": grammar_features["same_arity_hit"],
                    "grammar_ordered_path_hit": grammar_features["ordered_path_hit"],
                    "grammar_weak_label_match": grammar_features["weak_label_match"],
                    "grammar_matched_count": grammar_features["matched_count"],
                    "relation_ngram_score": grammar_features.get("relation_ngram_score", 0.0),
                    "relation_ngram_order": grammar_features.get("relation_ngram_order", 0),
                    "llm_rerank_score": 0.0,
                    "ranking_key": ranking_key,
                })

        needs_kg_fallback = (
            self.use_deterministic_valid_chain_fallback
            and ranked_candidates
            and (
                not any(c["kb_result"]["valid"] for c in ranked_candidates)
                or self._should_add_low_confidence_fallback(ranked_candidates)
            )
        )
        if needs_kg_fallback:
            print("[Fallback] Enumerating KG-valid fallback chains under confidence/budget gate...", flush=True)
            fallback_seed_candidates = list(candidates)
            fallback_seed_candidates.extend({
                "entity": row["entity"],
                "chain": row["chain"],
                "source": row.get("source", "ranked"),
                "confidence": row.get("confidence", 0.0),
            } for row in ranked_candidates)

            kg_candidates = self._make_deterministic_valid_chain_candidates(
                user_prompt,
                fallback_seed_candidates,
                top_k=self.valid_chain_fallback_topk,
            )
            existing_keys = {
                (row["entity"], tuple(row["chain"]))
                for row in ranked_candidates
            }
            kg_candidates = [
                cand for cand in self._dedup_candidates(kg_candidates)
                if (cand["entity"], tuple(cand["chain"])) not in existing_keys
            ]
            print(f"[Fallback] Generated {len(kg_candidates)} KG-valid candidates.", flush=True)

            for idx, cand in enumerate(kg_candidates):
                cand["_fallback_idx"] = idx
                cand["kb_result"] = self._check_chain_validity(cand["entity"], cand["chain"])

            rerank_scores: Dict[int, float] = {}
            if self.use_valid_chain_llm_rerank and len(kg_candidates) > 1:
                rerank_t0 = time.perf_counter()
                rerank_order, rerank_tokens = await self._rerank_valid_chain_candidates_with_llm(
                    user_prompt,
                    kg_candidates,
                )
                parse_latency += time.perf_counter() - rerank_t0
                correction_tokens += rerank_tokens
                order_pos = {idx: pos for pos, idx in enumerate(rerank_order)}
                denom = max(len(kg_candidates) - 1, 1)
                rerank_scores = {
                    idx: 1.0 - (pos / denom)
                    for idx, pos in order_pos.items()
                }
                kg_candidates.sort(
                    key=lambda cand: (
                        order_pos.get(cand.get("_fallback_idx", 0), len(kg_candidates)),
                        cand.get("_fallback_idx", 0),
                    )
                )
                print(f"[Fallback] LLM reranked KG-valid candidates | tokens~{rerank_tokens}", flush=True)

            base_len = len(ranked_candidates)
            for idx, cand in enumerate(kg_candidates):
                entity = cand["entity"]
                requested_chain = cand["chain"]
                kb_result = cand.get("kb_result") or self._check_chain_validity(entity, requested_chain)
                chain = kb_result.get("resolved_chain", requested_chain)
                grammar_features = (
                    self._get_grammar_match_features(chain)
                    if self.use_grammar_rerank
                    else self._empty_grammar_features()
                )
                grammar_features = self._add_relation_ngram_feature(grammar_features, chain)
                grammar_score = grammar_features["score"]
                source = cand.get("source", "kg_valid_fallback")
                fallback_idx = int(cand.get("_fallback_idx", idx))
                llm_rerank_score = rerank_scores.get(fallback_idx, 0.0)
                confidence = max(float(cand.get("confidence", 0.0) or 0.0), llm_rerank_score)
                ranking_key = self._score_candidate(
                    entity, chain, kb_result, grammar_features, base_len + idx,
                    llm_confidence=confidence,
                    llm_rerank_score=llm_rerank_score,
                    source=source,
                )
                ranked_candidates.append({
                    "entity": entity,
                    "chain": chain,
                    "requested_chain": requested_chain,
                    "source": source,
                    "confidence": confidence,
                    "llm_rerank_score": llm_rerank_score,
                    "llm_rank_index": base_len + idx,
                    "kb_result": kb_result,
                    "grammar_score": grammar_score,
                    "grammar_hit": grammar_features["hit"],
                    "grammar_same_arity_hit": grammar_features["same_arity_hit"],
                    "grammar_ordered_path_hit": grammar_features["ordered_path_hit"],
                    "grammar_weak_label_match": grammar_features["weak_label_match"],
                    "grammar_matched_count": grammar_features["matched_count"],
                    "relation_ngram_score": grammar_features.get("relation_ngram_score", 0.0),
                    "relation_ngram_order": grammar_features.get("relation_ngram_order", 0),
                    "ranking_key": ranking_key,
                })

        ranked_candidates.sort(key=lambda x: x["ranking_key"], reverse=True)
        retrieval_latency += time.perf_counter() - rank_t0
        print("[Ranking] Top candidates:")
        for i, rc in enumerate(ranked_candidates[:10]):
            print(
                f"  [{i}] entity={rc['entity']} chain={rc['chain']} "
                f"source={rc['source']} valid={rc['kb_result']['valid']} "
                f"grammar={rc['grammar_score']:.4f} rank_key={rc['ranking_key']}"
            )

        best = ranked_candidates[0] if ranked_candidates else None
        valid_candidates = [c for c in ranked_candidates if c["kb_result"]["valid"]]
        if not best or not valid_candidates:
            return {
                "status": "no_valid_chain",
                "answer": "No matching facts found in KB for candidate chains.",
                "selected_entity": best["entity"] if best else None,
                "selected_chain": best["chain"] if best else [],
                "edges": [],
                "candidates": ranked_candidates,
                "matched_rules": [],
                "token_usage": {
                    "parse1_tokens": p1_tokens,
                    "correction_tokens": correction_tokens,
                },
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_f1": 0.0,
                "subgraph_size": 0,
                "grammar_hit": False,
                "has_references": bool(references),
                "references": references or [],
                "spine_edges": [],
                "expanded_edges": [],
                "retrieval_policy": "none",
                "expansion_gate": "no_valid_chain",
                "selected_candidate": best or {},
                "parse_latency": parse_latency,
                "retrieval_latency": retrieval_latency,
                "total_prepare_latency": time.perf_counter() - total_t0,
            }

        retrieval_t0 = time.perf_counter()
        subgraph_candidates = []
        for cand in valid_candidates:
            subgraph = self._build_candidate_subgraph(
                cand["entity"],
                cand["chain"],
                references=references,
            )
            subgraph_ranking_key = self._score_subgraph_candidate(cand, subgraph, has_references=bool(references))
            row = dict(cand)
            row.update({
                "subgraph_ranking_key": subgraph_ranking_key,
                "grammar_hit": subgraph["grammar_hit"],
                "matched_rules": subgraph["matched_rules"],
                "retrieval_recall": subgraph["retrieval_recall"],
                "retrieval_precision": subgraph["retrieval_precision"],
                "retrieval_f1": subgraph["retrieval_f1"],
                "subgraph_size": subgraph["subgraph_size"],
                "edges": subgraph["final_edges"],
                "spine_edges": subgraph["spine_edges"],
                "expanded_edges": subgraph["expanded_edges"],
                "retrieval_policy": subgraph.get("retrieval_policy", "unknown"),
                "expansion_gate": subgraph.get("expansion_gate", ""),
                "spine_edge_count": subgraph.get("spine_edge_count", len(subgraph["spine_edges"])),
                "expanded_edge_count": subgraph.get("expanded_edge_count", len(subgraph["expanded_edges"])),
                "raw_expanded_edge_count": subgraph.get("raw_expanded_edge_count", len(subgraph["expanded_edges"])),
                "raw_final_edge_count": subgraph.get("raw_final_edge_count", len(subgraph["final_edges"])),
                "context_truncated": subgraph.get("context_truncated", False),
            })
            subgraph_candidates.append(row)

        if not subgraph_candidates:
            return {
                "status": "no_edges_after_selection",
                "answer": "No matching facts found in KB after candidate selection.",
                "selected_entity": best["entity"],
                "selected_chain": best["chain"],
                "edges": [],
                "candidates": ranked_candidates,
                "matched_rules": [],
                "token_usage": {
                    "parse1_tokens": p1_tokens,
                    "correction_tokens": correction_tokens,
                },
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_f1": 0.0,
                "subgraph_size": 0,
                "grammar_hit": False,
                "has_references": bool(references),
                "references": references or [],
                "spine_edges": [],
                "expanded_edges": [],
                "retrieval_policy": "none",
                "expansion_gate": "no_edges_after_selection",
                "selected_candidate": best or {},
                "parse_latency": parse_latency,
                "retrieval_latency": retrieval_latency + (time.perf_counter() - retrieval_t0),
                "total_prepare_latency": time.perf_counter() - total_t0,
            }

        subgraph_candidates.sort(key=lambda x: x["subgraph_ranking_key"], reverse=True)
        print("[Subgraph Ranking] Top candidates:")
        for i, rc in enumerate(subgraph_candidates[:10]):
            print(
                f"  [{i}] entity={rc['entity']} chain={rc['chain']} "
                f"source={rc['source']} policy={rc.get('retrieval_policy')} edges={len(rc['edges'])} "
                f"rr={rc['retrieval_recall']:.3f} rp={rc['retrieval_precision']:.3f} "
                f"rf1={rc['retrieval_f1']:.3f} s_key={rc['subgraph_ranking_key']}",
                flush=True,
            )

        chosen = subgraph_candidates[0]
        selected_entity = chosen["entity"]
        selected_chain = chosen["chain"]
        matched_rules = chosen["matched_rules"]
        grammar_hit = chosen["grammar_hit"]
        final_edges = chosen["edges"]
        spine_edges = chosen["spine_edges"]
        expanded_edges = chosen["expanded_edges"]
        retrieval_policy = chosen.get("retrieval_policy", "unknown")
        expansion_gate = chosen.get("expansion_gate", "")

        if matched_rules:
            print(self.hrg.summarize_matched(matched_rules), flush=True)

        print(
            f"[Edges] policy={retrieval_policy} gate={expansion_gate} "
            f"spine={len(spine_edges)} expanded={len(expanded_edges)} "
            f"total={len(final_edges)}",
            flush=True
        )

        retrieval_recall = chosen["retrieval_recall"]
        retrieval_precision = chosen["retrieval_precision"]
        retrieval_f1 = chosen["retrieval_f1"]
        subgraph_size = chosen["subgraph_size"]
        if references:
            print(
                f"[Metrics] recall={retrieval_recall:.3f} | precision={retrieval_precision:.3f} | "
                f"f1={retrieval_f1:.3f} | subgraph_size={subgraph_size}",
                flush=True
            )

        return {
            "status": "ok",
            "answer": None,
            "selected_entity": selected_entity,
            "selected_chain": selected_chain,
            "edges": final_edges,
            "candidates": subgraph_candidates,
            "matched_rules": matched_rules[: self.topk_expansion_rules],
            "token_usage": {
                "parse1_tokens": p1_tokens,
                "correction_tokens": correction_tokens,
            },
            "retrieval_recall": retrieval_recall,
            "retrieval_precision": retrieval_precision,
            "retrieval_f1": retrieval_f1,
            "subgraph_size": subgraph_size,
            "grammar_hit": grammar_hit,
            "has_references": bool(references),
            "references": references or [],
            "spine_edges": spine_edges,
            "expanded_edges": expanded_edges,
            "retrieval_policy": retrieval_policy,
            "expansion_gate": expansion_gate,
            "selected_candidate": chosen,
            "spine_edge_count": chosen.get("spine_edge_count", len(spine_edges)),
            "expanded_edge_count": chosen.get("expanded_edge_count", len(expanded_edges)),
            "raw_expanded_edge_count": chosen.get("raw_expanded_edge_count", len(expanded_edges)),
            "raw_final_edge_count": chosen.get("raw_final_edge_count", len(final_edges)),
            "context_truncated": chosen.get("context_truncated", False),
            "parse_latency": parse_latency,
            "retrieval_latency": retrieval_latency + (time.perf_counter() - retrieval_t0),
            "total_prepare_latency": time.perf_counter() - total_t0,
        }

    async def answer_from_prepared(
        self,
        user_prompt: str,
        prepared: Dict[str, Any],
        save_path: Optional[str] = None,
        serialization_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        generation_t0 = time.perf_counter()
        if serialization_format is None:
            serialization_format = self.serialization_format
        retrieval_diagnostics = {
            "spine_edge_count": prepared.get("spine_edge_count", len(prepared.get("spine_edges", []))),
            "expanded_edge_count": prepared.get("expanded_edge_count", len(prepared.get("expanded_edges", []))),
            "raw_expanded_edge_count": prepared.get("raw_expanded_edge_count", len(prepared.get("expanded_edges", []))),
            "raw_final_edge_count": prepared.get("raw_final_edge_count", len(prepared.get("edges", []))),
            "context_truncated": prepared.get("context_truncated", False),
        }

        original_format = self.serialization_format
        self.serialization_format = serialization_format
        try:
            status = prepared.get("status")
            if status != "ok":
                answer = prepared.get("answer") or "No matching facts found in KB."
                parse2_tokens = 0
                context_tokens = 0
                final_context = ""
                if save_path:
                    self._dump(
                        save_path=save_path,
                        question=user_prompt,
                        selected_entity=prepared.get("selected_entity"),
                        selected_chain=prepared.get("selected_chain", []),
                        edges=[],
                        answer=answer,
                        candidates=prepared.get("candidates", []),
                        matched_rules=prepared.get("matched_rules", []),
                        token_usage={
                            "parse1_tokens": prepared.get("token_usage", {}).get("parse1_tokens", 0),
                            "correction_tokens": prepared.get("token_usage", {}).get("correction_tokens", 0),
                            "parse2_tokens": 0,
                            "context_tokens": 0,
                        },
                        retrieval_recall=prepared.get("retrieval_recall", 0.0),
                        retrieval_precision=prepared.get("retrieval_precision", 0.0),
                        retrieval_f1=prepared.get("retrieval_f1", 0.0),
                        subgraph_size=prepared.get("subgraph_size", 0),
                        failure_stage=status or "unknown",
                        grammar_hit=prepared.get("grammar_hit", False),
                        serialization_format=serialization_format,
                        references=prepared.get("references", []),
                        spine_edges=prepared.get("spine_edges", []),
                        expanded_edges=prepared.get("expanded_edges", []),
                        selected_candidate=prepared.get("selected_candidate", {}),
                        final_context=final_context,
                        timing={
                            "parse_latency": prepared.get("parse_latency", 0.0),
                            "retrieval_latency": prepared.get("retrieval_latency", 0.0),
                            "generation_latency": 0.0,
                        },
                        retrieval_policy=prepared.get("retrieval_policy", ""),
                        expansion_gate=prepared.get("expansion_gate", ""),
                        retrieval_diagnostics=retrieval_diagnostics,
                    )
            else:
                final_edges = prepared.get("edges", [])
                context_str = self._serialize_edges(final_edges)
                context_tokens = self._estimate_tokens(context_str)
                print(f"[Serialize] format={serialization_format} | ctx_tokens~{context_tokens}", flush=True)
                answer, parse2_tokens = await self._generate_rag_response(user_prompt, context_str)
                print(f"[Parse 2] response generated | tokens~{parse2_tokens}", flush=True)
                if save_path:
                    self._dump(
                        save_path=save_path,
                        question=user_prompt,
                        selected_entity=prepared.get("selected_entity"),
                        selected_chain=prepared.get("selected_chain", []),
                        edges=final_edges,
                        answer=answer,
                        candidates=prepared.get("candidates", []),
                        matched_rules=prepared.get("matched_rules", []),
                        token_usage={
                            "parse1_tokens": prepared.get("token_usage", {}).get("parse1_tokens", 0),
                            "correction_tokens": prepared.get("token_usage", {}).get("correction_tokens", 0),
                            "parse2_tokens": parse2_tokens,
                            "context_tokens": context_tokens,
                        },
                        retrieval_recall=prepared.get("retrieval_recall", 0.0),
                        retrieval_precision=prepared.get("retrieval_precision", 0.0),
                        retrieval_f1=prepared.get("retrieval_f1", 0.0),
                        subgraph_size=prepared.get("subgraph_size", 0),
                        failure_stage=prepared.get("status", "ok"),
                        grammar_hit=prepared.get("grammar_hit", False),
                        serialization_format=serialization_format,
                        references=prepared.get("references", []),
                        spine_edges=prepared.get("spine_edges", []),
                        expanded_edges=prepared.get("expanded_edges", []),
                        selected_candidate=prepared.get("selected_candidate", {}),
                        final_context=context_str,
                        timing={
                            "parse_latency": prepared.get("parse_latency", 0.0),
                            "retrieval_latency": prepared.get("retrieval_latency", 0.0),
                            "generation_latency": time.perf_counter() - generation_t0,
                        },
                        retrieval_policy=prepared.get("retrieval_policy", ""),
                        expansion_gate=prepared.get("expansion_gate", ""),
                        retrieval_diagnostics=retrieval_diagnostics,
                    )
        finally:
            self.serialization_format = original_format

        self._accumulate_prepared_stats(prepared, parse2_tokens, context_tokens)
        elapsed = time.time() - start_time
        return {
            "answer": answer,
            "parse2_tokens": parse2_tokens,
            "context_tokens": context_tokens,
            "elapsed": elapsed,
            "retrieval_recall": prepared.get("retrieval_recall", 0.0),
            "retrieval_precision": prepared.get("retrieval_precision", 0.0),
            "retrieval_f1": prepared.get("retrieval_f1", 0.0),
            "subgraph_size": prepared.get("subgraph_size", 0),
            "status": prepared.get("status"),
            "retrieval_policy": prepared.get("retrieval_policy", ""),
            "expansion_gate": prepared.get("expansion_gate", ""),
            "generation_failed": not bool((answer or "").strip()) and prepared.get("status") == "ok",
            "generation_latency": time.perf_counter() - generation_t0,
        }

    async def ask(
        self,
        user_prompt: str,
        save_path: Optional[str] = None,
        references: Optional[List[str]] = None,
    ) -> str:
        prepared = await self.prepare_retrieval(user_prompt, references=references)
        result = await self.answer_from_prepared(
            user_prompt,
            prepared,
            save_path=save_path,
            serialization_format=self.serialization_format,
        )
        return result["answer"]

    async def ask_detailed(
        self,
        user_prompt: str,
        save_path: Optional[str] = None,
        references: Optional[List[str]] = None,
        hop_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        prepared = await self.prepare_retrieval(user_prompt, references=references)
        result = await self.answer_from_prepared(
            user_prompt,
            prepared,
            save_path=save_path,
            serialization_format=self.serialization_format,
        )
        return {
            "answer": result["answer"],
            "failure_stage": prepared.get("status", "ok"),
            "parse_latency": prepared.get("parse_latency", 0.0),
            "retrieval_latency": prepared.get("retrieval_latency", 0.0),
            "generation_latency": result.get("generation_latency", result.get("elapsed", 0.0)),
            "generation_failed": result.get("generation_failed", False),
            "answerable": bool((result.get("answer") or "").strip()),
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

    async def _correct_candidates_with_llm(
        self,
        user_prompt: str,
        failed_candidates: List[Dict[str, Any]],
        max_new_candidates: int = 5
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Optional correction step: ask LLM to revise failed chains.
        This is still lighter than grammar-first because it happens only when needed.
        """
        allowed_rels_str = json.dumps(self._select_relation_prompt_candidates(user_prompt), ensure_ascii=False)
        failed_json = json.dumps(failed_candidates, ensure_ascii=False)

        developer_instruction = (
            "You are a KG chain correction module.\n"
            "You are given a question and several FAILED candidate chains.\n"
            "Your task is to propose corrected alternatives.\n"
            "Rules:\n"
            f"  - Prefer relations from this candidate set: {allowed_rels_str}\n"
            "  - If needed, you may use another KG relation token only when clearly more accurate.\n"
            "  - Use only bare relation names.\n"
            "  - You may replace the chain with a better one if needed.\n"
            "  - Prefer same hop count as the question requires.\n"
            "Return ONLY a valid JSON array with corrected candidates.\n"
            "Format:\n"
            '[{"entity": "<topic entity>", "chain": ["<rel1>", "<rel2>", ...]}, ...]'
        )
        user_content = f"Question: {user_prompt}\nFailed candidates: {failed_json}"

        input_tokens = self._estimate_tokens(developer_instruction, user_content)
        raw = await self._inference(developer_instruction, user_content)
        clean = self._extract_final_content(raw)
        total_tokens = input_tokens + self._estimate_tokens(clean)

        try:
            arr = json.loads(clean)
            if not isinstance(arr, list):
                return [], total_tokens
        except Exception:
            return [], total_tokens

        repaired = []
        for item in arr[:max_new_candidates]:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            chain = item.get("chain", [])
            if not entity or not isinstance(chain, list):
                continue
            chain = [str(x).strip() for x in chain if str(x).strip()]
            if not chain:
                continue
            repaired.append({
                "entity": entity,
                "chain": chain,
                "source": "llm_correction",
                "confidence": max(0.0, min(1.0, self._safe_float(item.get("confidence", 0.0), 0.0))),
            })
        return repaired, total_tokens


# ============================================================
# Debug Entry Point
# ============================================================
if __name__ == "__main__":
    import asyncio

    GRAMMAR_PATH = "../hrg_grammar/metaqa_phrg_grammar.json"
    KB_PATH = "../Datasets/MetaQA/kb.txt"
    REL_PATH = "../Datasets/MetaQA/relations.json"

    TEST_QUESTION = "What movies did [Tom Hanks] star in?"

    async def run_test():
        agent = KnowledgeGraphAgent(
            model_id="openai/gpt-oss-20b",
            kb_path=KB_PATH,
            relation_path=REL_PATH,
            grammar_path=GRAMMAR_PATH,
            serialization_format="json",   # or "triples"
            num_candidates=5,
            use_grammar_rerank=True,
            use_grammar_expansion=True,
            use_fallback_correction=True,
            topk_expansion_rules=1,
        )

        print("=" * 80)
        print(f"[Test Question] {TEST_QUESTION}")
        print("=" * 80)

        answer = await agent.ask(TEST_QUESTION, save_path=None)
        print(f"\n[Final Answer] {answer}\n")

        print("=" * 80)
        print("[Token Stats]")
        print("=" * 80)
        print(f"total_questions       = {agent.total_questions}")
        print(f"total_parse1_tokens   = {agent.total_parse1_tokens}")
        print(f"total_correction_tokens = {agent.total_correction_tokens}")
        print(f"total_parse2_tokens   = {agent.total_parse2_tokens}")
        print(f"total_context_length  = {agent.total_context_length}")
        print(f"hit_grammar_count     = {agent.hit_grammar_count}")

    asyncio.run(run_test())
