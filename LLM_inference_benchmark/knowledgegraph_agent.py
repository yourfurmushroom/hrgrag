from __future__ import annotations

import json
import re
import pickle
import time
import random
import hashlib
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any, Set

from pathlib import Path

from agent_factory import build_llm_strategy
from dataset_utils import build_node_index, load_kb_adjacency, load_relation_list, normalize_lookup_key


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
        for rule in self.grammar:
            rule["_cached_labels"] = set(self._extract_labels(rule))

    def _extract_labels(self, rule: Dict[str, Any]) -> List[str]:
        rhs = rule.get("rhs", {})
        labels = []
        if "terminal_edges" in rhs:
            for t in rhs["terminal_edges"]:
                labels.append(t.get("label"))
        elif "terminals" in rhs:
            for t in rhs["terminals"]:
                labels.append(t.get("rel"))
        return [l for l in labels if l]

    def _chain_to_bare(self, chain: List[str]) -> List[str]:
        return [c[:-3] if c.endswith("^-1") else c for c in chain]

    def match_rules(self, chain: List[str]) -> List[Dict[str, Any]]:
        if not chain:
            return []
        bare_chain = set(self._chain_to_bare(chain))
        matched = []

        for rule in self.grammar:
            labels = rule.get("_cached_labels", set())
            if bare_chain.issubset(labels):
                matched.append(rule)

        matched.sort(key=lambda r: r.get("probability", r.get("count", 0)), reverse=True)
        print(f"[HRGMatcher] chain={chain} -> bare={bare_chain} -> matched {len(matched)} rules")
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


# ============================================================
# KnowledgeGraphAgent
# ============================================================

class KnowledgeGraphAgent:
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
        use_random_expansion: bool = False,
        use_frequency_expansion: bool = False,
        random_expansion_seed: int = 0,
        # ---- Ablation D: context top-K edge truncation ----
        top_k_edges: Optional[int] = None,
        # ---- HRG-D: grammar-guided constrained BFS retrieval (original design) ----
        use_grammar_guided_retrieval: bool = False,
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
        self.use_random_expansion = use_random_expansion
        self.use_frequency_expansion = use_frequency_expansion
        self.random_expansion_seed = random_expansion_seed
        self.top_k_edges = top_k_edges
        self.use_grammar_guided_retrieval = use_grammar_guided_retrieval

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

        print("[Init] Loading KB + building adjacency index...")
        self.kb_out, self.kb_in, self.all_nodes, derived_relations, self.alias_map = load_kb_adjacency(
            self.kb_path,
            max_triples=self.max_kb_triples,
            sanitize_entity_fn=self._sanitize_entity,
        )
        self.relations = self._load_relations(relation_path, derived_relations)
        self.allowed_rel_set = set(self.relations)
        self.allowed_rel_tokens = set(self.relations) | {f"{r}^-1" for r in self.relations}
        self._node_index = build_node_index(self.all_nodes, self.alias_map)
        self.relation_frequency = self._compute_relation_frequency()

        if grammar_path and Path(grammar_path).exists():
            self.hrg = HRGMatcher(grammar_path)
        else:
            self.hrg = None
            print("[Init] No grammar path provided or file not found. HRG disabled.")

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
        return rels or derived_relations

    def _compute_relation_frequency(self) -> Dict[str, int]:
        freq: Dict[str, int] = defaultdict(int)
        for head_rels in self.kb_out.values():
            for rel, tails in head_rels.items():
                freq[rel] += len(tails)
        return dict(freq)

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
            relation_scores[rel] += 100 + min(len(tails), 10)
            inv = f"{rel}^-1"
            if inv in self.allowed_rel_tokens:
                relation_scores[inv] += 10

        for rel, heads in self.kb_in.get(node, {}).items():
            inv = f"{rel}^-1"
            if inv in self.allowed_rel_tokens:
                relation_scores[inv] += 100 + min(len(heads), 10)
            if rel in self.allowed_rel_tokens:
                relation_scores[rel] += 10

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

    def _fuzzy_match_relation(self, raw: str) -> Optional[str]:
        if raw in self.allowed_rel_tokens:
            return raw
        norm = raw.strip().lower().replace(" ", "_")
        norm_inv = norm + "^-1" if not norm.endswith("^-1") else norm
        for tok in self.allowed_rel_tokens:
            tnorm = tok.lower().replace(" ", "_")
            if tnorm == norm or tnorm == norm_inv:
                return tok
        return None

    def _normalize_entity_from_question(self, user_prompt: str, entity: Optional[str]) -> Optional[str]:
        if entity and entity.strip():
            return entity.strip()
        m = re.search(r"\[(.*?)\]", user_prompt)
        if m:
            return m.group(1).strip()
        return entity

    def _flip_relation_direction(self, rel: str) -> str:
        return rel[:-3] if rel.endswith("^-1") else rel + "^-1"

    def _chain_to_bare(self, chain: List[str]) -> List[str]:
        return [c[:-3] if c.endswith("^-1") else c for c in chain]

    def _dedup_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for c in candidates:
            entity = (c.get("entity") or "").strip()
            chain = c.get("chain") or []
            key = (entity, tuple(chain))
            if key not in seen and entity and chain:
                seen.add(key)
                deduped.append({
                    "entity": entity,
                    "chain": chain,
                    "source": c.get("source", "llm"),
                    "confidence": float(c.get("confidence", 0.0) or 0.0),
                })
        return deduped

    def _normalize_match_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("^-1", " inverse ")
        text = text.replace("_", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _select_relation_prompt_candidates(self, user_prompt: str, limit: int = 64) -> List[str]:
        if len(self.allowed_rel_tokens) <= limit:
            return sorted(self.allowed_rel_tokens)

        question_terms = set(self._normalize_match_text(user_prompt).split())
        scored: List[Tuple[int, str]] = []
        prioritized_entity = self._normalize_entity_from_question(user_prompt, None)

        for rel in self._relation_neighbors_for_entity(prioritized_entity, max_relations=min(limit, 48)):
            scored.append((40, rel))

        for rel in sorted(self.allowed_rel_tokens):
            rel_terms = set(self._normalize_match_text(rel).split())
            overlap = len(question_terms & rel_terms)
            inverse_bonus = 1 if rel.endswith("^-1") and "inverse" in question_terms else 0
            score = overlap * 10 + inverse_bonus
            if score > 0:
                scored.append((score, rel))

        if self.hrg:
            for rule in self.hrg.topk_rules(k=min(self.grammar_hint_topk, 10)):
                for label in sorted(rule.get("_cached_labels", set())):
                    if label in self.allowed_rel_tokens:
                        scored.append((5, label))
                    inv = f"{label}^-1"
                    if inv in self.allowed_rel_tokens:
                        scored.append((4, inv))

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

        rel_semantics = (
            "\n[Relation Direction Guide]\n"
            "Each base relation is stored as: (Film) --relation--> (Value).\n"
            "Use relation^-1 when starting from the Value side and moving back to Film.\n"
            "Examples:\n"
            "  written_by         : Film -> Writer      | written_by^-1      : Writer -> Film\n"
            "  release_year       : Film -> Year        | release_year^-1    : Year -> Film\n"
            "  has_imdb_rating    : Film -> Rating      | has_imdb_rating^-1 : Rating -> Film\n"
            "  starred_actors     : Film -> Actor       | starred_actors^-1  : Actor -> Film\n"
            "  has_imdb_votes     : Film -> Votes       | has_imdb_votes^-1  : Votes -> Film\n"
            "  directed_by        : Film -> Director    | directed_by^-1     : Director -> Film\n"
            "  has_tags           : Film -> Tag         | has_tags^-1        : Tag -> Film\n"
            "  has_genre          : Film -> Genre       | has_genre^-1       : Genre -> Film\n"
            "  in_language        : Film -> Language    | in_language^-1     : Language -> Film\n"
        )

        few_shot = (
            "\n[Examples]\n"
            'Q: "What genre is [The Matrix]?"\n'
            '[{"entity": "The Matrix", "chain": ["has_genre"]}]\n\n'
            'Q: "What movies did [Tom Hanks] star in?"\n'
            '[{"entity": "Tom Hanks", "chain": ["starred_actors^-1"]}]\n\n'
            'Q: "Who directed the films that [Tom Hanks] starred in?"\n'
            '[{"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"]}]\n\n'
            'Q: "What genres do the films written by the writer of [Cast Away] belong to?"\n'
            '[{"entity": "Cast Away", "chain": ["written_by", "written_by^-1", "has_genre"]}]\n'
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
            "  - If needed, you may use another KG relation token only when clearly more accurate.\n"
            "  - If needed, use ^-1 for reverse traversal.\n"
            "  - 1-hop question -> 1 relation, 2-hop question -> 2 relations, 3-hop question -> 3 relations.\n"
            "  - Prefer structurally coherent chains that are likely to recur in the KG.\n"
            "  - Return multiple diverse candidates ranked from most likely to less likely.\n"
            "  - Do NOT use markdown fences or explanations.\n"
            f"{rel_semantics}"
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

    def _neighbors_for_token(self, ent: str, rel_token: str) -> List[Tuple[str, str, str]]:
        if rel_token.endswith("^-1"):
            rel = rel_token[:-3]
            heads = list(self.kb_in.get(ent, {}).get(rel, []))
            if self.per_entity_cap and len(heads) > self.per_entity_cap:
                heads = heads[:self.per_entity_cap]
            return [(h, rel, ent) for h in heads]
        else:
            rel = rel_token
            tails = list(self.kb_out.get(ent, {}).get(rel, []))
            if self.per_entity_cap and len(tails) > self.per_entity_cap:
                tails = tails[:self.per_entity_cap]
            return [(ent, rel, t) for t in tails]

    def _check_chain_validity(self, entity: str, chain: List[str]) -> Dict[str, Any]:
        """
        Returns whether the chain is executable from the entity in the KB.
        """
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
                    next_frontier.add(h if rel.endswith("^-1") else t)

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
                    next_frontier.add(h if rel_token.endswith("^-1") else t)

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

    def _get_grammar_match_features(self, chain: List[str]) -> Dict[str, Any]:
        if not self.hrg or not chain:
            return {
                "score": 0.0,
                "hit": 0,
                "same_arity_hit": 0,
                "matched_count": 0,
            }
        matched = self.hrg.match_rules(chain)
        if not matched:
            return {
                "score": 0.0,
                "hit": 0,
                "same_arity_hit": 0,
                "matched_count": 0,
            }

        same_arity_rules = []
        for rule in matched:
            lhs = rule.get("lhs", {})
            arity = lhs.get("rank", rule.get("arity", -1)) if isinstance(lhs, dict) else rule.get("arity", -1)
            if arity == len(chain):
                same_arity_rules.append(rule)

        preferred = same_arity_rules[0] if same_arity_rules else matched[0]
        return {
            "score": float(preferred.get("probability", preferred.get("count", 0))),
            "hit": 1,
            "same_arity_hit": 1 if same_arity_rules else 0,
            "matched_count": len(matched),
        }

    def _select_matched_rules(
        self,
        chain: List[str],
        top_k: Optional[int] = None,
        require_same_arity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Select the most relevant HRG rules for a predicted relation chain.

        We prefer rules whose arity matches the chain length so that grammar is
        used as a retrieval constraint around the predicted path, rather than as
        a broad high-frequency neighborhood prior.
        """
        if not self.hrg or not chain:
            return []

        matched_rules = self.hrg.match_rules(chain)
        if not matched_rules:
            return []

        if top_k is None:
            top_k = self.topk_expansion_rules

        if require_same_arity:
            chain_len = len(chain)
            filtered = []
            for rule in matched_rules:
                lhs = rule.get("lhs", {})
                arity = lhs.get("rank", rule.get("arity", -1)) if isinstance(lhs, dict) else rule.get("arity", -1)
                if arity == chain_len:
                    filtered.append(rule)
            if filtered:
                matched_rules = filtered

        return matched_rules[:top_k]

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
                lhs = r.get("lhs", {})
                arity = lhs.get("rank", r.get("arity", -1)) if isinstance(lhs, dict) else r.get("arity", -1)
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
        Generate candidates by flipping one relation direction at a time.
        """
        out = []
        for i in range(len(chain)):
            new_chain = list(chain)
            new_chain[i] = self._flip_relation_direction(new_chain[i])
            out.append({"entity": entity, "chain": new_chain, "source": f"flip_hop_{i+1}"})
        return out

    def _make_grammar_fallback_candidates(self, entity: str, chain: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback candidates from grammar rules by matching bare relations.
        We keep the original direction when possible, and fill missing direction
        using the original chain if relation names overlap.
        """
        if not self.hrg:
            return []

        bare_chain = self._chain_to_bare(chain)
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

            # reconstruct directions based on original chain when possible
            reconstructed = []
            for rel in rels[:max(1, len(chain))]:
                # if same bare relation appeared in original chain, reuse its direction
                found = None
                for old_rel in chain:
                    if old_rel[:-3] if old_rel.endswith("^-1") else old_rel == rel:
                        found = old_rel
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
        source: str = "llm"
    ) -> Tuple[Any, ...]:
        """
        Interpretable lexicographic ranking key.

        Higher tuple values are preferred.
        Priority order:
        1. KB executability
        2. Same-arity HRG compatibility
        3. HRG compatibility strength
        4. Later failure hop (or fully executable)
        5. Retrieval survivability across hops
        6. Final frontier size
        7. Candidate source prior
        8. LLM confidence / rank as tie-breakers
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
        grammar_score = self._safe_float(grammar_features.get("score", 0.0), 0.0)
        grammar_matched_count = min(self._safe_int(grammar_features.get("matched_count", 0), 0), 20)
        step_survival = sum(min(self._safe_int(s, 0), 10) for s in step_sizes)
        final_size = min(self._safe_int(kb_result.get("final_size", 0), 0), 20)

        source_priority = 0
        if source == "llm":
            source_priority = 3
        elif source == "llm_correction":
            source_priority = 2
        elif source == "grammar_fallback":
            source_priority = 0
        elif source.startswith("flip_hop_"):
            source_priority = 1

        llm_prior = -self._safe_int(llm_rank_index, 0)

        return (
            valid,
            same_arity_hit,
            grammar_hit,
            grammar_score,
            grammar_matched_count,
            failure_progress,
            step_survival,
            final_size,
            source_priority,
            llm_confidence,
            llm_prior,
        )

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
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_f1": 0.0,
                "subgraph_size": 0,
            }

        matched_rules = []
        expanded_edges = []
        is_single_hop = len(chain) == 1
        if not self.use_grammar_guided_retrieval and self.hrg and self.use_grammar_expansion and not is_single_hop:
            matched_rules = selected_rules
            if matched_rules:
                expanded_edges = self._expand_subgraph_by_grammar(spine_edges, matched_rules, chain=chain)
        elif self.use_frequency_expansion and not is_single_hop:
            expanded_edges = self._expand_subgraph_by_relation_frequency(spine_edges)
        elif self.use_random_expansion and not is_single_hop:
            expanded_edges = self._expand_subgraph_random(spine_edges, chain=chain)
        elif is_single_hop:
            matched_rules = selected_rules

        final_edges = spine_edges + expanded_edges
        if self.top_k_edges is not None and len(final_edges) > self.top_k_edges:
            final_edges = sorted(final_edges, key=lambda e: e.get("count", 0), reverse=True)
            final_edges = final_edges[: self.top_k_edges]

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
            "retrieval_recall": retrieval_recall,
            "retrieval_precision": retrieval_precision,
            "retrieval_f1": retrieval_f1,
            "subgraph_size": len(final_edges),
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
        spine_size = min(len(subgraph.get("spine_edges", [])), 10000)
        expanded_size = min(len(subgraph.get("expanded_edges", [])), 10000)
        # Smaller subgraphs are preferred once support quality is comparable.
        compactness = -min(subgraph.get("subgraph_size", 0), 10000)

        return (
            has_edges,
            same_arity_hit,
            grammar_hit,
            grammar_score,
            spine_size,
            expanded_size,
            compactness,
            chain_row.get("ranking_key", ()),
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
        }
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

        parse_t0 = time.perf_counter()
        candidates, p1_tokens = await self._parse_intent_candidates(user_prompt, self.num_candidates)
        parse_latency += time.perf_counter() - parse_t0
        print(f"[Parse 1] got {len(candidates)} candidates | tokens~{p1_tokens}", flush=True)

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
                "selected_candidate": {},
                "parse_latency": parse_latency,
                "retrieval_latency": 0.0,
            }

        rank_t0 = time.perf_counter()
        ranked_candidates = []
        failed_candidates = []

        for rank_idx, cand in enumerate(candidates):
            entity = cand["entity"]
            chain = cand["chain"]
            kb_result = self._check_chain_validity(entity, chain)
            grammar_features = self._get_grammar_match_features(chain) if self.use_grammar_rerank else {
                "score": 0.0, "hit": 0, "same_arity_hit": 0, "matched_count": 0
            }
            grammar_score = grammar_features["score"]
            source = cand.get("source", "llm")
            confidence = float(cand.get("confidence", 0.0) or 0.0)
            ranking_key = self._score_candidate(
                entity, chain, kb_result, grammar_features, rank_idx,
                llm_confidence=confidence, source=source
            )
            row = {
                "entity": entity,
                "chain": chain,
                "source": source,
                "confidence": confidence,
                "llm_rank_index": rank_idx,
                "kb_result": kb_result,
                "grammar_score": grammar_score,
                "grammar_hit": grammar_features["hit"],
                "grammar_same_arity_hit": grammar_features["same_arity_hit"],
                "grammar_matched_count": grammar_features["matched_count"],
                "ranking_key": ranking_key,
            }
            ranked_candidates.append(row)
            if not kb_result["valid"]:
                failed_candidates.append(row)

        correction_tokens = 0
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
                chain = cand["chain"]
                kb_result = self._check_chain_validity(entity, chain)
                grammar_features = self._get_grammar_match_features(chain) if self.use_grammar_rerank else {
                    "score": 0.0, "hit": 0, "same_arity_hit": 0, "matched_count": 0
                }
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
                    "source": source,
                    "confidence": confidence,
                    "llm_rank_index": base_len + idx,
                    "kb_result": kb_result,
                    "grammar_score": grammar_score,
                    "grammar_hit": grammar_features["hit"],
                    "grammar_same_arity_hit": grammar_features["same_arity_hit"],
                    "grammar_matched_count": grammar_features["matched_count"],
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
                "selected_candidate": best or {},
                "parse_latency": parse_latency,
                "retrieval_latency": retrieval_latency,
            }

        retrieval_t0 = time.perf_counter()
        subgraph_candidates = []
        for cand in valid_candidates:
            subgraph = self._build_candidate_subgraph(cand["entity"], cand["chain"], references=references)
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
                "grammar_hit": grammar_hit,
                "has_references": bool(references),
                "references": references or [],
                "spine_edges": [],
                "expanded_edges": [],
                "selected_candidate": best or {},
                "parse_latency": parse_latency,
                "retrieval_latency": retrieval_latency + (time.perf_counter() - retrieval_t0),
            }

        subgraph_candidates.sort(key=lambda x: x["subgraph_ranking_key"], reverse=True)
        print("[Subgraph Ranking] Top candidates:")
        for i, rc in enumerate(subgraph_candidates[:10]):
            print(
                f"  [{i}] entity={rc['entity']} chain={rc['chain']} "
                f"source={rc['source']} edges={len(rc['edges'])} "
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

        if matched_rules:
            print(self.hrg.summarize_matched(matched_rules), flush=True)

        print(
            f"[Edges] spine={len(spine_edges)} expanded={len(expanded_edges)} "
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
            "selected_candidate": chosen,
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
            "edges": prepared.get("edges", []),
            "candidates": prepared.get("candidates", []),
            "spine_edges": prepared.get("spine_edges", []),
            "expanded_edges": prepared.get("expanded_edges", []),
            "selected_candidate": prepared.get("selected_candidate", {}),
            "selected_chain": prepared.get("selected_chain", []),
            "selected_entity": prepared.get("selected_entity"),
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
            "  - You may change relation direction using ^-1.\n"
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
