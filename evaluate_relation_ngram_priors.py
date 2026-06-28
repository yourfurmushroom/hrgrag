from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_TAGS = [
    "metaqa-vanilla-test",
    "wikimovies-wiki_entities-test",
    "mlpq-en-zh-en-ills",
    "kqapro-validation",
]


def normalize_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def split_answer_items(text: Any) -> List[str]:
    out = []
    seen = set()
    for part in re.split(r"\s*\|\s*|\s*;\s*|\n+", str(text or "")):
        norm = normalize_text(part)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def edge_answer_coverage(edges: List[Dict[str, Any]], references: List[str]) -> float:
    if not edges or not references:
        return 0.0

    nodes = set()
    for edge in edges:
        for key in ("head", "tail"):
            norm = normalize_text(edge.get(key, ""))
            if norm:
                nodes.add(norm)

    ref_norms = [normalize_text(ref) for ref in references if normalize_text(ref)]
    if not nodes or not ref_norms:
        return 0.0

    hit = 0
    for ref in ref_norms:
        if any(ref in node or node in ref for node in nodes):
            hit += 1
    return hit / len(ref_norms)


def extract_rule_edges(rule: Dict[str, Any]) -> List[Tuple[Any, str, Any]]:
    rhs = rule.get("rhs", {}) if isinstance(rule, dict) else {}
    raw_edges = rhs.get("terminal_edges") or rhs.get("terminals") or []
    edges = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        rel = edge.get("rel") or edge.get("label") or edge.get("relation")
        if not rel:
            continue
        src = edge.get("a", edge.get("src", edge.get("source", edge.get("head"))))
        dst = edge.get("b", edge.get("dst", edge.get("target", edge.get("tail"))))
        edges.append((src, str(rel), dst))
    return edges


def load_grammar_rules(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "grammar" in data:
        return list(data.get("grammar", {}).get("production_rules", []))
    if isinstance(data, list):
        return data
    return []


class RelationNgramPrior:
    def __init__(self, rules: List[Dict[str, Any]], alpha: float = 0.1):
        self.alpha = alpha
        self.unigrams: Counter[Tuple[str, ...]] = Counter()
        self.bigrams: Counter[Tuple[str, ...]] = Counter()
        self.trigrams: Counter[Tuple[str, ...]] = Counter()
        self.prefix_1: Counter[Tuple[str, ...]] = Counter()
        self.prefix_2: Counter[Tuple[str, ...]] = Counter()
        self.vocab = set()

        for rule in rules:
            labels = [rel for _, rel, _ in extract_rule_edges(rule) if rel]
            if not labels:
                continue
            weight = float(rule.get("probability", rule.get("count", 1)) or 1)
            for label in labels:
                self.vocab.add(label)
                self.unigrams[(label,)] += weight
            for ngram in zip(labels, labels[1:]):
                self.bigrams[tuple(ngram)] += weight
                self.prefix_1[(ngram[0],)] += weight
            for ngram in zip(labels, labels[1:], labels[2:]):
                self.trigrams[tuple(ngram)] += weight
                self.prefix_2[(ngram[0], ngram[1])] += weight

        self.total_unigrams = sum(self.unigrams.values())
        self.vocab_size = max(1, len(self.vocab))

    def _log_unigram_prob(self, rel: str) -> float:
        return math.log(
            (self.unigrams.get((rel,), 0.0) + self.alpha)
            / (self.total_unigrams + self.alpha * self.vocab_size)
        )

    def _log_bigram_prob(self, prev: str, rel: str) -> float:
        prefix = self.prefix_1.get((prev,), 0.0)
        return math.log(
            (self.bigrams.get((prev, rel), 0.0) + self.alpha)
            / (prefix + self.alpha * self.vocab_size)
        )

    def _log_trigram_prob(self, prev2: str, prev1: str, rel: str) -> float:
        prefix = self.prefix_2.get((prev2, prev1), 0.0)
        return math.log(
            (self.trigrams.get((prev2, prev1, rel), 0.0) + self.alpha)
            / (prefix + self.alpha * self.vocab_size)
        )

    def score(self, chain: List[str], order: int) -> float:
        if not chain:
            return float("-inf")
        if order <= 1 or len(chain) == 1:
            return sum(self._log_unigram_prob(rel) for rel in chain) / len(chain)
        if order == 2 or len(chain) == 2:
            vals = [self._log_unigram_prob(chain[0])]
            vals.extend(self._log_bigram_prob(a, b) for a, b in zip(chain, chain[1:]))
            return sum(vals) / len(vals)

        vals = [self._log_unigram_prob(chain[0])]
        vals.append(self._log_bigram_prob(chain[0], chain[1]))
        vals.extend(
            self._log_trigram_prob(a, b, c)
            for a, b, c in zip(chain, chain[1:], chain[2:])
        )
        return sum(vals) / len(vals)

    def observed_ngram_count(self, chain: List[str], order: int) -> int:
        if order == 1:
            return sum(1 for rel in chain if (rel,) in self.unigrams)
        if order == 2:
            return sum(1 for bg in zip(chain, chain[1:]) if tuple(bg) in self.bigrams)
        return sum(1 for tg in zip(chain, chain[1:], chain[2:]) if tuple(tg) in self.trigrams)


def candidate_chain(candidate: Dict[str, Any]) -> List[str]:
    chain = candidate.get("chain") or candidate.get("requested_chain") or []
    return [str(rel) for rel in chain if rel]


def candidate_metric(candidate: Dict[str, Any], key: str) -> float:
    value = candidate.get(key)
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def candidate_valid(candidate: Dict[str, Any]) -> int:
    return 1 if (candidate.get("kb_result") or {}).get("valid") else 0


def selected_candidate_key(candidate: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], str]:
    return (
        str(candidate.get("entity") or ""),
        tuple(candidate_chain(candidate)),
        str(candidate.get("source") or ""),
    )


def rank_candidates(
    candidates: List[Dict[str, Any]],
    mode: str,
    prior: RelationNgramPrior,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    scored = []
    for idx, candidate in enumerate(candidates):
        chain = candidate_chain(candidate)
        valid = candidate_valid(candidate)
        confidence = float(candidate.get("confidence") or 0.0)
        if mode == "hrg_score":
            score = (
                float(candidate.get("grammar_score") or 0.0)
                + 2.0 * float(candidate.get("grammar_same_arity_hit") or 0.0)
                + 1.0 * float(candidate.get("grammar_ordered_path_hit") or 0.0)
                + 0.1 * min(float(candidate.get("grammar_matched_count") or 0.0), 20.0)
            )
        elif mode == "llm_confidence":
            score = confidence
        elif mode == "relation_unigram":
            score = prior.score(chain, 1)
        elif mode == "relation_bigram":
            score = prior.score(chain, 2)
        elif mode == "relation_trigram":
            score = prior.score(chain, 3)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Keep the method surface close to the current pipeline: candidates are
        # first KB-validated, then ranked by the tested structural prior.
        scored.append((valid, score, confidence, -idx, candidate))

    scored.sort(reverse=True)
    return scored[0][-1]


def iter_question_dumps(model_dir: Path) -> Iterable[Path]:
    for path in sorted(model_dir.glob("*-hop/q_*.pkl")):
        yield path


def evaluate_run(
    run_tag: str,
    model_dir: Path,
    grammar_path: Path,
    modes: List[str],
    alpha: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rules = load_grammar_rules(grammar_path)
    prior = RelationNgramPrior(rules, alpha=alpha)
    rows = []

    for dump_path in iter_question_dumps(model_dir):
        with dump_path.open("rb") as f:
            dump = pickle.load(f)
        candidates = dump.get("candidates") or []
        references = [str(ref) for ref in (dump.get("references") or [])]
        original_key = selected_candidate_key(dump.get("selected_candidate") or {})
        hop = dump_path.parent.name
        qid = dump_path.stem

        for mode in modes:
            selected = rank_candidates(candidates, mode, prior)
            if not selected:
                continue
            chain = candidate_chain(selected)
            selected_key = selected_candidate_key(selected)
            rows.append({
                "run_tag": run_tag,
                "model": model_dir.name,
                "hop": hop,
                "qid": qid,
                "mode": mode,
                "candidate_count": len(candidates),
                "selected_valid": candidate_valid(selected),
                "selected_source": selected.get("source", ""),
                "selected_chain": " | ".join(chain),
                "selected_same_as_original": 1 if selected_key == original_key else 0,
                "retrieval_recall": candidate_metric(selected, "retrieval_recall"),
                "retrieval_precision": candidate_metric(selected, "retrieval_precision"),
                "retrieval_f1": candidate_metric(selected, "retrieval_f1"),
                "answer_in_edges": edge_answer_coverage(selected.get("edges", []), references),
                "subgraph_size": len(selected.get("edges") or []),
                "ngram_score_1": prior.score(chain, 1),
                "ngram_score_2": prior.score(chain, 2),
                "ngram_score_3": prior.score(chain, 3),
                "observed_unigrams": prior.observed_ngram_count(chain, 1),
                "observed_bigrams": prior.observed_ngram_count(chain, 2),
                "observed_trigrams": prior.observed_ngram_count(chain, 3),
            })

    meta = {
        "rules": len(rules),
        "relation_vocab": len(prior.vocab),
        "unigrams": len(prior.unigrams),
        "bigrams": len(prior.bigrams),
        "trigrams": len(prior.trigrams),
    }
    return rows, meta


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_tag"], row["mode"])].append(row)

    out = []
    for (run_tag, mode), items in sorted(grouped.items()):
        out.append({
            "run_tag": run_tag,
            "mode": mode,
            "n": len(items),
            "valid_rate": mean(float(r["selected_valid"]) for r in items),
            "same_as_original": mean(float(r["selected_same_as_original"]) for r in items),
            "retrieval_recall": mean(float(r["retrieval_recall"]) for r in items),
            "retrieval_precision": mean(float(r["retrieval_precision"]) for r in items),
            "retrieval_f1": mean(float(r["retrieval_f1"]) for r in items),
            "answer_in_edges": mean(float(r["answer_in_edges"]) for r in items),
            "avg_subgraph_size": mean(float(r["subgraph_size"]) for r in items),
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: List[Dict[str, Any]], meta: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Relation N-gram Prior Ablation",
        "",
        "Candidate-level reranking over existing HRG-Proposed dumps.",
        "The tested priors use only KB relation labels from offline grammar rules; no new relation labels are created.",
        "",
        "## Grammar Statistics",
        "",
        "| run_tag | rules | relation_vocab | unigrams | bigrams | trigrams |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run_tag, stats in sorted(meta.items()):
        lines.append(
            f"| {run_tag} | {stats['rules']} | {stats['relation_vocab']} | "
            f"{stats['unigrams']} | {stats['bigrams']} | {stats['trigrams']} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        "| run_tag | mode | n | valid | same_as_original | RetR | RetP | RetF1 | ans_in_edges | avg_edges |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['run_tag']} | {row['mode']} | {row['n']} | "
            f"{row['valid_rate']:.3f} | {row['same_as_original']:.3f} | "
            f"{row['retrieval_recall']:.4f} | {row['retrieval_precision']:.4f} | "
            f"{row['retrieval_f1']:.4f} | {row['answer_in_edges']:.4f} | "
            f"{row['avg_subgraph_size']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relation unigram/bigram/trigram priors on existing candidate dumps.")
    parser.add_argument("--artifect-root", default="artifect_all")
    parser.add_argument("--grammar-root", default="artifacts")
    parser.add_argument("--out-dir", default="artifacts/ngram_prior_ablation")
    parser.add_argument("--model-name", default="HRG-Proposed-gpt-oss-triple")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--run-tags", nargs="*", default=RUN_TAGS)
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["hrg_score", "llm_confidence", "relation_unigram", "relation_bigram", "relation_trigram"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.artifect_root)
    grammar_root = Path(args.grammar_root)
    out_dir = Path(args.out_dir)

    all_rows: List[Dict[str, Any]] = []
    meta: Dict[str, Dict[str, Any]] = {}

    for run_tag in args.run_tags:
        model_dir = root / run_tag / "dumps" / "per_model" / f"{args.model_name}@{run_tag}"
        grammar_path = grammar_root / run_tag / "grammar" / "hrg_grammar.json"
        if not model_dir.exists():
            print(f"[skip] missing model dir: {model_dir}")
            continue
        if not grammar_path.exists():
            print(f"[skip] missing grammar: {grammar_path}")
            continue
        rows, stats = evaluate_run(run_tag, model_dir, grammar_path, args.modes, args.alpha)
        print(f"[ok] {run_tag}: rows={len(rows)} stats={stats}")
        all_rows.extend(rows)
        meta[run_tag] = stats

    summary_rows = summarize(all_rows)
    write_csv(out_dir / "relation_ngram_prior_details.csv", all_rows)
    write_csv(out_dir / "relation_ngram_prior_summary.csv", summary_rows)
    write_markdown(out_dir / "relation_ngram_prior_summary.md", summary_rows, meta)
    print(f"[write] {out_dir / 'relation_ngram_prior_details.csv'}")
    print(f"[write] {out_dir / 'relation_ngram_prior_summary.csv'}")
    print(f"[write] {out_dir / 'relation_ngram_prior_summary.md'}")


if __name__ == "__main__":
    main()
