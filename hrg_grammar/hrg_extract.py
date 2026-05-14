# -*- coding: utf-8 -*-
"""
PHRG / HRG for labeled KG triples, robust + fast.

Originally tuned for MetaQA, now generalized so the same extractor can
also be used on other KGQA datasets such as WebQSP, as long as a triple
file is provided.

Fixes for hub-induced clique explosion:
- Build undirected skeleton degrees
- Prefer low/medium-degree nodes as BFS seeds
- During BFS, cap branching factor (sample neighbors) to avoid hub blow-up
- No hard reject loop that can fail 40 times; we make sampling itself safe

Also includes previous speedups:
- Fast clique candidates from elimination (no nx.find_cliques)
- edge->bag assignment via node->bags inverted index
- no factorial canonicalization
- clique tree binarize without repeated root recomputation
- Save grammar to JSON/TXT
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple, Optional, Any, Set

import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_naming import build_run_tag


# ================= USER CONFIG =================
KB_PATH = "../Datasets/MetaQA/kb.txt"
MAX_TRIPLES = None                     # e.g. 200000 for quick test

K_SAMPLES = 4
S_SAMPLE_SIZE = 500
RANDOM_SEED = 0

# --- Robust sampling controls ---
SEED_DEGREE_QUANTILE = 0.80    # only choose seeds with degree <= this quantile (avoid hubs)
BFS_MAX_BRANCH = 30            # cap neighbors expanded per node (very effective)
BFS_SHUFFLE_NEIGHBORS = True   # randomize neighbor exploration

# --- Output files ---
OUT_DIR = "artifacts"
OUT_GRAMMAR_JSON = "metaqa_phrg_grammar.json"
OUT_GRAMMAR_TXT = "metaqa_phrg_grammar.txt"

# --- Optional generation ---
DO_GENERATE = False
GEN_MAX_STEPS = 200000
# ==============================================


# ============================================================
# 0) Load labeled directed MultiDiGraph
# ============================================================

FREEBASE_NS_PREFIXES = (
    "http://rdf.freebase.com/ns/",
    "https://rdf.freebase.com/ns/",
)

WIKIMOVIES_RELATIONS = {
    "directed_by",
    "written_by",
    "starred_actors",
    "release_year",
    "in_language",
    "has_tags",
    "has_genre",
    "has_imdb_votes",
    "has_imdb_rating",
    "has_plot",
}


def _normalize_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""

    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1].strip()

    for prefix in FREEBASE_NS_PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix):]
            break

    if token.startswith('"'):
        m = re.match(r'^"(.*)"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?$', token)
        if m:
            token = m.group(1)

    if "://" in token:
        token = re.split(r"[/#]", token.rstrip("/"))[-1]

    return token.strip()


def _parse_triple_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if re.match(r"^\d+\s+", line):
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0].isdigit():
            body = tokens[1:]
            for idx, tok in enumerate(body):
                if tok in WIKIMOVIES_RELATIONS:
                    head = " ".join(body[:idx]).strip()
                    rel = tok.strip()
                    tail = " ".join(body[idx + 1:]).strip()
                    if head and rel and tail:
                        return tuple(_normalize_token(x) for x in (head, rel, tail))

    if "|" in line:
        parts = line.split("|", 2)
        if len(parts) == 3:
            return tuple(_normalize_token(x) for x in parts)

    if "\t" in line:
        parts = line.split("\t", 2)
        if len(parts) == 3:
            return tuple(_normalize_token(x) for x in parts)

    nt_match = re.match(r'^(<[^>]+>)\s+(<[^>]+>)\s+(.+?)\s*\.\s*$', line)
    if nt_match:
        return tuple(_normalize_token(x) for x in nt_match.groups())

    parts = re.split(r"\s+", line, maxsplit=2)
    if len(parts) == 3:
        return tuple(_normalize_token(x) for x in parts)

    return None


def load_labeled_kb_graph(kb_path: str, max_triples: Optional[int] = None) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    n = 0
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            triple = _parse_triple_line(line)
            if triple is None:
                continue
            h, r, t = triple
            if not h or not r or not t:
                continue

            G.add_edge(h, t, key=r, rel=r)
            n += 1
            if max_triples is not None and n >= max_triples:
                break
    return G


def to_undirected_skeleton(G: nx.MultiDiGraph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, _ in G.edges(keys=True):
        if u != v:
            H.add_edge(u, v)
    return H


# ============================================================
# 1) Robust BFS node-induced sampling
# ============================================================

def bfs_node_induced_sample_capped(
    G: nx.MultiDiGraph,
    seed,
    s: int,
    rng: random.Random,
    max_branch: int = 30,
    shuffle_neighbors: bool = True
) -> nx.MultiDiGraph:
    """
    BFS but cap branching per node to avoid hub explosion.
    """
    visited = {seed}
    q = deque([seed])

    while q and len(visited) < s:
        v = q.popleft()
        neigh = list(set(G.successors(v)) | set(G.predecessors(v)))

        if shuffle_neighbors:
            rng.shuffle(neigh)

        if max_branch is not None and max_branch > 0 and len(neigh) > max_branch:
            neigh = neigh[:max_branch]

        for u in neigh:
            if u not in visited:
                visited.add(u)
                q.append(u)
                if len(visited) >= s:
                    break

    return G.subgraph(visited).copy()


def pick_seed_avoid_hubs(nodes: List[Any], degrees: Dict[Any, int], quantile: float, rng: random.Random):
    """
    Choose seed from nodes with degree <= quantile threshold.
    """
    deg_vals = sorted(degrees[n] for n in nodes)
    if not deg_vals:
        return rng.choice(nodes)

    idx = int((len(deg_vals) - 1) * quantile)
    thr = deg_vals[idx]

    candidates = [n for n in nodes if degrees[n] <= thr]
    if not candidates:
        candidates = nodes
    return rng.choice(candidates)


def k_bfs_samples_robust(
    G: nx.MultiDiGraph,
    Gs: nx.Graph,
    k: int,
    s: int,
    rng: random.Random
) -> List[nx.MultiDiGraph]:
    nodes = list(G.nodes())
    degrees = dict(Gs.degree())

    samples = []
    for _ in range(k):
        seed = pick_seed_avoid_hubs(nodes, degrees, SEED_DEGREE_QUANTILE, rng)
        samples.append(
            bfs_node_induced_sample_capped(
                G, seed, s, rng,
                max_branch=BFS_MAX_BRANCH,
                shuffle_neighbors=BFS_SHUFFLE_NEIGHBORS
            )
        )
    return samples


# ============================================================
# 2) Clique tree via MCS (triangulation + fast maximal cliques)
# ============================================================

def mcs_ordering(G: nx.Graph) -> list:
    label = defaultdict(int)
    unnumbered = set(G.nodes())
    order = []
    for _ in range(len(G)):
        v = max(unnumbered, key=lambda x: (label[x], str(x)))
        unnumbered.remove(v)
        order.append(v)
        for u in G.neighbors(v):
            if u in unnumbered:
                label[u] += 1
    return order


def triangulate_from_order(G: nx.Graph, order: list) -> Tuple[nx.Graph, List[frozenset]]:
    """
    Triangulate using elimination order AND return maximal clique candidates fast.
    Avoid nx.find_cliques().
    """
    H = G.copy()
    pos = {v: i for i, v in enumerate(order)}
    clique_cands = []

    for v in order:
        later = [u for u in H.neighbors(v) if pos[u] > pos[v]]
        for i in range(len(later)):
            ui = later[i]
            for j in range(i + 1, len(later)):
                uj = later[j]
                if not H.has_edge(ui, uj):
                    H.add_edge(ui, uj)

        if later:
            clique_cands.append(frozenset([v, *later]))
        else:
            clique_cands.append(frozenset([v]))

    cands = sorted(set(clique_cands), key=len, reverse=True)
    maximal = []
    for c in cands:
        if not any(c.issubset(m) for m in maximal):
            maximal.append(c)
    return H, maximal


def build_clique_tree_from_cliques(cliques: List[frozenset]) -> nx.Graph:
    CG = nx.Graph()
    n = len(cliques)
    for i in range(n):
        CG.add_node(i)
    for i in range(n):
        ci = cliques[i]
        for j in range(i + 1, n):
            w = len(ci & cliques[j])
            if w > 0:
                CG.add_edge(i, j, weight=w)
    if CG.number_of_nodes() <= 1:
        return CG
    return nx.maximum_spanning_tree(CG, weight="weight")


# ============================================================
# 3) Binarization + Pruning
# ============================================================

def root_tree(T: nx.Graph, root: int = 0):
    parent = {root: None}
    children = defaultdict(list)
    q = deque([root])
    while q:
        x = q.popleft()
        for y in T.neighbors(x):
            if y not in parent:
                parent[y] = x
                children[x].append(y)
                q.append(y)
    return parent, children


def binarize_clique_tree(T: nx.Graph, bags: Dict[int, frozenset], root: int = 0):
    if T.number_of_nodes() == 0:
        return T, bags

    T2 = T.copy()
    bags2 = dict(bags)
    next_id = max(T2.nodes()) + 1

    parent, children = root_tree(T2, root)
    bfs = list(nx.bfs_tree(T2, root))

    for x in bfs:
        while len(children.get(x, [])) > 2:
            ch_list = children[x]
            keep = ch_list[0]
            move = ch_list[1:]

            clone = next_id
            next_id += 1
            T2.add_node(clone)
            bags2[clone] = bags2[x]
            T2.add_edge(x, clone)

            parent[clone] = x
            children[clone] = []

            for c in move:
                if T2.has_edge(x, c):
                    T2.remove_edge(x, c)
                T2.add_edge(clone, c)
                parent[c] = clone
                children[clone].append(c)

            children[x] = [keep, clone]

    return T2, bags2


def prune_leaf_no_internal(T: nx.Graph, bags: Dict[int, frozenset], root: int = 0):
    if T.number_of_nodes() <= 1:
        return T, bags

    T2 = T.copy()
    bags2 = dict(bags)

    changed = True
    while changed and T2.number_of_nodes() > 1:
        changed = False
        parent, _ = root_tree(T2, root)
        for v in list(T2.nodes()):
            if v == root:
                continue
            if T2.degree(v) == 1:
                p = parent.get(v)
                if p is not None and bags2[v].issubset(bags2[p]):
                    T2.remove_node(v)
                    bags2.pop(v, None)
                    changed = True
                    break
    return T2, bags2


# ============================================================
# 4) HRG structures
# ============================================================

@dataclass(frozen=True)
class Nonterminal:
    name: str
    rank: int


@dataclass(frozen=True)
class RHS:
    terminals: Tuple[Tuple[int, str, int], ...]
    nonterms: Tuple[Tuple[Nonterminal, Tuple[int, ...]], ...]


@dataclass
class Rule:
    lhs: Nonterminal
    rhs: RHS
    count: int = 1


def canonicalize_rhs_fast(rhs: RHS, rank: int) -> tuple:
    terms = tuple(sorted(rhs.terminals))
    nts = tuple(sorted((nt.name, nt.rank, att) for nt, att in rhs.nonterms))
    return (rank, terms, nts)


# ============================================================
# 5) HRG extraction (fast edge->bag assignment)
# ============================================================

def extract_hrg_rules_labeled(G: nx.MultiDiGraph, T: nx.Graph, bags: Dict[int, frozenset], root: int = 0) -> List[Rule]:
    parent, children = root_tree(T, root)

    node2bags = defaultdict(set)
    bag_sizes = {}
    for bid, b in bags.items():
        bag_sizes[bid] = len(b)
        for x in b:
            node2bags[x].add(bid)

    edge_to_bag = {}
    for u, v, rel in G.edges(keys=True):
        cands = node2bags.get(u, set()) & node2bags.get(v, set())
        if cands:
            edge_to_bag[(u, v, rel)] = min(cands, key=lambda i: bag_sizes[i])

    rules: List[Rule] = []
    for eta in T.nodes():
        p = parent.get(eta)
        bag = bags[eta]

        if p is None:
            lhs = Nonterminal("S", 0)
            ext: List = []
        else:
            inter = list(bag & bags[p])
            lhs = Nonterminal("N", len(inter))
            ext = inter

        ext_set = set(ext)
        internal = [x for x in bag if x not in ext_set]
        verts = ext + internal
        idx = {v: i for i, v in enumerate(verts)}

        terminals = []
        for (u, v, rel), b_id in edge_to_bag.items():
            if b_id == eta and u in idx and v in idx:
                terminals.append((idx[u], rel, idx[v]))

        nonterms = []
        for ch in children.get(eta, []):
            att_nodes = (bag & bags[ch])
            att = tuple(idx[x] for x in att_nodes)
            nonterms.append((Nonterminal("N", len(att)), att))

        rhs = RHS(tuple(sorted(terminals)), tuple(sorted(nonterms, key=lambda x: (x[0].rank, x[1]))))
        rules.append(Rule(lhs, rhs))

    return rules


def merge_duplicate_rules(rules: List[Rule]) -> List[Rule]:
    acc: Dict[tuple, Rule] = {}
    for r in rules:
        sig = (r.lhs.name, r.lhs.rank, canonicalize_rhs_fast(r.rhs, r.lhs.rank))
        if sig not in acc:
            acc[sig] = Rule(r.lhs, r.rhs, 0)
        acc[sig].count += r.count
    return list(acc.values())


# ============================================================
# 6) Learning pipeline
# ============================================================

def learn_phrg_from_labeled_graph(G: nx.MultiDiGraph) -> Tuple[List[Rule], Dict[str, int]]:
    Gs = to_undirected_skeleton(G)
    order = mcs_ordering(Gs)
    _, cliques = triangulate_from_order(Gs, order)

    max_clique = max((len(c) for c in cliques), default=0)

    T = build_clique_tree_from_cliques(cliques)
    bags = {i: cliques[i] for i in range(len(cliques))}
    if T.number_of_nodes() > 0:
        T, bags = binarize_clique_tree(T, bags, root=0)
        T, bags = prune_leaf_no_internal(T, bags, root=0)

    rules = extract_hrg_rules_labeled(G, T, bags, root=0)
    rules = merge_duplicate_rules(rules)

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "cliques": len(cliques),
        "max_clique": max_clique,
        "rules": len(rules),
    }
    return rules, stats


def learn_phrg_from_k_bfs_samples(G: nx.MultiDiGraph, k: int, s: int, seed: int) -> List[Rule]:
    rng = random.Random(seed)

    # build skeleton once for degrees / seed selection
    Gs_full = to_undirected_skeleton(G)

    samples = k_bfs_samples_robust(G, Gs_full, k, s, rng)

    all_rules: List[Rule] = []
    for i, Gi in enumerate(samples, 1):
        gi_rules, st = learn_phrg_from_labeled_graph(Gi)
        all_rules.extend(gi_rules)
        print(f"[sample {i}/{k}] nodes={st['nodes']} edges={st['edges']} "
              f"cliques={st['cliques']} max_clique={st['max_clique']} rules={st['rules']}",
              flush=True)

    return merge_duplicate_rules(all_rules)


# ============================================================
# 7) Save grammar
# ============================================================

def rule_to_json(r: Rule) -> dict:
    return {
        "lhs": {"name": r.lhs.name, "rank": r.lhs.rank},
        "rhs": {
            "terminals": [{"a": a, "rel": rel, "b": b} for (a, rel, b) in r.rhs.terminals],
            "nonterms": [{"name": nt.name, "rank": nt.rank, "att": list(att)} for (nt, att) in r.rhs.nonterms],
        },
        "count": r.count,
    }


def save_grammar(grammar: List[Rule], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, OUT_GRAMMAR_JSON)
    txt_path = os.path.join(out_dir, OUT_GRAMMAR_TXT)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([rule_to_json(r) for r in grammar], f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in sorted(grammar, key=lambda x: (-x.count, x.lhs.rank, len(x.rhs.terminals), len(x.rhs.nonterms))):
            f.write(f"{r.lhs.name}/{r.lhs.rank}  count={r.count}\n")
            f.write(f"  T: {list(r.rhs.terminals)}\n")
            f.write(f"  N: {[(nt.name, nt.rank, att) for nt, att in r.rhs.nonterms]}\n\n")

    return json_path, txt_path


# ============================================================
# 8) MAIN
# ============================================================

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Extract PHRG/HRG grammar from a labeled KG triple file.")
    parser.add_argument("--kb-path", default=KB_PATH, help="Path to the triple file.")
    parser.add_argument("--dataset", choices=["metaqa", "wikimovies", "mlpq", "wqsp", "cwq", "kqapro", "mintaka", "custom"], default="metaqa")
    parser.add_argument("--split", default=None)
    parser.add_argument("--metaqa-variant", default="vanilla")
    parser.add_argument("--wikimovies-subset", default="wiki_entities")
    parser.add_argument("--mlpq-pair", default="en-zh")
    parser.add_argument("--mlpq-question-lang", default="en")
    parser.add_argument("--mlpq-fusion", default="ills")
    parser.add_argument("--mlpq-kb-mode", default="bilingual")
    parser.add_argument("--mlpq-kb-lang", default="auto")
    parser.add_argument("--custom-dataset-name", default="custom")
    parser.add_argument("--run-tag", default=None, help="Override auto-generated dataset run tag.")
    parser.add_argument("--out-root", default=OUT_DIR, help="Root directory for dataset-tagged outputs.")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory. Overrides --out-root/run_tag.")
    parser.add_argument("--out-prefix", default=None, help="Output filename prefix, e.g. wikimovies_phrg_grammar")
    parser.add_argument("--max-triples", type=int, default=MAX_TRIPLES, help="Optional limit on loaded triples.")
    parser.add_argument("--k-samples", type=int, default=K_SAMPLES, help="Number of BFS samples.")
    parser.add_argument("--sample-size", type=int, default=S_SAMPLE_SIZE, help="Target nodes per BFS sample.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()

    kb_path = args.kb_path
    run_tag = args.run_tag or build_run_tag(
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
    )
    out_dir = args.out_dir or os.path.join(args.out_root, run_tag)
    max_triples = args.max_triples
    k_samples = args.k_samples
    sample_size = args.sample_size
    random_seed = args.seed

    if args.out_prefix:
        OUT_GRAMMAR_JSON = f"{args.out_prefix}.json"
        OUT_GRAMMAR_TXT = f"{args.out_prefix}.txt"
    elif OUT_GRAMMAR_JSON == "metaqa_phrg_grammar.json":
        OUT_GRAMMAR_JSON = "hrg_grammar.json"
        OUT_GRAMMAR_TXT = "hrg_grammar.txt"

    t0 = time.perf_counter()
    print(f"Run tag: {run_tag}", flush=True)
    print(f"Loading KB from {kb_path} ...", flush=True)
    G_full = load_labeled_kb_graph(kb_path, max_triples)
    t1 = time.perf_counter()
    print(f"KB loaded: nodes={G_full.number_of_nodes()}, edges={G_full.number_of_edges()} (time={t1 - t0:.2f}s)",
          flush=True)

    print("Learning PHRG...", flush=True)
    t2 = time.perf_counter()
    grammar = learn_phrg_from_k_bfs_samples(G_full, k_samples, sample_size, random_seed)
    t3 = time.perf_counter()
    print(f"Grammar learned: rules={len(grammar)} (time={t3 - t2:.2f}s)", flush=True)

    abs_out_dir = os.path.abspath(out_dir)
    json_path, txt_path = save_grammar(grammar, abs_out_dir)
    print("CWD =", os.getcwd(), flush=True)
    print("Saved grammar JSON:", json_path, flush=True)
    print("Saved grammar TXT :", txt_path, flush=True)

    t_end = time.perf_counter()
    print(f"Total runtime: {t_end - t0:.2f}s", flush=True)
