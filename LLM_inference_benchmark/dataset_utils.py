from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


FREEBASE_NS_PREFIXES = (
    "http://rdf.freebase.com/ns/",
    "https://rdf.freebase.com/ns/",
)

ALIAS_RELATIONS = {
    "type.object.name",
    "common.topic.alias",
    "name",
}


def normalize_kb_token(token: str) -> str:
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


def parse_kb_triple_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    wikimovies_match = re.match(r"^\d+\s+(.*?)\s+([^\s]+)\s+(.*)$", line)
    if wikimovies_match:
        h, r, t = wikimovies_match.groups()
        return normalize_kb_token(h), normalize_kb_token(r), normalize_kb_token(t)

    if "|" in line:
        parts = line.split("|", 2)
        if len(parts) == 3:
            h, r, t = parts
            return normalize_kb_token(h), normalize_kb_token(r), normalize_kb_token(t)

    if "\t" in line:
        parts = line.split("\t", 2)
        if len(parts) == 3:
            h, r, t = parts
            return normalize_kb_token(h), normalize_kb_token(r), normalize_kb_token(t)

    nt_match = re.match(
        r'^(<[^>]+>)\s+(<[^>]+>)\s+(.+?)\s*\.\s*$',
        line,
    )
    if nt_match:
        h, r, t = nt_match.groups()
        return normalize_kb_token(h), normalize_kb_token(r), normalize_kb_token(t)

    parts = re.split(r"\s+", line, maxsplit=2)
    if len(parts) == 3:
        h, r, t = parts
        return normalize_kb_token(h), normalize_kb_token(r), normalize_kb_token(t)

    return None


def iter_kb_triples(kb_path: str, max_triples: Optional[int] = None) -> Iterable[Tuple[str, str, str]]:
    count = 0
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            triple = parse_kb_triple_line(line)
            if triple is None:
                continue
            h, r, t = triple
            if not h or not r or not t:
                continue
            yield h, r, t
            count += 1
            if max_triples is not None and count >= max_triples:
                break


def load_relation_list(
    relation_path: Optional[str] = None,
    kb_path: Optional[str] = None,
    max_triples: Optional[int] = None,
) -> List[str]:
    if relation_path and Path(relation_path).exists():
        with open(relation_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception:
            pass
        return [line.strip() for line in raw.splitlines() if line.strip()]

    if not kb_path:
        return []

    rels = sorted({r for _, r, _ in iter_kb_triples(kb_path, max_triples=max_triples)})
    return rels


def build_node_index(nodes: Iterable[str], alias_map: Optional[Dict[str, Set[str]]] = None) -> Dict[str, str]:
    index: Dict[str, str] = {}

    def add_key(key: str, canonical: str):
        key = (key or "").strip().lower()
        if not key:
            return
        index.setdefault(key, canonical)
        index.setdefault(key.replace(" ", "_"), canonical)
        index.setdefault(key.replace("_", " "), canonical)
        index.setdefault(re.sub(r"[\s_]+", "", key), canonical)

    for node in nodes:
        add_key(node, node)
        if alias_map:
            for alias in alias_map.get(node, set()):
                add_key(alias, node)

    return index


def load_kb_adjacency(
    kb_path: str,
    max_triples: Optional[int] = None,
    sanitize_entity_fn: Optional[Callable[[str], str]] = None,
    collect_aliases: bool = True,
) -> Tuple[
    DefaultDict[str, DefaultDict[str, Set[str]]],
    DefaultDict[str, DefaultDict[str, Set[str]]],
    Set[str],
    List[str],
    Dict[str, Set[str]],
]:
    kb_out: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    kb_in: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    all_nodes: Set[str] = set()
    relations: Set[str] = set()
    alias_map: Dict[str, Set[str]] = defaultdict(set)

    sanitize = sanitize_entity_fn or (lambda x: x)

    for h, r, t in iter_kb_triples(kb_path, max_triples=max_triples):
        hs = sanitize(h)
        ts = sanitize(t)
        rs = r.strip()
        if not hs or not ts or not rs:
            continue

        kb_out[hs][rs].add(ts)
        kb_in[ts][rs].add(hs)
        all_nodes.add(hs)
        all_nodes.add(ts)
        relations.add(rs)

        if collect_aliases and rs in ALIAS_RELATIONS:
            alias = t.strip()
            if alias and not alias.startswith("m.") and not alias.startswith("g."):
                alias_map[hs].add(alias)

    return kb_out, kb_in, all_nodes, sorted(relations), dict(alias_map)


def load_metaqa_dataset(file_path: str) -> List[Tuple[str, List[str]]]:
    dataset = []
    if not Path(file_path).exists():
        return dataset

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0]
            answers = [a for a in parts[1].split("|") if a]
            dataset.append((question, answers))
    return dataset


def _display_form(token: str) -> str:
    return normalize_kb_token(token).replace("_", " ").strip()


def load_wikimovies_dataset(file_path: str) -> Dict[str, List[Tuple[str, List[str]]]]:
    grouped: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    path = Path(file_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, rest = line.split(" ", 1)
                question, raw_answers = rest.split("\t", 1)
            except ValueError:
                continue
            answers: List[str] = []
            for answer in raw_answers.split(","):
                answer = answer.strip()
                if not answer:
                    continue
                for candidate in (answer, answer.replace("_", " ")):
                    if candidate and candidate not in answers:
                        answers.append(candidate)
            if answers:
                grouped["1-hop"].append((question.strip(), answers))
    return dict(grouped)


def _resolve_mlpq_question_file(root: str, pair: str, hop: int, question_lang: str) -> Path:
    pair_dir = pair.lower()
    pair_prefix = pair.lower().replace("-", "_")
    filename = f"{pair_prefix}_{hop}h_{question_lang.lower()}_question.txt"
    return Path(root) / "datasets" / "Questions" / pair_dir / f"{hop}-hop" / filename


def resolve_mlpq_kb_path(root: str, pair: str, fusion: str) -> str:
    pair_key = pair.lower().replace("-", "_")
    fusion_dir = "ILLs_fusion" if fusion.lower() == "ills" else "NMN_fusion"
    file_prefix = "merged_ILLs_KG" if fusion.lower() == "ills" else "merged_NMN_KG"
    return str(Path(root) / "datasets" / "KGs" / "fusion_bilingual_KGs" / fusion_dir / f"{file_prefix}_{pair_key}.txt")


def load_mlpq_dataset(
    root: str,
    pair: str = "en-zh",
    question_lang: str = "en",
    inject_topic_entity: bool = True,
) -> Dict[str, List[Tuple[str, List[str]]]]:
    grouped: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for hop in (2, 3):
        question_file = _resolve_mlpq_question_file(root, pair, hop, question_lang)
        if not question_file.exists():
            continue
        with open(question_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                question, answer_raw, path_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
                path_parts = [normalize_kb_token(p) for p in path_raw.split("#") if p.strip()]
                topic_entity = path_parts[0] if len(path_parts) >= 1 else ""
                question_text = question
                if inject_topic_entity and topic_entity:
                    question_text += f"\nTopic entity: [{_display_form(topic_entity)}]"

                answers: List[str] = []
                normalized_answer = normalize_kb_token(answer_raw)
                display_answer = _display_form(answer_raw)
                for candidate in (normalized_answer, display_answer):
                    if candidate and candidate not in answers:
                        answers.append(candidate)
                if answers:
                    grouped[f"{hop}-hop"].append((question_text, answers))
    return dict(grouped)


def load_custom_dataset(
    file_path: str,
    fmt: str = "auto",
    default_hop: int = 1,
) -> Dict[str, List[Tuple[str, List[str]]]]:
    grouped: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    path = Path(file_path)
    if not path.exists():
        return {}

    normalized_fmt = fmt.lower()
    if normalized_fmt == "auto":
        normalized_fmt = "jsonl" if path.suffix.lower() == ".jsonl" else "tsv"

    def add_record(question: str, answers: List[str], hop_value: int):
        q = (question or "").strip()
        clean_answers = [str(a).strip() for a in answers if str(a).strip()]
        if not q or not clean_answers:
            return
        hop = hop_value if hop_value and hop_value > 0 else default_hop
        grouped[f"{hop}-hop"].append((q, clean_answers))

    if normalized_fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                question = record.get("question", "")
                raw_answers = record.get("answers", [])
                if isinstance(raw_answers, str):
                    answers = [a.strip() for a in raw_answers.split("|") if a.strip()]
                elif isinstance(raw_answers, list):
                    answers = [str(a).strip() for a in raw_answers if str(a).strip()]
                else:
                    answers = []
                hop = record.get("hop", default_hop)
                try:
                    hop = int(hop)
                except Exception:
                    hop = default_hop
                add_record(question, answers, hop)
        return dict(grouped)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0]
            answers = [a.strip() for a in parts[1].split("|") if a.strip()]
            hop = default_hop
            if len(parts) >= 3 and parts[2].strip():
                try:
                    hop = int(parts[2].strip())
                except Exception:
                    hop = default_hop
            add_record(question, answers, hop)

    return dict(grouped)


def load_normalized_jsonl_dataset(
    file_path: str,
    default_hop: int = 1,
) -> Dict[str, List[Tuple[str, List[str]]]]:
    grouped: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    path = Path(file_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            question = str(record.get("question", "")).strip()
            raw_answers = record.get("answers", [])
            if isinstance(raw_answers, str):
                answers = [a.strip() for a in raw_answers.split("|") if a.strip()]
            elif isinstance(raw_answers, list):
                answers = [str(a).strip() for a in raw_answers if str(a).strip()]
            else:
                answers = []

            hop = record.get("hop", default_hop)
            try:
                hop = int(hop)
            except Exception:
                hop = default_hop
            if hop <= 0:
                hop = default_hop

            if question and answers:
                grouped[f"{hop}-hop"].append((question, answers))

    return dict(grouped)
