from __future__ import annotations

import json
import re
import pickle
import time
from collections import deque
from typing import Optional, List, Tuple, Dict, Any, Set

from agent_factory import build_llm_strategy
from dataset_utils import build_node_index, load_kb_adjacency, load_relation_list, normalize_lookup_key


class BaselineKnowledgeGraphAgent:
    """
    Baseline = 標準 KG-RAG (KAG) 真正的 BFS 展開：
      1) LLM parse intent: entity (不依賴 chain)
      2) 從 entity 出發，做無引導 BFS，展開 depth=hop_count 層
         收集所有到達的 (head, relation, tail) 三元組
      3) 把找到的 edges 當 context，叫 LLM 回答

    Notes:
    - Entity lookup 支援底線↔空格互查
    - 完全不靠 relation chain / grammar 引導，是真正的 KAG baseline
    - 支援 retrieval_recall / precision / F1 / subgraph_size 指標
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        kb_path: str = "../Datasets/MetaQA/kb.txt",
        relation_path: Optional[str] = "../Datasets/MetaQA/relations.json",
        max_new_tokens: int = 1024,
        use_model_sharding: bool = False,
        strict_gpu_sharding: bool = False,
        target_device: Optional[str] = None,
        bfs_depth: int = 2,             # BFS 展開最大深度
        max_edges_per_hop: int = 500,   # 防爆：每 hop 最多保留多少條邊
        max_frontier: int = 5000,       # 防爆：frontier node 上限
    ):
        print(f"[Init] Loading LLM strategy for: {model_id} ...")
        self.llm = build_llm_strategy(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            use_model_sharding=use_model_sharding,
            strict_gpu_sharding=strict_gpu_sharding,
            target_device=target_device,
        )

        self.bfs_depth = bfs_depth
        self.max_edges_per_hop = max_edges_per_hop
        self.max_frontier = max_frontier

        print("[Init] Loading Knowledge Graph data...")
        self.kb_out, self.kb_in, self.all_nodes, self.relations, self.alias_map = self._load_data(kb_path, relation_path)
        self.allowed_relations = set(self.relations)
        self.allowed_rel_tokens = set(self.relations) | {f"{r}^-1" for r in self.relations}

        # 建立 entity → normalized 的快速查詢索引
        self._node_index = build_node_index(self.all_nodes, self.alias_map)

        # =========================
        # 給 benchmark 用的統計欄位
        # =========================
        self.total_questions = 0
        self.hit_grammar_count = 0          # baseline 沒 grammar，永遠是 0
        self.total_context_length = 0
        self.total_parse1_tokens = 0
        self.total_correction_tokens = 0    # baseline 沒 correction，永遠是 0
        self.total_parse2_tokens = 0

        # Retrieval Recall / Precision / F1
        self.total_retrieval_recall = 0.0
        self.total_retrieval_precision = 0.0
        self.total_retrieval_f1 = 0.0
        self.total_retrieval_questions = 0

        # Subgraph size (triple count)
        self.total_subgraph_size = 0

        print(f"[Init] KB Stats: nodes={len(self.all_nodes)}, "
              f"relations={len(self.relations)}")
        print("[Init] Ready (True BFS Baseline).")

    # ============================================================
    # 基本工具
    # ============================================================

    def _estimate_tokens(self, *texts: str) -> int:
        return sum(len(t) for t in texts) // 4

    async def _inference(self, developer_instruction: str, user_content: str) -> str:
        try:
            return await self.llm.inference(developer_instruction, user_content)
        except Exception as e:
            print(f"[Inference Error] {e}")
            return ""

    def _extract_final_content(self, raw_text: str) -> str:
        pattern = r"<\|channel\|>final<\|message\|>\s*(.*?)(?=<\||$)"
        match = re.search(pattern, raw_text, re.DOTALL)
        text = match.group(1).strip() if match else (raw_text or "").strip()
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

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        candidates = self._extract_balanced_json_segments(text, "{", "}")
        for s in candidates:
            try:
                data = json.loads(s)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

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

    def _normalize_entity_from_question(self, user_prompt: str, entity: Optional[str]) -> Optional[str]:
        if entity and entity.strip():
            return entity.strip()
        m = re.search(r"\[(.*?)\]", user_prompt)
        if m:
            return m.group(1).strip()
        return entity

    # ============================================================
    # 載入資料（adjacency dict，速度比 nx 快）
    # ============================================================

    def _load_data(self, kb_path: str, relation_path: Optional[str]):
        """
        Returns:
            kb_out: {head: {rel: set(tails)}}
            kb_in:  {tail: {rel: set(heads)}}
            all_nodes: set of all node strings
            relations: list of relation strings
        """
        try:
            kb_out, kb_in, all_nodes, derived_relations, alias_map = load_kb_adjacency(kb_path)
            relations = load_relation_list(relation_path=relation_path, kb_path=kb_path)
            if not relations:
                relations = derived_relations
            return kb_out, kb_in, all_nodes, relations, alias_map
        except FileNotFoundError as e:
            print(f"[Error] File not found: {e}")
            return {}, {}, set(), [], {}

    def _display_node(self, node: str) -> str:
        aliases = sorted(self.alias_map.get(node, set()))
        if aliases:
            return aliases[0]
        return node.replace("_", " ")

    # ============================================================
    # Intent Parsing（只需要 entity，不需要 chain）
    # ============================================================

    async def _parse_entity(self, user_prompt: str) -> Tuple[Optional[str], int]:
        """
        從問題中抽出 topic entity。
        直接用 regex 抓 [...] 內容，不需要 LLM chain。
        若沒有 [] 再用 LLM 確認。
        """
        # Fast path: entity in brackets
        m = re.search(r"\[(.*?)\]", user_prompt)
        if m:
            entity = m.group(1).strip()
            tokens = 0
            return entity, tokens

        # LLM fallback for entity extraction
        developer_instruction = (
            "You are a knowledge graph entity extractor.\n"
            "Task: Extract ONLY the main topic entity from the user question.\n"
            "Rules:\n"
            "1. Return ONLY valid JSON: {\"entity\": \"...\"}\n"
            "2. No explanation, no markdown.\n"
            "CRITICAL: Do not use <think> tags. Output final JSON only.\n"
        )
        input_tokens = self._estimate_tokens(developer_instruction, user_prompt)
        raw = await self._inference(developer_instruction, user_prompt)
        clean = self._extract_final_content(raw)
        total_tokens = input_tokens + self._estimate_tokens(clean)

        data = self._extract_json_object(clean)
        if data:
            entity = data.get("entity", "").strip()
            if entity:
                return entity, total_tokens

        return None, total_tokens

    # ============================================================
    # Entity Lookup（底線↔空格 + case-insensitive）
    # ============================================================

    def _find_entity_in_kb(self, entity: str) -> Optional[str]:
        if not entity:
            return None

        # 1. 精確匹配
        if entity in self.all_nodes:
            return entity

        normalized = normalize_lookup_key(entity)

        # 2. index 查 (handles 底線↔空格 + lowercase)
        for key in (
            entity.lower().strip(),
            entity.lower().replace(" ", "_"),
            entity.lower().replace("_", " "),
            re.sub(r"[^\w\s]", "", entity).lower().replace(" ", "_"),
            normalized,
            normalized.replace(" ", "_"),
            normalized.replace(" ", ""),
        ):
            if key in self._node_index:
                return self._node_index[key]

        print(f"[Lookup] Entity '{entity}' not found in KB")
        return None

    # ============================================================
    # 真正的 BFS 展開（不靠 chain，按深度展開）
    # ============================================================

    def _bfs_expand(
        self,
        start_entity: str,
        depth: int,
    ) -> List[Tuple[str, str, str]]:
        """
        從 start_entity 出發，雙向 BFS 展開 depth 層。
        回傳所有經過的 (head, relation, tail) edges。

        這是真正的 KAG baseline：
        - 不依賴任何 chain 或 grammar
        - 展開所有方向（outgoing + incoming）
        - 用 depth 控制子圖大小（depth = hop count of the question）
        """
        node = self._find_entity_in_kb(start_entity)
        if not node:
            return []

        print(f"[BFS] Start: '{node}' | depth={depth}")

        visited_nodes: Set[str] = {node}
        frontier: Set[str] = {node}
        collected_edges: List[Tuple[str, str, str]] = []
        visited_edges: Set[Tuple[str, str, str]] = set()

        for d in range(depth):
            if len(frontier) > self.max_frontier:
                frontier = set(list(frontier)[:self.max_frontier])
                print(f"  [BFS d={d+1}] Frontier capped at {self.max_frontier}")

            next_frontier: Set[str] = set()
            hop_edges: List[Tuple[str, str, str]] = []

            for ent in frontier:
                # Outgoing edges: ent --[rel]--> tail
                for rel, tails in self.kb_out.get(ent, {}).items():
                    for tail in tails:
                        edge = (ent, rel, tail)
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            hop_edges.append(edge)
                            if tail not in visited_nodes:
                                visited_nodes.add(tail)
                                next_frontier.add(tail)

                # Incoming edges: head --[rel]--> ent
                for rel, heads in self.kb_in.get(ent, {}).items():
                    for head in heads:
                        edge = (head, rel, ent)
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            hop_edges.append(edge)
                            if head not in visited_nodes:
                                visited_nodes.add(head)
                                next_frontier.add(head)

            # Cap edges per hop to avoid explosion
            if len(hop_edges) > self.max_edges_per_hop:
                hop_edges = hop_edges[:self.max_edges_per_hop]
                print(f"  [BFS d={d+1}] Hop edges capped at {self.max_edges_per_hop}")

            print(f"  [BFS d={d+1}] found={len(hop_edges)} edges, new_nodes={len(next_frontier)}")
            collected_edges.extend(hop_edges)

            if not next_frontier:
                print(f"  [BFS d={d+1}] Frontier exhausted.")
                break

            frontier = next_frontier

        print(f"[BFS] Total collected edges: {len(collected_edges)}")
        return collected_edges

    # ============================================================
    # Retrieval 指標計算（Recall / Precision / F1）
    # ============================================================

    def _compute_retrieval_metrics(
        self,
        edges: List[Tuple[str, str, str]],
        references: List[str],
    ) -> Tuple[float, float, float]:
        """
        計算 retrieval recall, precision, F1。

        - Recall:    gold answers 中有多少被子圖涵蓋
        - Precision: 子圖中有多少「節點」是 gold answer
                     (這裡用 binary precision：子圖中是否有任何 gold answer)
        - F1:        harmonic mean

        注意：這是 KG 子圖的 binary/set-level 指標，
              衡量的是子圖「是否覆蓋答案」，而非詞語 overlap。
        """
        if not references or not edges:
            return 0.0, 0.0, 0.0

        # 收集子圖中所有節點
        nodes_in_subgraph = set()
        for h, r, t in edges:
            nodes_in_subgraph.add(self._display_node(h).lower().replace("_", " "))
            nodes_in_subgraph.add(self._display_node(t).lower().replace("_", " "))
            nodes_in_subgraph.add(h.lower().replace("_", " "))
            nodes_in_subgraph.add(t.lower().replace("_", " "))

        # Recall：有多少 gold answer 被涵蓋
        hit_count = 0
        for ref in references:
            ref_lower = ref.lower().replace("_", " ")
            for n in nodes_in_subgraph:
                if ref_lower in n or n in ref_lower:
                    hit_count += 1
                    break

        recall = hit_count / len(references) if references else 0.0

        # Precision：子圖節點中有多少是 gold answer
        # (Binary: 1 if any gold answer in subgraph, else 0)
        # 更精確的版本可以算：gold_nodes / all_nodes in subgraph
        gold_nodes_in_subgraph = 0
        refs_lower = [r.lower().replace("_", " ") for r in references]
        for n in nodes_in_subgraph:
            for ref_lower in refs_lower:
                if ref_lower in n or n in ref_lower:
                    gold_nodes_in_subgraph += 1
                    break

        precision = gold_nodes_in_subgraph / len(nodes_in_subgraph) if nodes_in_subgraph else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return recall, precision, f1

    # ============================================================
    # 回答生成
    # ============================================================

    async def _generate_rag_response(
        self,
        user_prompt: str,
        context_edges: List[Tuple[str, str, str]],
    ) -> Tuple[str, int, int]:
        """
        Returns: (answer, parse2_tokens, context_tokens)
        """
        if not context_edges:
            context_str = "No information found."
        else:
            lines = ["Knowledge Graph Context:"]
            for i, (s, r, o) in enumerate(context_edges, 1):
                lines.append(f"{i}. {self._display_node(s)} --[{r}]--> {self._display_node(o)}")
            context_str = "\n".join(lines)

        context_tokens = self._estimate_tokens(context_str)

        dev_instruction = (
            "You are a strict answer formatter for KGQA.\n"
            "Use only the provided Knowledge Graph Context to answer the question.\n"
            "Output rules:\n"
            "1. Return only the final answer.\n"
            "2. Do not output reasoning, explanations, analysis, intermediate steps, or chain-of-thought.\n"
            "3. Do not output <think> tags or any hidden-thinking markers.\n"
            "4. Do not output full sentences, prefixes, markdown, bullets, numbering, quotes, or the question.\n"
            "5. Copy answer strings verbatim from the Knowledge Graph Context; do not paraphrase, translate, or add outside knowledge.\n"
            "6. For a single answer, output exactly that answer string.\n"
            "7. For multiple answers, output only the answer strings joined by ' | '.\n"
            "8. Remove duplicates and keep answer strings concise.\n"
            "9. If the context is insufficient, output exactly: I don't know"
        )
        user_content = f"{context_str}\n\nQuestion: {user_prompt}"

        parse2_tokens = self._estimate_tokens(dev_instruction, user_content)
        raw = await self._inference(dev_instruction, user_content)
        answer = self._extract_final_content(raw)
        parse2_tokens += self._estimate_tokens(answer)

        return answer, parse2_tokens, context_tokens

    # ============================================================
    # Dump
    # ============================================================

    def _dump(self, save_path: str, payload: dict):
        try:
            with open(save_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"[Dump Error] {e}")

    # ============================================================
    # Main Ask
    # ============================================================

    async def ask(
        self,
        user_prompt: str,
        save_path: Optional[str] = None,
        references: Optional[List[str]] = None,
        hop_override: Optional[int] = None,    # 可從 benchmark 傳入 hop count
    ) -> str:
        details = await self.ask_detailed(
            user_prompt,
            save_path=save_path,
            references=references,
            hop_override=hop_override,
        )
        return details["answer"]

    async def ask_detailed(
        self,
        user_prompt: str,
        save_path: Optional[str] = None,
        references: Optional[List[str]] = None,
        hop_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.total_questions += 1
        print(f"\n{'=' * 70}")
        print(f"[Question] {user_prompt}")
        print(f"{'=' * 70}\n")

        # 1. Extract entity (no chain needed)
        parse_t0 = time.perf_counter()
        entity, p1_tokens = await self._parse_entity(user_prompt)
        parse_latency = time.perf_counter() - parse_t0
        self.total_parse1_tokens += p1_tokens
        print(f"[Entity] '{entity}' | tokens~{p1_tokens}")

        if not entity:
            answer = "Sorry, I couldn't identify the entity in your question."
            if save_path:
                self._dump(save_path, {
                    "question": user_prompt,
                    "entity": entity,
                    "edges": [],
                    "answer": answer,
                    "failure_stage": "entity_parse_failure",
                    "references": references or [],
                    "retrieval_recall": 0.0,
                    "retrieval_precision": 0.0,
                    "retrieval_f1": 0.0,
                    "subgraph_size": 0,
                    "selected_depth": hop_override if hop_override is not None else self.bfs_depth,
                    "serialization_format": "baseline_bfs_text",
                    "final_context": "",
                    "timing": {
                        "parse_latency": parse_latency,
                        "retrieval_latency": 0.0,
                        "generation_latency": 0.0,
                    },
                    "token_usage": {
                        "parse1_tokens": p1_tokens,
                        "correction_tokens": 0,
                        "parse2_tokens": 0,
                        "context_tokens": 0,
                    }
                })
            return {
                "answer": answer,
                "failure_stage": "entity_parse_failure",
                "parse_latency": parse_latency,
                "retrieval_latency": 0.0,
                "generation_latency": 0.0,
                "generation_failed": False,
                "answerable": True,
            }

        # 2. True BFS expand (no chain guidance)
        retrieval_t0 = time.perf_counter()
        depth = hop_override if hop_override is not None else self.bfs_depth
        edges = self._bfs_expand(entity, depth=depth)
        retrieval_latency = time.perf_counter() - retrieval_t0
        subgraph_size = len(edges)
        self.total_subgraph_size += subgraph_size

        # 3. Retrieval metrics
        retrieval_recall = 0.0
        retrieval_precision = 0.0
        retrieval_f1 = 0.0
        if references:
            retrieval_recall, retrieval_precision, retrieval_f1 = self._compute_retrieval_metrics(
                edges, references
            )
            self.total_retrieval_recall += retrieval_recall
            self.total_retrieval_precision += retrieval_precision
            self.total_retrieval_f1 += retrieval_f1
            self.total_retrieval_questions += 1
            print(f"[Metrics] recall={retrieval_recall:.3f} | "
                  f"precision={retrieval_precision:.3f} | f1={retrieval_f1:.3f} | "
                  f"subgraph_size={subgraph_size}")

        # 4. Generate answer
        generation_t0 = time.perf_counter()
        if edges:
            answer, p2_tokens, context_tokens = await self._generate_rag_response(user_prompt, edges)
        else:
            answer = "I couldn't find any information in the Knowledge Graph matching your query."
            p2_tokens = 0
            context_tokens = 0
        generation_latency = time.perf_counter() - generation_t0
        generation_failed = bool(edges) and not bool((answer or "").strip())

        self.total_parse2_tokens += p2_tokens
        self.total_context_length += context_tokens

        print(f"[Answer] {answer} | tokens~{p2_tokens} | ctx_tokens~{context_tokens}\n")

        if save_path:
            context_str = ""
            if edges:
                lines = ["Knowledge Graph Context:"]
                for i, (s, r, o) in enumerate(edges, 1):
                    lines.append(f"{i}. {self._display_node(s)} --[{r}]--> {self._display_node(o)}")
                context_str = "\n".join(lines)
            self._dump(save_path, {
                "question": user_prompt,
                "entity": entity,
                "edges": edges,
                "answer": answer,
                "failure_stage": "retrieval_empty" if not edges else "ok",
                "references": references or [],
                "retrieval_recall": retrieval_recall,
                "retrieval_precision": retrieval_precision,
                "retrieval_f1": retrieval_f1,
                "subgraph_size": subgraph_size,
                "selected_depth": depth,
                "serialization_format": "baseline_bfs_text",
                "final_context": context_str,
                "timing": {
                    "parse_latency": parse_latency,
                    "retrieval_latency": retrieval_latency,
                    "generation_latency": generation_latency,
                },
                "token_usage": {
                    "parse1_tokens": p1_tokens,
                    "correction_tokens": 0,
                    "parse2_tokens": p2_tokens,
                    "context_tokens": context_tokens,
                }
            })

        return {
            "answer": answer,
            "failure_stage": "retrieval_empty" if not edges else "ok",
            "parse_latency": parse_latency,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "generation_failed": generation_failed,
            "answerable": bool((answer or "").strip()),
        }


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = BaselineKnowledgeGraphAgent(
            model_id="openai/gpt-oss-20b",
            kb_path="../Datasets/MetaQA/kb.txt",
            relation_path="../Datasets/MetaQA/relations.json",
            bfs_depth=2,
        )

        questions = [
            ("the films that share directors with the film [Black Snake Moan] were in which genres", ["Drama", "Action"]),
            ("what movies did the director of [The Dark Knight] direct?", ["Memento"]),
            ("What movies did [Tom Hanks] star in?", ["Cast Away"]),
        ]

        for q, refs in questions:
            res = await agent.ask(q, references=refs)
            print(f"\nQ: {q}")
            print(f"A: {res}")
            print("-" * 70)

        print("\n[Stats]")
        print(f"total_questions         = {agent.total_questions}")
        print(f"total_parse1_tokens     = {agent.total_parse1_tokens}")
        print(f"total_parse2_tokens     = {agent.total_parse2_tokens}")
        print(f"total_context_length    = {agent.total_context_length}")
        print(f"total_subgraph_size     = {agent.total_subgraph_size}")
        if agent.total_retrieval_questions > 0:
            n = agent.total_retrieval_questions
            print(f"avg_retrieval_recall    = {agent.total_retrieval_recall/n:.4f}")
            print(f"avg_retrieval_precision = {agent.total_retrieval_precision/n:.4f}")
            print(f"avg_retrieval_f1        = {agent.total_retrieval_f1/n:.4f}")
            print(f"avg_subgraph_size       = {agent.total_subgraph_size/agent.total_questions:.1f}")

    asyncio.run(main())
