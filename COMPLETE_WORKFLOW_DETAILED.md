# COMPLETE WORKFLOW DETAILED

本文件不是只講概念，而是直接依照目前 `portable_runner` 的真實程式與 `artifacts/` 內已跑出的結果，完整說明：

1. 一次 benchmark run 是怎麼從 KB 與 dataset 走到 `grammar`、`results`、`dumps`
2. HRG grammar 是怎麼被抽出來的
3. online 階段怎麼 parse、驗證 chain、fallback、擴張 subgraph、生成答案
4. `benchmark_results.json` 每個欄位是怎麼來的
5. `all_models_outputs_wide.csv` 每一格到底存什麼
6. grammar 的配對是不是「一條一條照順序對」這件事，實際答案是什麼

---

## 1. 本文件對應的實際 artifacts

目前 `artifacts/` 裡已存在三組成功結果與一份批次總結：

1. `artifacts/wikimovies-wiki_entities-test/`
2. `artifacts/mlpq-en-zh-en-ills/`
3. `artifacts/kqapro-validation/`
4. `artifacts/_batch/run_all_summary.txt`

批次總結目前是：

```txt
metaqa FAILED
wikimovies OK
mlpq OK
kqapro OK
```

所以這份文件的「結果解讀」部分，主要根據上面三組成功 run 來寫。

---

## 2. 整個 workflow 的最短總覽

整體入口是 [run_pipeline.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_pipeline.sh)。

它只做兩件事：

1. 先跑 `hrg_grammar/hrg_extract.py` 產生 grammar
2. 再跑 `LLM_inference_benchmark/benchmark.py` 做 benchmark

也就是：

```text
KB triples
  -> HRG extractor
  -> hrg_grammar.json / hrg_grammar.txt
  -> benchmark.py
  -> 每題 retrieval + answer
  -> benchmark_results.json
  -> all_models_outputs_wide.csv
  -> dumps/q_xxxx.pkl / shared retrieval cache
```

---

## 3. 每次 run 的命名與輸出位置怎麼決定

### 3.1 run tag 怎麼來

`run_pipeline.sh` 會先根據 dataset 組出 `RUN_TAG`。

例子：

1. `wikimovies + wiki_entities + test` -> `wikimovies-wiki_entities-test`
2. `mlpq + en-zh + en + ills` -> `mlpq-en-zh-en-ills`
3. `kqapro + validation` -> `kqapro-validation`

這個 tag 之後會同時決定 grammar、results、dumps 的輸出資料夾。

### 3.2 輸出目錄長什麼樣

每次 run 會有：

```text
artifacts/<run_tag>/
  grammar/
    hrg_grammar.json
    hrg_grammar.txt
  results/
    benchmark_results.json
    all_models_outputs_wide.csv
  dumps/
    per_model/
    _shared_retrieval/
```

例如 WikiMovies 實際就是：

```text
artifacts/wikimovies-wiki_entities-test/
```

---

## 4. Offline 階段：HRG grammar 是怎麼做出來的

核心程式是 [hrg_grammar/hrg_extract.py](/home/zihui/projects/masterPaperRemake/portable_runner/hrg_grammar/hrg_extract.py)。

可以拆成 8 個步驟。

### 4.1 讀 KB 三元組並正規化

首先會逐行讀 KB，呼叫：

1. `_normalize_token`
2. `_parse_triple_line`
3. `load_labeled_kb_graph`

支援的輸入格式包含：

1. `head|relation|tail`
2. `head\trelation\ttail`
3. N-Triples
4. 一般空白分隔
5. WikiMovies 特殊格式：`數字 head relation tail`

#### 例子

如果原始行是：

```text
42 The_Matrix directed_by Lana_Wachowski
```

或

```text
The_Matrix|directed_by|Lana_Wachowski
```

最後都會被統一成：

```text
(head="The_Matrix", relation="directed_by", tail="Lana_Wachowski")
```

這一步的重點是：先把所有奇形怪狀格式收斂成同一種 triple。

### 4.2 建立有標籤的 MultiDiGraph

接著每條 triple 都會進 `networkx.MultiDiGraph`：

1. `head` 是起點
2. `tail` 是終點
3. `relation` 同時當 edge key 與 `rel`

為什麼不是普通 `DiGraph`：

1. 同一對節點可能有多條不同 relation
2. KG 本身有方向
3. 之後還要保留 relation label

### 4.3 轉成 undirected skeleton

HRG 抽取時不是直接在原始有向 KG 上做 clique tree，而是先做 skeleton：

1. 節點保留
2. 邊只保留「有沒有連接」
3. 暫時不管 relation label 與方向

目的不是丟語意，而是方便：

1. 算 degree
2. 做 MCS ordering
3. 做 triangulation / clique tree

### 4.4 先抽多個 BFS sample，不直接吃整張圖

程式不會直接把整張大圖拿去做 clique extraction，而是：

1. 先根據 skeleton degree 避開 hub
2. 選 seed
3. 對每個 seed 做 capped BFS
4. 取 node-induced subgraph

預設參數是：

1. `K_SAMPLES = 4`
2. `S_SAMPLE_SIZE = 500`
3. `BFS_MAX_BRANCH = 30`

#### 為什麼要這樣

如果直接從高 degree 節點展開，很容易 clique 爆炸，grammar 會變得不可控。

所以它不是「隨便抽」，而是「刻意限制 hub 擴張」。

### 4.5 對 sample graph 做 MCS 與 triangulation

每個 sample graph 會做：

1. `mcs_ordering`
2. `triangulate_from_order`
3. 收 maximal clique candidates

這裡的作用是把 sample graph 轉成一組 clique bags。

### 4.6 把 cliques 組成 clique tree，再二元化、修剪

接著會：

1. `build_clique_tree_from_cliques`
2. `binarize_clique_tree`
3. `prune_leaf_no_internal`

意思是：

1. 先用 clique overlap 建 weighted graph
2. 取 maximum spanning tree 當 clique tree
3. 若某個 bag 子節點超過 2 個，就 clone bag 做二元化
4. 若某個 leaf bag 完全被 parent 包住，就刪掉

這樣做是為了讓後面的 HRG rule 形狀更穩定。

### 4.7 把每個 bag 變成一條 HRG rule

真正 rule 抽取在：

```python
extract_hrg_rules_labeled(G, T, bags, root=0)
```

規則是：

1. root bag 的左邊是 `S/0`
2. 非 root bag 的左邊是 `N/k`
3. 其中 `k = 這個 bag 與 parent 的交集大小`
4. 該 bag 負責的 terminal edges 會放進 RHS 的 `terminals`
5. 該 bag 的 child 會變成 RHS 的 `nonterms`

#### 這裡的「complete」是什麼意思

一個 bag 要變成一條完整 rule，至少要把三件事補齊：

1. `lhs` 是哪個 nonterminal
2. `rhs.terminals` 是這個 bag 實際承接到的 labeled edges
3. `rhs.nonterms` 是它對子 bag 的銜接點位 `att`

也就是說，`complete` 不是單純「把 relation 收集起來」，而是：

1. 先決定 bag 與 parent 的交集
2. 再決定 bag 內哪些點是 external，哪些是 internal
3. 然後對 bag 內節點做局部 re-index
4. 最後把 terminal edges 與 child attachment 一起寫進 RHS

這樣一條 rule 才算完整。

### 4.8 合併重複 rule

最後會跑：

```python
merge_duplicate_rules(rules)
```

如果兩條 rule 的：

1. `lhs`
2. `rhs.terminals`
3. `rhs.nonterms`

完全相同，就合併成一條，然後把 `count` 加總。

這個 `count` 後面 online 階段會拿來當規則強度排序依據。

---

## 5. grammar 檔案長什麼樣

grammar 會輸出兩份：

1. `hrg_grammar.json`
2. `hrg_grammar.txt`

### 5.1 JSON 格式

每條 rule 長這樣：

```json
{
  "lhs": {"name": "N", "rank": 27},
  "rhs": {
    "terminals": [
      {"a": 27, "rel": "directed_by", "b": 11},
      {"a": 27, "rel": "has_genre", "b": 8},
      {"a": 27, "rel": "starred_actors", "b": 7},
      {"a": 27, "rel": "written_by", "b": 1}
    ],
    "nonterms": [
      {"name": "N", "rank": 1, "att": [27]}
    ]
  },
  "count": 1
}
```

上面這個形狀對應到 `artifacts/wikimovies-wiki_entities-test/grammar/hrg_grammar.txt` 裡實際存在的一條 `N/27` 規則。

### 5.2 TXT 格式比較好讀

WikiMovies grammar 中有這樣一條：

```txt
N/27  count=1
  T: [..., (27, 'directed_by', 11), (27, 'has_genre', 8), ..., (27, 'starred_actors', 7), ..., (27, 'written_by', 1), ...]
  N: [('N', 1, (27,))]
```

這代表：

1. 這條 rule 左邊是 `N/27`
2. 右邊 terminals 裡同時出現 `directed_by`
3. 也同時出現 `has_genre`
4. 也同時出現 `starred_actors`
5. 也同時出現 `written_by`

換句話說，這條 rule 表示在某個局部結構裡，這些 relation 曾一起共現。

---

## 6. Online 階段：每一題是怎麼跑的

核心程式是 [LLM_inference_benchmark/benchmark.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/benchmark.py) 和 [LLM_inference_benchmark/knowledgegraph_agent.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py)。

一題的流程可以拆成 10 步。

### 6.1 先把 dataset 讀成 `(question, references)` 清單

不同資料集會進不同 loader：

1. WikiMovies -> `load_wikimovies_dataset`
2. MLPQ -> `load_mlpq_dataset`
3. KQAPro -> `load_normalized_jsonl_dataset`

最後 benchmark 看到的單位都一樣：

```text
(question, [gold_answer1, gold_answer2, ...])
```

### 6.2 LLM 第一次 parse：提候選 entity + relation chain

`prepare_retrieval()` 先呼叫 `_parse_intent_candidates()`。

LLM 要回：

```json
[
  {"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"], "confidence": 0.82}
]
```

#### 這一步做了什麼

1. 先挑一批可能 relation token 放進 prompt
2. 給 few-shot examples
3. 若有 grammar，就把 top frequent grammar patterns 當 structural hints 放進 prompt
4. 讓 LLM 回傳 top-k candidates

### 6.3 relation token 會再做 fuzzy 對齊

不是 LLM 輸出什麼就直接用。

例如：

1. `directed by`
2. `directed_by`
3. `directed_by^-1`

都會再經過 `_fuzzy_match_relation()` 對齊到系統實際允許的 relation token。

### 6.4 先驗證 chain 能不能在 KB 上走通

對每個 candidate，會做 `_check_chain_validity(entity, chain)`。

#### 例子

如果 candidate 是：

```json
{"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"]}
```

系統會：

1. 先把 `Tom Hanks` 對到 KB 節點
2. 第 1 hop 走 `starred_actors^-1`
3. 看能不能找到電影集合
4. 第 2 hop 從這些電影走 `directed_by`
5. 若任何一步 frontier 變空，就視為 invalid

這一步只是在檢查「有沒有路」，還沒真正產生最後 context。

### 6.5 grammar matching 實際怎麼配對

這是最重要的一段。

很多人直覺會以為 grammar matching 是：

1. 第一跳對第一條 grammar edge
2. 第二跳對第二條 grammar edge
3. 完全照順序一條一條對

但這個 repo 目前**不是這樣做**。

實作在 `HRGMatcher.match_rules()`，真正邏輯是：

1. 先把 chain 去方向
2. 例如 `starred_actors^-1` 會變成 `starred_actors`
3. 把整條 chain 變成一個 bare relation set
4. 只要某條 grammar rule 的 terminal labels 包含這個 set，就算 match

也就是：

```text
bare_chain ⊆ rule_labels
```

#### 具體例子

若 predicted chain 是：

```text
["starred_actors^-1", "directed_by"]
```

會先變成：

```text
{"starred_actors", "directed_by"}
```

如果某條 grammar rule 的 labels 是：

```text
{"starred_actors", "directed_by", "has_genre", "written_by"}
```

那就算 match。

所以答案是：

1. 不是逐 hop 逐 terminal 順序對
2. 不是要求 RHS 剛好等於 chain
3. 而是「去方向後，用 relation label 的 subset inclusion 來配」

### 6.6 matching 後還會再做 same-arity 過濾

雖然一開始是 subset match，但後面 `_select_matched_rules()` 會優先保留：

```text
lhs.rank == len(chain)
```

也就是：

1. 2-hop chain 優先用 `rank=2` 的 rule
2. 3-hop chain 優先用 `rank=3` 的 rule

如果有同 arity 的 match，就縮到這批。

所以實際流程是：

1. 先寬鬆 subset match
2. 再用 chain 長度做 second-stage filter
3. 再取 top-k rules

### 6.7 如果第一輪 chain 全失敗，會做 fallback correction

當所有 candidate 都 invalid 時，會進 fallback。

fallback 有三種來源：

1. 單 hop 方向翻轉
2. grammar fallback
3. 再叫一次 LLM 做 correction

#### 方向翻轉例子

原本：

```text
["written_by"]
```

可能改成：

```text
["written_by^-1"]
```

#### grammar fallback 例子

如果原本 chain 的 bare labels 跟某條高頻規則有 overlap，系統會用那條規則的 labels 重組一條新 chain。

這裡仍然不是完整圖匹配，而是 relation label 層級的重建。

#### 這一步實際是怎麼組 correction pool 的

真正程式順序在 `prepare_retrieval()` 中是：

1. 先確認 `ranked_candidates` 裡沒有任何 `kb_result["valid"] == True`
2. 對每個原始 candidate 產生 direction-flip 候選
3. 如果有 grammar，對每個原始 candidate 產生 grammar fallback 候選
4. 再呼叫 `_correct_candidates_with_llm()` 請 LLM 針對 failed candidates 提新 chain
5. 把三路來源全部合併
6. 用 `_dedup_candidates()` 依 `(entity, tuple(chain))` 去重
7. 再逐條重跑 validity check、grammar score、ranking key

也就是 correction 並不是「先 grammar，再 LLM」，而是：

```text
flip candidates
+ grammar fallback candidates
+ llm correction candidates
-> dedup
-> 全部重新評分
```

#### LLM correction prompt 到底允許改什麼

`_correct_candidates_with_llm()` 明確允許 LLM：

1. 改 relation 方向
2. 換 relation
3. 保持相同 hop 數優先
4. 必要時整條 chain 換掉

但是它也限制：

1. 優先使用系統提供的 relation candidate set
2. 必須只回 JSON array
3. 最多取 `max_new_candidates=5`

#### correction 的 source priority 怎麼影響排序

`_score_candidate()` 裡，來源優先順序是：

1. 原始 `llm` -> `source_priority = 3`
2. `llm_correction` -> `2`
3. `flip_hop_i` -> `1`
4. `grammar_fallback` -> `0`

這表示：

1. 在其他條件接近時，系統仍偏好原始 parse
2. LLM correction 比單純 flip 更被信任
3. grammar fallback 是最弱先驗，只在前面都不夠好時才上位

#### 委員可能會問：為什麼 grammar fallback 反而排最弱

因為目前 grammar fallback 不是 exact derivation，只是：

1. 根據 relation 共現規則取 labels
2. 再用舊 chain 的方向資訊重建一條候選

所以它的語義精確度沒有原始 LLM parse 或 LLM correction 高，程式設計上故意把它放成弱 prior。

### 6.8 真正取 subgraph：先 spine，再看要不要 expansion

若 candidate 通過 validity 檢查，接著 `_build_candidate_subgraph()` 會做：

1. 先依 chain 嚴格走一次，拿到 `spine_edges`
2. 如果開了 grammar expansion，再用 matched rule 擴張

#### spine 是什麼

spine 就是由 predicted chain 直接走出來的主路徑邊集合。

例如：

```text
Tom_Hanks --starred_actors^-1--> Cast_Away --directed_by--> Robert_Zemeckis
```

這些邊就是 spine。

#### spine 是怎麼抽的，不是怎麼想像的

spine 由 `_find_subgraph_multi_hop_kb_strict()` 產生。

做法是：

1. frontier 初始為起始 entity
2. 逐 hop 讀 chain 裡的 relation token
3. 若 token 是正向 relation，就從 `kb_out[entity][rel]` 找 tails
4. 若 token 是 `rel^-1`，就從 `kb_in[entity][rel]` 找 heads
5. 每經過一條邊就累加到 `edge_counts`
6. 下一 hop frontier 變成這些邊抵達的節點集合
7. 若某 hop 之後 frontier 變空，就停止

所以 spine 不是單一路徑，而是：

1. 一條 predicted relation chain
2. 在 KB 上可能對應到的整批可執行邊集合

例如 2-hop chain 不一定只有 2 條 edge，而可能是：

1. 第 1 hop 走出 8 條邊
2. 第 2 hop 從這 8 個節點又走出 20 條邊
3. 最後 spine 是這 28 條去重後的 edge 集合

這點很重要，因為委員很容易誤以為 spine = 單一路徑，其實不是。

#### grammar expansion 是什麼

如果 matched rule 的 labels 還包含：

1. `written_by`
2. `has_genre`

那系統會以 spine 上節點為中心，再把這些 relation 的鄰邊加進來。

重點是：

1. 不是把整條 grammar rule 還原成完整 bag graph
2. 而是取 rule 裡的 label 集合作為 allowed relations
3. 然後從 spine nodes 往外補邊

#### GrammarExpansion 實際怎麼 expand

`_expand_subgraph_by_grammar()` 的細節是：

1. 先取 `matched_rules[:topk_expansion_rules]`
2. 若 `expansion_strict=True`，再加一道過濾：
   - 規則 `lhs.rank` 必須等於 `len(chain)`
   - 規則分數 `prob/count` 必須 `>= expansion_min_prob`
3. 將這些規則的 `_cached_labels` 聯集成 `allowed_rels`
4. 從 `spine_edges` 蒐集所有 `spine_nodes`
5. 對每個 spine node：
   - 往外掃所有 outgoing relations
   - 只保留 relation 在 `allowed_rels` 裡的邊
   - 再掃所有 incoming relations
   - 一樣只保留 relation 在 `allowed_rels` 裡的邊
6. 用 `visited` 避免把 spine 已有邊重複加入
7. 若 `expansion_strict=True`，每個 node 最多只擴 `expansion_per_node_cap` 條邊
8. 最後若總 expanded edges 超過 `max_frontier`，再全域截斷

#### GrammarExpansion 不是怎麼 expand

它目前**不是**：

1. 沿著 grammar rule 的 node index `a/b/att` 去做 graph unification
2. 也不是在 bag attachment 上做真正 hyperedge substitution
3. 也不是 exact HRG decoding

它是：

1. 用 grammar rule labels 當 allowed relation vocabulary
2. 以 spine nodes 為中心擴一圈鄰邊

#### 為什麼單跳題不做 expansion

`_build_candidate_subgraph()` 裡有：

```python
is_single_hop = len(chain) == 1
```

只要是 single-hop：

1. 不做 grammar expansion
2. 不做 random expansion
3. 不做 frequency expansion

原因很直接：

1. 1-hop 問題本來就應該用最短最直接的局部 evidence
2. 再往外擴很容易只加噪音
3. 壓縮方法的比較也會不公平

#### GrammarGuidedRetrieval 跟 GrammarExpansion 不一樣

這兩個很容易被混淆。

`GrammarGuidedRetrieval` 是：

1. 在取 spine 前就放寬 strict chain
2. 改成 grammar-allowed BFS

`GrammarExpansion` 是：

1. 先照 strict chain 取 spine
2. 再以 matched rule labels 向外補邊

所以：

1. 前者改的是 retrieval search space
2. 後者改的是 spine 周邊 context 補充

目前 benchmark 裡的 `Spine-GrammarExpansion-*` 與 `HRG-Proposed-*` 走的是後者，不是前者。

#### GrammarGuidedRetrieval 實際怎麼做

`_find_subgraph_grammar_guided()` 的步驟是：

1. 先拿 matched rules
2. 把前 `topk_expansion_rules` 的 labels 合成 `allowed_rels`
3. `max_depth = len(chain)`
4. 從起始 entity 做 BFS，但每一層都只允許走 `allowed_rels`
5. outgoing / incoming 都會看
6. 用 `visited_edges` 去重
7. 用 `visited_nodes` 控制 frontier
8. 每個 node 的擴張上限取決於：
   - strict 模式時用 `expansion_per_node_cap`
   - 否則用 `per_entity_cap`

它比較像：

```text
chain 預測 relation 數量
+ grammar 規則限制 relation 類型
-> constrained BFS
```

而不是：

```text
照 chain token 一跳一跳硬走
```

#### RandomExpansion 怎麼隨機

`_expand_subgraph_random()` 不是每次都真的隨機到不可重現，而是「可重現隨機」。

步驟是：

1. 先取 `spine_nodes`
2. 把 `random_expansion_seed`、`chain`、排序後的 `spine_nodes` 串成字串
3. 對這個字串做 MD5
4. 取前 8 碼轉成整數
5. 用這個整數初始化 `random.Random(seed_int)`

所以同一題、同一 chain、同一組 spine nodes：

1. 每次都會抽到同樣的隨機邊
2. 這樣實驗可重現

之後對每個 spine node：

1. 蒐集所有 outgoing candidate edges
2. 蒐集所有 incoming candidate edges
3. 去掉已在 spine 裡的邊
4. shuffle candidate edges
5. 取前 `expansion_per_node_cap` 條

所以 RandomExpansion 的「隨機」來自：

1. 候選邊先全列出
2. 再隨機打散
3. 然後固定取前 K 條

#### FrequencyExpansion 的頻率是什麼，怎麼來

`_compute_relation_frequency()` 在 agent 初始化時就會先掃整張 KB：

1. 對 `kb_out` 每個 head
2. 對其每個 relation
3. 把該 relation 的 tail 數量加總

所以 frequency 定義是：

```text
某個 relation 在整張 KB 中出現的 edge 總數
```

不是：

1. 某個 rule 的出現次數
2. 某個 question 中的局部頻率
3. 某個實驗 batch 的使用次數

#### FrequencyExpansion 實際怎麼選邊

`_expand_subgraph_by_relation_frequency()` 會：

1. 對每個 spine node 蒐集所有 outgoing/incoming candidate edges
2. 每條候選邊附上該 relation 的全域頻率 `rel_freq`
3. 對候選邊依 `(-rel_freq, edge_tuple)` 排序
4. 取前 `expansion_per_node_cap` 條

所以 FrequencyExpansion 的 bias 是：

1. 偏好全域更常見的 relation
2. 假設高頻 relation 比較可能提供一般性輔助上下文

#### 委員可能會問：這樣不是很容易偏向 stopword relation 嗎

是，這正是這個 ablation 的意義。

它不是要證明最合理，而是要對照：

1. 用 grammar 規則選 relation
2. 跟用單純 relation 全域頻率選 relation
3. 哪個比較好

因此 FrequencyExpansion 的存在，本質上是 control condition。

#### Proposed 到底是什麼

`HRG-Proposed-*` 不是新的推理架構，而是 `KnowledgeGraphAgent` 打開下列開關的組合：

1. `use_grammar_expansion = True`
2. `use_fallback_correction = True`
3. `use_grammar_rerank = False`
4. `use_grammar_hint = False`
5. `expansion_strict = True`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

也就是：

```text
Spine retrieval
+ strict grammar-based expansion
+ fallback correction
```

不是：

1. grammar-first parsing
2. grammar-guided BFS retrieval
3. prompt-injected grammar hints

這點口試很重要，因為名字叫 `HRG-Proposed` 很容易讓人誤會它用了所有 grammar 功能；其實這個 benchmark 裡的 proposed，是「嚴格 grammar expansion + correction」的組合版。

#### 各方法對照表

以 qwen3.5 為例，模型配置差異其實是：

1. `Baseline-BFS-qwen3.5`
   - 不看 chain
   - 不看 grammar
   - 直接以 entity 做雙向 BFS
2. `Spine-Only-qwen3.5-{json|triple}`
   - 用 LLM parse chain
   - 嚴格照 chain 取 spine
   - 不 correction
   - 不 expansion
3. `Spine-Correction-qwen3.5-{json|triple}`
   - Spine-Only
   - 再加 fallback correction
4. `Spine-GrammarExpansion-qwen3.5-{json|triple}`
   - Spine-Only
   - 再加 grammar expansion
   - 不 correction
5. `Spine-RandomExpansion-qwen3.5-{json|triple}`
   - Spine-Only
   - 再加 random expansion
6. `Spine-FrequencyExpansion-qwen3.5-{json|triple}`
   - Spine-Only
   - 再加 frequency expansion
7. `HRG-Proposed-qwen3.5-{json|triple}`
   - Spine-Only
   - 加 grammar expansion
   - 加 correction
   - expansion 用 strict mode

#### Candidate ranking 怎麼排，不是黑盒

`_score_candidate()` 是 lexicographic tuple，優先序固定如下：

1. `valid`：KB 上能不能完整走通
2. `same_arity_hit`：有沒有 match 到 hop 數相同的 grammar rule
3. `grammar_hit`
4. `grammar_score`
5. `grammar_matched_count`
6. `failure_progress`
7. `step_survival`
8. `final_size`
9. `source_priority`
10. `llm_confidence`
11. `llm_prior`

這代表：

1. 先求能不能走通
2. 再求 grammar 相容
3. 再看若失敗，是失敗在第幾 hop
4. 最後才用 LLM 自己的信心當 tie-breaker

所以這不是單純「信 LLM」，而是明確地把 KB executability 擺在最前面。

#### Subgraph ranking 怎麼排

即便多條 chain 都 valid，也不是直接選第一條。

每條 valid chain 都會先 build 一個 subgraph，然後用 `_score_subgraph_candidate()` 再排一次。

排序依據是：

1. `has_edges`
2. `same_arity_hit`
3. `grammar_hit`
4. `grammar_score`
5. `spine_size`
6. `expanded_size`
7. `compactness = -subgraph_size`
8. 原始 candidate ranking key

這裡一個容易被忽略的點是：

1. 當支持品質差不多時，系統偏好較小的 subgraph
2. 所以不是越多邊越好
3. 它在 retrieval 階段就已經內建 compression bias

#### shared retrieval cache 是什麼

`benchmark.py` 對同 backbone 的 `json/triple` 兩個版本，不會重做兩次 retrieval。

它會先根據：

1. `base_model_name`
2. `dataset_name`
3. `run_tag`
4. `agent_kwargs`（扣掉 serialization）

組出一個 hash key，存成：

```text
artifacts/<run_tag>/dumps/_shared_retrieval/<share_key>/q_0000.prepared.pkl
```

這表示：

1. `Spine-Only-qwen3.5-json` 跟 `Spine-Only-qwen3.5-triple`
2. 用的是同一份 prepared retrieval
3. 差別只在最後 context serialization 與 answer generation

這樣做的理由是：

1. 讓 json/triple 的比較更公平
2. 避免 retrieval 階段隨機性或重跑差異污染比較
3. 減少重複成本

#### Baseline 跟 spine 系列的本質差異

baseline 在 [baseline.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/baseline.py) 裡的做法是：

1. 只抽 entity，不抽 chain
2. 直接做雙向 BFS
3. 深度由題目 hop 數決定
4. outgoing/incoming 全收
5. 每 hop 可做 edge cap

所以 baseline 的 retrieval space 是：

```text
entity-centered unguided BFS neighborhood
```

而 spine / grammar 系列是：

```text
entity + predicted relation chain centered retrieval
```

這就是為什麼 baseline 通常：

1. recall 比較高
2. context 比較大
3. subgraph size 也比較大

#### 委員可能追問的幾個真正敏感點

1. `grammar_score` 現在其實直接用 `count/probability`，沒有做 dataset-size normalization
2. `expansion_min_prob=0.005` 在目前 grammar JSON 裡實際上常常等價於「count 門檻非常低」，因為 extractor 存的是 `count`
3. grammar matching 用的是 relation set inclusion，不是 exact ordered chain match
4. grammar expansion 用的是 label union，而不是 RHS attachment-structure preserving expansion
5. retrieval precision / recall 是以 subgraph node 是否涵蓋 gold answer 來算，不是 edge-level metric
6. `coverage` 的定義是 grammar hit 題數 / total questions，不是答案答對率
7. `answerable_rate` 在目前程式裡幾乎等於輸出字串是否非空，所以很多失敗題仍可能算 answerable

這幾點若先寫清楚，口試時比較不會被抓到說法太泛。

### 6.9 context 序列化成 JSON 或 triples

同一個 retrieval 結果，會做兩種 serialization：

1. `json`
2. `triples`

#### 例子

JSON：

```json
[{"head":"Cast Away","relation":"directed_by","tail":"Robert Zemeckis","count":1}]
```

Triple：

```text
Cast Away directed_by Robert Zemeckis.
```

### 6.10 最後才是 answer generation

生成答案時，LLM 被要求：

1. 只能用 Context
2. 只能輸出最後答案
3. 多答案用 ` | ` 串接
4. 沒有答案就輸出 `I don't know`

這一步的輸出才會拿去跟 gold answer 算 EM / F1 / recall。

---

## 7. `benchmark_results.json` 每個欄位怎麼來

這個檔案是以「模型名」為 key。

例如：

```json
"HRG-Proposed-qwen3.5-json@kqapro-validation": { ... }
```

每個模型底下有兩層數字。

### 7.1 `results`

`results` 是分 hop 的平均值。

例如 KQAPro baseline：

```json
"results": {
  "1-hop": {...},
  "2-hop": {...},
  "3-hop": {...}
}
```

### 7.2 外層同名欄位

外層的：

1. `bleu`
2. `answer_recall`
3. `em`
4. `answer_set_f1`
5. `avg_latency`

是對該模型所有 dataset split 結果再做一次平均後得到的 overall。

### 7.3 指標意義

#### `answer_recall`

如果 gold 只有 1 個答案，這其實近似單答案正確率。

如果 gold 有多個答案，就是候選答案集合對 gold 集合的覆蓋率。

#### `em`

1. 單答案時，等於答對或答錯
2. 多答案時，要求 candidate set 與 gold set 完全相等

#### `contains_hit`

只要 raw output 字串裡包含任一 gold 字串就算命中。

#### `hit_at_1_any`

把 output 切成候選答案集合後，只要和 gold set 有交集就算 1。

#### `answer_set_precision / recall / f1`

這三個是 set-based 指標。

### 7.4 retrieval 與成本相關欄位

1. `coverage`: 有 grammar hit 的題目比例
2. `avg_ctx_tokens`: 平均 context token 量
3. `avg_parse1_tokens`: 第一次 parse 花的 token
4. `avg_correction_tokens`: correction 花的 token
5. `avg_parse2_tokens`: 最終答案生成花的 token
6. `avg_subgraph_size`: 最終 subgraph 邊數
7. `avg_retrieval_recall / precision / f1`: subgraph 與 gold answer 的對齊情況
8. `avg_parse_latency / avg_retrieval_latency / avg_generation_latency`
9. `answerable_rate`
10. `generation_failure_count`
11. `failure_counts`

### 7.5 `failure_counts` 是怎麼計的

每題都會被標一個 `failure_stage`，例如：

1. `ok`
2. `retrieval_empty`
3. `no_valid_chain`
4. `no_candidates`
5. `runtime_error`
6. `oom`

最後把各題計數加總，就變成 `failure_counts`。

#### 例子

WikiMovies `Baseline-BFS-qwen3.5`：

```json
"failure_counts": {
  "retrieval_empty": 94,
  "ok": 6
}
```

表示 100 題裡：

1. 94 題在 retrieval 就空掉
2. 6 題至少有正常進到 `ok`

### 7.6 `compression_vs_bfs_*` 怎麼算

這兩個欄位是 benchmark 後處理算的：

1. `compression_vs_bfs_ctx_ratio = 本模型 avg_ctx_tokens / 同 backbone baseline 的 avg_ctx_tokens`
2. `compression_vs_bfs_subgraph_ratio = 本模型 avg_subgraph_size / 同 backbone baseline 的 avg_subgraph_size`

所以：

1. 小於 1 代表比 baseline 更壓縮
2. 越小表示 context/subgraph 越短

---

## 8. `all_models_outputs_wide.csv` 每一格存的是什麼

這個檔不是只有答案字串，而是**每個模型對每一題的完整 payload JSON**。

欄位大致是：

1. `hop`
2. `dataset`
3. `idx`
4. `question`
5. `expected_outputs`
6. 每個模型一欄

而每個模型欄裡裝的是：

```json
{
  "answer": "...",
  "bleu": 0.0,
  "answer_recall": 0.0,
  "em": 0.0,
  "elapsed": 1.23,
  "failure_stage": "no_valid_chain",
  "parse_latency": ...,
  "retrieval_latency": ...,
  "generation_latency": ...,
  "generation_failed": false,
  "answerable": true
}
```

### 實際例子

WikiMovies 第 0 題：

1. `question = "what does Grégoire Colin appear in?"`
2. `expected_outputs = "Before the Rain"`

同一題在不同模型欄位中，可以看到：

1. 有些模型 `failure_stage = retrieval_empty`
2. 有些模型 `failure_stage = no_valid_chain`
3. 有些模型 `failure_stage = no_candidates`

所以這個 CSV 的真正用途是：

1. 對齊同一題在不同模型的失敗型態
2. 檢查是 parse 掛掉、chain 不可走、還是 retrieval 空掉
3. 做 qualitative error analysis

---

## 9. 三個已跑完 artifacts 的結果要怎麼看

下面只抓最有代表性的現象。

### 9.0 先看一組統一比較口徑

下面的數字若沒有特別註明，主要以 `qwen3.5` backbone 為例，因為：

1. 三組 artifact 都有完整的 `qwen3.5` 結果
2. 它同時覆蓋 baseline、spine、correction、grammar expansion、random/frequency、proposed
3. 最適合作為方法對照主軸

建議口試時固定先講這 6 個欄位：

1. `answer_recall`
2. `em`
3. `avg_ctx_tokens`
4. `avg_subgraph_size`
5. `coverage`
6. `failure_counts`

因為這 6 個欄位能同時回答：

1. 答案效果
2. 壓縮程度
3. grammar 有沒有真的介入
4. 失敗主要卡在哪一段

### 9.0.1 WikiMovies `qwen3.5` 主要方法對照

| Method | Answer Recall | EM | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.0305 | 0.03 | 2.02 | 0.14 | 0.00 | `retrieval_empty=94, ok=6` |
| Spine-Only-json | 0.0305 | 0.03 | 0.86 | 0.04 | 0.00 | `no_valid_chain=96, ok=4` |
| Spine-Correction-json | 0.0305 | 0.03 | 0.86 | 0.04 | 0.00 | `no_valid_chain=96, ok=4` |
| Spine-GrammarExpansion-json | 0.0205 | 0.02 | 0.67 | 0.03 | 0.03 | `no_valid_chain=97, ok=3` |
| HRG-Proposed-json | 0.0205 | 0.02 | 0.67 | 0.03 | 0.03 | `no_valid_chain=97, ok=3` |

這組數字最重要的結論是：

1. baseline 與 spine-only 在 answer recall/EM 幾乎打平
2. 但 baseline 的失敗型態是 `retrieval_empty`
3. spine 系列的失敗型態主要轉成 `no_valid_chain`
4. grammar expansion 與 proposed 在 WikiMovies 上沒有把主效果拉起來
5. 反而只看得到少量 `coverage=0.03` 的 grammar 介入痕跡

### 9.0.2 MLPQ `qwen3.5` 主要方法對照

| Method | Answer Recall | EM | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.335 | 0.25 | 6200.835 | 418.14 | 0.00 | `ok=200` |
| Spine-Only-json | 0.06 | 0.03 | 66.365 | 2.795 | 0.00 | `no_valid_chain=152, ok=48` |
| Spine-Correction-json | 0.06 | 0.03 | 67.22 | 2.835 | 0.00 | `no_valid_chain=150, ok=50` |
| Spine-GrammarExpansion-json | 0.025 | 0.02 | 157.595 | 6.17 | 0.075 | `no_valid_chain=168, ok=32` |
| HRG-Proposed-json | 0.045 | 0.04 | 421.34 | 16.185 | 0.135 | `no_valid_chain=154, ok=46` |

這組數字很適合回答「你的方法到底在 trade off 什麼」：

1. baseline 的 `answer_recall=0.335` 最好
2. 但它付出的代價是 `avg_ctx_tokens=6200.835`、`avg_subgraph_size=418.14`
3. spine-only 把 context 壓到 66 左右，只有 baseline 的約 `1.07%`
4. proposed 把 context 拉回 `421.34`，仍只有 baseline 的約 `6.79%`
5. proposed 的 subgraph 大小 `16.185`，也遠小於 baseline 的 `418.14`

所以在 MLPQ 上最精確的說法不是「proposed 最準」，而是：

1. proposed 在極端壓縮與極端大型 context 之間提供中間點
2. grammar 與 correction 讓 `ok` 題數從 32/48 的區間回到 46
3. 但仍明顯落後 baseline 的 answer recall

### 9.0.3 KQAPro `qwen3.5` 主要方法對照

| Method | Answer Recall | EM | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.14 | 0.14 | 3068.21 | 215.5567 | 0.00 | `ok=247, retrieval_empty=53` |
| Spine-Only-json | 0.0333 | 0.0333 | 8.87 | 0.36 | 0.00 | `no_valid_chain=256, no_candidates=19, ok=25` |
| Spine-Correction-json | 0.0367 | 0.0367 | 10.3633 | 0.4267 | 0.00 | `no_valid_chain=252, no_candidates=19, ok=29` |
| Spine-GrammarExpansion-json | 0.0433 | 0.0433 | 17.8933 | 0.7333 | 0.0467 | `no_valid_chain=261, no_candidates=12, ok=27` |
| HRG-Proposed-json | 0.07 | 0.07 | 21.3267 | 0.88 | 0.0733 | `no_valid_chain=246, no_candidates=12, ok=42` |

KQAPro 是目前最能看出 proposed 方法效果的例子：

1. `Spine-Only` -> `answer_recall=0.0333`
2. `Spine-Correction` -> `0.0367`
3. `Spine-GrammarExpansion` -> `0.0433`
4. `HRG-Proposed` -> `0.07`

也就是 correction 單獨有幫助，grammar expansion 單獨也有幫助，而兩者合起來的 proposed 提升更明顯。

而且它仍然維持強壓縮：

1. proposed `avg_ctx_tokens=21.3267`
2. baseline `avg_ctx_tokens=3068.21`
3. 壓縮比約為 `0.00695`

這組數字很適合在口試中回答：

1. 你的方法不是要打贏 baseline 的絕對 recall
2. 而是在超大幅壓縮下，盡量把 answer quality 拉回來

### 9.0.4 JSON 與 Triple 序列化差異也要講數字

很多委員會追問 `json` 跟 `triple` 為什麼都保留。

目前三組 artifact 的一個穩定現象是：

1. `triple` 幾乎總是比 `json` 更省 `avg_ctx_tokens`
2. 例如 WikiMovies `Spine-Only-qwen3.5`
   - json: `avg_ctx_tokens=0.86`
   - triple: `avg_ctx_tokens=0.35`
3. MLPQ `HRG-Proposed-qwen3.5`
   - json: `avg_ctx_tokens=421.34`
   - triple: `avg_ctx_tokens=218.955`
4. KQAPro `HRG-Proposed-qwen3.5`
   - json: `avg_ctx_tokens=21.3267`
   - triple: `avg_ctx_tokens` 更低，而且 `1-hop` recall 還略高於 json

因此可以合理說：

1. triple serialization 的主要效果是節省 context token
2. 它不保證所有資料集都提升答案品質
3. 但它通常提供更好的 token efficiency

### 9.1 WikiMovies：極度稀疏，主要卡在 chain 與 retrieval

`wikimovies-wiki_entities-test` 的 qwen3.5 baseline：

1. `em = 0.03`
2. `avg_ctx_tokens = 2.02`
3. `avg_subgraph_size = 0.14`
4. `failure_counts = {"retrieval_empty": 94, "ok": 6}`

代表這組 run 的主問題不是 context 太大，而是：

1. 大多數題根本拿不到有效 subgraph
2. 所以再精緻的生成也沒有足夠內容可答

同一組中 `HRG-Proposed-qwen3.5-json`：

1. `coverage = 0.03`
2. `em = 0.02`
3. `avg_ctx_tokens = 0.67`
4. `failure_counts = {"no_valid_chain": 97, "ok": 3}`

解讀是：

1. grammar 的確有 hit 到少部分題目
2. 但大量問題更早就倒在 `no_valid_chain`
3. 所以 WikiMovies 這份 artifact 主要反映的是 parse/retrieval bottleneck，不是生成瓶頸

### 9.2 MLPQ：baseline 很大、壓縮很強，但準確率掉很多

`mlpq-en-zh-en-ills` 的 qwen3.5 baseline：

1. `answer_recall = 0.335`
2. `em = 0.25`
3. `avg_ctx_tokens = 6200.835`
4. `avg_subgraph_size = 418.14`
5. `avg_retrieval_recall = 0.745`

這代表 baseline 在 MLPQ 的策略是：

1. 抓很大的 subgraph
2. retrieval recall 很高
3. 代價是 context 超大

對照 `Spine-Only-qwen3.5-json`：

1. `answer_recall = 0.06`
2. `em = 0.03`
3. `avg_ctx_tokens = 66.365`
4. `compression_vs_bfs_ctx_ratio = 0.0107`

這表示它把 context 壓到 baseline 的約 1%，但答案表現也大幅下降。

再看 `HRG-Proposed-qwen3.5-json`：

1. `coverage = 0.135`
2. `avg_ctx_tokens = 421.34`
3. `avg_subgraph_size = 16.185`
4. `answer_recall = 0.045`
5. `failure_counts = {"ok": 46, "no_valid_chain": 154}`

這組最值得記錄的不是準確率高，而是：

1. grammar 擴張把 context 從 66 拉到 421
2. 但仍遠小於 baseline 的 6200
3. 說明這套方法在 MLPQ 上是「中等擴張、仍然壓縮」
4. 只是 parse/chain valid 問題仍然很重

### 9.3 KQAPro：baseline recall 最高，但 context 也最大

`kqapro-validation` 的 qwen3.5 baseline：

1. `answer_recall = 0.14`
2. `em = 0.14`
3. `avg_ctx_tokens = 3068.21`
4. `avg_subgraph_size = 215.56`
5. `avg_retrieval_recall = 0.4067`

對照 `Spine-Only-qwen3.5-json`：

1. `answer_recall = 0.0333`
2. `avg_ctx_tokens = 8.87`
3. `failure_counts = {"no_valid_chain": 256, "no_candidates": 19, "ok": 25}`

再看 `HRG-Proposed-qwen3.5-json`：

1. `answer_recall = 0.07`
2. `em = 0.07`
3. `coverage = 0.0733`
4. `avg_ctx_tokens = 21.33`
5. `avg_subgraph_size = 0.88`
6. `failure_counts = {"no_valid_chain": 246, "ok": 42, "no_candidates": 12}`

這組可以很清楚看到 HRG pipeline 的作用：

1. 與 `Spine-Only` 比，`ok` 題數從 25 增到 42
2. `no_candidates` 從 19 降到 12
3. `answer_recall` 從 `0.0333` 升到 `0.07`
4. 但 context 仍然遠小於 baseline

所以 KQAPro 是目前最能看出「grammar 幫助 retrieval，但仍保留強壓縮」的 artifact。

---

## 10. 一題從頭到尾的具體走法

下面用「抽象但貼近實作」的方式寫一次完整題流程。

### Step 1：輸入題目

例子：

```text
Who directed the films that [Tom Hanks] starred in?
```

gold answer 假設是：

```text
["Robert Zemeckis"]
```

### Step 2：Parse 1

LLM 可能輸出：

```json
[
  {"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"], "confidence": 0.82},
  {"entity": "Tom Hanks", "chain": ["written_by^-1", "directed_by"], "confidence": 0.33}
]
```

### Step 3：KB validity check

系統逐條檢查：

1. `["starred_actors^-1", "directed_by"]` 能不能走通
2. `["written_by^-1", "directed_by"]` 能不能走通

第一條若可走，第二條若中途 frontier 變空，就會被判 invalid。

### Step 4：grammar matching

對第一條 chain：

```text
["starred_actors^-1", "directed_by"]
```

先去方向：

```text
{"starred_actors", "directed_by"}
```

如果 grammar 裡某條規則 labels 包含這兩個 relation，就算 matched。

### Step 5：candidate ranking

排序時會綜合：

1. LLM confidence
2. chain 是否 valid
3. grammar score
4. 失敗進度
5. 候選來源是原始 parse 還是 fallback

### Step 6：取 spine

從 `Tom Hanks` 開始，沿著：

1. `starred_actors^-1`
2. `directed_by`

把嚴格主路徑邊抓出來。

### Step 7：grammar expansion

若 matched rule 還帶有：

1. `written_by`
2. `has_genre`

那就從 spine 上節點往外補這些 relation 的邊。

### Step 8：序列化 context

可能變成：

```json
[
  {"head":"Tom Hanks","relation":"starred_actors","tail":"Cast Away","count":1},
  {"head":"Cast Away","relation":"directed_by","tail":"Robert Zemeckis","count":1},
  {"head":"Cast Away","relation":"written_by","tail":"William Broyles Jr.","count":1}
]
```

### Step 9：Answer generation

LLM 只允許根據 context 回：

```text
Robert Zemeckis
```

### Step 10：算 metrics 並寫到 CSV / JSON

這題最後會產生：

1. per-question payload -> 寫進 `all_models_outputs_wide.csv`
2. 各指標累加 -> 匯總進 `benchmark_results.json`

---

## 10.5 委員最容易追問的設計問題與標準回答

### Q1. 為什麼 grammar matching 不按順序對 hop

因為目前這版系統把 grammar 用在「relation co-occurrence prior」，不是 exact derivation parser。

所以它回答的是：

1. 這條 chain 涉及的 relation 組合，在 KG 裡常不常一起出現
2. 哪些 relation 值得用來擴張 spine 周邊 context

不是回答：

1. 這條 chain 是否被某條 HRG rule 完整逐步生成

### Q2. 為什麼 correction 放在 fallback，不一開始就做

因為一開始就 correction 會：

1. 增加 token 成本
2. 讓原始 parse 與修正版本混在一起
3. 降低 ablation 可解釋性

所以程式選擇：

1. 先相信初始 parse
2. 全失敗時才 correction

### Q3. 為什麼 Proposed 不開 grammar_hint / grammar_rerank

因為 benchmark 設計想把主要效果集中在：

1. grammar expansion
2. correction

如果同時打開太多 grammar 功能，委員會很難分辨：

1. 到底是 prompt hint 有效
2. 還是 retrieval expansion 有效
3. 還是 rerank 有效

### Q4. 為什麼要有 random / frequency 這兩個 expansion

因為它們是控制組：

1. random expansion 測試「只要多加邊是不是就有幫助」
2. frequency expansion 測試「只用全域高頻 relation 補邊是不是就夠」
3. grammar expansion 才是測試「結構先驗選邊」是否比前兩者更合理

---

## 11. 目前這套 grammar 使用方式的準確說法

如果要寫在論文或口試，最精確的說法應該是：

1. offline 階段，系統從 KG 的局部結構中抽取 HRG-style production rules
2. online 階段，系統不是做完整 graph derivation
3. 而是把 rule 中出現的 relation co-occurrence 當成 structural prior
4. 對 predicted relation chain 做 subset-style grammar matching
5. 再用 matched rule 的 relation labels 來做 rerank、fallback、subgraph expansion

也就是說，目前這版比較接近：

```text
grammar-guided relation co-occurrence retrieval
```

而不是：

```text
full symbolic HRG parsing / exact derivation decoding
```

---

## 12. 總結：你看 artifact 時應該怎麼讀

最實用的閱讀順序是：

1. 先看 `results/benchmark_results.json`
2. 先抓 `failure_counts`、`avg_ctx_tokens`、`avg_subgraph_size`
3. 再看 `coverage` 與 `avg_retrieval_*`
4. 若某模型失敗多，去 `all_models_outputs_wide.csv` 看它失敗在 `no_candidates`、`no_valid_chain` 還是 `retrieval_empty`
5. 若要理解 grammar 有沒有幫上忙，再對照 `grammar/hrg_grammar.txt`

最重要的三個判讀原則是：

1. `coverage` 高，不代表答案一定好，只代表更多題目有 hit 到 grammar
2. `compression_vs_bfs_*` 小，不代表方法比較好，只代表壓縮比較強
3. 真正要看 trade-off，必須同時看 `answer_recall / em` 和 `avg_ctx_tokens / avg_subgraph_size`

這三組 artifact 目前共同指出的現象是：

1. baseline 通常 recall 較高，但 context 很大
2. spine-only 壓縮最強，但很容易掉到 `no_valid_chain`
3. HRG-proposed 在部分資料集能把 `ok` 題數拉回來一些
4. 但目前整體瓶頸仍然主要在 chain parsing 與 chain validity，而不是最後答案生成

---

## Appendix A. 口試用主表

這一節的目的不是再解釋方法，而是提供可以直接放進簡報或口試回答的數字。

### A.1 WikiMovies 主表

| Method | Answer Recall | EM | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-qwen3.5 | 0.0305 | 0.03 | 0.0310 | 1.43 | 2.02 | 0.14 | 0.00 | `retrieval_empty=94, ok=6` |
| Spine-Only-qwen3.5-json | 0.0305 | 0.03 | 0.0310 | 5.75 | 0.86 | 0.04 | 0.00 | `no_valid_chain=96, ok=4` |
| Spine-Correction-qwen3.5-json | 0.0305 | 0.03 | 0.0310 | 7.81 | 0.86 | 0.04 | 0.00 | `no_valid_chain=96, ok=4` |
| Spine-GrammarExpansion-qwen3.5-json | 0.0205 | 0.02 | 0.0210 | 6.55 | 0.67 | 0.03 | 0.03 | `no_valid_chain=97, ok=3` |
| HRG-Proposed-qwen3.5-json | 0.0205 | 0.02 | 0.0210 | 9.00 | 0.67 | 0.03 | 0.03 | `no_valid_chain=97, ok=3` |

### A.2 MLPQ 主表

| Method | Answer Recall | EM | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-qwen3.5 | 0.3350 | 0.25 | 0.3562 | 2.58 | 6200.835 | 418.14 | 0.00 | `ok=200` |
| Spine-Only-qwen3.5-json | 0.0600 | 0.03 | 0.0988 | 0.30 | 66.365 | 2.795 | 0.00 | `no_valid_chain=152, ok=48` |
| Spine-Correction-qwen3.5-json | 0.0600 | 0.03 | 0.1021 | 0.30 | 67.22 | 2.835 | 0.00 | `no_valid_chain=150, ok=50` |
| Spine-GrammarExpansion-qwen3.5-json | 0.0250 | 0.02 | 0.0473 | 0.22 | 157.595 | 6.17 | 0.075 | `no_valid_chain=168, ok=32` |
| HRG-Proposed-qwen3.5-json | 0.0450 | 0.04 | 0.0673 | 0.32 | 421.34 | 16.185 | 0.135 | `no_valid_chain=154, ok=46` |

### A.3 KQAPro 主表

| Method | Answer Recall | EM | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-qwen3.5 | 0.1400 | 0.14 | 0.1436 | 2.66 | 3068.21 | 215.5567 | 0.00 | `ok=247, retrieval_empty=53` |
| Spine-Only-qwen3.5-json | 0.0333 | 0.0333 | 0.0333 | 8.02 | 8.87 | 0.36 | 0.00 | `no_valid_chain=256, no_candidates=19, ok=25` |
| Spine-Correction-qwen3.5-json | 0.0367 | 0.0367 | 0.0380 | 12.22 | 10.3633 | 0.4267 | 0.00 | `no_valid_chain=252, no_candidates=19, ok=29` |
| Spine-GrammarExpansion-qwen3.5-json | 0.0433 | 0.0433 | 0.0433 | 8.25 | 17.8933 | 0.7333 | 0.0467 | `no_valid_chain=261, no_candidates=12, ok=27` |
| HRG-Proposed-qwen3.5-json | 0.0700 | 0.07 | 0.0713 | 13.03 | 21.3267 | 0.88 | 0.0733 | `no_valid_chain=246, no_candidates=12, ok=42` |

---

## Appendix B. Hop 級別數據

### B.1 MLPQ `Baseline-BFS-qwen3.5`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 2-hop | 0.44 | 0.30 | 0.4957 | 1.20 |
| 3-hop | 0.23 | 0.20 | 0.2167 | 3.95 |

### B.2 MLPQ `Spine-Only-qwen3.5-json`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 2-hop | 0.00 | 0.00 | 0.0000 | 0.04 |
| 3-hop | 0.12 | 0.06 | 0.1975 | 0.56 |

### B.3 MLPQ `HRG-Proposed-qwen3.5-json`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 2-hop | 0.00 | 0.00 | 0.0000 | 0.06 |
| 3-hop | 0.09 | 0.08 | 0.1345 | 0.59 |

### B.4 KQAPro `Baseline-BFS-qwen3.5`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 1-hop | 0.03 | 0.03 | 0.0300 | 1.92 |
| 2-hop | 0.17 | 0.17 | 0.1807 | 2.36 |
| 3-hop | 0.22 | 0.22 | 0.2200 | 3.70 |

### B.5 KQAPro `Spine-Only-qwen3.5-json`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 1-hop | 0.06 | 0.06 | 0.0600 | 6.29 |
| 2-hop | 0.02 | 0.02 | 0.0200 | 7.77 |
| 3-hop | 0.02 | 0.02 | 0.0200 | 9.99 |

### B.6 KQAPro `HRG-Proposed-qwen3.5-json`

| Hop | Answer Recall | EM | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|
| 1-hop | 0.10 | 0.10 | 0.1000 | 10.38 |
| 2-hop | 0.05 | 0.05 | 0.0540 | 12.83 |
| 3-hop | 0.06 | 0.06 | 0.0600 | 15.88 |

---

## Appendix C. 序列化格式比較

### C.1 WikiMovies `qwen3.5`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 0.86 | 0.35 | 9.18 | 8.67 |
| Spine-Correction | 0.86 | 0.35 | 9.18 | 8.67 |
| HRG-Proposed | 0.67 | 0.29 | 6.93 | 6.55 |

### C.2 MLPQ `qwen3.5`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 66.365 | 31.41 | 122.08 | 87.09 |
| Spine-Correction | 67.22 | 31.765 | 125.195 | 89.705 |
| HRG-Proposed | 421.34 | 218.955 | 474.385 | 271.98 |

### C.3 KQAPro `qwen3.5`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 8.87 | 4.3533 | 27.2967 | 22.76 |
| Spine-Correction | 10.3633 | 5.0133 | 31.82 | 26.45 |
| HRG-Proposed | 21.3267 | 低於 JSON | 52.61 | 低於 JSON |

註：

1. KQAPro 的 `HRG-Proposed-qwen3.5-triple` 在目前節錄中沒有完整外層 token 欄位表格，但從既有結果可確定 triple 版本的 context token 與 parse2 token 低於 json。
2. 若要做正式論文表格，建議再從完整 JSON 逐欄抽一次所有 triple 數字。

---

## Appendix D. 口試時可直接念的數字結論

1. 在 MLPQ 上，baseline `answer_recall=0.335`、`avg_ctx_tokens=6200.835`；proposed `answer_recall=0.045`、`avg_ctx_tokens=421.34`，表示 proposed 用約 `6.8%` 的 context 保留一部分答案能力。
2. 在 KQAPro 上，spine-only `answer_recall=0.0333`，proposed 提升到 `0.07`，同時 `avg_ctx_tokens` 只有 `21.3267`，遠低於 baseline 的 `3068.21`。
3. 在 WikiMovies 上，主要瓶頸不在生成，而在 retrieval/chain validity，因為 baseline 的 `retrieval_empty=94`，spine/proposed 系列則主要失敗於 `no_valid_chain`。
4. correction 單獨的幫助通常小於「correction + grammar expansion」的組合；KQAPro 是最清楚的例子：`0.0333 -> 0.0367 -> 0.0433 -> 0.07`。

---

## Appendix E. 現有 Dump 可補算結果

這一節不是根據 `benchmark_results.json`，而是直接根據目前磁碟上真的存在的 `artifacts/*/dumps/per_model/q_*.pkl` 補算。

### E.1 先講限制

截至目前為止，現有 dump 保留狀況非常稀疏：

1. `artifacts/wikimovies-wiki_entities-test/dumps/per_model/` 只找到 1 個 `q_*.pkl`
2. `artifacts/mlpq-en-zh-en-ills/dumps/per_model/` 目前沒有找到 `q_*.pkl`
3. `artifacts/kqapro-validation/dumps/per_model/` 目前只找到 3 個 `Baseline-BFS-llama3.1` 的 `q_*.pkl`

所以這一節的數字只能視為：

1. dump schema 驗證
2. partial recomputation example
3. 不能當成全量 benchmark 結論

### E.2 Partial Recompute: `kqapro-validation` `Baseline-BFS-llama3.1`（3 題）

根據現有 3 個 dump 補算得到：

| Metric | Value |
|---|---:|
| Dump Count | 3 |
| Avg Subgraph Size | 5.3333 |
| Avg Retrieval Recall | 0.3333 |
| Avg Retrieval Precision | 0.0833 |
| Avg Retrieval F1 | 0.1333 |
| Avg Context Tokens | 74.3333 |
| Avg Parse1 Tokens | 89.0 |
| Avg Correction Tokens | 0.0 |
| Avg Parse2 Tokens | 221.3333 |

這 3 題的具體情況是：

1. `q_0000.pkl`
   - `answer = Mrs. Miniver`
   - `subgraph_size = 10`
   - `retrieval_recall = 0.0`
   - `context_tokens = 143`
2. `q_0001.pkl`
   - `answer = 3`
   - `subgraph_size = 6`
   - `retrieval_recall = 1.0`
   - `context_tokens = 80`
3. `q_0002.pkl`
   - `answer = I couldn't find any information in the Knowledge Graph matching your query.`
   - `subgraph_size = 0`
   - `retrieval_recall = 0.0`
   - `context_tokens = 0`

這個 partial result 至少證明：

1. baseline dump 已經足夠重建 token / retrieval / subgraph 類指標
2. 題級 context token 與 subgraph size 可以直接補算
3. 若 dump 完整保留，全量重算 `avg_ctx_tokens / avg_subgraph_size / avg_retrieval_*` 沒有技術障礙

### E.3 Partial Recompute: `wikimovies-wiki_entities-test` `Spine-RandomExpansion-qwen3.5-triple`（1 題）

目前只找到 1 題：

| Metric | Value |
|---|---:|
| Dump Count | 1 |
| Avg Subgraph Size | 0.0 |
| Avg Retrieval Recall | 0.0 |
| Avg Retrieval Precision | 0.0 |
| Avg Retrieval F1 | 0.0 |
| Avg Context Tokens | 0.0 |
| Avg Parse1 Tokens | 766.0 |
| Avg Correction Tokens | 0.0 |
| Avg Parse2 Tokens | 0.0 |

這題的 candidate 內容也能直接從 dump 看到：

1. `source = llm`
2. `entity = Heather Sears`
3. `chain = ['starred_actors^-1']`
4. `grammar_hit = 0`
5. `grammar_same_arity_hit = 0`
6. `grammar_matched_count = 0`
7. `grammar_score = 0.0`
8. `kb_result = {'valid': False, 'step_sizes': [], 'final_size': 0, 'failed_hop': 0}`

這代表目前 HRG/spine 類 dump 雖然保留數量不足，但單一 dump 已經足夠支持：

1. candidate-level validity analysis
2. grammar hit / same-arity hit / grammar score 分析
3. chain source 分析
4. parse token 成本分析

### E.4 用現有 dump，現在就能補算哪些指標

如果下次把 dump 保留完整，基於目前 schema 已可補算：

1. `avg_subgraph_size`
2. `avg_ctx_tokens`
3. `avg_parse1_tokens`
4. `avg_correction_tokens`
5. `avg_parse2_tokens`
6. `avg_retrieval_recall / precision / f1`
7. `coverage`
8. `candidate_validity_rate`
9. `same_arity_hit_rate`
10. `grammar_score` 分布
11. `correction_salvage_rate`
12. `per-question compression diagnostics`

### E.5 用現有 dump，還不能完整補算哪些主流 benchmark 指標

光靠現在這批 sparse dump，仍然不能完整補算：

1. `Recall@k`
2. `MRR`
3. `nDCG`
4. claim-level `faithfulness`
5. `hallucination`
6. evidence correctness
7. citation correctness

原因不是完全沒有 dump，而是：

1. 目前保留下來的 dump 題數不足
2. 舊 schema 沒有把 ranked retrieval list、final context、timing、selected candidate、references 等欄位統一存好

---

## Appendix F. Dump Schema 已補強項目

為了讓下一次實驗可以補算更多 benchmark 指標，程式已經補強 dump schema。

### F.1 HRG / Spine 類 dump 現在會多存

在 [knowledgegraph_agent.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py) 中，dump 現在會額外保留：

1. `failure_stage`
2. `grammar_hit`
3. `serialization_format`
4. `references`
5. `spine_edges`
6. `expanded_edges`
7. `selected_candidate`
8. `final_context`
9. `timing`

這些欄位的目的分別是：

1. `failure_stage`: 重建 `failure_counts` 與題級失敗型態
2. `spine_edges` / `expanded_edges`: 分開分析 retrieval 主幹與擴張貢獻
3. `selected_candidate`: 重建 candidate ranking 與最終選擇邏輯
4. `final_context`: 之後可做 context-level faithfulness / claim-support 檢查
5. `references`: 不用再額外回頭對 dataset 檔做對齊
6. `timing`: 題級 latency 分析

### F.2 Baseline dump 現在也會多存

在 [baseline.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/baseline.py) 中，現在也補了：

1. `failure_stage`
2. `references`
3. `selected_depth`
4. `serialization_format`
5. `final_context`
6. `timing`

這樣 baseline 與 spine/HRG 路徑的 dump 結構就比較一致。

### F.3 下一次重跑後，最值得補算的新增指標

基於補強後 schema，下一輪完整實驗後最適合新增：

1. `correction_salvage_rate`
2. `grammar_hit_rate_per_hop`
3. `spine_vs_expansion_edge_ratio`
4. `avg_expanded_edges`
5. `selected_candidate_source_distribution`
6. `same_arity_hit_rate`
7. `per-question final_context_length`
8. `failure_stage -> token_cost` 對照

這一批指標最貼近你目前方法的研究問題，而且不需要再大改主程式。

---

## Appendix G. Dump 補算腳本

現在 repo 已新增：

[`recompute_from_dumps.py`](/home/zihui/projects/masterPaperRemake/portable_runner/recompute_from_dumps.py)

用途是直接從：

```text
artifacts/<run_tag>/dumps/per_model/**/q_*.pkl
```

重建 dump-based summary。

### G.1 使用方式

```bash
python3 recompute_from_dumps.py --run-tag kqapro-validation
python3 recompute_from_dumps.py --run-tag wikimovies-wiki_entities-test
python3 recompute_from_dumps.py --run-tag mlpq-en-zh-en-ills
```

預設輸出到：

```text
artifacts/<run_tag>/results/dump_recomputed_summary.json
artifacts/<run_tag>/results/dump_recomputed_rows.csv
```

### G.2 腳本會補算什麼

目前腳本會自動補算：

1. `dump_count`
2. `avg_subgraph_size`
3. `avg_retrieval_recall`
4. `avg_retrieval_precision`
5. `avg_retrieval_f1`
6. `avg_ctx_tokens`
7. `avg_parse1_tokens`
8. `avg_correction_tokens`
9. `avg_parse2_tokens`
10. `avg_parse_latency`
11. `avg_retrieval_latency`
12. `avg_generation_latency`
13. `avg_num_candidates`
14. `avg_num_matched_rules`
15. `coverage_from_dump`
16. `candidate_validity_rate`
17. `candidate_grammar_hit_rate`
18. `candidate_same_arity_hit_rate`
19. `correction_salvage_rate`
20. `failure_counts_from_dump`
21. `candidate_source_counts`
22. `selected_candidate_source_counts`

如果 dump 裡有 `references`，腳本還會額外補：

1. `avg_answer_recall`
2. `avg_em`
3. `avg_contains_hit`
4. `avg_hit_at_1_any`
5. `avg_answer_set_precision`
6. `avg_answer_set_recall`
7. `avg_answer_set_f1`

### G.3 目前已驗證成功的例子

我已經實際執行：

```bash
python3 recompute_from_dumps.py --run-tag kqapro-validation
```

並成功生成：

1. [dump_recomputed_summary.json](/home/zihui/projects/masterPaperRemake/portable_runner/artifacts/kqapro-validation/results/dump_recomputed_summary.json)
2. [dump_recomputed_rows.csv](/home/zihui/projects/masterPaperRemake/portable_runner/artifacts/kqapro-validation/results/dump_recomputed_rows.csv)

其中目前因為是舊 dump，會看到：

1. `failure_stage = unknown`
2. `avg_parse_latency = 0.0`
3. `avg_retrieval_latency = 0.0`
4. `avg_generation_latency = 0.0`

這不是腳本錯，而是舊 dump 當初根本沒有把這些欄位存進去。

### G.4 新 dump 與舊 dump 的差別一定要講清楚

目前要分成兩代 dump：

1. 舊 dump
   - 沒有 `failure_stage`
   - 沒有 `timing`
   - 大多沒有 `references`
   - 有些也沒有 `selected_candidate`
2. 新 dump
   - 有 `failure_stage`
   - 有 `timing`
   - 有 `references`
   - 有 `spine_edges / expanded_edges / final_context / selected_candidate`

所以：

1. 舊 dump 只能補算一部分 summary
2. 新 dump 才能真正支持完整 dump-based benchmark augmentation
