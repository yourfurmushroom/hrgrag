# COMPLETE WORKFLOW DETAILED

本文件不是只講概念，而是直接依照目前 `portable_runner` 的真實程式與 `artifacts/` 內已跑出的結果，完整說明：

1. 一次 benchmark run 是怎麼從 KB 與 dataset 走到 `grammar`、`results`、`dumps`
2. HRG grammar 是怎麼被抽出來的
3. online 階段怎麼 parse、驗證 chain、fallback、擴張 subgraph、生成答案
4. `benchmark_results.json` 每個欄位是怎麼來的
5. `all_models_outputs_wide.csv` 每一格到底存什麼
6. grammar 的配對是不是「一條一條照順序對」這件事，實際答案是什麼

---

## 0. 快速看懂版

### 0.1 一句話流程

先把 KB 抽成 HRG grammar，再讓 LLM 從問題預測 `entity + relation chain`，到 KB 驗證這條 chain 能不能走通，必要時做 correction 與 expansion，最後把取到的子圖餵給 LLM 產生答案。

### 0.2 一句話流程圖

```text
KB -> 抽 grammar -> LLM parse entity/chain -> KB 驗證 -> correction / expansion -> 組 context -> LLM answer -> 算指標
```

### 0.3 `HRG-Proposed` 快速流程

1. `Parse`：先讓 LLM 輸出 top-k `{"entity": ..., "chain": [...]}` 候選，不是只輸出一條 chain。
做法：prompt 裡會先放 relation shortlist、few-shot format、必要時再放 grammar hints，要求模型回 JSON array。

2. `Ground`：把每個 candidate 的 entity 對到 KB node，避免後面拿錯起點。
做法：先 exact / normalize lookup，再用 token-overlap fallback 補救 alias、重音、composite alias 這類差一點命中的情況。

3. `Validate`：對每條 candidate chain 做 KB walk，檢查這條路在圖上能不能真的走通。
做法：從 grounded entity 出發，逐 hop 依 relation 或 `^-1` 方向更新 frontier；只要中途 frontier 變空，這條 chain 就判成 invalid。

4. `Rerank`：對所有初始 candidates 排序，不是單看 LLM 原始信心分數。
做法：排序時會綜合 `kb_result.valid`、LLM confidence、candidate 來源、grammar hit、same-arity hit、matched rule 數量。

5. `Branch A`：如果初始 candidates 裡已經有至少一條 valid chain，流程**不做 correction**，直接往下建 subgraph。
做法：程式是先看 `any(c["kb_result"]["valid"] for c in ranked_candidates)`，只要為真就跳過 correction。

6. `Correct`：只有在**所有初始 candidates 都 invalid** 時，才會產生補救 candidates。
做法：correction pool 會合併三路來源：
1. direction flip
2. grammar fallback
3. LLM correction

7. `Branch B`：如果 correction 後還是沒有任何 valid chain，流程直接結束，狀態是 `no_valid_chain`。
做法：這時不會進 subgraph construction，也不會進 expansion，更不會進 answer generation with KG context。

8. `Spine`：只對 valid candidates 建 subgraph，而且每條 valid chain 都先取 strict spine。
做法：strict spine 只保留這條 chain 真正走到的核心邊，不會一開始就把周邊鄰居全部攤開。

9. `Expand`：只有當 candidate 已經 valid、而且方法有開 expansion 時，才在 spine 外補邊。
做法：`HRG-Proposed` 用 strict grammar expansion，只補符合 matched rules、機率門檻、per-node cap 的邊；不是所有成功題都無限制擴張。

10. `Subgraph Rank`：對每個 valid candidate 建好的 subgraph 再做一次排序，最後只選一個最好的。
做法：排序時會看 `has_edges`、same-arity、grammar hit、grammar score、spine size、expanded size、compactness，不是第一條 valid chain 自動獲勝。

11. `Serialize`：把最後選中的 subgraph 轉成 `json` 或 `triple` context。
做法：retrieval 本身不會因 `json/triple` 改變；差異只出在最後怎麼把同一個子圖餵給 answer generator。

12. `Answer`：讓 LLM 只根據這份 final context 輸出答案。
做法：prompt 明確要求只輸出 final answer，不要 reasoning，不要 `<think>`，不足就答 `I don't know`。

13. `Measure`：最後統計答案、retrieval、token、latency、failure。
做法：同時寫回 `benchmark_results.json`、`all_models_outputs_wide.csv`、每題 dump，方便後面做 error analysis。

#### `HRG-Proposed` 真正的條件分支

```text
Parse top-k candidates
-> ground entity to KB node
-> validate 每條 chain
-> 只要初始 candidates 裡已有 valid chain
   -> 不做 correction
   -> 直接拿 valid candidates 建 subgraph
-> 只有當初始 candidates 全部 invalid
   -> 才做 correction
   -> correction 後若仍無 valid chain，直接結束
-> 對 valid candidates 建 spine / expansion subgraph
-> subgraph rerank
-> 只選最佳一個 subgraph 去 answer
```

#### 這段最容易講錯的地方

1. `Correction` 不是每題都做，而是**只有全部初始 candidates 都失敗時才做**。
2. `Expansion` 不是 parse 一成功就立刻做，而是**先要有 valid chain，然後才對 valid candidate 建 subgraph 時做**。
3. `HRG-Proposed` 不是看到第一條 valid chain 就直接停，而是**會先收所有 valid candidates，再做 subgraph ranking，最後只選最好的一個**。
4. `json / triple` 不會改 retrieval 結果，只會改最後 answer generation 看到的 context 文字格式。

#### `HRG-Proposed` 一句話版本

`HRG-Proposed = entity/chain parse + grammar-aware rerank + fallback correction + strict grammar expansion + answer generation`

### 0.4 一句話消融實驗

1. `Baseline-BFS`：不用 chain、也不用 grammar，直接從 entity 做 BFS 撈子圖。
2. `Spine-Only`：只信 LLM 預測的 chain，嚴格照 chain 取 spine。
3. `Spine-Correction`：先照 chain 取 spine，失敗時再用 direction flip、grammar fallback、LLM correction 補救。
4. `Spine-GrammarExpansion`：先照 chain 取 spine，再用 grammar 命中的 rule 去擴周邊鄰邊。
5. `Spine-RandomExpansion`：先照 chain 取 spine，再隨機補一些鄰邊當對照組。
6. `Spine-FrequencyExpansion`：先照 chain 取 spine，再補高頻 relation 鄰邊當對照組。
7. `HRG-Proposed`：先照 chain 取 spine，再加 grammar-aware rerank、strict grammar expansion 和 correction。
8. `json / triple`：不是不同 retrieval，而是同一個子圖用兩種不同序列化格式餵給 LLM。

### 0.5 一句話資料集

1. `MetaQA`：乾淨的單語 atomic triples，最接近你方法原本假設，弱結果通常比較像方法問題。
   例子：`[Tom Hanks] starred_actors^-1 directed_by ?` 這類 2-hop 問題，在圖上就是很標準的 chain traversal。
2. `WikiMovies`：原始 KB 不是天然 atomic triples，還有 composite tail，所以要先做 parser 修正與 KB normalization 才能公平比較。
   例子：`The Inbetweeners 2 starred_actors James Buckley, Simon Bird, Blake Harrison, Joe Thomas` 原始上是一行字串，但對圖推理來說應該拆成 4 條 atomic triples。
3. `MLPQ`：問題語言可分開，但 KB 可以是 bilingual fused 或 monolingual sampled，所以它同時在測跨語言 grounding 與 chain executability。
   例子：英文問題可能要走 `dbpedia.org/property/team` 再接 `zh.dbpedia.org/property/capital`，也就是 question 是英文，但 relation chain 本身跨到中文 relation。
4. `KQAPro`：格式比 WikiMovies 乾淨，但 compositional reasoning 壓力最大，弱結果通常比較像 chain / reasoning 能力不足。
   例子：問題常不是單純問一跳鄰居，而是要先找中間實體，再接第二跳或第三跳條件，才會落到答案節點。

### 0.6 一句話 benchmark 項目

#### 目前 `results/benchmark_results.json` 真的有輸出的

1. `EM / Hits@1 / Hits@3 / Hits@5 / MRR / Answer-Set Precision / Recall / F1`：看答案有沒有答對、正確答案排得多前面。
2. `avg_retrieval_recall / precision / f1 / coverage`：看取到的子圖有沒有碰到答案，grammar 有沒有介入。
3. `avg_ctx_tokens / avg_subgraph_size / compression_vs_bfs_*`：看 context 與子圖有沒有真的被壓小。
4. `avg_latency / avg_parse_latency / avg_retrieval_latency / avg_generation_latency`：看時間花在哪一段。
5. `avg_parse1_tokens / avg_correction_tokens / avg_parse2_tokens`：看 token 成本花在 parse、correction 還是 answer generation。
6. `failure_counts / generation_failure_count / answerable_rate`：看 pipeline 主要死在哪一段。

#### `benchmark.py` 裡還留著，但目前主結果沒有正式輸出的

1. `bleu`：n-gram 型答案相似度，現在這批主結果沒有寫出來。
2. `contains_hit`：只要輸出字串包含任一 gold answer 就算命中，現在這批主結果也沒有寫出來。

#### 現在已補上，但要注意是現有 pipeline 上的近似版本

1. `retrieval_recall_at_k / retrieval_ndcg_at_k`：是對 ranked candidate subgraphs 的 answer-support 排名評估，不是全文檢索 document ranking。
2. `claim_faithfulness / claim_hallucination`：是把輸出答案拆成 claims 後，用 final context/subgraph 做 support 檢查的 heuristic 版本。
3. `evidence_precision / recall / f1 / citation_correctness`：只有有 gold path/evidence supervision 的資料集才會有值；目前最直接的是 `MLPQ`。

### 0.7 一句話 grammar matching

目前 grammar matching 不是逐 hop 一條一條照順序精確配對，而是先把 chain 去方向後做 relation label 的 subset match，再用 same-arity 優先過濾。

---

## 1. 本文件對應的實際 artifacts

目前 `artifacts/` 裡已存在三組主結果、一次 `MetaQA` 部分成功 run，與一份批次總結：

1. `artifacts/wikimovies-wiki_entities-test/`
2. `artifacts/mlpq-en-zh-en-ills/`
3. `artifacts/kqapro-validation/`
4. `artifacts/wikimovies-wiki_entities-test/metaqa-vanilla-test/`
5. `artifacts/_batch/run_all_summary.txt`

批次總結目前是：

```txt
metaqa FAILED
wikimovies OK
mlpq OK
kqapro OK
```

這份 `_batch/run_all_summary.txt` 是較早批次留下來的摘要；本次文件更新時，主表與分析一律以各 dataset `results/benchmark_results.json` 的最新內容為準。

這次 artifact refresh 後，四組資料集都有完整的 `llama3.1 / llama3.2 / qwen2.5` benchmark 結果。

所以下面的主表與分析，會以這一批更新後的完整結果為主，而不再沿用舊的 `qwen3.5` 局部 run。

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

1. MetaQA -> `load_metaqa_dataset`
1. WikiMovies -> `load_wikimovies_dataset`
2. MLPQ -> `load_mlpq_dataset`
3. KQAPro -> `load_normalized_jsonl_dataset`

最後 benchmark 看到的單位都一樣：

```text
(question, [gold_answer1, gold_answer2, ...])
```

### 6.1.1 各資料集在這個 repo 裡怎麼被轉成 benchmark 可用格式

這一段很重要，因為你的 benchmark 不是直接吃原始 dataset，而是先做不同程度的整理、補欄位、正規化。

#### MetaQA

資料路徑：

1. KB: `Datasets/MetaQA/kb.txt`
2. QA: `Datasets/MetaQA/<hop>/qa_*.txt` 類型檔案

在這個 repo 裡，`MetaQA` 的 QA loader 很直接：

1. 每行用 `\t` 切成 `question` 和 `answers`
2. `answers` 再用 `|`
   切成多答案列表
3. 最後變成 `(question, [answers...])`

它的好處是：

1. KB 本來就是 atomic triples：`head|relation|tail`
2. QA 配對規則很單純
3. 幾乎不需要額外資料轉換

它的限制是：

1. benchmark 端主要靠檔案路徑或外部 run 配置來決定 `1-hop / 2-hop / 3-hop`
2. loader 本身不會從單行 QA 內容裡重新推理 hop 數
3. 所以若資料路徑或 split 組織錯了，hop 標籤也可能跟內容不一致

但整體上，`MetaQA` 是目前最接近「原始資料本身就天然符合你方法假設」的資料集。

#### WikiMovies

資料路徑：

1. KB: `Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt`
2. 正規化後 KB: `Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb_normalized.txt`
3. QA: `Datasets/WikiMovies/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt`

在這個 repo 裡，`WikiMovies` 的 QA loader 會：

1. 先把每行前面的題號去掉
2. 用 `\t` 切成 `question` 和 `raw_answers`
3. `raw_answers` 用 `,` 切成多答案
4. 每個答案同時保留：
   - 原字串
   - 把 `_` 換成空格後的版本
5. 最後全部都放到 `1-hop`

這裡有幾個非常重要的限制：

1. `WikiMovies` 在這個 pipeline 裡被當成 `1-hop` 資料
2. 所以很多 expansion 方法天生就不容易展現優勢
3. 原始 KB 不是乾淨的 atomic triple 風格，而是有 `數字 + head + relation + tail` 這種特殊行格式
4. 而且某些 relation 的 tail 不是單一 entity，而是逗號串起來的 composite value

例如：

```text
The Inbetweeners 2 starred_actors James Buckley, Simon Bird, Blake Harrison, Joe Thomas
```

這種格式的問題是：

1. 它語意上像很多 triples
2. 但檔案裡實際上只是一行
3. 若不轉換成 atomic triples，`starred_actors^-1` 這種反向查詢在圖上不夠自然

所以這個 repo 才額外做了兩件事：

1. 修 parser，先找到 canonical relation token 再切 head / relation / tail
2. 新增 `normalize_wikimovies_kb.py`，把安全的多值 relation 拆成 atomic triples

也就是說，`WikiMovies` 的一大限制是：

1. 原始資料表示本身就不是最適合 graph query 的形式
2. 若不做 KB 轉換，弱結果不一定代表方法弱

#### MLPQ

資料路徑：

1. QA: `Datasets/MLPQ/datasets/Questions/<pair>/<hop>-hop/...`
2. KB: 由 `resolve_mlpq_kb_path()` 根據語言 pair 與 fusion 類型動態組出

在這個 repo 裡，`MLPQ` loader 會：

1. 只載入 `2-hop` 和 `3-hop`
2. 每行切成：
   - `question`
   - `answer_raw`
   - `path_raw`
3. `path_raw` 會用 `#` 切開
4. 第一段當成 `topic_entity`
5. 若 `inject_topic_entity=True`，會把 topic entity 額外插進 question：

```text
Topic entity: [entity]
```

這代表 `MLPQ` 在這個 repo 裡不是完全原樣 benchmark，而是做了「顯式 topic entity 注入」。

這個設計的意義是：

1. 降低 topic entity 沒被問句清楚表達時的解析難度
2. 讓比較聚焦在 relation chain 與 retrieval，而不是純 NER

但它也帶來限制：

1. 這和完全不注入 topic entity 的 setting 不完全可比
2. 若和外部論文直接比較，要先講清楚這裡有 prompt-level side information
3. 問句本身與 gold answer 的配對，已經部分依賴原始資料提供的 path 結構

另外，`MLPQ` 的答案也會同時保留：

1. normalize 後版本
2. display-form 版本

所以它的 QA 配對比 MetaQA 稍寬鬆，但也更依賴跨語言 surface-form 對齊是否穩定。

補充：目前 pipeline 也已支援把 `MLPQ` 的 KB 在兩種模式間切換：

1. `bilingual`：沿用原本的 fused bilingual KG
2. `monolingual`：改用 `Sampled_<lang>.txt` 單語 KG

這是專門用來檢查 `MLPQ` 弱結果到底有多少是來自 mixed-language KG，而不是只來自方法本身。

#### KQAPro

資料路徑：

1. QA: `Datasets/KQAPro/normalized/validation.jsonl`
2. KB: `Datasets/KQAPro/kqapro_kb_triples.tsv`

在這個 repo 裡，`KQAPro` 不是直接吃原始複雜結構，而是吃 `normalized jsonl`。

loader 會：

1. 逐行讀 JSON
2. 抽出 `question`
3. 抽出 `answers`
4. 抽出 `hop`
5. 依 hop 分桶成 `1-hop / 2-hop / 3-hop`

這代表它的限制是：

1. benchmark 相依於前處理後的 normalized 檔案品質
2. 若 normalization 階段把 hop 或答案整理錯，benchmark 不會再自己糾正
3. 它不是「原始 KQAPro 全資訊直接進 pipeline」，而是先經過資料格式標準化

但相較於 WikiMovies，它的好處是：

1. QA 配對欄位比較明確
2. hop 標記是顯式欄位
3. 資料格式風險較低

### 6.1.2 這些資料集限制對結果解讀有什麼影響

最重要的是：不同資料集弱，不代表同一種原因。

#### MetaQA 弱時，優先懷疑方法

因為：

1. triple 很乾淨
2. QA 配對很直接
3. hop 結構本來就和方法假設接近

所以 MetaQA 若弱，通常比較像：

1. chain parse 不夠好
2. grammar matching / expansion 設計不夠好
3. generation 沒有把 retrieval 用好

#### WikiMovies 弱時，不能先怪方法

因為：

1. 原始 KB 不是天然 atomic triples
2. 有 composite tail 問題
3. actor / movie 這類反向查詢常被 KB 表示方式卡住
4. 這個 repo 還必須額外做 parser 修正與 KB normalization

所以 WikiMovies 若弱，先問：

1. KB 是不是還在用原始未正規化版本
2. alias / composite tail 支援有沒有吃到
3. 失敗是 `retrieval_empty` 還是 `no_valid_chain`

#### MLPQ 弱時，先看跨語言對齊與 injected topic entity

因為：

1. 它有 cross-lingual entity surface form 壓力
2. 同時又把 topic entity 額外注入 question

所以 MLPQ 的結果要小心說：

1. 它不是純粹的「模型自己從問題找 topic entity」
2. 也不是單純 parser bug
3. 更像 multilingual grounding + chain executability 的綜合壓力測試

#### KQAPro 弱時，先看 compositional reasoning

因為：

1. 它的格式比 WikiMovies 乾淨
2. 但語義與 hop 組合更複雜

所以 KQAPro 弱比較合理的解讀是：

1. method 對複合語義與可執行 chain 的能力不足
2. 而不是簡單的資料 parser 爆掉

### 6.2 LLM 第一次 parse：提候選 entity + relation chain

`prepare_retrieval()` 先呼叫 `_parse_intent_candidates()`。

LLM 要回：

```json
[
  {"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"], "confidence": 0.82}
]
```

#### 這一步做了什麼

1. 先從 question 抽 `topic entity`
2. 先把這個 entity 嘗試對到 KB node
3. 若對得到，就先收這個 node 周邊一跳內的 outgoing / incoming relation，優先放進 relation shortlist
4. 再用 question 與 relation token 的 lexical overlap 補 shortlist
5. 若有 grammar，再把 top frequent grammar patterns 的 labels 補進 shortlist
6. 把這批 relation token 放進 prompt
7. 給 few-shot examples
8. 讓 LLM 回傳 top-k candidates

也就是現在不是單純 `question -> lexical shortlist -> parse`，而是：

```text
question
  -> topic entity 抽取
  -> KB entity grounding
  -> local neighborhood relation shortlist
  -> lexical / grammar 補充 shortlist
  -> LLM parse top-k chains
```

#### 例子

若 question 裡的 topic entity 是：

```text
[Tom Hanks]
```

而這個節點在 KB 裡的一跳鄰邊包含：

1. `starred_actors^-1`
2. `written_by^-1`
3. `directed_by^-1`

那這些 relation 會先被放進 shortlist，再由 lexical overlap 與 grammar labels 補其他候選。

這樣做的目的不是保證 parse 正確，而是避免 LLM 一開始就從整張 KB 的 relation 空間亂猜。

### 6.3 entity grounding 現在不只做 exact / normalized lookup

過去 entity lookup 比較像：

1. exact match
2. 空格 / 底線 / 大小寫 normalize 後查 `_node_index`

現在又多了一層 token-overlap fallback。

#### 具體做法

若：

1. exact node 找不到
2. alias index 也找不到

系統會再做：

1. 把 query entity normalize 成 token 集合
2. 掃描 `_node_index` 內的 alias / node key
3. 比較 token overlap
4. 若 alias 近似包含 query，或 query 近似包含 alias，給 bonus
5. 取最高分 candidate 當 fallback node

#### 例子

若模型輸出：

```text
Gregoire Colin
```

而 KB / alias 裡是：

```text
Grégoire Colin
```

或 composite alias 是：

```text
James Buckley, Simon Bird, Blake Harrison, Joe Thomas
```

這一層 fallback 會比單純 exact / underscore lookup 更容易把 query 拉回正確 node 或 composite alias。

### 6.4 relation token 會再做 fuzzy 對齊

不是 LLM 輸出什麼就直接用。

例如：

1. `directed by`
2. `directed_by`
3. `directed_by^-1`

都會再經過 `_fuzzy_match_relation()` 對齊到系統實際允許的 relation token。

### 6.5 先驗證 chain 能不能在 KB 上走通

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

### 6.6 grammar matching 實際怎麼配對

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

### 6.7 matching 後還會再做 same-arity 過濾

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

### 6.8 如果第一輪 chain 全失敗，會做 fallback correction

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

### 6.9 真正取 subgraph：先 spine，再看要不要 expansion

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
3. `use_grammar_rerank = True`
4. `use_grammar_hint = False`
5. `expansion_strict = True`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

也就是：

```text
Spine retrieval
+ strict grammar-based expansion
+ grammar-aware rerank
+ fallback correction
```

不是：

1. grammar-first parsing
2. grammar-guided BFS retrieval
3. prompt-injected grammar hints

這點口試很重要，因為名字叫 `HRG-Proposed` 很容易讓人誤會它用了所有 grammar 功能；其實這個 benchmark 裡的 proposed，是「嚴格 grammar expansion + grammar-aware rerank + correction」的組合版。

#### 各方法對照表

以目前這版 `benchmark.py` 為例，模型配置差異其實是：

1. `Baseline-BFS-<backbone>`
   - 不看 chain
   - 不看 grammar
   - 直接以 entity 做雙向 BFS
2. `Spine-Only-<backbone>-{json|triple}`
   - 用 LLM parse chain
   - 嚴格照 chain 取 spine
   - 不 correction
   - 不 expansion
3. `Spine-Correction-<backbone>-{json|triple}`
   - Spine-Only
   - 再加 fallback correction
4. `Spine-GrammarExpansion-<backbone>-{json|triple}`
   - Spine-Only
   - 再加 grammar expansion
   - 再加 grammar rerank
   - 不 correction
5. `Spine-RandomExpansion-<backbone>-{json|triple}`
   - Spine-Only
   - 再加 random expansion
6. `Spine-FrequencyExpansion-<backbone>-{json|triple}`
   - Spine-Only
   - 再加 frequency expansion
7. `HRG-Proposed-<backbone>-{json|triple}`
   - Spine-Only
   - 加 grammar expansion
   - 加 grammar rerank
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

1. `Spine-Only-<backbone>-json` 跟 `Spine-Only-<backbone>-triple`
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

### 6.10 context 序列化成 JSON 或 triples

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

### 6.11 最後才是 answer generation

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
"HRG-Proposed-llama3.1-json@kqapro-validation": { ... }
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

1. `em`
2. `hits_at_1`
3. `hits_at_3`
4. `hits_at_5`
5. `mrr`
6. `answer_set_precision`
7. `answer_set_recall`
8. `answer_set_f1`
9. `avg_latency`

是對該模型所有 dataset split 結果再做一次平均後得到的 overall。

### 7.3 指標意義

下面這些是 `benchmark.py` 真的會寫進 `benchmark_results.json` 的主欄位。

#### `em`

1. 單答案時，等於答對或答錯
2. 多答案時，要求 candidate set 與 gold set 完全相等

#### `hits_at_1 / hits_at_3 / hits_at_5`

這批更新後的 artifact 已經改成排名式答案指標。

意思是：

1. 把模型輸出的答案候選視為一個 ranked list
2. 只要 top-`k` 裡出現任一正確答案，就算命中

所以：

1. `hits_at_1` 最嚴格
2. `hits_at_3` 看前三個
3. `hits_at_5` 看前五個

如果資料集大多是一題一答案，那 `hits_at_1` 會很接近單答案 accuracy，但仍不完全等於 `EM`，因為 `EM` 會對整個答案集合做完全比對。

#### `mrr`

`MRR` 是 Mean Reciprocal Rank。

它的意思是：

1. 找到第一個正確答案出現在第幾名
2. 若在第 1 名，得分是 `1`
3. 若在第 2 名，得分是 `1/2`
4. 若在第 5 名，得分是 `1/5`
5. 全題平均後就是 `mrr`

所以它比 `hits_at_k` 更細，因為它不只看「有沒有進 top-k」，也看正確答案出現得有多前面。

#### `answer_set_precision / recall / f1`

這三個是 set-based 指標。

公式是：

```text
precision = |pred ∩ gold| / |pred|
recall    = |pred ∩ gold| / |gold|
f1        = 2PR / (P + R)
```

它們比 `EM` 更適合多答案題，因為不要求完全一致，而是允許部分命中。

#### `avg_latency`

這是單題總耗時的平均值。

它不是只算 generation，而是：

1. parse
2. retrieval
3. generation
4. 以及其中的 Python-side 控制流程

一起加總後的 end-to-end 時間。

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

這些欄位裡最容易被問細節的，下面逐一拆開。

#### `coverage`

在目前實作裡，`coverage = hit_grammar_count / total_q`。

也就是：

1. 有多少題至少 hit 到一次 grammar
2. 除以總題數

它不是：

1. grammar 正確率
2. grammar 讓答案答對的比例
3. retrieval recall

所以 `coverage` 高只能說 grammar 有介入，不能直接說方法成功。

#### `avg_ctx_tokens`

這是最後送進 answer generation prompt 的 context token 數平均值。

它反映的是：

1. retrieval 最終留下多少 evidence
2. `json` 與 `triple` 序列化哪個更省
3. compression 是否真的發生

#### `avg_parse1_tokens`

第一次 chain parse 的 token 使用量平均值。

也就是：

1. LLM 讀 question
2. 預測 entity 與 relation chain
3. 所消耗的 token

#### `avg_correction_tokens`

只有開了 correction 的方法才會非零。

它代表：

1. 初始 parse 不夠用時
2. 進入 fallback correction prompt
3. 再請 LLM 改 chain
4. 額外花掉多少 token

#### `avg_parse2_tokens`

這個命名容易誤會。

它在目前程式裡實際代表的是：

1. answer generation 那一步
2. 讀最終 context 後輸出答案
3. 花掉的 token 數

所以它比較像 `generation_tokens`，不是第二次 chain parse。

#### `avg_subgraph_size`

這是最終送去序列化前的 subgraph 邊數平均值。

它不是 node 數，也不是 frontier 大小，而是：

1. strict spine edges
2. 加上 expansion edges
3. 去重後
4. 的 edge 數量

#### `avg_retrieval_recall / precision / f1`

這組值是 retrieval 評估，但要特別注意它不是主流 IR 的 `Recall@k / MRR / nDCG`。

在你現在這份 code 裡，它更接近：

1. 最終 subgraph 是否覆蓋到 gold answer 對應節點
2. 以及 subgraph 中非答案節點的比例有多高

所以它是 answer-grounded subgraph metric，不是 ranked retrieval metric。

#### `avg_parse_latency / avg_retrieval_latency / avg_generation_latency`

這三個是把總耗時拆段後的平均值：

1. parse latency
2. retrieval latency
3. generation latency

用途是回答：

1. 到底慢在 parse 還是 retrieval
2. correction / grammar expansion 到底把時間花去哪

#### `answerable_rate`

目前程式邏輯下，它幾乎等於：

1. 模型有沒有產生非空答案字串

所以它不是嚴格的「這題可回答率」，而比較像：

1. pipeline 最後有沒有吐出 answer string

#### `generation_failure_count`

這是 answer generation 階段明確標成 failure 的題數。

例如：

1. runtime exception
2. OOM
3. 後處理判定生成失敗

#### `failure_counts`

這是最值得做 error analysis 的欄位，因為它保留了 pipeline 在哪一段掛掉。

常見類別包括：

1. `ok`
2. `retrieval_empty`
3. `no_valid_chain`
4. `no_candidates`
5. `runtime_error`
6. `oom`

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

WikiMovies `Baseline-BFS-llama3.1`：

```json
"failure_counts": {
  "ok": 100
}
```

表示 100 題裡：

1. 100 題都有正常進到 `ok`
2. 這也代表目前 refresh 後的 WikiMovies 已不再被 `retrieval_empty` 主導

### 7.6 `compression_vs_bfs_*` 怎麼算

這兩個欄位是 benchmark 後處理算的：

1. `compression_vs_bfs_ctx_ratio = 本模型 avg_ctx_tokens / 同 backbone baseline 的 avg_ctx_tokens`
2. `compression_vs_bfs_subgraph_ratio = 本模型 avg_subgraph_size / 同 backbone baseline 的 avg_subgraph_size`

所以：

1. 小於 1 代表比 baseline 更壓縮
2. 越小表示 context/subgraph 越短

### 7.7 benchmark 項目最適合怎麼分組講

如果是口試或論文 methods/experiments 章節，最實用的分法是：

#### A. 答案品質

1. `em`
2. `hits_at_1`
3. `hits_at_3`
4. `hits_at_5`
5. `mrr`
4. `answer_set_precision`
5. `answer_set_recall`
6. `answer_set_f1`

#### B. retrieval / grounding

1. `avg_retrieval_recall`
2. `avg_retrieval_precision`
3. `avg_retrieval_f1`
4. `coverage`

#### C. 成本 / 壓縮

1. `avg_ctx_tokens`
2. `avg_subgraph_size`
3. `compression_vs_bfs_ctx_ratio`
4. `compression_vs_bfs_subgraph_ratio`

#### D. 系統效率

1. `avg_latency`
2. `avg_parse_latency`
3. `avg_retrieval_latency`
4. `avg_generation_latency`
5. `avg_parse1_tokens`
6. `avg_correction_tokens`
7. `avg_parse2_tokens`

#### E. failure analysis

1. `failure_counts`
2. `generation_failure_count`
3. `answerable_rate`

這樣分類的好處是，委員比較不會把：

1. `coverage`
2. `hits_at_1 / mrr`
3. `compression ratio`

混成同一種「準確率」概念。

### 7.8 目前 benchmark 還缺什麼

這裡也要誠實寫，因為委員很可能會問。

這次更新後，benchmark 已經補上：

1. `retrieval_recall_at_1/3/5`
2. `retrieval_ndcg_at_1/3/5`
3. `claim_faithfulness`
4. `claim_hallucination`
5. `evidence_precision / recall / f1`
6. `citation_correctness`

但要注意，它們目前是建立在這個 repo 現有資料結構上的近似定義，不是外部 RAG benchmark 套件那種 fully supervised 標準版。

目前仍然缺或仍不完整的，是：

1. 全 corpus document-level `Recall@k / nDCG`
2. 更嚴格的 claim-to-evidence alignment
3. human 或 LLM judge 的語義 faithfulness
4. 跨資料集一致的 gold supporting evidence

所以目前它還是偏：

1. answer quality
2. retrieval/subgraph coverage
3. token/compression
4. pipeline failure diagnosis

而不是完全等價於主流 RAG 論文裡那種標準化 grounding benchmark。

## 7.9 消融實驗設計：每個方法到底只改了什麼

你這份 `benchmark.py` 不是隨便拼模型名稱，而是固定用一組開關組合做 ablation。

最重要的是：**所有 spine 系列都共用同一個核心 pipeline，只改少數功能開關。**

### 7.9.1 Baseline-BFS

`Baseline-BFS-*` 不走 chain parsing，不走 grammar。

它的設定是：

1. 只做 entity-centered BFS
2. `bfs_depth = 3` 作為預設上限
3. retrieval 結果直接序列化並生成答案

所以它測的是：

1. 不用 relation chain
2. 不用 grammar
3. 單純大範圍 BFS neighborhood

### 7.9.2 Spine-Only

`Spine-Only-*` 的設定是：

1. `use_grammar_rerank = False`
2. `use_grammar_expansion = False`
3. `use_fallback_correction = False`
4. `use_grammar_hint = False`

意思是：

1. 只有 parse chain
2. 只有 strict spine retrieval
3. 不做 correction
4. 不做任何 expansion
5. grammar 完全不介入

它是最乾淨的 chain-guided retrieval 基線。

### 7.9.3 Spine-Correction

`Spine-Correction-*` 只比 `Spine-Only` 多開：

1. `use_fallback_correction = True`

其他仍然是：

1. `use_grammar_rerank = False`
2. `use_grammar_expansion = False`
3. `use_grammar_hint = False`

所以它測的是：

1. 如果只補 correction
2. 不讓 grammar 介入
3. 能不能救回 invalid chain

### 7.9.4 Spine-GrammarExpansion

`Spine-GrammarExpansion-*` 的設定是：

1. `use_grammar_rerank = True`
2. `use_grammar_expansion = True`
3. `use_fallback_correction = False`
4. `use_grammar_hint = False`
5. `expansion_strict = True`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

所以它不是「只開 expansion」這麼簡單，而是：

1. grammar 已經前移到 rerank
2. retrieval 成功後再做 strict grammar expansion
3. 但不開 correction

它測的是：

1. 單靠 grammar prior
2. 不靠 correction
3. 能不能改善 chain selection 與 context 補充

### 7.9.5 Spine-RandomExpansion

`Spine-RandomExpansion-*` 的設定是：

1. `use_grammar_rerank = False`
2. `use_grammar_expansion = False`
3. `use_random_expansion = True`
4. `use_fallback_correction = False`
5. `use_grammar_hint = False`
6. `expansion_per_node_cap = 5`

它的作用是控制組：

1. 檢查是不是只要在 spine 周邊多補邊就會變好
2. 不讓 grammar 或 correction 進來

### 7.9.6 Spine-FrequencyExpansion

`Spine-FrequencyExpansion-*` 的設定是：

1. `use_grammar_rerank = False`
2. `use_grammar_expansion = False`
3. `use_frequency_expansion = True`
4. `use_fallback_correction = False`
5. `use_grammar_hint = False`
6. `expansion_per_node_cap = 5`

它測的是：

1. 若只用全域高頻 relation 擴邊
2. 不用 grammar
3. 不用 correction
4. 效果會不會已經夠好

### 7.9.7 HRG-Proposed

`HRG-Proposed-*` 的設定是：

1. `use_grammar_rerank = True`
2. `use_grammar_expansion = True`
3. `use_fallback_correction = True`
4. `use_grammar_hint = False`
5. `expansion_strict = True`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

這代表它同時測三件事：

1. grammar 前移到 rerank
2. grammar 擴張 spine 周邊 context
3. invalid chain 時用 correction 補救

所以它是目前 benchmark 裡最完整的 proposed 組合。

### 7.9.8 為什麼每個方法都有 `json` / `triple`

除了 baseline 之外，所有 spine 系列都會再拆成兩種 serialization：

1. `-json`
2. `-triple`

這不是另一種方法，而是同一 retrieval 結果的兩種 context 表示。

因為 benchmark 有 shared retrieval cache，所以：

1. `Spine-Only-<backbone>-json`
2. `Spine-Only-<backbone>-triple`

會共用同一份 prepared retrieval。

也就是說，它們的差異應該被解讀成：

1. context 表示差異
2. 不是 retrieval 差異

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
  "em": 0.0,
  "hits_at_1": 0.0,
  "hits_at_3": 0.0,
  "hits_at_5": 0.0,
  "mrr": 0.0,
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

這次 artifact 更新後，主結果已經不再是 `qwen3.5`，而是：

1. `llama3.1`
2. `llama3.2`
3. `qwen2.5`

下面主文一律先以 `llama3.1` 當主要對照 backbone，理由是：

1. 四個資料集都有完整結果
2. 它是目前整體最穩定、最有代表性的 backbone
3. `llama3.2` 太弱，比較像 capacity 下界
4. `qwen2.5` 會在個別地方另外補充，尤其是 KQAPro 的最佳壓縮結果

建議口試時固定先講這 7 個欄位：

1. `em`
2. `hits_at_1`
3. `mrr`
4. `answer_set_f1`
5. `avg_ctx_tokens`
6. `avg_subgraph_size`
7. `coverage / failure_counts`

因為這 7 個欄位能同時回答：

1. 最終答案品質
2. 正確答案排得多前面
3. context 壓縮程度
4. grammar 是否真的介入
5. pipeline 主要失敗在哪

### 9.0.0 MetaQA `llama3.1` 主要方法對照

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.4433 | 0.6800 | 0.7148 | 0.5917 | 4374.22 | 364.7467 | 0.0000 | `ok=300` |
| Spine-Only-json | 0.5233 | 0.6600 | 0.7019 | 0.6702 | 1219.5926 | 54.8889 | 0.0000 | `ok=247, no_valid_chain=50, oom=3` |
| Spine-Correction-json | 0.5667 | 0.7133 | 0.7619 | 0.7342 | 1259.4579 | 56.5690 | 0.0000 | `ok=269, no_valid_chain=28, oom=3` |
| Spine-GrammarExpansion-json | 0.4033 | 0.5667 | 0.6119 | 0.5750 | 1612.1959 | 71.0653 | 0.7698 | `ok=224, no_valid_chain=67, oom=9` |
| HRG-Proposed-json | 0.4633 | 0.6433 | 0.7002 | 0.6640 | 1984.3196 | 87.2405 | 0.8866 | `ok=258, no_valid_chain=33, oom=9` |

這組結果現在非常清楚：

1. `Spine-Correction-llama3.1-json` 是目前 MetaQA 上的最佳整體組合
2. 它同時比 baseline 更準，且 context 只有 baseline 的約 `28.8%`
3. `Spine-Only` 已經能贏 baseline，表示 chain-guided retrieval 在 MetaQA 上是成立的
4. `GrammarExpansion` 與 `HRG-Proposed` 雖然讓 `coverage` 很高，但沒有把答案表現推到最好，反而把 context 拉大

如果補看 `qwen2.5`，有一個額外重點：

1. `Spine-Correction-qwen2.5-json` 的 `em=0.53`
2. `avg_ctx_tokens=277.47`
3. 它比 `llama3.1` 更省 context，但絕對分數略低

所以 MetaQA 現在最準確的說法是：

1. chain-guided retrieval 在這組資料上確實有效
2. correction 是最穩定的增益來源
3. grammar 介入不是完全沒用，但在目前實作下沒有超過 correction-only

### 9.0.1 WikiMovies `llama3.1` 主要方法對照

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.1800 | 0.4500 | 0.4514 | 0.3085 | 23.23 | 1.30 | 0.0000 | `ok=100` |
| Spine-Only-json | 0.2800 | 0.4400 | 0.4400 | 0.3489 | 34.56 | 1.48 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-Correction-json | 0.2800 | 0.4400 | 0.4400 | 0.3489 | 34.56 | 1.48 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-GrammarExpansion-json | 0.2400 | 0.4200 | 0.4200 | 0.3129 | 217.77 | 8.78 | 0.9300 | `ok=93, no_valid_chain=7` |
| Spine-RandomExpansion-json | 0.3000 | 0.4800 | 0.4800 | 0.3775 | 127.36 | 5.09 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-FrequencyExpansion-json | 0.2900 | 0.4800 | 0.4800 | 0.3704 | 129.27 | 5.09 | 0.0000 | `ok=96, no_valid_chain=4` |
| HRG-Proposed-json | 0.2500 | 0.4300 | 0.4350 | 0.3296 | 234.28 | 9.44 | 0.9700 | `ok=97, no_valid_chain=3` |

這次 WikiMovies 和之前最大的差異是：

1. baseline 不再是幾乎全空的 `retrieval_empty`
2. normalized KB 與 parser 修正後，所有方法都能正常取到可答的子圖
3. `Spine-Only` 已經比 baseline 好
4. `RandomExpansion` 和 `FrequencyExpansion` 反而是這組裡最好的兩個擴張對照
5. `HRG-Proposed` 的 `coverage` 幾乎滿了，但 EM 沒有贏過 `Spine-Only` / `RandomExpansion`

這組最值得講的結論是：

1. WikiMovies 的資料表示問題修正後，benchmark 變得可信
2. 但在 1-hop 為主的 setting 下，grammar 介入不一定比簡單擴邊更有利
3. 這也符合前面的方法分析：1-hop 題本來就不太能充分展現 grammar expansion 的優勢

### 9.0.2 MLPQ `llama3.1` 主要方法對照

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.2750 | 0.2900 | 0.3100 | 0.3112 | 6215.66 | 418.14 | 0.0000 | `ok=200` |
| Spine-Only-json | 0.0050 | 0.0650 | 0.0909 | 0.0720 | 40.89 | 1.80 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| Spine-Correction-json | 0.0100 | 0.0700 | 0.0959 | 0.0770 | 41.52 | 1.825 | 0.0000 | `no_valid_chain=145, ok=54, no_candidates=1` |
| Spine-GrammarExpansion-json | 0.0150 | 0.0300 | 0.0375 | 0.0303 | 39.98 | 1.72 | 0.1400 | `no_valid_chain=154, ok=42, no_candidates=4` |
| Spine-RandomExpansion-json | 0.0250 | 0.0700 | 0.0909 | 0.0804 | 175.06 | 7.28 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| Spine-FrequencyExpansion-json | 0.0100 | 0.0650 | 0.0875 | 0.0706 | 177.76 | 7.28 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| HRG-Proposed-json | 0.0200 | 0.0350 | 0.0425 | 0.0353 | 143.545 | 5.665 | 0.1650 | `no_valid_chain=149, ok=47, no_candidates=4` |

MLPQ 這次結果更直接，甚至比上一版更能說明問題：

1. baseline 依然遠遠最好
2. 所有 spine 系列都被 `no_valid_chain` 壓住
3. `HRG-Proposed` 不是這組裡最強的壓縮方法
4. 在 `llama3.1` 上，`RandomExpansion` 反而比 `HRG-Proposed` 的 EM 和 Ans-F1 更高

如果把 `triple` 一起看進來，另一個值得記錄的點是：

1. `Spine-RandomExpansion-llama3.1-triple` 的 `em=0.045`
2. `avg_ctx_tokens=83.965`
3. 這是目前 MLPQ 非 baseline 方法裡最好的 EM

所以在 MLPQ 上最誠實的說法是：

1. 你的方法還沒有證明 grammar prior 比簡單擴邊更好
2. 現階段最大瓶頸仍然是 multilingual chain executability
3. 這組比較像暴露方法限制，而不是成功案例

### 9.0.3 KQAPro：目前最好的壓縮組合出現在 `qwen2.5`

這組要分成兩層看：

1. `llama3.1` 當主 backbone，方便和其他資料集對齊
2. `qwen2.5` 的 `HRG-Proposed` 是目前非 baseline 的最佳結果，必須額外點名

| Method | Backbone | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx Tokens | Avg Subgraph Size | Coverage | 主要 Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | llama3.1 | 0.1100 | 0.1167 | 0.1208 | 0.1172 | 3173.4733 | 219.8767 | 0.0000 | `ok=236, retrieval_empty=64` |
| Spine-Only-json | llama3.1 | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0733 | 0.0033 | 0.0000 | `no_valid_chain=243, no_candidates=56, ok=1` |
| Spine-Correction-json | llama3.1 | 0.0233 | 0.0233 | 0.0242 | 0.0247 | 11.8967 | 0.4633 | 0.0000 | `no_valid_chain=227, no_candidates=56, ok=17` |
| Spine-GrammarExpansion-json | llama3.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.14 | 0.0067 | 0.0033 | `no_valid_chain=254, no_candidates=45, ok=1` |
| HRG-Proposed-json | llama3.1 | 0.0233 | 0.0233 | 0.0242 | 0.0247 | 13.4233 | 0.54 | 0.0533 | `no_valid_chain=233, ok=22, no_candidates=45` |
| HRG-Proposed-triple | llama3.1 | 0.0333 | 0.0333 | 0.0342 | 0.0347 | 6.6533 | 0.54 | 0.0533 | `no_valid_chain=233, ok=22, no_candidates=45` |
| Baseline-BFS | qwen2.5 | 0.0833 | 0.1000 | 0.1053 | 0.1019 | 3226.6167 | 226.92 | 0.0000 | `ok=251, retrieval_empty=49` |
| HRG-Proposed-json | qwen2.5 | 0.0433 | 0.0500 | 0.0508 | 0.0502 | 3.98 | 0.17 | 0.0367 | `no_valid_chain=261, no_candidates=15, ok=24` |
| HRG-Proposed-triple | qwen2.5 | 0.0433 | 0.0467 | 0.0467 | 0.0456 | 1.84 | 0.17 | 0.0367 | `no_valid_chain=261, no_candidates=15, ok=24` |

KQAPro 這組現在最值得講的是：

1. baseline 還是明顯最強
2. `llama3.1` 上的 spine-only 幾乎完全失效
3. correction 有幫助，但 grammar expansion 單開幾乎沒用
4. `qwen2.5` 的 `HRG-Proposed` 是目前所有非 baseline 方法裡最好的組合
5. `qwen2.5` `HRG-Proposed-triple` 只用 `1.84` 個平均 context tokens，就拿到 `em=0.0433`

所以 KQAPro 目前最安全的說法是：

1. 你的方法確實能在極小 context 下保留一點 compositional QA 能力
2. 但絕對效果仍遠低於 baseline
3. 現階段最成功的非 baseline 組合不是 `llama3.1`，而是 `qwen2.5 + HRG-Proposed`

### 9.0.4 JSON 與 Triple 序列化差異也要講數字

這次更新後，`json` 和 `triple` 的差異比上一版更明確：

1. `triple` 多數情況下仍然更省 context token
2. 但省 token 不一定讓 EM 更高
3. `KQAPro` 是 `triple` 最划算的例子
4. `MetaQA` 則是 `json` 常常更準的例子

例子：

1. MetaQA `Spine-Correction-llama3.1`
   - json: `em=0.5667`, `avg_ctx_tokens=1259.46`
   - triple: `em=0.4933`, `avg_ctx_tokens=781.94`
2. WikiMovies `Spine-Only-llama3.1`
   - json: `em=0.28`, `avg_ctx_tokens=34.56`
   - triple: `em=0.20`, `avg_ctx_tokens=15.89`
3. MLPQ `HRG-Proposed-llama3.1`
   - json: `em=0.02`, `avg_ctx_tokens=143.545`
   - triple: `em=0.02`, `avg_ctx_tokens=72.66`
4. KQAPro `HRG-Proposed-qwen2.5`
   - json: `em=0.0433`, `avg_ctx_tokens=3.98`
   - triple: `em=0.0433`, `avg_ctx_tokens=1.84`

所以現在可以更精確地說：

1. `triple` 的核心價值是 token efficiency
2. `json` 在某些 backbone/dataset 上會保留更好的生成穩定性
3. 如果目標是極端壓縮，KQAPro 會偏向 `triple`
4. 如果目標是 MetaQA 上的最佳答案表現，目前偏向 `json`

### 9.1 MetaQA：目前最支持方法假設，但最佳組合是 Correction，不是 Proposed

這次 `MetaQA` 最重要的訊號比上一版更清楚：

1. baseline `em=0.4433`
2. `Spine-Only-llama3.1-json` 已經升到 `0.5233`
3. `Spine-Correction-llama3.1-json` 再升到 `0.5667`

這表示：

1. chain-guided retrieval 本身有效
2. correction 能穩定救回一批原本的 invalid chain
3. grammar expansion 並沒有帶來同等級提升

換句話說，在 MetaQA 這個最乾淨的資料集上，目前最強證據其實是：

1. entity + chain parsing
2. strict spine retrieval
3. correction 補救

而不是 grammar expansion 本身。

### 9.2 WikiMovies：資料修正後，結果已可用，但 grammar 不是主贏點

更新後的 WikiMovies 已經和之前完全不同：

1. baseline `em=0.18`
2. `Spine-Only-llama3.1-json` `em=0.28`
3. `Spine-RandomExpansion-llama3.1-json` `em=0.30`

這代表：

1. parser 與 normalized KB 修正後，WikiMovies benchmark 現在是可用的
2. 失敗不再主要來自資料讀壞
3. 但 grammar-based expansion 不是這組裡最強的 gain source

所以這組現在該講成：

1. dataset-side 問題已經被大幅排除
2. 但 1-hop 問題上，簡單的 random/frequency expansion 也能拿到很強結果
3. 這使得 WikiMovies 不適合當 HRG 優勢的主證據

### 9.3 MLPQ：仍然是最難、也最不利於當前方法的一組

這次更新沒有改變 MLPQ 的主結論，反而讓它更明確：

1. baseline `em=0.275`
2. 最好的非 baseline 也只有 `em=0.045`
3. 大量失敗仍然是 `no_valid_chain`

而且一個關鍵細節是：

1. `HRG-Proposed` 不是最佳非 baseline
2. `Spine-RandomExpansion-llama3.1-triple` 才是目前這組中最好的非 baseline EM

所以 MLPQ 最合理的分析是：

1. grammar prior 還不足以穩定改善 multilingual chain execution
2. 方法目前對這組資料仍然過度脆弱
3. 這組最能暴露方法限制

### 9.4 KQAPro：最佳壓縮結果存在，但絕對效果仍低

KQAPro 的結論也比上一版更清楚：

1. baseline 仍然遠勝所有 spine 系列
2. 但 `qwen2.5 + HRG-Proposed` 是目前最好的非 baseline 組合
3. `HRG-Proposed-qwen2.5-triple` 用極小 context 保住了一點答案能力

最有代表性的數字是：

1. baseline `llama3.1`: `em=0.11`, `avg_ctx_tokens=3173.47`
2. proposed `qwen2.5-triple`: `em=0.0433`, `avg_ctx_tokens=1.84`

所以這組最值得講的不是「接近 baseline」，而是：

1. 在極端壓縮下，方法還保留了非零 compositional QA 能力
2. 但如果目標是追求絕對分數，現在還遠遠不夠

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

### Q3. 為什麼 Proposed 不開 grammar_hint，但會開 grammar_rerank

因為 benchmark 設計目前想把主要效果集中在：

1. grammar expansion
2. correction
3. grammar rerank

其中 `grammar_rerank` 已經打開，因為它現在被視為較輕量、較直接的「讓 grammar 前移」做法；但 `grammar_hint` 仍然關閉，原因是如果連 prompt hint 也一起打開，委員會很難分辨：

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
3. 真正要看 trade-off，必須同時看 `em / hits_at_1 / mrr` 和 `avg_ctx_tokens / avg_subgraph_size`

這三組 artifact 目前共同指出的現象是：

1. baseline 通常答案分數較高，但 context 很大
2. spine-only 壓縮最強，但很容易掉到 `no_valid_chain`
3. HRG-proposed 在部分資料集能把 `ok` 題數拉回來一些
4. 但目前整體瓶頸仍然主要在 chain parsing 與 chain validity，而不是最後答案生成

---

## Appendix A. 口試用主表

這一節的目的不是再解釋方法，而是提供可以直接放進簡報或口試回答的數字。

### A.0 MetaQA 主表

| Method | EM | Hits@1 | Hits@3 | Hits@5 | MRR | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-llama3.1 | 0.4433 | 0.6800 | 0.7533 | 0.7567 | 0.7148 | 0.5917 | 2.95 | 4374.22 | 364.7467 | 0.0000 | `ok=300` |
| Spine-Only-llama3.1-json | 0.5233 | 0.6600 | 0.7467 | 0.7467 | 0.7019 | 0.6702 | 4.14 | 1219.5926 | 54.8889 | 0.0000 | `ok=247, no_valid_chain=50, oom=3` |
| Spine-Correction-llama3.1-json | 0.5667 | 0.7133 | 0.8133 | 0.8133 | 0.7619 | 0.7342 | 4.51 | 1259.4579 | 56.5690 | 0.0000 | `ok=269, no_valid_chain=28, oom=3` |
| Spine-GrammarExpansion-llama3.1-json | 0.4033 | 0.5667 | 0.6567 | 0.6633 | 0.6119 | 0.5750 | 4.41 | 1612.1959 | 71.0653 | 0.7698 | `ok=224, no_valid_chain=67, oom=9` |
| HRG-Proposed-llama3.1-json | 0.4633 | 0.6433 | 0.7567 | 0.7633 | 0.7002 | 0.6640 | 5.18 | 1984.3196 | 87.2405 | 0.8866 | `ok=258, no_valid_chain=33, oom=9` |

### A.1 WikiMovies 主表

| Method | EM | Hits@1 | Hits@3 | Hits@5 | MRR | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-llama3.1 | 0.1800 | 0.4500 | 0.4500 | 0.4500 | 0.4514 | 0.3085 | 2.02 | 23.23 | 1.30 | 0.0000 | `ok=100` |
| Spine-Only-llama3.1-json | 0.2800 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.3489 | 2.17 | 34.56 | 1.48 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-Correction-llama3.1-json | 0.2800 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.3489 | 2.21 | 34.56 | 1.48 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-GrammarExpansion-llama3.1-json | 0.2400 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.3129 | 3.51 | 217.77 | 8.78 | 0.9300 | `ok=93, no_valid_chain=7` |
| Spine-RandomExpansion-llama3.1-json | 0.3000 | 0.4800 | 0.4800 | 0.4800 | 0.4800 | 0.3775 | 2.18 | 127.36 | 5.09 | 0.0000 | `ok=96, no_valid_chain=4` |
| Spine-FrequencyExpansion-llama3.1-json | 0.2900 | 0.4800 | 0.4800 | 0.4800 | 0.4800 | 0.3704 | 2.19 | 129.27 | 5.09 | 0.0000 | `ok=96, no_valid_chain=4` |
| HRG-Proposed-llama3.1-json | 0.2500 | 0.4300 | 0.4400 | 0.4400 | 0.4350 | 0.3296 | 3.63 | 234.28 | 9.44 | 0.9700 | `ok=97, no_valid_chain=3` |

### A.2 MLPQ 主表

| Method | EM | Hits@1 | Hits@3 | Hits@5 | MRR | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS-llama3.1 | 0.2750 | 0.2900 | 0.3250 | 0.3350 | 0.3100 | 0.3112 | 1.42 | 6215.66 | 418.14 | 0.0000 | `ok=200` |
| Spine-Only-llama3.1-json | 0.0050 | 0.0650 | 0.1150 | 0.1250 | 0.0909 | 0.0720 | 3.80 | 40.89 | 1.80 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| Spine-Correction-llama3.1-json | 0.0100 | 0.0700 | 0.1200 | 0.1300 | 0.0959 | 0.0770 | 4.83 | 41.52 | 1.825 | 0.0000 | `no_valid_chain=145, ok=54, no_candidates=1` |
| Spine-GrammarExpansion-llama3.1-json | 0.0150 | 0.0300 | 0.0400 | 0.0500 | 0.0375 | 0.0303 | 3.71 | 39.98 | 1.72 | 0.1400 | `no_valid_chain=154, ok=42, no_candidates=4` |
| Spine-RandomExpansion-llama3.1-json | 0.0250 | 0.0700 | 0.1150 | 0.1150 | 0.0909 | 0.0804 | 3.82 | 175.06 | 7.28 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| Spine-FrequencyExpansion-llama3.1-json | 0.0100 | 0.0650 | 0.1150 | 0.1150 | 0.0875 | 0.0706 | 3.82 | 177.76 | 7.28 | 0.0000 | `no_valid_chain=146, ok=53, no_candidates=1` |
| HRG-Proposed-llama3.1-json | 0.0200 | 0.0350 | 0.0450 | 0.0550 | 0.0425 | 0.0353 | 4.98 | 143.545 | 5.665 | 0.1650 | `no_valid_chain=149, ok=47, no_candidates=4` |

### A.3 KQAPro 主表

| Method | Backbone | EM | Hits@1 | Hits@3 | Hits@5 | MRR | Ans-F1 | Avg Latency | Avg Ctx | Avg Subgraph | Coverage | Failure Counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | llama3.1 | 0.1100 | 0.1167 | 0.1233 | 0.1267 | 0.1208 | 0.1172 | 1.31 | 3173.4733 | 219.8767 | 0.0000 | `ok=236, retrieval_empty=64` |
| Spine-Only-json | llama3.1 | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 6.60 | 0.0733 | 0.0033 | 0.0000 | `no_valid_chain=243, no_candidates=56, ok=1` |
| Spine-Correction-json | llama3.1 | 0.0233 | 0.0233 | 0.0233 | 0.0267 | 0.0242 | 0.0247 | 10.01 | 11.8967 | 0.4633 | 0.0000 | `no_valid_chain=227, no_candidates=56, ok=17` |
| Spine-GrammarExpansion-json | llama3.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.56 | 0.14 | 0.0067 | 0.0033 | `no_valid_chain=254, no_candidates=45, ok=1` |
| HRG-Proposed-json | llama3.1 | 0.0233 | 0.0233 | 0.0233 | 0.0267 | 0.0242 | 0.0247 | 9.44 | 13.4233 | 0.54 | 0.0533 | `no_valid_chain=233, ok=22, no_candidates=45` |
| HRG-Proposed-json | qwen2.5 | 0.0433 | 0.0500 | 0.0500 | 0.0533 | 0.0508 | 0.0502 | 2.76 | 3.98 | 0.17 | 0.0367 | `no_valid_chain=261, no_candidates=15, ok=24` |
| HRG-Proposed-triple | qwen2.5 | 0.0433 | 0.0467 | 0.0467 | 0.0467 | 0.0467 | 0.0456 | 0.01 | 1.84 | 0.17 | 0.0367 | `no_valid_chain=261, no_candidates=15, ok=24` |

---

## Appendix B. Hop 級別數據

### B.0 MetaQA `Baseline-BFS-llama3.1`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 1-hop | 0.72 | 0.88 | 0.8900 | 0.8352 | 1.57 |
| 2-hop | 0.54 | 0.68 | 0.7158 | 0.6967 | 1.06 |
| 3-hop | 0.07 | 0.48 | 0.5385 | 0.2431 | 6.22 |

### B.0.1 MetaQA `Spine-Correction-llama3.1-json`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 1-hop | 0.92 | 0.96 | 0.9650 | 0.9570 | 2.34 |
| 2-hop | 0.61 | 0.69 | 0.7750 | 0.8125 | 4.55 |
| 3-hop | 0.17 | 0.49 | 0.5458 | 0.4330 | 6.65 |

### B.1 MLPQ `Baseline-BFS-llama3.1`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 2-hop | 0.29 | 0.32 | 0.3475 | 0.3457 | 0.21 |
| 3-hop | 0.26 | 0.26 | 0.2725 | 0.2767 | 2.63 |

### B.2 MLPQ `HRG-Proposed-llama3.1-json`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 2-hop | 0.00 | 0.00 | 0.0000 | 0.0000 | 4.85 |
| 3-hop | 0.04 | 0.07 | 0.0850 | 0.0706 | 5.11 |

### B.3 KQAPro `Baseline-BFS-llama3.1`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 1-hop | 0.07 | 0.07 | 0.0750 | 0.0709 | 0.55 |
| 2-hop | 0.09 | 0.10 | 0.1075 | 0.1057 | 1.30 |
| 3-hop | 0.17 | 0.18 | 0.1800 | 0.1750 | 2.09 |

### B.4 KQAPro `HRG-Proposed-qwen2.5-json`

| Hop | EM | Hits@1 | MRR | Ans-F1 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| 1-hop | 0.08 | 0.09 | 0.0900 | 0.0814 | 0.01 |
| 2-hop | 0.03 | 0.03 | 0.0300 | 0.0300 | 0.01 |
| 3-hop | 0.02 | 0.03 | 0.0325 | 0.0391 | 8.24 |

---

## Appendix C. 序列化格式比較

### C.0 MetaQA `llama3.1`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 1219.5926 | 763.1678 | 1425.0034 | 991.1275 |
| Spine-Correction | 1259.4579 | 781.9430 | 1482.2660 | 1034.4832 |
| HRG-Proposed | 1984.3196 | 2394.2500 | 2218.3162 | 2936.6689 |

### C.1 WikiMovies `llama3.1`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 34.56 | 15.89 | 236.59 | 405.53 |
| Spine-Correction | 34.56 | 15.89 | 236.59 | 405.53 |
| HRG-Proposed | 234.28 | 116.02 | 442.97 | 781.14 |

### C.2 MLPQ `llama3.1`

| Method | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---:|---:|---:|---:|
| Spine-Only | 40.89 | 18.34 | 109.94 | 83.94 |
| Spine-Correction | 41.52 | 18.655 | 111.72 | 85.54 |
| HRG-Proposed | 143.545 | 72.66 | 215.75 | 170.13 |

### C.3 KQAPro

| Method | Backbone | JSON Ctx | Triple Ctx | JSON Parse2 | Triple Parse2 |
|---|---|---:|---:|---:|---:|
| Spine-Only | llama3.1 | 0.0733 | 0.03 | 0.77 | 0.7367 |
| Spine-Correction | llama3.1 | 11.8967 | 6.0967 | 24.67 | 20.7167 |
| HRG-Proposed | qwen2.5 | 3.98 | 1.84 | 21.8033 | 19.57 |

---

## Appendix D. 口試時可直接念的數字結論

1. 在 MetaQA 上，`Spine-Correction-llama3.1-json` 的 `EM=0.5667`、`Hits@1=0.7133`、`MRR=0.7619`，高於 baseline 的 `0.4433 / 0.68 / 0.7148`；同時 `avg_ctx_tokens=1259.46`，只有 baseline `4374.22` 的約 `28.8%`。
2. 在 WikiMovies 上，parser 與 normalized KB 修正後，`Spine-RandomExpansion-llama3.1-json` 的 `EM=0.30` 已高於 baseline `0.18`；表示這組資料現在已可用，但最佳 gain 不是來自 grammar。
3. 在 MLPQ 上，baseline `EM=0.275`、`avg_ctx_tokens=6215.66`；`HRG-Proposed-llama3.1-json` 只有 `EM=0.02`、`avg_ctx_tokens=143.545`，說明方法保住了極強壓縮，但答案品質仍大幅落後 baseline。
4. 在 KQAPro 上，最佳非 baseline 是 `HRG-Proposed-qwen2.5-triple`：`EM=0.0433`、`Hits@1=0.0467`、`avg_ctx_tokens=1.84`；相比 baseline `llama3.1` 的 `EM=0.11`、`avg_ctx_tokens=3173.47`，它展示的是極端壓縮下的非零能力，而不是接近 baseline。
5. 這次更新後，`hits_at_1 / hits_at_3 / hits_at_5 / mrr` 已經成為主結果指標，因此目前的 artifact 更像 ranking-style answer evaluation，而不再是先前那組以 `answer_recall` 為主的報表。

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

1. 全 corpus document-level `Recall@k`
2. 全 corpus document-level `nDCG`
3. 嚴格版 claim-level `faithfulness`
4. 嚴格版 `hallucination`
5. 跨資料集一致的 gold-evidence `evidence correctness`
6. 跨資料集一致的 `citation correctness`

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

---

## Appendix H. 全模型結果總表

這一節補上四個資料集所有 backbone 的方法結果，避免主文只看 `llama3.1` 時遺漏其他模型。

為了控制表格長度，這裡統一只保留：

1. `EM`
2. `Hits@1`
3. `MRR`
4. `Ans-F1`
5. `Avg Ctx`
6. `Coverage`

若要看完整 `Hits@3 / Hits@5 / Avg Subgraph / Failure Counts`，仍以各 dataset 的 `results/benchmark_results.json` 為準。

### H.1 MetaQA

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-BFS-llama3.1 | 0.4433 | 0.6800 | 0.7148 | 0.5917 | 4374.22 | 0.0000 |
| Baseline-BFS-llama3.2 | 0.0767 | 0.2533 | 0.3019 | 0.2291 | 4374.22 | 0.0000 |
| Baseline-BFS-qwen2.5 | 0.4033 | 0.6400 | 0.6590 | 0.5410 | 4376.19 | 0.0000 |
| Spine-Only-llama3.1-json | 0.5233 | 0.6600 | 0.7019 | 0.6702 | 1219.5926 | 0.0000 |
| Spine-Only-llama3.1-triple | 0.4400 | 0.6533 | 0.6836 | 0.6052 | 763.1678 | 0.0000 |
| Spine-Only-llama3.2-json | 0.0033 | 0.0133 | 0.0156 | 0.0101 | 2894.5754 | 0.0000 |
| Spine-Only-llama3.2-triple | 0.0033 | 0.0067 | 0.0090 | 0.0051 | 1907.9126 | 0.0000 |
| Spine-Only-qwen2.5-json | 0.4567 | 0.5567 | 0.5708 | 0.5339 | 257.50 | 0.0000 |
| Spine-Only-qwen2.5-triple | 0.3967 | 0.5467 | 0.5683 | 0.5167 | 113.9867 | 0.0000 |
| Spine-Correction-llama3.1-json | 0.5667 | 0.7133 | 0.7619 | 0.7342 | 1259.4579 | 0.0000 |
| Spine-Correction-llama3.1-triple | 0.4933 | 0.7100 | 0.7452 | 0.6690 | 781.9430 | 0.0000 |
| Spine-Correction-llama3.2-json | 0.0100 | 0.0200 | 0.0250 | 0.0200 | 2966.4912 | 0.0000 |
| Spine-Correction-llama3.2-triple | 0.0100 | 0.0133 | 0.0157 | 0.0118 | 2931.1789 | 0.0000 |
| Spine-Correction-qwen2.5-json | 0.5300 | 0.6367 | 0.6508 | 0.6096 | 277.4733 | 0.0000 |
| Spine-Correction-qwen2.5-triple | 0.4633 | 0.6267 | 0.6483 | 0.5906 | 123.1133 | 0.0000 |
| Spine-GrammarExpansion-llama3.1-json | 0.4033 | 0.5667 | 0.6119 | 0.5750 | 1612.1959 | 0.7698 |
| Spine-GrammarExpansion-llama3.1-triple | 0.3500 | 0.5500 | 0.5939 | 0.5279 | 2227.2128 | 0.7736 |
| Spine-GrammarExpansion-llama3.2-json | 0.0033 | 0.0133 | 0.0182 | 0.0132 | 3033.9489 | 0.0584 |
| Spine-GrammarExpansion-llama3.2-triple | 0.0067 | 0.0133 | 0.0217 | 0.0193 | 9411.2340 | 0.0780 |
| Spine-GrammarExpansion-qwen2.5-json | 0.3900 | 0.5300 | 0.5594 | 0.5056 | 599.8467 | 0.7033 |
| Spine-GrammarExpansion-qwen2.5-triple | 0.3433 | 0.4967 | 0.5294 | 0.4745 | 270.92 | 0.7033 |
| Spine-RandomExpansion-llama3.1-json | 0.4733 | 0.6333 | 0.6781 | 0.6400 | 1822.5172 | 0.0000 |
| Spine-RandomExpansion-llama3.1-triple | 0.4033 | 0.5967 | 0.6355 | 0.5810 | 1965.9459 | 0.0000 |
| Spine-RandomExpansion-llama3.2-json | 0.0033 | 0.0067 | 0.0097 | 0.0081 | 4236.4093 | 0.0000 |
| Spine-RandomExpansion-llama3.2-triple | 0.0000 | 0.0033 | 0.0090 | 0.0050 | 5177.5649 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-json | 0.4100 | 0.5500 | 0.5783 | 0.5218 | 1058.8967 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-triple | 0.3567 | 0.5100 | 0.5361 | 0.4811 | 470.2967 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-json | 0.4633 | 0.6333 | 0.6750 | 0.6328 | 1836.80 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-triple | 0.3933 | 0.5833 | 0.6255 | 0.5735 | 2002.8581 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-json | 0.0000 | 0.0033 | 0.0056 | 0.0033 | 4292.7544 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-triple | 0.0000 | 0.0067 | 0.0128 | 0.0057 | 6694.7587 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-json | 0.4133 | 0.5500 | 0.5794 | 0.5209 | 743.2709 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-triple | 0.3567 | 0.5167 | 0.5417 | 0.4814 | 479.9433 | 0.0000 |
| HRG-Proposed-llama3.1-json | 0.4633 | 0.6433 | 0.7002 | 0.6640 | 1984.3196 | 0.8866 |
| HRG-Proposed-llama3.1-triple | 0.4133 | 0.6400 | 0.6839 | 0.6110 | 2394.25 | 0.8885 |
| HRG-Proposed-llama3.2-json | 0.0133 | 0.0367 | 0.0752 | 0.0702 | 3131.8571 | 0.3480 |
| HRG-Proposed-llama3.2-triple | 0.0133 | 0.0500 | 0.0731 | 0.0532 | 9483.3203 | 0.3594 |
| HRG-Proposed-qwen2.5-json | 0.4500 | 0.6233 | 0.6528 | 0.5865 | 704.86 | 0.8067 |
| HRG-Proposed-qwen2.5-triple | 0.3933 | 0.5900 | 0.6228 | 0.5527 | 317.4433 | 0.8067 |

### H.2 WikiMovies

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-BFS-llama3.1 | 0.1800 | 0.4500 | 0.4514 | 0.3085 | 23.23 | 0.0000 |
| Baseline-BFS-llama3.2 | 0.0100 | 0.0200 | 0.0733 | 0.0736 | 15.02 | 0.0000 |
| Baseline-BFS-qwen2.5 | 0.1600 | 0.4000 | 0.4000 | 0.2714 | 23.24 | 0.0000 |
| Spine-Only-llama3.1-json | 0.2800 | 0.4400 | 0.4400 | 0.3489 | 34.56 | 0.0000 |
| Spine-Only-llama3.1-triple | 0.2000 | 0.4700 | 0.4818 | 0.3193 | 15.89 | 0.0000 |
| Spine-Only-llama3.2-json | 0.0000 | 0.0000 | 0.0100 | 0.0113 | 1.4040 | 0.0000 |
| Spine-Only-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2641.33 | 0.0000 |
| Spine-Only-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2812 | 25.13 | 0.0000 |
| Spine-Only-qwen2.5-triple | 0.0800 | 0.2600 | 0.2600 | 0.1683 | 11.94 | 0.0000 |
| Spine-Correction-llama3.1-json | 0.2800 | 0.4400 | 0.4400 | 0.3489 | 34.56 | 0.0000 |
| Spine-Correction-llama3.1-triple | 0.2000 | 0.4700 | 0.4818 | 0.3193 | 15.89 | 0.0000 |
| Spine-Correction-llama3.2-json | 0.0000 | 0.0000 | 0.0100 | 0.0113 | 1.4040 | 0.0000 |
| Spine-Correction-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2641.33 | 0.0000 |
| Spine-Correction-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2812 | 25.13 | 0.0000 |
| Spine-Correction-qwen2.5-triple | 0.0800 | 0.2600 | 0.2600 | 0.1683 | 11.94 | 0.0000 |
| Spine-GrammarExpansion-llama3.1-json | 0.2400 | 0.4200 | 0.4200 | 0.3129 | 217.77 | 0.9300 |
| Spine-GrammarExpansion-llama3.1-triple | 0.1200 | 0.4200 | 0.4250 | 0.2600 | 107.76 | 0.9300 |
| Spine-GrammarExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0067 | 0.0090 | 18.79 | 0.0800 |
| Spine-GrammarExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0050 | 0.0040 | 9.52 | 0.0800 |
| Spine-GrammarExpansion-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2795 | 25.67 | 0.9500 |
| Spine-GrammarExpansion-qwen2.5-triple | 0.0800 | 0.2500 | 0.2500 | 0.1617 | 12.23 | 0.9500 |
| Spine-RandomExpansion-llama3.1-json | 0.3000 | 0.4800 | 0.4800 | 0.3775 | 127.36 | 0.0000 |
| Spine-RandomExpansion-llama3.1-triple | 0.2100 | 0.4600 | 0.4718 | 0.3098 | 63.51 | 0.0000 |
| Spine-RandomExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0033 | 0.0040 | 5.6869 | 0.0000 |
| Spine-RandomExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7677 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2812 | 25.64 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-triple | 0.0800 | 0.2600 | 0.2600 | 0.1683 | 12.20 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-json | 0.2900 | 0.4800 | 0.4800 | 0.3704 | 129.27 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-triple | 0.1800 | 0.4500 | 0.4668 | 0.2942 | 65.39 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0033 | 0.0040 | 5.6869 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0025 | 0.0033 | 2.7677 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2812 | 25.64 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-triple | 0.0800 | 0.2600 | 0.2600 | 0.1683 | 12.20 | 0.0000 |
| HRG-Proposed-llama3.1-json | 0.2500 | 0.4300 | 0.4350 | 0.3296 | 234.28 | 0.9700 |
| HRG-Proposed-llama3.1-triple | 0.1300 | 0.4300 | 0.4400 | 0.2767 | 116.02 | 0.9700 |
| HRG-Proposed-llama3.2-json | 0.0000 | 0.0000 | 0.0492 | 0.0548 | 22.52 | 0.2100 |
| HRG-Proposed-llama3.2-triple | 0.0000 | 0.0000 | 0.0183 | 0.0170 | 11.23 | 0.2100 |
| HRG-Proposed-qwen2.5-json | 0.2200 | 0.3700 | 0.3700 | 0.2795 | 25.67 | 0.9500 |
| HRG-Proposed-qwen2.5-triple | 0.0800 | 0.2500 | 0.2500 | 0.1617 | 12.23 | 0.9500 |

### H.3 MLPQ

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-BFS-llama3.1 | 0.2750 | 0.2900 | 0.3100 | 0.3112 | 6215.66 | 0.0000 |
| Baseline-BFS-llama3.2 | 0.0400 | 0.0400 | 0.0425 | 0.0433 | 6215.66 | 0.0000 |
| Baseline-BFS-qwen2.5 | 0.2050 | 0.2700 | 0.3010 | 0.2892 | 6215.66 | 0.0000 |
| Spine-Only-llama3.1-json | 0.0050 | 0.0650 | 0.0909 | 0.0720 | 40.89 | 0.0000 |
| Spine-Only-llama3.1-triple | 0.0200 | 0.0550 | 0.0775 | 0.0722 | 18.34 | 0.0000 |
| Spine-Only-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.035 | 0.0000 |
| Spine-Only-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.08 | 0.0000 |
| Spine-Only-qwen2.5-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.87 | 0.0000 |
| Spine-Only-qwen2.5-triple | 0.0000 | 0.0000 | 0.0025 | 0.0034 | 2.03 | 0.0000 |
| Spine-Correction-llama3.1-json | 0.0100 | 0.0700 | 0.0959 | 0.0770 | 41.52 | 0.0000 |
| Spine-Correction-llama3.1-triple | 0.0250 | 0.0600 | 0.0825 | 0.0771 | 18.655 | 0.0000 |
| Spine-Correction-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.94 | 0.0000 |
| Spine-Correction-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.425 | 0.0000 |
| Spine-Correction-qwen2.5-json | 0.0000 | 0.0100 | 0.0100 | 0.0066 | 6.66 | 0.0000 |
| Spine-Correction-qwen2.5-triple | 0.0050 | 0.0050 | 0.0100 | 0.0117 | 2.82 | 0.0000 |
| Spine-GrammarExpansion-llama3.1-json | 0.0150 | 0.0300 | 0.0375 | 0.0303 | 39.98 | 0.1400 |
| Spine-GrammarExpansion-llama3.1-triple | 0.0150 | 0.0250 | 0.0312 | 0.0282 | 18.415 | 0.1400 |
| Spine-GrammarExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.295 | 0.0300 |
| Spine-GrammarExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.03 | 0.0300 |
| Spine-GrammarExpansion-qwen2.5-json | 0.0050 | 0.0050 | 0.0050 | 0.0050 | 10.815 | 0.0450 |
| Spine-GrammarExpansion-qwen2.5-triple | 0.0050 | 0.0050 | 0.0075 | 0.0083 | 5.035 | 0.0450 |
| Spine-RandomExpansion-llama3.1-json | 0.0250 | 0.0700 | 0.0909 | 0.0804 | 175.06 | 0.0000 |
| Spine-RandomExpansion-llama3.1-triple | 0.0450 | 0.0650 | 0.0784 | 0.0740 | 83.965 | 0.0000 |
| Spine-RandomExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0010 | 0.0016 | 33.245 | 0.0000 |
| Spine-RandomExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0013 | 0.0020 | 15.235 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.495 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-triple | 0.0000 | 0.0000 | 0.0025 | 0.0025 | 7.145 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-json | 0.0100 | 0.0650 | 0.0875 | 0.0706 | 177.76 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-triple | 0.0300 | 0.0650 | 0.0804 | 0.0754 | 86.69 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-json | 0.0050 | 0.0050 | 0.0060 | 0.0066 | 33.515 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0013 | 0.0020 | 15.51 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.65 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-triple | 0.0000 | 0.0000 | 0.0025 | 0.0025 | 7.31 | 0.0000 |
| HRG-Proposed-llama3.1-json | 0.0200 | 0.0350 | 0.0425 | 0.0353 | 143.545 | 0.1650 |
| HRG-Proposed-llama3.1-triple | 0.0200 | 0.0300 | 0.0362 | 0.0331 | 72.66 | 0.1650 |
| HRG-Proposed-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.595 | 0.0650 |
| HRG-Proposed-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.755 | 0.0650 |
| HRG-Proposed-qwen2.5-json | 0.0150 | 0.0150 | 0.0150 | 0.0150 | 14.69 | 0.0500 |
| HRG-Proposed-qwen2.5-triple | 0.0150 | 0.0150 | 0.0175 | 0.0184 | 6.715 | 0.0500 |

### H.4 KQAPro

| Method | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-BFS-llama3.1 | 0.1100 | 0.1167 | 0.1208 | 0.1172 | 3173.4733 | 0.0000 |
| Baseline-BFS-llama3.2 | 0.0500 | 0.0567 | 0.0600 | 0.0589 | 1661.96 | 0.0000 |
| Baseline-BFS-qwen2.5 | 0.0833 | 0.1000 | 0.1053 | 0.1019 | 3226.6167 | 0.0000 |
| Spine-Only-llama3.1-json | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0733 | 0.0000 |
| Spine-Only-llama3.1-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.03 | 0.0000 |
| Spine-Only-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0767 | 0.0000 |
| Spine-Only-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0367 | 0.0000 |
| Spine-Only-qwen2.5-json | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.2633 | 0.0000 |
| Spine-Only-qwen2.5-triple | 0.0000 | 0.0033 | 0.0033 | 0.0022 | 0.1367 | 0.0000 |
| Spine-Correction-llama3.1-json | 0.0233 | 0.0233 | 0.0242 | 0.0247 | 11.8967 | 0.0000 |
| Spine-Correction-llama3.1-triple | 0.0233 | 0.0233 | 0.0242 | 0.0247 | 6.0967 | 0.0000 |
| Spine-Correction-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0767 | 0.0000 |
| Spine-Correction-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0367 | 0.0000 |
| Spine-Correction-qwen2.5-json | 0.0167 | 0.0167 | 0.0175 | 0.0180 | 2.1133 | 0.0000 |
| Spine-Correction-qwen2.5-triple | 0.0133 | 0.0167 | 0.0167 | 0.0156 | 0.9667 | 0.0000 |
| Spine-GrammarExpansion-llama3.1-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.14 | 0.0033 |
| Spine-GrammarExpansion-llama3.1-triple | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0567 | 0.0033 |
| Spine-GrammarExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| Spine-GrammarExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| Spine-GrammarExpansion-qwen2.5-json | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.4733 | 0.0000 |
| Spine-GrammarExpansion-qwen2.5-triple | 0.0000 | 0.0033 | 0.0033 | 0.0022 | 0.2667 | 0.0000 |
| Spine-RandomExpansion-llama3.1-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.14 | 0.0000 |
| Spine-RandomExpansion-llama3.1-triple | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0567 | 0.0000 |
| Spine-RandomExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0767 | 0.0000 |
| Spine-RandomExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0367 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-json | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.8967 | 0.0000 |
| Spine-RandomExpansion-qwen2.5-triple | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.4367 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.14 | 0.0000 |
| Spine-FrequencyExpansion-llama3.1-triple | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0567 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0767 | 0.0000 |
| Spine-FrequencyExpansion-llama3.2-triple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0367 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-json | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.8967 | 0.0000 |
| Spine-FrequencyExpansion-qwen2.5-triple | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.4367 | 0.0000 |
| HRG-Proposed-llama3.1-json | 0.0233 | 0.0233 | 0.0242 | 0.0247 | 13.4233 | 0.0533 |
| HRG-Proposed-llama3.1-triple | 0.0333 | 0.0333 | 0.0342 | 0.0347 | 6.6533 | 0.0533 |
| HRG-Proposed-llama3.2-json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5667 | 0.0033 |
| HRG-Proposed-llama3.2-triple | 0.0000 | 0.0000 | 0.0011 | 0.0013 | 0.2733 | 0.0033 |
| HRG-Proposed-qwen2.5-json | 0.0433 | 0.0500 | 0.0508 | 0.0502 | 3.98 | 0.0367 |
| HRG-Proposed-qwen2.5-triple | 0.0433 | 0.0467 | 0.0467 | 0.0456 | 1.84 | 0.0367 |
