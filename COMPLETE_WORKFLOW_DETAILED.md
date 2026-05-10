# 完整流程與消融實驗說明

## 1. 文件定位

本文件整理目前 `portable_runner` 的完整流程，目的是提供一份可直接用於：

1. 論文章節整理
2. 口試簡報底稿
3. 方法章與實驗章的統一說明

本文涵蓋的範圍包括：

1. 環境與資料集準備
2. offline HRG grammar 生成
3. online LLM inference 與 KG retrieval
4. benchmark 執行流程
5. 消融實驗設計
6. 評估指標
7. 輸出 artifact 結構
8. 現階段可安全宣稱的範圍與限制

目前整套系統最適合的定位是：

**training-free、model-agnostic 的 KGQA framework**

也就是說，本研究不額外 fine-tune LLM，不另外訓練新的 end-to-end QA 模型，而是：

1. 在 offline 階段從知識圖譜中抽取結構先驗
2. 在 online 階段讓 LLM 負責語意解析與答案生成
3. 讓 HRG 作為 retrieval constraint 與 context compression 機制

本研究關心的重點不只有最終答案正確率，也包括：

1. 答案品質
2. context 大小
3. token 消耗
4. 執行時間

因此，本研究更適合被表述為一個「效果與成本 trade-off」的研究，而不是單純追求最高準確率的模型競賽。

## 2. 整體執行入口

### 2.1 Docker 版本

若使用 Docker，目前的主要入口為：

```bash
cd portable_runner
docker compose up --build
```

這條命令現在預設會跑目前可完整閉環的 4 個資料集：

1. `metaqa`
2. `wikimovies`
3. `mlpq`
4. `kqapro`

預設批次清單定義在 [run_all_benchmarks.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_all_benchmarks.sh)。

### 2.2 非 Docker 版本

若機器沒有 Docker，現在可直接使用本機一鍵入口：

```bash
cd portable_runner
bash run_local_all.sh
```

這個腳本會自動完成：

1. 準備 Python 環境
2. 安裝依賴
3. 下載與整理資料集
4. 生成 config
5. 依序執行 benchmark

### 2.3 核心腳本

目前流程由以下腳本共同組成：

1. [setup_env.sh](/home/zihui/projects/masterPaperRemake/portable_runner/setup_env.sh)
2. [download_datasets.sh](/home/zihui/projects/masterPaperRemake/portable_runner/download_datasets.sh)
3. [download_datasets.py](/home/zihui/projects/masterPaperRemake/portable_runner/download_datasets.py)
4. [generate_configs.py](/home/zihui/projects/masterPaperRemake/portable_runner/generate_configs.py)
5. [resolve_kb.py](/home/zihui/projects/masterPaperRemake/portable_runner/resolve_kb.py)
6. [run_pipeline.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_pipeline.sh)
7. [auto_benchmark.sh](/home/zihui/projects/masterPaperRemake/portable_runner/auto_benchmark.sh)
8. [run_all_benchmarks.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_all_benchmarks.sh)
9. [run_local_all.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_local_all.sh)

## 3. 資料集準備與目前覆蓋範圍

### 3.1 目前可完整執行的資料集

目前 `portable_runner` 已接好的資料集為：

1. `MetaQA`
2. `WikiMovies`
3. `MLPQ`
4. `KQAPro`

這些資料的固定位置都在 `portable_runner/Datasets/` 底下。

### 3.2 各資料集現況

#### MetaQA

目前固定路徑為：

1. dataset root: `Datasets/MetaQA`
2. KB: `Datasets/MetaQA/kb.txt`
3. relation list: `Datasets/MetaQA/relations.json`
4. benchmark split: `test`
5. variant: `vanilla`

這是目前與整套 benchmark 最自然、最乾淨對齊的一組資料。

#### WikiMovies

目前固定路徑為：

1. dataset root: `Datasets/WikiMovies`
2. KB: `Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt`
3. 問題檔：`Datasets/WikiMovies/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt`
4. benchmark split: `test`

WikiMovies 也能直接對接現在的 line-based triple parser。

#### MLPQ

目前固定路徑為：

1. dataset root: `Datasets/MLPQ`
2. KB: `Datasets/MLPQ/datasets/KGs/fusion_bilingual_KGs/ILLs_fusion/merged_ILLs_KG_en_zh.txt`
3. 預設問題設定：`en-zh / en / ills`

MLPQ 的特性是跨語知識圖譜問答，因此答案字串正規化比 MetaQA 與 WikiMovies 更敏感。

#### KQAPro

KQAPro 目前不是原生直接接上，而是透過 compatibility adaptation 接進來。原因是原始 snapshot 中的：

1. `hf_snapshot/kb.json`

無法直接被目前 benchmark 的 line-based triple loader 讀取，因此 portable workflow 會：

1. 下載 snapshot
2. 將 `kb.json` 轉成 `Datasets/KQAPro/kqapro_kb_triples.tsv`
3. 使用 `validation` 而不是 `test`

之所以改用 `validation`，是因為官方 `test.json` 沒有 gold answer，無法用來做完整 benchmark 評估。

因此，在論文或口試中，KQAPro 較穩妥的說法是：

**adapted compatibility setting**

而不是完全原生 benchmark setting。

### 3.3 尚未完整閉環的資料集

目前 portable workflow 也能下載：

1. `WQSP`
2. `CWQ`
3. `Mintaka`

但這三組目前只有 question / answer 資料，沒有隨附 benchmark 可直接使用的 KG triples，因此沒有被放進預設可執行清單。

## 4. Offline 階段：HRG Grammar 生成

### 4.1 Offline 階段的角色

offline 階段負責把原始知識圖譜轉成一組可重用的結構規則。這些 grammar 不是直接用來回答問題，而是用來：

1. 提供結構先驗
2. 捕捉 relation 共現模式
3. 作為 online 檢索限制與局部擴張依據

因此，offline 的產物不是答案模型，而是一個「圖結構樣板庫」。

### 4.2 主要程式與資料流

核心實作在：

1. [portable_runner/hrg_grammar/hrg_extract.py](/home/zihui/projects/masterPaperRemake/portable_runner/hrg_grammar/hrg_extract.py)

整個 offline pipeline 的實際步驟是：

1. 讀入 triple file
2. 將 triple 正規化為 `(head, relation, tail)`
3. 建立帶標籤有向多重圖 `MultiDiGraph`
4. 忽略方向與 relation label，轉成 undirected skeleton
5. 從 skeleton degree 分布中選種子，做多次 BFS node-induced sampling
6. 對每個 sample graph 建立 chordal-like 結構
7. 從 triangulation 過程中提取 maximal clique candidates
8. 建 clique tree
9. 將 clique tree 做二元化與冗餘 leaf 修剪
10. 把每個 bag 轉成一條 HRG rule
11. 合併重複 rule
12. 輸出 grammar JSON / TXT

下面把每一步拆開說。

### 4.3 第一步：讀入 triple 並正規化 token

在 `hrg_extract.py` 中，最先做的是：

1. `_normalize_token(token)`
2. `_parse_triple_line(line)`
3. `load_labeled_kb_graph(kb_path, max_triples)`

這裡的工作包含：

1. 去除 `<...>` 包裝
2. 處理 Freebase URI 前綴
3. 處理 literal、語言標記、datatype 標記
4. 支援多種輸入格式
   - `head|relation|tail`
   - `head\trelation\ttail`
   - N-Triples
   - 一般空白分隔三元組
   - WikiMovies 特殊 line format

這一步完成後，會把資料統一成：

1. 節點 `head`
2. 關係 `relation`
3. 節點 `tail`

### 4.4 第二步：建立 labeled directed MultiDiGraph

`load_labeled_kb_graph()` 會把所有 triples 放進 `networkx.MultiDiGraph`：

1. `u = head`
2. `v = tail`
3. `key = relation`
4. `rel = relation`

之所以使用 `MultiDiGraph` 而不是一般 `DiGraph`，是因為：

1. 同一對節點之間可能有多種 relation
2. graph 是有方向的
3. 後續要保留 relation label

到這一步為止，圖仍然是語意完整的有向 labeled KG。

### 4.5 第三步：轉成 undirected skeleton

接著會透過 `to_undirected_skeleton(G)` 建立 `Graph H`：

1. 節點沿用原 KG 節點
2. 若 `(u, v, rel)` 存在，就在 `H` 中加入無向邊 `{u, v}`
3. relation label 與方向此時暫時被忽略

這一步的目的不是要丟掉語意，而是：

1. 方便後續做圖分解
2. 方便計算節點 degree
3. 方便做 MCS 與 triangulation

也就是說，offline grammar induction 的結構分解階段，是先在 skeleton 上做，而不是直接在 labeled directed graph 上做 clique 分析。

### 4.6 第四步：robust BFS node-induced sampling

這一步對應：

1. `pick_seed_avoid_hubs(...)`
2. `bfs_node_induced_sample_capped(...)`
3. `k_bfs_samples_robust(...)`

#### 為什麼要 sampling

如果直接在整張圖上做 clique 與 grammar induction，很容易發生：

1. clique explosion
2. 高 degree hub 主導結構
3. bag 太大、規則太大

所以目前做法是先抽樣幾個局部子圖，再從這些子圖學高頻結構樣板。

#### 具體怎麼抽

1. 先在 skeleton 上計算所有節點 degree
2. 用 `SEED_DEGREE_QUANTILE = 0.80` 過濾 seed 候選
   - 只優先從 degree 不高於 80% 分位數的節點中挑 seed
3. 對每個 seed 做 BFS
4. 每個節點展開鄰居時：
   - 若鄰居太多，先 shuffle
   - 再截斷到 `BFS_MAX_BRANCH = 30`
5. 直到收集到大約 `S_SAMPLE_SIZE = 500` 個節點
6. 最後取 induced subgraph

#### 為什麼說是 node-induced sample

因為最後保留的是：

1. 被 BFS 納入的節點集合
2. 原圖中這些節點之間的所有邊

所以不是只保留 BFS tree 本身，而是保留這批節點在原圖中的 induced subgraph。

#### 現行預設值

1. `K_SAMPLES = 4`
2. `S_SAMPLE_SIZE = 500`
3. `SEED_DEGREE_QUANTILE = 0.80`
4. `BFS_MAX_BRANCH = 30`
5. `RANDOM_SEED = 0`

這些值較適合表述為：

1. 結構膨脹控制參數
2. 計算預算控制參數
3. 可重現的保守預設值

### 4.7 第五步：MCS ordering

每個 sample graph 會先忽略 edge direction 與 relation label，進入：

`mcs_ordering(G)`

這裡使用的是 `Maximum Cardinality Search, MCS`。

具體做法：

1. 每個未編號節點一開始 label = 0
2. 每一輪選出 label 最大的未編號節點 `v`
3. 把 `v` 加入 ordering
4. 對 `v` 的所有未編號鄰居，其 label 加 1
5. 重複直到所有節點都被編號

產物是一個 elimination ordering。

這個 ordering 之後會拿來做 triangulation。

### 4.8 第六步：triangulation，讓圖變成 chordal-like 結構

對應函式：

`triangulate_from_order(G, order)`

這一步是你特別提到的重點。實際上做的是：

1. 依照 ordering 逐一處理節點 `v`
2. 找出在 ordering 中排在 `v` 後面的鄰居 `later`
3. 將 `later` 裡的節點彼此補邊，使其形成 clique
4. 這些補上的邊就是 fill-in edges

這一步可視為：

1. 利用 elimination ordering 對圖做 triangulation
2. 讓圖趨近 chordal graph 的結構

#### 為什麼要 triangulation

因為：

1. chordal / triangulated graph 比較容易抽 clique 結構
2. elimination ordering 可以自然提供 clique 候選
3. 不需要直接呼叫昂貴的 `nx.find_cliques()`

### 4.9 第七步：從 triangulation 過程收集 maximal clique candidates

在 `triangulate_from_order()` 中，每當處理一個節點 `v`，就會收集：

1. `clique_candidate = {v} ∪ later_neighbors`

然後：

1. 將所有 candidate 去重
2. 依大小排序
3. 過濾掉被其他更大 clique 完全包含的 candidate

最後保留 maximal clique 候選。

這裡要強調：

1. 不是直接暴力列舉所有 clique
2. 而是利用 elimination 過程得到 clique 候選，再保留 maximal clique

### 4.10 第八步：建立 clique tree

對應函式：

`build_clique_tree_from_cliques(cliques)`

具體做法：

1. 每個 clique 視為一個 node
2. 任兩個 clique 若有交集，則在它們之間連邊
3. 邊權重設為交集大小
4. 在這張 clique graph 上取 maximum spanning tree

最後得到的就是 clique tree。

這個 clique tree 的意義是：

1. 用一棵樹組織所有局部 clique bag
2. 讓不同 bag 之間共享的節點能被明確表示
3. 方便後續將 bag 轉成具有 external nodes 的 graph grammar rule

### 4.11 第九步：binarize clique tree

對應函式：

`binarize_clique_tree(T, bags, root=0)`

若 clique tree 中某個 bag 有超過兩個 child，程式會：

1. 複製一個 bag clone
2. 把原先多出來的 children 移到 clone 底下
3. 直到每個節點最多只保留兩個 child

這一步的目的不是改變 bag 本身內容，而是：

1. 控制 tree branching factor
2. 避免後續 grammar RHS 結構過於複雜

### 4.12 第十步：prune redundant leaves

對應函式：

`prune_leaf_no_internal(T, bags, root=0)`

若某個 leaf bag：

1. 不是 root
2. 只有一個 parent
3. 且該 bag 完全被 parent bag 包含

那就刪掉它。

這一步等於把沒有帶來新 internal node 的冗餘葉節點去掉，避免生成資訊增益太低的規則。

### 4.13 第十一步：把 clique-tree bag 轉成 HRG rule

這一步對應：

1. `Nonterminal`
2. `RHS`
3. `Rule`
4. `extract_hrg_rules_labeled(...)`

這是整個 offline 最關鍵的一步。

#### 先建 node-to-bags index

程式會先建立：

1. `node2bags`
   - 紀錄每個原圖節點出現在哪些 bag
2. `bag_sizes`
   - 每個 bag 的大小

#### edge-to-bag assignment

接著會把原始有向 labeled edge `(u, v, rel)` 指派給某個 bag：

1. 找同時包含 `u` 與 `v` 的 bag 候選
2. 從中選最小的 bag
3. 把這條 edge 指派到該 bag

這一步很重要，因為它決定：

1. 某條 terminal edge 最後屬於哪一條規則
2. 避免同一條 edge 被重複放進多個 bag

#### 對每個 bag 建 rule

對每個 clique-tree 節點 `eta`：

1. 找 parent `p`
2. 取目前 bag = `bags[eta]`

##### lhs 怎麼決定

如果 `eta` 是 root：

1. `lhs = Nonterminal("S", 0)`

如果 `eta` 不是 root：

1. 找 `bag ∩ parent_bag`
2. 這個交集就是 external nodes
3. `lhs = Nonterminal("N", len(intersection))`

所以：

1. `lhs.name`
   - `S` 代表起始規則
   - `N` 代表一般中間規則
2. `lhs.rank`
   - 代表此規則有多少 external attachment points

##### external / internal nodes 怎麼分

1. 與 parent bag 的交集 = external nodes
2. bag 中其餘節點 = internal nodes

然後程式會建立：

`verts = external nodes + internal nodes`

再用這個順序賦予局部索引。

##### rhs.terminals 怎麼來

對所有被指派到這個 bag 的 edge `(u, v, rel)`：

1. 若 `u` 與 `v` 都在目前 bag 的局部節點集合中
2. 就將它轉成 `(idx[u], rel, idx[v])`
3. 放進 `rhs.terminals`

所以 `rhs.terminals` 是：

1. bag 內明確出現的 relation edge
2. 用局部節點索引來表示

##### rhs.nonterms 怎麼來

對每個 child bag：

1. 取 `bag ∩ child_bag`
2. 把交集對應成目前 bag 裡的局部索引 tuple
3. 形成一個 nonterminal attachment

程式中用的是：

1. `Nonterminal("N", len(att))`
2. `att = tuple(idx[x] for x in att_nodes)`

所以 `rhs.nonterms` 表示：

1. 此規則右側還接了一個未展開的子結構
2. 且那個子結構透過哪些 attachment points 與目前規則連接

### 4.14 第十二步：合併重複規則

對應函式：

`merge_duplicate_rules(rules)`

它會以：

1. `lhs.name`
2. `lhs.rank`
3. canonicalized `rhs`

作為規則 signature，將完全相同的規則合併，累積 `count`。

因此最終 grammar 不是一堆 bag 的原始逐條轉錄，而是一個：

1. 去重後的 rule set
2. 每條 rule 帶有出現次數

### 4.15 Grammar JSON 的具體例子

一條 HRG 規則的 JSON 形式如下：

```json
{
  "lhs": {"name": "S", "rank": 0},
  "rhs": {
    "terminals": [
      {"a": 231, "rel": "directed_by", "b": 37},
      {"a": 231, "rel": "has_tags", "b": 110},
      {"a": 231, "rel": "written_by", "b": 37}
    ],
    "nonterms": [
      {"name": "N", "rank": 288, "att": [0, 1, 2, 3]}
    ]
  },
  "count": 1
}
```

欄位含義：

1. `lhs`
   - 這條規則左側非終結符
2. `rhs.terminals`
   - bag 內顯式出現的 terminal relation edges
3. `rhs.nonterms`
   - 尚未展開的子結構插槽
4. `count`
   - 相同規則在樣本中累積出現的次數

### 4.16 規則中 `a`、`b`、`att` 的數字到底是什麼

這些數字不是原始 KG entity id，而是：

1. 目前這條 rule 內部的局部節點編號

也就是說，規則描述的是：

1. 局部結構中的連接方式
2. relation 共現模式

而不是原始實體名稱本身。

### 4.17 Offline 最終輸出位置

每個 run tag 會輸出：

1. `artifacts/<run_tag>/grammar/hrg_grammar.json`
2. `artifacts/<run_tag>/grammar/hrg_grammar.txt`

其中：

1. JSON 主要供程式載入
2. TXT 主要供人閱讀與檢查 rule 內容

## 5. Online 階段：LLM Inference 與 Retrieval

### 5.1 主要程式位置

online KGQA pipeline 的核心在：

1. [portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py)
2. [portable_runner/LLM_inference_benchmark/benchmark.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/benchmark.py)
3. [portable_runner/LLM_inference_benchmark/baseline.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/baseline.py)
4. [portable_runner/LLM_inference_benchmark/LLM_stratiges.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/LLM_stratiges.py)

### 5.2 Agent 初始化時做了什麼

`KnowledgeGraphAgent.__init__()` 啟動時會：

1. 載入指定 LLM strategy
2. 載入 KB adjacency
3. 載入 relation list
4. 建立 `allowed_rel_set`
5. 建立 `allowed_rel_tokens`
   - 包含正向 relation 與 `relation^-1`
6. 建立 entity alias / node index
7. 若 grammar 檔存在，載入 `HRGMatcher`

所以 agent 一開始就同時擁有：

1. LLM semantic parsing 能力
2. KG traversal 能力
3. grammar matching 能力

### 5.3 Prompt 格式化

HF 模型現在會優先使用：

`tokenizer.apply_chat_template(...)`

而不是手動字串拼 prompt。這對：

1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`

都比較重要，因為這類模型在原生 instruct template 下通常更穩。

### 5.4 Candidate Parsing：模型怎麼先抓 entity 與 relation chain

這一步主要在：

`_parse_intent_candidates(user_prompt, num_candidates)`

#### 先建立 relation prompt 候選集合

程式會先呼叫：

`_select_relation_prompt_candidates(user_prompt, limit=64)`

這一步會：

1. 先看問題文字和 relation token 的 lexical overlap
2. 若 grammar 存在，也把 top grammar rules 裡常出現的 relation 加入 shortlist
3. 產生一個 prompt 內給 LLM 參考的候選 relation 集合

也就是說，LLM 並不是對整個 relation universe 完全無限制亂猜，而是：

1. 優先從 shortlisted relation candidate set 中選

#### Prompt 要求模型輸出什麼

Developer prompt 會要求 LLM 輸出 JSON array，每個物件包含：

1. `entity`
2. `chain`
3. `confidence`

例如：

```json
[{"entity": "Tom Hanks", "chain": ["starred_actors^-1", "directed_by"], "confidence": 0.82}]
```

其中：

1. `entity`
   - 題目的 topic entity
2. `chain`
   - relation traversal 序列
3. `confidence`
   - LLM 自估可信度

#### 解析與清理

模型輸出後，程式會：

1. 用 balanced JSON segment parser 抽出 JSON array 或 JSON object
2. 逐個 relation 做 `_fuzzy_match_relation()`
3. 把不合法的 relation token 丟掉
4. 做 candidate 去重 `_dedup_candidates()`

若完全解析失敗，還會退化成：

1. 只保留 entity
2. chain 為空

### 5.5 Candidate 在 KG 上怎麼檢查可執行性

這一步在：

`_check_chain_validity(entity, chain)`

流程如下：

1. 先把 entity 對到 KB 節點
2. 將 frontier 初始化為 `{start_entity}`
3. 對 chain 每一 hop：
   - 取目前 frontier 中每個節點
   - 用 `_neighbors_for_token(ent, rel_token)` 找下一跳
   - 若 relation 是 `rel^-1`，則走 backward
   - 否則走 forward
4. 逐 hop 更新 frontier
5. 若某 hop 後 frontier 變空，該 chain 視為失敗

回傳會包括：

1. `valid`
2. `step_sizes`
3. `final_size`
4. `failed_hop`

所以這一步不是語意判斷，而是：

**該 chain 在 KG 上到底走不走得通**

### 5.6 Grammar 怎麼跟 chain 做 matching

這一步由 `HRGMatcher` 負責。

#### Grammar 載入時會做什麼

每條 rule 載入後，會先抽出：

1. `rhs.terminals` 中所有 relation label

並存成：

`rule["_cached_labels"]`

#### chain matching 怎麼做

當給定一條 chain，例如：

1. `["written_by", "written_by^-1", "has_genre"]`

程式會先把它轉成 bare relation：

1. 去掉 `^-1`

也就是：

1. `["written_by", "written_by", "has_genre"]`

然後與每條 grammar rule 的 `_cached_labels` 做比對：

1. 若 bare chain 是 rule label set 的子集，視為 match

接著依：

1. `probability`
2. 若沒有 probability，則 `count`

做排序。

### 5.7 Same-arity rule selection

接著在：

`_select_matched_rules(chain, top_k, require_same_arity=True)`

會再做一層過濾：

1. 優先只保留 `lhs.rank == len(chain)` 的 rule

這是為了讓 grammar 更像：

1. 和目前 chain 長度一致的局部結構約束

而不是使用一個很大、很泛的高頻鄰域 prior。

### 5.8 Candidate ranking：怎麼對多個 chain 排名

這是 online 階段最關鍵的部分之一，對應：

`_score_candidate(...)`

排序是用一個可解釋的 lexicographic tuple，不是單純線性分數。

排序優先順序大致是：

1. `valid`
   - 能不能在 KG 上完整執行
2. `same_arity_hit`
   - 是否有同 hop 長度的 grammar 支持
3. `grammar_hit`
   - 是否有 grammar 支持
4. `grammar_score`
   - grammar match 強度
5. `grammar_matched_count`
   - 匹配到多少 rule
6. `failure_progress`
   - 若失敗，是在哪一 hop 才失敗
7. `step_survival`
   - 每一 hop 的 frontier 存活程度
8. `final_size`
   - 最後 frontier 大小
9. `source_priority`
   - 來源是原始 LLM、correction、flip 或 grammar fallback
10. `llm_confidence`
11. `llm_prior`

這意味著：

1. 可執行性優先於表面語意合理性
2. grammar compatibility 是次要但重要的 tie-breaker
3. LLM 原始排序與 confidence 只放在比較後面

### 5.9 Correction：當第一輪候選全失敗時怎麼補救

如果第一輪所有 candidate 都失敗，且 `use_fallback_correction=True`，程式會啟動 correction。

目前 correction pool 來源有三種：

#### 1. Direction flip

`_make_direction_flip_candidates(entity, chain)`

做法是：

1. 對 chain 中每一 hop 各自翻轉方向一次
2. 例如 `written_by` ↔ `written_by^-1`

#### 2. Grammar fallback

`_make_grammar_fallback_candidates(entity, chain, top_k=5)`

做法是：

1. 先找與原始 chain 相關的 grammar rules
2. 若找不到，就退化用 top frequent rules
3. 從 rule 的 label set 中重建一條較合理的 relation sequence
4. 若某個 bare relation 在原 chain 裡出現過，就盡量保留原來方向

#### 3. LLM correction

`_correct_candidates_with_llm(...)`

做法是：

1. 把失敗候選與題目重新丟給 LLM
2. 請它修正 entity / chain
3. 再把結果併回 correction pool

最後 correction pool 也會做去重，然後和原始 candidate 一起重排。

### 5.10 選到 valid chain 後，怎麼建 strict spine

當有 candidate 被判定為 valid，就會建立對應 subgraph。

如果不是 grammar-guided retrieval 模式，主路徑是：

`_find_subgraph_multi_hop_kb_strict(start_entity, relation_chain)`

做法是：

1. frontier 初始化為起點 entity
2. 對每一 hop relation：
   - 只沿著這個 relation 走
   - 收集符合該 relation 的邊
   - 更新 next frontier
3. 所有被走過的 edge 形成 strict spine

這個 subgraph 只有 chain 指定的邊，不含自由補邊。

### 5.11 Grammar-guided expansion：怎麼從 spine 擴張

對應：

`_expand_subgraph_by_grammar(spine_edges, matched_rules, chain)`

流程如下：

1. 從 matched rules 中選 top-k rules
2. 若 `expansion_strict=True`
   - 只保留 `lhs.rank == len(chain)` 的 rules
   - 且其 `probability/count >= expansion_min_prob`
3. 將這些 rule 的 `_cached_labels` 合併成 `allowed_rels`
4. 取所有出現在 spine 裡的節點 `spine_nodes`
5. 對每個 spine node：
   - 檢查 `kb_out` 中 relation 是否屬於 `allowed_rels`
   - 檢查 `kb_in` 中 relation 是否屬於 `allowed_rels`
6. 只把符合 allowed relation 的局部邊補進來
7. 每個 spine node 最多加入 `expansion_per_node_cap` 條邊

所以 grammar expansion 不是重新跑 BFS，而是：

**以 spine 為中心，只補 grammar 覺得合理的局部關係邊**

### 5.12 Random expansion：對照組怎麼做

對應：

`_expand_subgraph_random(spine_edges, chain)`

流程是：

1. 收集 spine_nodes
2. 對每個 node 蒐集其所有 local candidate edges
3. 用固定 seed 與 chain/states 生成可重現亂數
4. shuffle candidate edges
5. 每個 node 只取前 `expansion_per_node_cap` 條

這個設計的重點是：

1. 和 grammar expansion 用同樣的 local edge budget
2. 但完全不依賴 HRG

### 5.13 Subgraph ranking：不只比 chain，還要比建出來的 subgraph

當 valid candidate 都建好 subgraph 之後，還會做第二層排序：

`_score_subgraph_candidate(chain_row, subgraph, has_references)`

這一步考慮的因素包括：

1. `has_edges`
2. `same_arity_hit`
3. `grammar_hit`
4. `grammar_score`
5. `spine_size`
6. `expanded_size`
7. `compactness`
8. 原始 chain ranking key

其中：

1. 更小的 subgraph 在支持品質相近時會比較被偏好
2. 這使得方法不只是追求多抓邊，而是追求較精簡且有支撐力的 context

### 5.14 Serialization：最後送進 LLM 前怎麼表示 subgraph

對應：

`_serialize_edges(edges)`

目前兩種格式為：

1. `triples`
   - 格式如：`head relation tail.`
2. `json`
   - 直接把 edge list 轉成 JSON string

### 5.15 Answer generation：最後一輪 LLM 怎麼回答

對應：

`_generate_rag_response(user_prompt, context_json)`

Developer prompt 會要求模型：

1. 只能依據 Context 回答
2. 不要輸出 reasoning
3. 不要輸出 markdown
4. 多答案時用 ` | ` 連接
5. 若 context 不足，輸出 `I don't know`

也就是說，online 階段其實有兩輪 LLM：

1. 第一輪：semantic parsing / candidate chain generation
2. 第二輪：讀取最終子圖並輸出答案

### 5.16 Token 與 latency 是怎麼記錄的

目前 agent 會分別記錄：

1. `parse1_tokens`
   - 第一輪 candidate parsing
2. `correction_tokens`
   - correction prompt 開銷
3. `parse2_tokens`
   - 最終 answer generation
4. `context_tokens`
   - 子圖序列化後的 context 長度估計

時間則分成：

1. `parse_latency`
2. `retrieval_latency`
3. `generation_latency`

這樣之後不只能比較答案，也能比較整體成本。

## 6. 目前 Benchmark 矩陣

### 6.1 Backbone 模型

目前 benchmark 透過一個模型陣列自動展開所有固定方法。模型定義在：

1. [portable_runner/LLM_inference_benchmark/benchmark.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/benchmark.py)

目前 backbone 包括：

1. `meta-llama/Llama-3.1-8B-Instruct`
2. `Qwen/Qwen2.5-7B-Instruct`

之後若要新增模型，只需要在 `MODEL_BACKBONES` 陣列中新增一個項目，不需要重寫整個 benchmark matrix。

### 6.2 Baseline

目前非 HRG 基線為：

1. `Baseline-BFS-{backbone}`

它使用 BFS 類 retrieval，並由 benchmark 提供 oracle hop depth。

### 6.3 核心消融實驗

目前每個 backbone 都會自動展開以下方法：

1. `Spine-Only-{backbone}-{json,triple}`
2. `Spine-Correction-{backbone}-{json,triple}`
3. `Spine-GrammarExpansion-{backbone}-{json,triple}`
4. `Spine-RandomExpansion-{backbone}-{json,triple}`
5. `Spine-FrequencyExpansion-{backbone}-{json,triple}`
6. `HRG-Proposed-{backbone}-{json,triple}`

也就是說，現在可以同時比較：

1. 方法消融
2. serialization 消融
3. backbone 消融

### 6.4 各方法的意義

#### Spine-Only

啟用：

1. LLM semantic parsing
2. strict spine retrieval

關閉：

1. correction
2. grammar expansion
3. random expansion

用途：

1. 作為最乾淨的 chain-first 下界基線

#### Spine-Correction

啟用：

1. LLM semantic parsing
2. correction
3. strict spine retrieval

關閉：

1. grammar expansion
2. random expansion

用途：

1. 單獨量化 correction 的貢獻

#### Spine-GrammarExpansion

啟用：

1. LLM semantic parsing
2. strict spine retrieval
3. grammar-guided expansion

關閉：

1. correction
2. random expansion

用途：

1. 單獨量化 grammar-based local context completion 的貢獻

#### Spine-RandomExpansion

啟用：

1. LLM semantic parsing
2. strict spine retrieval
3. random local expansion

關閉：

1. correction
2. grammar-guided expansion

用途：

1. 驗證 HRG 是否真的優於 naive expansion

#### Spine-FrequencyExpansion

啟用：

1. LLM semantic parsing
2. strict spine retrieval
3. relation-frequency-guided expansion

關閉：

1. correction
2. grammar-guided expansion
3. random expansion

用途：

1. 回答「HRG 的效果是否只是因為用了 relation 頻率資訊」
2. 建立比 random expansion 更強的對照組

具體做法是：

1. 先統計整張 KG 中各 relation 的全域出現次數
2. 對每個 spine node 收集其可擴張邊
3. 依 relation 全域頻率由高到低排序
4. 在與 HRG expansion 相同的 `expansion_per_node_cap` 預算下保留前幾條

因此這個方法不是看結構樣板，只看 relation popularity。

#### HRG-Proposed

啟用：

1. LLM semantic parsing
2. correction
3. strict spine retrieval
4. grammar-guided expansion

用途：

1. 評估完整主方法

### 6.5 為什麼 Random Expansion 很重要

如果沒有 `Spine-RandomExpansion`，委員或審稿人很容易質疑：

1. 你變好是不是只是因為多加了邊
2. 並不是 HRG 本身有幫助

因此 Random Expansion 的存在，是為了建立一個更乾淨的對照：

1. 同樣的 expansion budget
2. 沒有 HRG 指導
3. 可以直接比較「多加邊」和「有結構地加邊」的差異

而 `Spine-FrequencyExpansion` 更進一步回答：

1. 若只用 relation 頻率，而不用 HRG 結構樣板，效果是否已經足夠
2. 若 HRG 仍優於 frequency expansion，才更能說明 HRG 捕捉到的不只是統計頻率，而是更有用的局部圖結構

## 7. 評估指標

### 7.1 主要答案指標

目前 benchmark 主要輸出：

1. `em`
2. `answer_set_f1`
3. `answer_recall`
4. `hit_at_1_any`

含義如下：

#### EM

1. 對單答案題：等同 exact match
2. 對多答案題：要求整組答案完全一致

#### Answer-Set F1

1. 用於多答案題最重要
2. 會同時懲罰漏答與亂答

#### Answer Recall

這是目前原本 `acc` 重新定義後的版本：

1. 對單答案題：0 或 1
2. 對多答案題：`命中的 gold 答案數 / gold 答案總數`

例如 gold 有 4 個答案，模型答對其中 3 個，則：

`answer_recall = 0.75`

#### Hit@1-any

1. 只要至少命中一個 gold answer 就算 1
2. 屬於較寬鬆的輔助指標

### 7.2 輔助答案指標

benchmark 另外還保留：

1. `answer_set_precision`
2. `answer_set_recall`
3. `bleu`
4. `contains_hit`

其中：

1. `BLEU` 仍保留，但不建議作為 KGQA 主指標
2. `contains_hit` 保留在 JSON 裡以維持相容性，但已不再作為主表重點

### 7.3 答案切分規則

目前多答案切分採較保守設計，只切：

1. `|`
2. `;`
3. 換行

不再預設使用逗號切分，以避免像：

1. `Washington, D.C.`
2. 某些複雜片名

這種合法答案被錯切。

### 7.4 效率指標

目前 benchmark 同時記錄：

1. `avg_latency`
2. `avg_parse_latency`
3. `avg_retrieval_latency`
4. `avg_generation_latency`
5. `avg_ctx_tokens`
6. `avg_parse1_tokens`
7. `avg_correction_tokens`
8. `avg_parse2_tokens`
9. `avg_subgraph_size`
10. `answerable_rate`
11. `generation_failure_count`

### 7.5 Retrieval 指標

在有 reference answer 的資料集上，也會統計：

1. `avg_retrieval_recall`
2. `avg_retrieval_precision`
3. `avg_retrieval_f1`

這些指標的重要性在於，它能幫助區分：

1. retrieval 本身做得好不好
2. 最後答案生成是否因 LLM 回答階段而失真

## 8. Artifact 與輸出目錄

### 8.1 資料與輸出規則

目前 portable workflow 已經正規化為：

1. 資料集固定放在 `portable_runner/Datasets/...`
2. 所有產出固定放在 `portable_runner/artifacts/<run_tag>/...`

### 8.2 每個 run tag 的輸出

對每個 run tag，目前會產生：

1. grammar
   - `artifacts/<run_tag>/grammar/`
2. benchmark result JSON
   - `artifacts/<run_tag>/results/benchmark_results.json`
3. detail CSV
   - `artifacts/<run_tag>/results/all_models_outputs_wide.csv`
4. dump / shared retrieval caches
   - `artifacts/<run_tag>/dumps/`

### 8.3 Batch summary

整批執行後的摘要會寫在：

1. `artifacts/_batch/run_all_summary.txt`

## 9. 一次完整 run 的實際流程

對單一資料集而言，完整流程大致如下：

1. 建立或啟用 Python 環境
2. 安裝依賴
3. 下載並正規化資料集
4. 固定資料路徑與 KB 路徑
5. 根據 dataset / split 生成 `run_tag`
6. 依據該 dataset 的 KB 生成 grammar JSON / TXT
7. 載入 benchmark split
8. 初始化 baseline 與 KG agents
9. 對每個問題：
   - 產生 candidate chain
   - 檢查可執行性
   - 必要時做 correction
   - 建 strict spine
   - 視設定做 grammar 或 random expansion
   - 序列化成 `json` 或 `triples`
   - 用 LLM 生成答案
   - 計算 answer / retrieval / efficiency 指標
10. 聚合成 dataset-level 與 model-level 報告
11. 輸出結果 JSON、CSV、dump、grammar

## 10. 目前比較安全的詮釋邊界

### 10.1 可以合理主張的部分

目前整套流程較能支撐以下主張：

1. HRG 可作為 offline structural prior 用於 KGQA retrieval
2. chain-first retrieval 搭配局部 expansion，有機會在答案品質與 context 成本之間取得較好平衡
3. grammar-guided expansion 可以和 random expansion 在相同 budget 下做直接比較
4. `json` 與 `triples` 可作為 LLM-facing serialization choice 進行效率與效果比較

### 10.2 需要保守表述的部分

以下幾點應保守處理：

1. `KQAPro` 是 adapted setting，不是完全原生 benchmark
2. 跨資料集做單一總平均不夠穩，dataset-wise analysis 更安全
3. `json vs triple` 比較的是 LLM serialization 與 prompt 表達效果，不是理論上 KG formalism 的絕對優劣
4. 一些 offline 參數本質上是 heuristic / budget-control，而不是數學上最優值

### 10.3 容易看起來像 magic number 的參數

目前最容易被質疑為 heuristic 的值包括：

1. `K_SAMPLES = 4`
2. `S_SAMPLE_SIZE = 500`
3. `SEED_DEGREE_QUANTILE = 0.80`
4. `BFS_MAX_BRANCH = 30`
5. `topk_expansion_rules = 1`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

對這些值，較安全的說法是：

1. 它們是 computation-control / context-budget control 設定
2. 它們是保守預設
3. 它們不是理論最優解的宣稱

## 11. 輔助分析：HRG 必要性與 Sampling 穩定性

### 11.1 HRG 必要性分析

為了回應「你的方法是否只是利用頻率資訊，而不是利用圖結構」這個質疑，現在流程中已加入：

1. `Spine-RandomExpansion`
2. `Spine-FrequencyExpansion`

三者可形成一條更完整的證據鏈：

1. `Random Expansion`
   - 沒有結構，沒有頻率
2. `Frequency Expansion`
   - 有全域 relation 頻率，但沒有 HRG 結構
3. `Grammar Expansion / HRG-Proposed`
   - 有結構樣板與局部圖樣約束

因此，若 `HRG-Proposed` 明顯優於 `FrequencyExpansion`，就比較能支持：

1. HRG 的幫助不只是因為用了 relation 頻率
2. 而是因為它保留了更有辨識力的局部結構模式

### 11.2 Sampling 穩定性分析

為了回答「只抽 4 個 BFS sample、每個 500 節點，是否能代表整張 KG」這個問題，portable workflow 現在補了一個專門腳本：

1. [portable_runner/grammar_stability_analysis.py](/home/zihui/projects/masterPaperRemake/portable_runner/grammar_stability_analysis.py)

這個腳本可直接對同一個 KG：

1. 用不同 random seed 重複生成 grammar
2. 比較不同 seed 下 top-N rules 的 signature overlap
3. 比較不同 seed 下 top-N relation label 的 overlap

### 11.3 Stability script 實際做什麼

`grammar_stability_analysis.py` 會：

1. 載入同一份 KG
2. 對多個 seed 分別呼叫 `learn_phrg_from_k_bfs_samples(...)`
3. 對每份 grammar：
   - 統計 rule 數量
   - 取 top-N rules
   - 把 rule 轉成 signature
   - 收集 top-N rule 內出現的 relation labels
4. 對任兩個 seed：
   - 計算 top-rule signature Jaccard overlap
   - 計算 top-label Jaccard overlap
5. 將結果輸出成 JSON

### 11.4 Stability script 範例

可直接執行：

```bash
cd portable_runner
python3 grammar_stability_analysis.py \
  --kb-path Datasets/MetaQA/kb.txt \
  --seeds 0 1 2 \
  --k-samples 4 \
  --sample-size 500 \
  --top-n 100
```

預設輸出位置為：

1. `portable_runner/artifacts/_analysis/grammar_stability.json`

### 11.5 這份分析能回答什麼

它不能直接證明 sample grammar 完全代表整張 KG，但它可以補兩個很重要的輔助證據：

1. 不同 seed 下，top structural rules 是否穩定
2. 不同 seed 下，抽到的 relation pattern 是否高度重疊

若 overlap 很高，就可以說：

1. 在目前的採樣設定下，grammar 並不是完全 seed-sensitive 的偶然結果
2. 抽樣結果至少對高頻局部樣板有一定穩定性

## 12. 結果呈現建議

### 11.1 建議以資料集分開比較

目前最穩妥的論文寫法，不是強調四個資料集做單一總排名，而是：

**每個 dataset 內部做完整比較**

這樣的好處是：

1. `KQAPro` 的 adapted setting 不會污染其他資料集
2. metric 定義爭議較小
3. 不同資料集的 KG 結構與答案型態差異不會被硬平均掉

### 11.2 每個資料集內建議呈現的內容

對每個 dataset，可重點呈現：

1. 答案指標
   - `EM`
   - `Answer-Set F1`
   - `Answer Recall`
2. 效率指標
   - `Avg Context Tokens`
   - `Avg Latency`
   - `Avg Subgraph Size`
3. Retrieval 指標
   - `Subgraph Recall`
   - `Subgraph Precision`
   - `Subgraph F1`
4. Serialization 比較
   - `json`
   - `triples`

## 13. 快取與可攜性

目前模型與相關快取不再寫到機器預設的 `~/.cache`，而是固定在目前資料夾下：

1. `portable_runner/.cache/huggingface/`
2. `portable_runner/.cache/torch/`
3. `portable_runner/.cache/nltk/`

這樣做的目的，是讓整個 `portable_runner` 更容易整包搬到其他機器上使用，也讓 cache 行為更可控。

## 14. 總結

目前的 portable workflow 已經不是單純的 demo runner，而是一套相對完整、可分析的 benchmark pipeline。它目前具備：

1. 固定資料集路徑管理
2. offline grammar generation
3. adapted KQAPro support
4. training-free online KGQA inference
5. backbone comparison
6. 方法消融
7. serialization comparison
8. answer / retrieval / token / latency 多面向指標

因此，現在這套系統最適合支撐的論文敘事是：

1. offline 圖結構先驗抽取
2. online chain-first constrained retrieval
3. correction 與 expansion 的可拆解貢獻
4. grammar-guided vs random expansion 的對照
5. 效果與成本之間的 trade-off 分析
