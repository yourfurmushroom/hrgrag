# Spine-Correction KG-RAG Oral Defense Report Draft

> 立場設定：以下內容用「最嚴厲口試委員」的標準整理。不要把目前結果包裝成全面勝過 BFS 或全面 SOTA；目前最能 defended 的主軸是：**Spine-Correction 在 relation-chain-friendly 的 KGQA 任務上，用極小 context 保留可解釋推理路徑，並在 MetaQA multi-hop 尤其 3-hop 中比無結構 BFS 更有效。**  
> 主方法建議命名為 **Spine-Correction KG-RAG**。`HRG-Proposed` 可作為延伸方法或未完全成熟的 adaptive variant，不建議把它當作主要貢獻的唯一證據。

---

## 0. 報告的核心論點

### 一句話主張

給定知識圖譜問答問題，傳統 BFS KG-RAG 雖然召回高，但會把大量與答案無關的邊塞進 LLM context；本研究提出 **Spine-Correction**，先由 LLM 預測 topic entity 與 relation chain，再用 KB 執行驗證與 fallback correction 修正不可執行 chain，最後只保留可解釋 spine subgraph 作為回答 evidence。

### 目前數據能支持的 claim

1. 在 MetaQA 上，Spine-Correction-gpt-oss-json 達到 **EM 0.6133 / F1 0.6622**，高於 Baseline-BFS 的 **EM 0.5750 / F1 0.6109**。
2. 在 MetaQA 3-hop 上，Baseline-BFS 的 answer EM 只有 **0.005**，Spine-Correction-gpt-oss-json 達到 **0.160**，代表結構化 spine 對長 hop generation 比大量 BFS context 更有效。
3. Spine-Correction 在 MetaQA 使用的平均 context tokens 是 **212.03**，只有 BFS **4415.83** 的 **4.80%**；triple serialization 則是 **96.86 tokens**，只有 BFS 的 **2.19%**。
4. 在 WikiMovies 上 BFS 幾乎飽和：**EM 0.985 / F1 0.9913**。Spine-Correction-json 仍達 **EM 0.970 / F1 0.9740**，但因 WikiMovies 目前多為 1-hop，BFS 是合理強 baseline。
5. 在 MLPQ 與 KQAPro，BFS EM 高於 Spine-Correction 與 HRG-Proposed，這不是缺點要隱藏，而是論文需要誠實定義的 limitation：跨語言 relation normalization 與 KQAPro operator/statement qualifier semantics 尚未被 Spine-Correction 完整建模。

### 目前不能強講的 claim

1. 不能說 `HRG-Proposed` 全面優於 BFS。四個資料集上它的最佳 EM 都低於 BFS：
   - MetaQA: HRG-Proposed 0.5517 vs BFS 0.5750
   - WikiMovies: HRG-Proposed 0.9650 vs BFS 0.9850
   - MLPQ: HRG-Proposed 0.1950 vs BFS 0.2550
   - KQAPro: HRG-Proposed 0.0400 vs BFS 0.0700
2. 不能說 correction 明顯優於 Spine-Only。主結果中 Spine-Correction 與 Spine-Only 在多數資料集數字相同或非常接近，表示正式結果裡 correction 觸發率或有效增益需要補更多分析。
3. 不能把 qwen3.5 寫成有效實驗結果；目前 qwen3.5 在四個資料集所有列出的 runs 都是 `FAILED`。

---

## 1. 建議論文題目與摘要

### 建議題目

**Spine-Correction KG-RAG: Executable Relation-Chain Retrieval for Compact and Faithful Knowledge Graph Question Answering**

中文可寫：

**Spine-Correction KG-RAG：以可執行關係鏈修正實現精簡且忠實的知識圖譜問答**

### 摘要草稿

Knowledge Graph Question Answering (KGQA) requires retrieving evidence that is both relevant to the answer and compact enough for large language models to use reliably. A common KG-RAG baseline expands a breadth-first-search subgraph from the topic entity, which provides high recall but often introduces a large amount of irrelevant evidence. This work studies a spine-based KG-RAG framework that parses each question into a topic entity and an ordered relation chain, validates the chain by executing it over the KG, and uses fallback correction when the predicted chain is not executable. The resulting Spine-Correction method retrieves a compact evidence spine instead of an unconstrained BFS neighborhood. Experiments on MetaQA, WikiMovies, MLPQ, and KQAPro show that Spine-Correction is most effective when the dataset semantics align with relation-chain reasoning. On MetaQA, Spine-Correction improves answer EM from 0.5750 to 0.6133 over BFS while reducing average context tokens from 4415.83 to 212.03. Per-hop analysis further shows that the method is especially beneficial for 3-hop questions, where BFS has high retrieval recall but poor answer generation. The results also reveal important limitations on multilingual and operator-heavy datasets, motivating future work on relation canonicalization and statement-aware graph traversal.

---

## 2. Introduction 應該怎麼講

### Slide 1: Title

**核心訊息：本研究不是單純把 KG 丟給 LLM，而是讓 LLM 使用可執行的 relation spine。**

建議內容：

- Title: Spine-Correction KG-RAG
- Subtitle: Executable relation-chain retrieval for compact KGQA evidence
- Dataset logos/text: MetaQA, WikiMovies, MLPQ, KQAPro

建議圖片：

- **Figure 1: Compact KGQA Evidence Teaser**
- 圖片描述：左側是一個問題「Who directed the movies acted by [Tom Hanks]?」，中間是大型 KG neighborhood，灰色大量 BFS 邊被淡化；右側只保留一條高亮 relation spine：`Tom Hanks --starred_actors--> Movie --directed_by--> Director`。下方放兩個數字 callout：BFS context tokens high，Spine context tokens low。
- 注意：圖中每條邊都必須是本研究可解釋的 relation-chain traversal，不要畫 transformer block、神秘 attention cloud 或沒有對應模組的圖案。

### Slide 2: Real Problem

**核心訊息：KGQA 的困難不是只找資料，而是找「剛好足夠且可被 LLM 使用」的 evidence。**

白話說法：

使用 KG-RAG 時，如果從 topic entity 做 BFS，很容易拿到答案所在區域，但也會拿到大量不相關 facts。LLM 最後看到的是一個大 subgraph，不一定能穩定沿著正確 reasoning path 回答。

例子：

```text
Question: What is the genre of the movie directed by the person who wrote [Cast Away]?
BFS context: hundreds of neighboring facts
Needed evidence spine:
Cast Away --written_by--> Writer
Writer --written_by--> Other Movie
Other Movie --has_genre--> Genre
```

建議圖片：

- **Figure 2: BFS Noise vs Evidence Spine**
- 圖片描述：同一個 topic entity 展開 3-hop。左圖是 BFS：節點與邊很多，答案節點存在但被噪音包圍。右圖是 spine：只有 3-hop path 與必要的 relation labels。用紅色標示「answer exists but hard to use」，用綠色標示「answer path is explicit」。

### Slide 3: Why This Problem Matters

**核心訊息：長 hop KGQA 中，高 retrieval recall 不等於高 answer accuracy。**

用 MetaQA 3-hop 的實驗數字建立動機：

| Method | Retrieval R@5 | Answer EM | Avg ctx tokens | Insight |
|---|---:|---:|---:|---|
| Baseline-BFS-gpt-oss | 0.9700 | 0.0050 | 4415.83 overall | 找得到答案附近，但 LLM 幾乎不能從大量 context 產生正確答案 |
| Spine-Correction-gpt-oss-json | 0.3850 | 0.1600 | 212.03 overall | retrieval 較窄，但 evidence 結構更可用 |

這頁要講清楚：**BFS 的 retrieval recall 高，但 answer EM 極低，說明 retrieval metric 和 generation utility 之間有落差。**

建議圖片：

- **Figure 3: Recall-Accuracy Mismatch Bar Chart**
- 圖片描述：MetaQA 3-hop 的 grouped bar：BFS R@5=0.970 vs EM=0.005；Spine-Correction R@5=0.385 vs EM=0.160。圖上標註「High recall can still fail generation」。

### Slide 4: Challenges

**核心訊息：本研究處理的是 relation-chain retrieval 的三個 failure points。**

| Challenge | 具體問題 | 對應方法設計 | 對應實驗 |
|---|---|---|---|
| C1: Context explosion | BFS 取回太多 irrelevant edges | Relation spine retrieval | Context token/subgraph compression |
| C2: LLM chain may be invalid | LLM 預測 relation chain 可能在 KG 上走不通 | KB validation + fallback correction | Spine-Only vs Spine-Correction；failure counts |
| C3: Dataset semantics mismatch | WikiMovies composite tails、MLPQ multilingual relations、KQAPro qualifiers 不一定是普通 chain | Dataset risk audit + future semantic adapters | Cross-dataset analysis and limitations |

建議圖片：

- **Figure 4: Challenge-Method-Experiment Alignment Matrix**
- 圖片描述：三列 challenge，三欄 method component / measured metric / observed result。這張圖的目的不是漂亮，而是讓口委看到前後章節有一一對應。

### Slide 5: Research Goal

**核心訊息：我們不是最大化取回子圖大小，而是取回可執行、精簡、可回答的 evidence spine。**

Problem statement 白話版：

給定一個 KGQA 問題與知識圖譜，我們希望找出從 topic entity 出發、能在 KG 上實際走通的 relation chain，並用該 chain 對應的 facts 作為 LLM 回答依據。困難在於 LLM 預測的 chain 不一定可執行，而 BFS 雖然穩但 context 過大。

正式符號版：

- Knowledge graph: \(G=(V,E,R)\)，其中 \(E \subseteq V \times R \times V\)
- Question: \(q\)
- Gold answer set: \(A_q\)
- Parser output candidates: \(\mathcal{C}_q=\{(e_i, \mathbf{r}_i, s_i)\}_{i=1}^{K}\)
- Relation chain: \(\mathbf{r}_i=[r_{i,1},...,r_{i,L}]\)
- Chain execution:

```text
Frontier_0 = {ground(e_i)}
Frontier_t = {v' | exists v in Frontier_{t-1}, (v, r_{i,t}, v') in E or (v', r_{i,t}, v) in E}
```

- Valid chain:

```text
valid(e_i, r_i) = true iff Frontier_t is non-empty for every t = 1...L
```

- Objective:

```text
Select compact subgraph S_q from executable candidates
such that S_q supports answer A_q while minimizing irrelevant context.
```

---

## 3. Method: Spine-Correction KG-RAG

### Slide 6: Method Overview

**核心訊息：方法分成 parse、ground、validate、correct、retrieve spine、answer 六步。**

一定要放在方法一開始。

建議圖片：

- **Figure 5: Spine-Correction Architecture / Flowchart**
- 圖片描述：
  1. Input: Question \(q\), KG \(G\)
  2. LLM parser outputs top-k `(entity, relation chain, confidence)`
  3. Entity grounding maps entity surface to KG node
  4. KB validation executes relation chain
  5. If all initial candidates invalid, fallback correction produces revised chains
  6. Valid chain builds strict spine subgraph
  7. Subgraph serialization as JSON or triples
  8. LLM answer generation
  9. Metrics: EM/F1, retrieval R@k, context tokens, latency
- 圖中不要畫「HRG expansion」為主路徑，因為主要方法是 Spine-Correction；HRG-Proposed 可放在右側 extension dashed box。

### Slide 7: Why Not Simple BFS?

**核心訊息：BFS 是強 baseline，但會把 retrieval problem 轉成 context selection problem。**

BFS 流程：

```text
LLM extracts topic entity
-> BFS from entity up to depth h
-> collect all edges under budget
-> serialize context
-> LLM generates answer
```

BFS 優點：

- 不依賴 relation parsing
- Recall 通常高
- 對 1-hop WikiMovies 很強

BFS 缺點：

- Context tokens 大
- 3-hop noise 累積嚴重
- LLM 需要在大量 facts 中自行找 reasoning path

數據：

| Dataset | BFS EM | BFS F1 | BFS ctx tokens | BFS subgraph |
|---|---:|---:|---:|---:|
| MetaQA | 0.5750 | 0.6109 | 4415.83 | 369.09 |
| WikiMovies | 0.9850 | 0.9913 | 50.01 | 3.30 |
| MLPQ | 0.2550 | 0.3693 | 4992.29 | 333.59 |
| KQAPro | 0.0700 | 0.0700 | 2984.98 | 191.92 |

Insight:

- WikiMovies 是 1-hop 且子圖本來小，所以 BFS 不是問題。
- MetaQA/MLPQ/KQAPro 的 BFS context 很大，尤其 MetaQA 3-hop 顯示高 recall 不會自動帶來高答案正確率。

### Slide 8: Step 1 - LLM Parses Candidate Relation Chains

**核心訊息：方法不是只讓 LLM 直接回答，而是讓 LLM 產生可驗證的查詢意圖。**

程式實作：

- `KnowledgeGraphAgent._parse_intent_candidates`
- 輸出 top-k JSON array：

```json
[
  {"entity": "Cast Away", "chain": ["written_by", "written_by", "has_genre"], "confidence": 0.82},
  {"entity": "Cast Away", "chain": ["directed_by", "has_genre"], "confidence": 0.41}
]
```

設計理由：

- 直接回答不可驗證；relation chain 可以在 KG 上執行。
- top-k 不是只信第一個 LLM output，可降低 parse error 的單點失敗。
- relation shortlist 與 alias guide 讓輸出更接近 KG vocabulary。

口委可能問：「為什麼不用 end-to-end LLM 直接回答？」

答法：

> 因為本研究要控制 evidence。直接回答無法分辨模型是從 KG evidence 推理，還是靠參數記憶或 hallucination。Relation chain parsing 讓每一步 retrieval 可以被 KB 驗證。

建議圖片：

- **Figure 6: Parser Output Example**
- 圖片描述：左側 question，右側 JSON candidates，每個 candidate 下方接一個 mini KG validation result：valid / invalid。

### Slide 9: Step 2 - Entity Grounding

**核心訊息：relation chain 正確之前，topic entity 必須先對到 KG node。**

實作重點：

- exact lookup
- normalized lookup
- whitespace / underscore / punctuation normalization
- alias map
- token-overlap fallback
- WikiMovies composite-tail alias support

WikiMovies 風險例子：

```text
Raw KB:
The Inbetweeners 2 starred_actors James Buckley, Simon Bird, Blake Harrison, Joe Thomas

Graph reasoning should behave like:
The Inbetweeners 2 --starred_actors--> James Buckley
The Inbetweeners 2 --starred_actors--> Simon Bird
...
```

口委可能問：「這是方法貢獻還是資料清理？」

答法：

> 這不是主要方法貢獻，但它是讓 benchmark 公平的必要 adapter。若 entity 不能 ground 到 KG node，任何 retrieval method 都會失敗。因此報告中要把它放在 data adaptation / risk control，而不是誇大成模型創新。

建議圖片：

- **Figure 7: Entity Grounding and Alias Normalization**
- 圖片描述：多個 surface forms，如 `Tom_Hanks`, `Tom Hanks`, punctuation variants，匯入同一個 KG node。WikiMovies composite tail 用 dashed split arrows 表示是 adapter，不是原始 KG 天然結構。

### Slide 10: Step 3 - KB Chain Validation

**核心訊息：Spine-Correction 的關鍵不是相信 LLM chain，而是執行它。**

演算法：

```text
Input: entity e, chain [r1, ..., rL], KG adjacency
frontier = {ground(e)}
spine_edges = []
for each relation r_t:
    next_frontier = {}
    for node in frontier:
        collect outgoing and incoming edges labeled r_t
        add matched edges to spine_edges
        add neighbor nodes to next_frontier
    if next_frontier is empty:
        return invalid
    frontier = next_frontier
return valid with spine_edges
```

設計理由：

- LLM 的 chain 是 hypothesis；KB execution 是 verification。
- Invalid chain 不應進入 answer context，否則 LLM 會看到錯 evidence 或空 evidence。
- Validation 使 retrieval 可解釋，也讓 failure analysis 可定位到 no_candidates / no_valid_chain。

建議圖片：

- **Figure 8: Chain Execution Over KG**
- 圖片描述：三層 frontier。每一步 relation label 高亮，若某一步 frontier empty，就用紅色 cross 標示 invalid；若成功到答案節點，用綠色 path 標示 valid spine。

### Slide 11: Step 4 - Fallback Correction

**核心訊息：Correction 只在所有初始 candidates 都 invalid 時觸發，避免過度修改可執行 chain。**

目前正式 `Spine-Correction` 設定：

- `use_grammar_rerank=False`
- `use_grammar_expansion=False`
- `use_fallback_correction=True`
- `grammar_path=None`

Correction pool 來源：

- LLM correction：根據失敗 chain 重新產生候選
- direction flip 介面存在，但目前 `_make_direction_flip_candidates()` 在正式流程中沒有實質產生候選
- grammar fallback 只有在 agent 有 HRG matcher 時才會使用；Spine-Correction 正式設定 `grammar_path=None`，所以主要是 LLM correction

重要限制：

- 主結果中 Spine-Correction 與 Spine-Only 在 MetaQA / WikiMovies 幾乎相同，表示 correction 不是目前最大增益來源。
- 報告要說「correction 是 safety mechanism」，不要說「correction 已大幅改善所有結果」。

建議圖片：

- **Figure 9: Correction Trigger Condition**
- 圖片描述：decision tree。Initial candidates -> any valid? yes: skip correction; no: correction candidates -> validation again -> valid spine or no_valid_chain。圖上清楚標示 correction 不是每題都做。

### Slide 12: Step 5 - Strict Spine Retrieval

**核心訊息：Spine subgraph 只保留 chain 真正走到的 evidence edges。**

定義：

```text
S_spine(q) = union of edges traversed by the selected valid relation chain
```

與 BFS 差異：

| Method | Retrieval unit | Main risk | Strength |
|---|---|---|---|
| BFS | all reachable neighborhood edges | context noise | high recall |
| Spine | executable relation-chain edges | parse/chain error | compact evidence |

數據例子 MetaQA overall：

| Method | EM | F1 | Avg ctx tokens | Avg subgraph size |
|---|---:|---:|---:|---:|
| Baseline-BFS | 0.5750 | 0.6109 | 4415.83 | 369.09 |
| Spine-Correction-json | 0.6133 | 0.6622 | 212.03 | 9.20 |
| Spine-Correction-triple | 0.5950 | 0.6448 | 96.86 | 9.20 |

Insight:

Spine-Correction 不只是變小；在 MetaQA 也提升 answer quality。這是主方法最強證據。

### Slide 13: Step 6 - Serialization and Answer Generation

**核心訊息：json/triple 不是不同 retrieval，而是同一 retrieval evidence 的不同序列化。**

序列化方式：

- JSON：結構清楚，但 tokens 較多，latency 通常較高。
- Triples：tokens 較少，latency 較低，但 LLM 讀取結構可能較不穩。

MetaQA 結果：

| Method | Serialization | EM | F1 | Avg latency | Avg ctx tokens |
|---|---|---:|---:|---:|---:|
| Spine-Correction | json | 0.6133 | 0.6622 | 44.98s | 212.03 |
| Spine-Correction | triple | 0.5950 | 0.6448 | 10.63s | 96.86 |

Insight:

JSON 通常答案品質稍好，但 triple 成本更低。這可以作為 deployment trade-off，而不是方法輸贏。

建議圖片：

- **Figure 10: Serialization Trade-off**
- 圖片描述：x 軸 avg latency 或 ctx tokens，y 軸 EM，標出 JSON 與 triple。JSON 在 MetaQA EM 較高，triple 在成本較低。

---

## 4. HRG-Proposed 應如何放在報告裡

### Slide 14: HRG-Proposed as Extension

**核心訊息：HRG-Proposed 是 Spine-Correction 的延伸，但目前不是主實驗最佳方法。**

HRG-Proposed components:

```text
Spine retrieval
+ grammar-aware rerank
+ strict grammar expansion
+ fallback correction
+ deterministic KG-valid chain fallback
+ subgraph reranking
```

目前正式設定：

- `num_candidates=8`
- `use_grammar_rerank=True`
- `use_grammar_expansion=True`
- `use_fallback_correction=True`
- `use_deterministic_valid_chain_fallback=True`
- `expansion_strict=True`
- `expansion_min_prob=2.0`
- `expansion_per_node_cap=4`
- `max_total_context_edges=160`

結果解讀：

| Dataset | BFS EM | Spine-Correction best EM | HRG-Proposed best EM | HRG ctx ratio vs BFS |
|---|---:|---:|---:|---:|
| MetaQA | 0.5750 | 0.6133 | 0.5517 | 0.0384 |
| WikiMovies | 0.9850 | 0.9700 | 0.9650 | 1.3046 |
| MLPQ | 0.2550 | 0.1550 | 0.1950 | 0.0718 |
| KQAPro | 0.0700 | 0.0083 | 0.0400 | 0.0110 |

Insight:

- HRG-Proposed 在 MLPQ/KQAPro 比 narrow Spine-Correction 更好，但仍輸 BFS。
- HRG-Proposed 壓縮 context 很強，但答案品質尚未穩定。
- 因此論文應把 HRG-Proposed 寫成「toward adaptive grammar-guided retrieval」，不是主 claim。

建議圖片：

- **Figure 11: Method Family**
- 圖片描述：Baseline-BFS、Spine-Only、Spine-Correction、HRG-Proposed 四個方法並排，用勾選表顯示是否有 chain parse、KB validation、correction、grammar rerank、expansion、KG-valid fallback。

---

## 5. Baselines and Fairness

### Slide 15: Baselines

**核心訊息：Baselines 不是隨便挑，而是對應 retrieval design choices。**

| Category | Baseline | Why included | Fairness note |
|---|---|---|---|
| Unstructured KG-RAG | Baseline-BFS | 強 recall baseline；不依賴 chain parsing | BFS depth fixed to 3; same answer generator backbone |
| Narrow spine | Spine-Only | 測試只用 LLM chain 是否足夠 | 與 Spine-Correction 同 parser/retrieval serialization |
| Main method | Spine-Correction | 測試 invalid-chain correction 的價值 | 不使用 grammar rerank/expansion，讓主方法清楚 |
| Extended method | HRG-Proposed | 測試 grammar prior + adaptive fallback | 目前作為延伸，不作主優勢 claim |

目前 benchmark 中註解掉、可做 future ablation 的方法：

- Spine-GrammarExpansion
- Spine-RandomExpansion
- Spine-FrequencyExpansion

口委可能問：「你是不是只挑容易打敗的方法？」

答法：

> BFS 在 WikiMovies、MLPQ、KQAPro 都贏過我們的 spine variants，所以它不是弱 baseline。這反而幫助我們界定方法適用條件：當任務是 clean relation-chain KGQA，Spine-Correction 有優勢；當資料語義需要 operator、qualifier 或跨語言 relation canonicalization，BFS 的寬召回仍然較穩。

---

## 6. Experiment Setup

### Slide 16: Datasets

**核心訊息：四個資料集不是同質 benchmark，它們測到的方法能力不同。**

| Dataset | Split / setting | Main structure | Risk level | What it tests |
|---|---|---|---|---|
| MetaQA | vanilla test | clean movie KG, 1/2/3-hop | Low | relation-chain reasoning |
| WikiMovies | wiki_entities test | mostly 1-hop movie facts | High data encoding risk | entity grounding and KB normalization |
| MLPQ | en-zh / en / ills | multilingual KGQA | Medium-high | multilingual relation/entity matching |
| KQAPro | validation | compositional KBQA with operators/qualifiers | High semantic mismatch | statement-aware and operator reasoning |

重要口試說法：

> MetaQA 是驗證方法假設的 clean control。WikiMovies、MLPQ、KQAPro 是壓力測試，暴露資料語義和 chain abstraction 的 mismatch。

建議圖片：

- **Figure 12: Dataset Compatibility Spectrum**
- 圖片描述：橫軸是 relation-chain compatibility，MetaQA 在 high，WikiMovies partial，MLPQ partial，KQAPro low-to-partial。每個 dataset 下方放一個代表性風險 icon：atomic triples、composite tails、multilingual relations、statement qualifiers。

### Slide 17: Metrics

**核心訊息：同時評估答案、retrieval、faithfulness、成本，不只看 EM。**

| Metric group | Metrics | Purpose |
|---|---|---|
| Answer quality | EM, Hits@1/3/5, MRR, Answer-Set F1 | 是否答對 |
| Retrieval quality | Retrieval R@1/3/5, nDCG@1/3/5, retrieval precision/F1 | 子圖是否 support answer |
| Faithfulness | claim faithfulness / hallucination | 輸出是否被 context 支持 |
| Cost | avg ctx tokens, avg subgraph size, latency | 是否真的壓縮 |
| Robustness | answerable rate, failure counts | pipeline 死在哪裡 |

注意：

- `claim_faithfulness` 是 heuristic support check，不要講成人工標註 truth。
- MLPQ 的 `citation_correctness` 有數值；其他資料集多為空值。

---

## 7. Main Results

### Slide 18: Overall Main Results

**核心訊息：Spine-Correction 在 MetaQA 是最強；BFS 在非 chain-friendly datasets 仍是強 baseline。**

| Dataset | Best EM model | EM | Best F1 model | F1 | Lowest latency model | Latency |
|---|---|---:|---|---:|---|---:|
| MetaQA | Spine-Correction-gpt-oss-json | 0.6133 | Spine-Correction-gpt-oss-json | 0.6622 | Baseline-BFS-gpt-oss | 7.39s |
| WikiMovies | Baseline-BFS-gpt-oss | 0.9850 | Baseline-BFS-gpt-oss | 0.9913 | HRG-Proposed-gpt-oss-triple | 6.40s |
| MLPQ | Baseline-BFS-gpt-oss | 0.2550 | Baseline-BFS-gpt-oss | 0.3693 | Spine-Only-gpt-oss-triple | 7.85s |
| KQAPro | Baseline-BFS-gpt-oss | 0.0700 | Baseline-BFS-gpt-oss | 0.0700 | Spine-Only-gpt-oss-triple | 0.08s |

Insight:

- 主要成功案例是 MetaQA。
- WikiMovies BFS 幾乎飽和，因為平均 BFS subgraph 只有 3.3 edges。
- MLPQ/KQAPro 表明現有 chain abstraction 尚不足。

建議圖片：

- **Figure 13: Overall EM by Dataset and Method**
- 圖片描述：四個 dataset grouped bar：BFS、Spine-Only-json、Spine-Correction-json、HRG-Proposed-json。每組上方標註 best method。不要只畫方法第一名，要讓限制一眼可見。

### Slide 19: MetaQA Detailed Results

**核心訊息：MetaQA 支持 Spine-Correction 的主要 claim：更小 context 且更高答案品質。**

| Model | EM | Hits@1 | F1 | R@5 | Faithfulness | Latency | Ctx tokens | Subgraph | Ctx ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spine-Correction-json | 0.6133 | 0.6300 | 0.6622 | 0.7250 | 0.9784 | 44.98 | 212.03 | 9.20 | 0.0480 |
| Spine-Only-json | 0.6133 | 0.6300 | 0.6622 | 0.7250 | 0.9784 | 44.63 | 212.03 | 9.20 | 0.0480 |
| Spine-Correction-triple | 0.5950 | 0.6150 | 0.6448 | 0.7250 | 0.9524 | 10.63 | 96.86 | 9.20 | 0.0219 |
| Baseline-BFS | 0.5750 | 0.6133 | 0.6109 | 0.9783 | 0.9808 | 7.39 | 4415.83 | 369.09 | 1.0000 |
| HRG-Proposed-json | 0.5517 | 0.5733 | 0.5881 | 0.6575 | 0.9430 | 47.48 | 169.57 | 7.37 | 0.0384 |
| HRG-Proposed-triple | 0.5350 | 0.5583 | 0.5718 | 0.6575 | 0.9197 | 8.92 | 77.26 | 7.37 | 0.0175 |

Insight:

- BFS retrieval R@5 最高，但 answer F1 不是最高。
- Spine-Correction-json 用 4.8% 的 BFS context，F1 提升 0.0513。
- Spine-Only 與 Spine-Correction 相同，表示 correction 需要補充觸發率分析，不能誇大。

建議圖片：

- **Figure 14: MetaQA Accuracy-Compression Pareto Plot**
- 圖片描述：x 軸 context ratio，y 軸 Answer-Set F1。BFS 在 x=1, F1=0.6109；Spine-Correction-json 在 x=0.048, F1=0.6622；Spine-Correction-triple 在 x=0.0219, F1=0.6448。

### Slide 20: MetaQA Per-Hop Results

**核心訊息：Spine-Correction 的優勢主要出現在 3-hop；1-hop/2-hop BFS 仍很強。**

| Hop | Method | EM | F1 | R@5 | Latency |
|---|---|---:|---:|---:|---:|
| 1-hop | Baseline-BFS | 0.995 | 0.9950 | 1.000 | 6.06 |
| 1-hop | Spine-Correction-json | 0.985 | 0.9850 | 0.990 | 20.71 |
| 2-hop | Baseline-BFS | 0.725 | 0.8326 | 0.965 | 15.27 |
| 2-hop | Spine-Correction-json | 0.695 | 0.7405 | 0.800 | 47.60 |
| 3-hop | Baseline-BFS | 0.005 | 0.0050 | 0.970 | 0.84 |
| 3-hop | Spine-Correction-json | 0.160 | 0.2612 | 0.385 | 66.62 |

Insight:

- 3-hop 是最有說服力的 slide：BFS 找得到相關區域但回答崩潰，spine 的 explicit path 讓 LLM 更能使用 evidence。
- 2-hop BFS 仍強，顯示方法不是無條件優勢。

建議圖片：

- **Figure 15: MetaQA Per-Hop EM**
- 圖片描述：1/2/3-hop grouped bar，BFS vs Spine-Correction-json。3-hop bar 要用 callout 標示「0.005 -> 0.160」。

### Slide 21: WikiMovies Results

**核心訊息：在 1-hop 且子圖很小的 WikiMovies，BFS 是合理上限；Spine 方法維持接近表現但壓縮優勢有限。**

| Model | EM | F1 | R@5 | Latency | Ctx tokens | Subgraph | Ctx ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline-BFS | 0.985 | 0.9913 | 1.000 | 9.82 | 50.01 | 3.30 | 1.0000 |
| Spine-Correction-json | 0.970 | 0.9740 | 0.985 | 20.46 | 68.16 | 2.89 | 1.3628 |
| Spine-Correction-triple | 0.945 | 0.9586 | 0.985 | 6.44 | 31.83 | 2.89 | 0.6365 |
| HRG-Proposed-json | 0.965 | 0.9690 | 0.980 | 27.55 | 65.25 | 2.76 | 1.3046 |
| HRG-Proposed-triple | 0.940 | 0.9536 | 0.980 | 6.40 | 30.49 | 2.76 | 0.6097 |

Insight:

- WikiMovies 不是證明 spine 壓縮的好資料集，因為 BFS context 本來就很小。
- 它更適合作為 entity grounding / KB normalization 的 case study。

建議圖片：

- **Figure 16: WikiMovies Composite Tail Example**
- 圖片描述：raw composite tail 被拆成 atomic actor nodes。旁邊放結果表的一行：BFS already near ceiling due to 1-hop small subgraph。

### Slide 22: MLPQ Results

**核心訊息：MLPQ 顯示 multilingual relation/entity mismatch 會削弱 strict spine。**

| Model | EM | Hits@1 | F1 | R@5 | Faithfulness | Citation Correctness | Ctx tokens | Ctx ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-BFS | 0.2550 | 0.3275 | 0.3693 | 0.8550 | 0.9783 | 0.0000 | 4992.29 | 1.0000 |
| HRG-Proposed-triple | 0.1950 | 0.2525 | 0.2771 | 0.5297 | 0.7526 | 0.3840 | 358.41 | 0.0718 |
| HRG-Proposed-json | 0.1825 | 0.2650 | 0.3004 | 0.5278 | 0.7383 | 0.3911 | 472.70 | 0.0947 |
| Spine-Correction-triple | 0.1550 | 0.2050 | 0.2242 | 0.3625 | 0.9676 | 0.3550 | 28.59 | 0.0057 |
| Spine-Correction-json | 0.1450 | 0.2200 | 0.2466 | 0.3625 | 0.9882 | 0.3550 | 60.11 | 0.0120 |

Per-hop:

| Hop | Best method | EM | F1 | R@5 |
|---|---|---:|---:|---:|
| 2-hop | Baseline-BFS | 0.425 | 0.6307 | 0.995 |
| 3-hop | Baseline-BFS | 0.085 | 0.1078 | 0.715 |
| 2-hop | HRG-Proposed-triple | 0.315 | 0.3984 | 0.6245 |
| 3-hop | HRG-Proposed-json/triple | 0.075 | 0.1815 / 0.1558 | 0.4310 / 0.4349 |

Insight:

- BFS 的寬召回仍最有效。
- HRG-Proposed 比 Spine-Correction 更好，表示 adaptive fallback/expansion 對 multilingual graph 有幫助，但還不夠。
- 下一步應做 relation canonicalization，而不是盲目加大 context。

建議圖片：

- **Figure 17: MLPQ Multilingual Chain Mismatch**
- 圖片描述：英文問題連到 bilingual KG relation labels，relation surface variants 分裂成多個 label，導致 grammar support 和 chain matching 分散。

### Slide 23: KQAPro Results

**核心訊息：KQAPro 暴露 ordinary relation-chain abstraction 無法完整處理 statement qualifier/operator reasoning。**

| Model | EM | F1 | R@5 | Faithfulness | Latency | Ctx tokens | Ctx ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline-BFS | 0.0700 | 0.0700 | 0.3283 | 0.2234 | 26.93 | 2984.98 | 1.0000 |
| HRG-Proposed-triple | 0.0400 | 0.0400 | 0.0820 | 0.1303 | 4.36 | 32.83 | 0.0110 |
| HRG-Proposed-json | 0.0283 | 0.0283 | 0.0820 | 0.0860 | 83.98 | 33.44 | 0.0112 |
| Spine-Correction-triple | 0.0083 | 0.0083 | 0.0067 | 0.9000 | 0.09 | 0.47 | 0.0002 |
| Spine-Correction-json | 0.0067 | 0.0067 | 0.0067 | 0.8000 | 65.88 | 0.89 | 0.0003 |

Per-hop best:

| Hop | Best model | EM | F1 | R@5 |
|---|---|---:|---:|---:|
| 1-hop | HRG-Proposed-triple | 0.045 | 0.045 | 0.1082 |
| 2-hop | Baseline-BFS | 0.170 | 0.170 | 0.3800 |
| 3-hop | HRG-Proposed-triple | 0.025 | 0.025 | 0.0613 |

Insight:

- KQAPro 現有成績低，不能拿來主張方法成功。
- 但它是很好的 limitation analysis：普通 entity-relation chain 不足以處理 qualifier、attribute、count、comparison、verification 等 program semantics。

建議圖片：

- **Figure 18: KQAPro Statement Qualifier Mismatch**
- 圖片描述：普通 fact `entity --relation--> value` 與 KQAPro statement node encoding 對比。第二張小圖顯示 qualifier 需要走 `entity --relation::statement--> statement_node --qualifier_relation--> qualifier_value`，而非單純一條 relation chain。

---

## 8. Ablation and Robustness

### Slide 24: Method Ablation

**核心訊息：目前 ablation 顯示 spine 本身是主要有效元件，correction 增益尚未被充分量化。**

主實驗中：

| Dataset | Spine-Only-json EM | Spine-Correction-json EM | Delta |
|---|---:|---:|---:|
| MetaQA | 0.6133 | 0.6133 | 0.0000 |
| WikiMovies | 0.9700 | 0.9700 | 0.0000 |
| MLPQ | 0.1450 | 0.1450 | 0.0000 |
| KQAPro | 0.0067 | 0.0067 | 0.0000 |

Triple serialization:

| Dataset | Spine-Only-triple EM | Spine-Correction-triple EM | Delta |
|---|---:|---:|---:|
| MetaQA | 0.5950 | 0.5950 | 0.0000 |
| WikiMovies | 0.9450 | 0.9450 | 0.0000 |
| MLPQ | 0.1550 | 0.1550 | 0.0000 |
| KQAPro | 0.0083 | 0.0083 | 0.0000 |

嚴厲解讀：

- 這表示目前「Correction」這個名稱要小心。它是架構中的必要安全機制，但主結果尚未證明它帶來顯著效能提升。
- 報告中要補充 failure trigger analysis：多少題 no_valid_chain、多少題 correction 後變 valid、多少題 answer 改善。

建議圖片：

- **Figure 19: Spine-Only vs Spine-Correction Delta**
- 圖片描述：四個 dataset 的 delta bar 全部接近 0。這張圖的目的不是炫耀，而是誠實指出需要更多 correction-specific analysis。

### Slide 25: KB Ablation - MetaQA Drop Nodes 10%

**核心訊息：在 10% grammar-KB node drop 的 ablation 中，Spine 類方法仍維持接近原本表現；但此 ablation 目前只有一組成功結果。**

目前 artifact: `metaqa-vanilla-test-abl-drop_nodes-10pct-seed42`

| Method | Status | EM | F1 | R@5 | Ctx tokens | Failure counts |
|---|---|---:|---:|---:|---:|---|
| Baseline-BFS | OK | 0.5717 | 0.6049 | 0.9733 | 4445.38 | ok=600 |
| Spine-Only-json | OK | 0.6083 | 0.6635 | 0.7250 | 212.03 | ok=435, no_candidates=160, no_valid_chain=5 |
| Spine-Only-triple | OK | 0.5933 | 0.6471 | 0.7250 | 96.86 | ok=435, no_candidates=160, no_valid_chain=5 |
| HRG-Proposed-json | OK | 0.5983 | 0.6511 | 0.8089 | 359.05 | ok=598, oom=2 |
| Spine-Correction-json | FAILED | - | - | - | - | - |
| Spine-Correction-triple | FAILED | - | - | - | - | - |

與原始 MetaQA 對比：

| Method | Original EM | Ablated EM | Delta |
|---|---:|---:|---:|
| Baseline-BFS | 0.5750 | 0.5717 | -0.0033 |
| Spine-Only-json | 0.6133 | 0.6083 | -0.0050 |
| HRG-Proposed-json | 0.5517 | 0.5983 | +0.0466 |

嚴厲解讀：

- 此 ablation 不是完整 robustness conclusion，因為只有 drop_nodes 10% seed42 且 Spine-Correction failed。
- 可以作為「初步壓力測試」，不能作為主要 robustness claim。
- 需要補跑 drop_relations、20%、30%、多 seed，並修復 Spine-Correction failed run。

建議圖片：

- **Figure 20: MetaQA Node-Drop Ablation**
- 圖片描述：Original vs 10% node-drop 的 EM bar。用灰色或 warning 標示 Spine-Correction ablation failed，不要假裝缺失不存在。

---

## 9. Error Analysis and Limitations

### Slide 26: Failure Taxonomy

**核心訊息：失敗不是單一原因，而是可分成 parse、grounding、chain validity、dataset semantics、generation。**

| Failure type | Symptom | Example dataset | Fix direction |
|---|---|---|---|
| no_candidates | Parser 沒產生有效 entity/chain | MetaQA ablation has 160 no_candidates in Spine-Only | better parser prompt / entity fallback |
| no_valid_chain | Chain 在 KG 上走不通 | MetaQA ablation has 5 no_valid_chain | stronger correction / deterministic fallback |
| context overload | retrieval recall high but answer EM low | MetaQA 3-hop BFS | spine evidence selection |
| relation surface mismatch | semantically same relation appears as different labels | MLPQ | relation canonicalization |
| statement/operator mismatch | question semantics not ordinary chain | KQAPro | statement-aware traversal / query IR |
| data encoding mismatch | answer hidden in composite tail | WikiMovies | atomicization and alias adapter |

建議圖片：

- **Figure 21: Failure Taxonomy Sankey**
- 圖片描述：Questions 流入五種 failure buckets；每個 bucket 接到 method/future fix。若沒有精確比例，不要畫比例寬度，使用等寬 blocks。

### Slide 27: Dataset Risk Assessment

**核心訊息：跨資料集結果差異反映 method assumption 與 dataset semantics 的 alignment。**

| Dataset | Why aligned / misaligned | Oral-defense framing |
|---|---|---|
| MetaQA | Atomic triples; explicit hop reasoning | Clean control for relation-chain retrieval |
| WikiMovies | 1-hop and composite-tail risks | Not a pure multi-hop method benchmark |
| MLPQ | Multilingual relation/entity variants | Tests grounding and canonicalization more than pure spine retrieval |
| KQAPro | Operators, qualifiers, statement nodes | Requires query semantics beyond ordinary relation chain |

這頁很重要，因為口委很可能問：

> 你的方法如果只在 MetaQA 好，是不是不夠 general?

建議回答：

> 目前結果顯示方法不是 universal KGQA solver，而是對 relation-chain-compatible KGQA 有明確優勢。跨語言與 operator-heavy datasets 需要 semantic adapters。這也是本研究的 limitation 與未來工作，而不是硬把所有資料集都視為同一種問題。

---

## 10. Contributions

### Slide 28: Contributions

**核心訊息：貢獻要對應 challenge 與實驗，不要只是口號。**

建議寫法：

1. **Executable spine retrieval for KG-RAG**  
   We formulate KGQA retrieval as selecting an executable relation-chain spine rather than expanding an unconstrained BFS neighborhood.

   Evidence: MetaQA Spine-Correction-json EM 0.6133 vs BFS 0.5750; context ratio 0.0480.

2. **Validation-and-correction pipeline for LLM-generated KG queries**  
   The method explicitly validates LLM-predicted relation chains on the KG and uses fallback correction when all candidates are invalid.

   Evidence: Implemented pipeline reports no_candidates/no_valid_chain and supports correction tokens/failure analysis. Current performance gain of correction itself needs further quantification.

3. **Cross-dataset diagnostic analysis of relation-chain KGQA**  
   We show where relation-chain retrieval works and where it fails: MetaQA aligns well; WikiMovies needs KB normalization; MLPQ needs relation canonicalization; KQAPro needs statement/operator-aware traversal.

   Evidence: Cross-dataset results and dataset risk assessment.

不要寫：

- 「我們的方法全面超越 baseline」
- 「HRG-Proposed 是 SOTA」
- 「Correction 已證明是最關鍵模組」

---

## 11. Conclusion

### Slide 29: Conclusion

**核心訊息：Spine-Correction 是一個 compact and executable KG-RAG 方法，但它的適用條件必須清楚界定。**

結論草稿：

This work shows that KG-RAG should not only retrieve more facts, but retrieve facts in a structure that LLMs can use. Spine-Correction uses LLM parsing to propose relation chains, validates those chains against the KG, and retrieves compact evidence spines for answer generation. On MetaQA, this design improves answer quality while reducing context tokens by over 95% compared with BFS. The strongest evidence appears in 3-hop questions, where BFS has high retrieval recall but poor answer accuracy. At the same time, experiments on WikiMovies, MLPQ, and KQAPro show that relation-chain retrieval is sensitive to dataset semantics. Future work should incorporate relation canonicalization, statement-aware traversal, and stronger correction analysis.

### Slide 30: Future Work

**核心訊息：未來工作直接對應目前失敗點。**

| Limitation | Future work | Expected effect |
|---|---|---|
| Correction gain not isolated | Add correction trigger/success/error analysis | Prove whether correction is necessary |
| MLPQ relation fragmentation | Canonicalize multilingual relation families | Improve chain support and grammar matching |
| KQAPro qualifier/operator mismatch | Add statement-aware traversal or query IR | Support non-chain semantics |
| WikiMovies composite tails | Atomicize KB values systematically | Improve entity grounding and reverse traversal |
| HRG-Proposed unstable | Separate grammar rerank, expansion, KG-valid fallback ablations | Identify which component helps |

建議圖片：

- **Figure 22: Future Work Roadmap**
- 圖片描述：四個 blocks：Correction analysis、Relation canonicalization、Statement-aware traversal、Grammar-guided adaptive retrieval。每個 block 接到對應 dataset。

---

## 12. 建議整份投影片順序

1. Title
2. Real KGQA problem: evidence must be usable, not just retrieved
3. Recall-accuracy mismatch using MetaQA 3-hop
4. Challenges and challenge-method-experiment alignment
5. Problem statement: plain version then notation
6. Method overview architecture
7. Why not BFS
8. LLM candidate chain parsing
9. Entity grounding and alias normalization
10. KB chain validation
11. Fallback correction trigger
12. Strict spine retrieval
13. Serialization trade-off
14. HRG-Proposed as extension
15. Baselines and fairness
16. Dataset compatibility and risks
17. Metrics
18. Overall results
19. MetaQA main result
20. MetaQA per-hop result
21. WikiMovies result and composite-tail analysis
22. MLPQ result and multilingual mismatch
23. KQAPro result and statement qualifier mismatch
24. Method ablation: Spine-Only vs Spine-Correction
25. KB ablation: MetaQA drop_nodes 10%
26. Failure taxonomy
27. Dataset risk assessment
28. Contributions
29. Conclusion
30. Future work

每頁只講一個核心訊息。若時間只有 20 分鐘，保留 1-6、10-12、15-20、24、26、28-30。

---

## 13. 圖片總清單與製圖注意事項

| Figure | Slide | Purpose | Must show | Avoid |
|---|---:|---|---|---|
| Fig. 1 Compact KGQA Evidence Teaser | 1 | 建立主題 | large KG vs compact spine | 無意義 AI 背景圖 |
| Fig. 2 BFS Noise vs Evidence Spine | 2 | 說明問題 | same question, BFS noise, spine evidence | 沒有 relation labels 的漂亮網路圖 |
| Fig. 3 Recall-Accuracy Mismatch | 3 | 動機數據 | MetaQA 3-hop R@5 vs EM | 只畫 EM 不解釋 recall |
| Fig. 4 Challenge-Method-Experiment Matrix | 4 | 前後對齊 | C1/C2/C3 -> method -> metric | challenge 口號 |
| Fig. 5 Spine-Correction Flowchart | 6 | 方法總覽 | parse, ground, validate, correct, retrieve, answer | 多畫不存在模組 |
| Fig. 6 Parser Output Example | 8 | 方法細節 | JSON candidates and validation | chain 不可執行卻畫成成功 |
| Fig. 7 Entity Grounding | 9 | adapter 說明 | aliases to node, composite tail split | 把資料清理畫成模型 magic |
| Fig. 8 Chain Execution | 10 | validation | frontier updates and invalid chain | 沒有 empty frontier case |
| Fig. 9 Correction Trigger | 11 | correction 條件 | only all invalid triggers correction | 畫成每題都 correction |
| Fig. 10 Serialization Trade-off | 13 | 成本 vs 準確率 | JSON/triple EM/token/latency | 混淆 retrieval 差異 |
| Fig. 11 Method Family | 14 | baseline 比較 | component checklist | 把 HRG-Proposed 畫成必勝 |
| Fig. 12 Dataset Compatibility Spectrum | 16 | dataset risk | MetaQA high, KQAPro low | 所有 dataset 畫成同質 |
| Fig. 13 Overall EM | 18 | 主結果 | four datasets, methods | 隱藏 BFS 贏的資料集 |
| Fig. 14 MetaQA Pareto Plot | 19 | 主 claim | F1 vs context ratio | 只報第一名 |
| Fig. 15 MetaQA Per-Hop EM | 20 | 3-hop claim | BFS vs Spine-Correction by hop | 不標 3-hop insight |
| Fig. 16 WikiMovies Composite Tail | 21 | data risk | raw row to atomic triples | 用抽象圖不給例子 |
| Fig. 17 MLPQ Mismatch | 22 | limitation | multilingual relation fragmentation | 假裝已解決 |
| Fig. 18 KQAPro Qualifier | 23 | limitation | statement-node traversal | 畫成普通 chain |
| Fig. 19 Spine vs Correction Delta | 24 | ablation honesty | delta near zero | 隱藏無增益 |
| Fig. 20 Node-Drop Ablation | 25 | robustness | original vs ablated, failed rows | 略過 failed |
| Fig. 21 Failure Taxonomy | 26 | error analysis | parse/ground/validity/semantics/generation | 沒有 fix direction |
| Fig. 22 Future Roadmap | 30 | closing | limitation -> future work | 空泛 roadmap |

AI 生圖 prompt 建議只用來產生初稿，最後必須手動檢查：

1. 每個 box 是否對應真實程式模組。
2. 每條 edge label 是否出現在 KG/relation list 或是明確示意。
3. 是否畫了不存在的 attention、embedding、training loss 或 end-to-end learning。
4. 是否把 `HRG-Proposed` 的 extension flow 和 `Spine-Correction` 主方法混在一起。
5. 是否有數據 callout，且數據與本文件表格一致。

---

## 14. 口委可能追問與建議回答

### Q1: 你的方法不是只在 MetaQA 有效嗎？

建議回答：

> 目前最強證據確實來自 MetaQA，因為它最符合 relation-chain KGQA 假設。這不是我想迴避的限制，而是本研究的核心診斷：Spine-Correction 適用於可被 entity + relation chain 表示的問題；MLPQ 與 KQAPro 需要額外的 semantic adapter，例如 relation canonicalization 和 statement-aware traversal。

### Q2: BFS 在很多資料集比你強，為什麼你的方法有價值？

建議回答：

> BFS 是強 baseline，尤其在 1-hop 或需要寬召回時很有效。但 BFS 的高 recall 不保證 LLM 可用。MetaQA 3-hop 中 BFS R@5 是 0.970，但 EM 只有 0.005；Spine-Correction R@5 較低，EM 卻到 0.160。這表示我們的方法價值在於提供可用、可解釋、精簡的 evidence spine。

### Q3: Correction 和 Spine-Only 結果一樣，為什麼方法叫 Spine-Correction？

建議回答：

> 這是目前實驗的弱點。Correction 是 pipeline 裡處理 invalid chain 的 safety mechanism，但目前主結果尚未量化它的獨立增益。因此論文中我會誠實呈現 Spine-Only vs Spine-Correction 的 ablation，並補做 correction trigger/success rate 分析。若補完後仍無增益，名稱應改為 Executable-Spine KG-RAG。

### Q4: HRG-Proposed 為什麼沒有比較好？

建議回答：

> HRG-Proposed 加了 grammar rerank、strict expansion、KG-valid fallback，目的是處理 strict spine 過窄的問題。但目前 expansion 和 fallback 對不同資料集的作用尚未穩定，特別是 KQAPro 的 semantics 不是普通 relation chain。現在我會把 HRG-Proposed 放成 extension 和 future direction，而不是主 claim。

### Q5: 你的 retrieval recall 比 BFS 低，這不是壞事嗎？

建議回答：

> 不一定。KG-RAG 的目標不是最大 recall，而是取回能支持答案且 LLM 能使用的 evidence。MetaQA 3-hop 正好說明 retrieval recall 和 answer EM 不一致。當然，如果 recall 太低也會失敗，所以我們同時報 answer metrics、retrieval metrics 和 context size。

### Q6: 為什麼不用 graph neural network 或訓練一個 retriever？

建議回答：

> 本研究目標是 portable KG-RAG pipeline，不依賴 dataset-specific training。LLM parsing + KB validation 可以直接套用到不同 KG，且每一步可解釋。GNN 或 supervised retriever 可能提升特定資料集表現，但需要標註和訓練，並且不一定能解決 LLM context usability 問題。

### Q7: KQAPro 這麼低，是否代表方法失敗？

建議回答：

> KQAPro 的低分代表普通 relation-chain abstraction 不足以表示它的 program semantics，例如 qualifier、attribute、count 和 comparison。這不是單純 parser 問題，而是 query representation mismatch。因此我會把 KQAPro 定位為 limitation analysis，指出需要 statement-aware traversal 或 query IR。

---

## 15. 最低限度必補實驗

若要讓口試更穩，建議優先補：

1. **Correction effectiveness table**
   - 每個 dataset 統計 initial valid rate、correction triggered count、correction produced valid chain count、correction improved answer count。
2. **Failure breakdown**
   - no_candidates / no_valid_chain / no_edges_after_selection / generation failure by dataset and method。
3. **Full ablation restore**
   - 跑 Spine-GrammarExpansion、Spine-RandomExpansion、Spine-FrequencyExpansion，至少在 MetaQA。
4. **KB ablation completion**
   - drop_nodes/drop_relations at 10/20/30%，多 seed；修復 Spine-Correction failed rows。
5. **Representative qualitative cases**
   - MetaQA 3-hop BFS fail vs Spine-Correction success。
   - MLPQ relation mismatch fail。
   - KQAPro statement qualifier fail。

---

## 16. 最終報告章節建議

### Chapter 1 Introduction

1. KGQA and KG-RAG background
2. Problem: retrieval recall vs usable evidence
3. Failure example: BFS context overload
4. Research challenges C1-C3
5. Research goal and contributions

### Chapter 2 Related Work

1. KGQA
2. Retrieval-augmented generation
3. LLM-based query parsing for KG
4. Graph-constrained retrieval
5. Semantic parsing / program-based KBQA for KQAPro-like tasks

### Chapter 3 Problem Formulation

1. Plain-language formulation
2. KG notation
3. Relation chain execution
4. Evidence subgraph objective
5. Evaluation dimensions

### Chapter 4 Method

1. Overview architecture
2. Candidate relation-chain parsing
3. Entity grounding
4. KB chain validation
5. Fallback correction
6. Spine subgraph construction
7. Serialization and answer generation
8. HRG-Proposed extension

### Chapter 5 Experiments

1. Dataset setup and risk assessment
2. Baselines
3. Metrics
4. Main results
5. Per-hop analysis
6. Ablation
7. Cost/latency/context analysis
8. Error analysis

### Chapter 6 Discussion

1. When Spine-Correction works
2. When BFS remains better
3. Why MLPQ/KQAPro are harder
4. Threats to validity

### Chapter 7 Conclusion

1. Summary of findings
2. Limitations
3. Future work

