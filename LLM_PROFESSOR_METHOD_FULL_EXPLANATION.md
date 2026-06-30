# HRG-Proposed KG-RAG 完整方法說明稿

> 用途：這份文件是 HRG-Proposed KG-RAG 的完整技術說明。  
> 報告主軸：**以 HRG structural prior 引導可執行 relation spine retrieval，在答案品質貼近 BFS baseline 的條件下減少 context tokens，並輸出可檢查的 KG evidence structure。**

---

## 1. 研究主題與一句話摘要

本研究處理的是 Knowledge Graph Question Answering，也就是給定一個自然語言問題與一個知識圖譜，系統必須從 KG 中找出支持答案的 evidence，並讓 LLM 根據 evidence 輸出答案。

傳統 KG-RAG 常用做法是從題目中的 topic entity 出發做 BFS，取回附近多跳子圖後交給 LLM。這種方法通常召回率高，但缺點是 context 很容易爆炸，尤其在 multi-hop 問題中，LLM 會看到大量與答案無關的邊，最後不一定能沿著正確推理路徑回答。

本研究的主方法是 **HRG-Proposed KG-RAG**：offline 階段先從 KG 局部子圖抽取 HRG-like grammar rules，形成 relation structure prior；online 階段讓 LLM 把問題解析成 topic entity 與 ordered relation chain，再以 KB validation 確認這條 chain 是否可執行。如果 initial chain 走不通，系統會使用 correction、grammar prior 與 deterministic KG-valid fallback 產生修正候選；若 chain 走通，則以 strict evidence spine 為核心。HRG 的主要作用是 grammar-hit candidate selection、KG-valid fallback 與 ranking features。

目前程式另外加入一個更強的 **HRG-GrammarFirst** ablation。它不是把 `HRG-Proposed` 改名，而是新增一條 retrieval variant：只先做 entity grounding，不要求 LLM 先猜 relation chain / hop 數；系統會先從 offline HRG rules 抽 relation path-bank，再由 KG 驗證哪些 paths 可從 entity seed 執行，最後用 HRG grammar features、question-relation relevance、frontier compactness 與 optional LLM rerank 排序。這個 row 專門回答「LLM 一開始 hop 猜錯時，HRG 能不能自己產生候選路徑」。

本論文最安全的主張不是「全面超越 BFS」，而是：

```text
在 relation-chain-friendly 或 strict spine 可執行的 KGQA 任務上，HRG-Proposed 可以在維持貼近 BFS 的 answer quality 的同時，
用明顯小於 BFS 的 context 輸出 selected entity、relation chain、HRG structural prior 與 spine evidence，提供 evidence-level explainability。
```

一句話版本：

```text
把 KGQA retrieval 從「取很大的 BFS 子圖」改成「以 HRG structural prior 引導的可執行、精簡、可檢查 relation spine」。
```

### 1.1 Claim Boundary

本文 claim 收斂成四點：

1. **HRG-Proposed 是主方法，HRG-GrammarFirst 是新增 ablation**：本文主方法不是單純 Spine-Correction，而是結合 executable relation spine、HRG structural prior、KG-valid fallback 與 grammar-aware ranking signals 的 HRG-Proposed KG-RAG。Spine-Only、Spine-Correction、Spine-Correction-KGValidFallback、HRG-Proposed-NoExpansion/Expansion、HRG-GrammarFirst-NoExpansion/Expansion 是 ablation / variant，用來拆解 strict spine、fallback、HRG prior、expansion 與 grammar-first search 的貢獻。
2. **Compact retrieval**：在四個有效模型平均下，HRG-Proposed-triple 在 MetaQA 將 context tokens 從 BFS 的 `4429.98` 降到 `189.55`，約為 BFS 的 `4.28%`；在 KQAPro 從 `3595.48` 降到 `783.54`，約為 BFS 的 `21.79%`；在 MLPQ 從 `4983.34` 降到 `209.30`，約為 BFS 的 `4.20%`。
3. **Closer-to-BFS hard-case behavior**：HRG-Proposed 不宣稱全面最高分，而是在多數資料集以小於 BFS 的 context 取得更接近 BFS 的 EM/F1。KQAPro 必須保守解讀：BFS 本身 F1 也很低，因此 KQAPro 不是「方法解決 hard QA」的證據，而是 out-of-scope stress test；它只能用來說明 HRG prior / KG-valid fallback 比 strict spine ablation 補回更多 executable evidence coverage。
4. **Evidence-level explainability**：本研究不宣稱能解釋 LLM 內部思考，而是能解釋 retrieval evidence：每題可以輸出 topic entity、relation chain、KB validation 結果、matched grammar prior、spine edges 與 final context。

因此結果章不以 EM 作為唯一主表，而是採用以下比較方式：

```text
answer quality: EM / Hits@1 / answer-set F1
evidence quality: retrieval recall@5 / claim faithfulness / citation correctness when available
cost: avg_ctx_tokens / avg_subgraph_size / compression_vs_bfs_ctx_ratio
robustness: generation_failure_count / answerable_rate / perturbation trends
```

這樣不是迴避 EM，而是符合本文的研究問題：本方法要解的是 **KGQA evidence retrieval 的品質、成本與可檢查性 trade-off**，不是單純把 LLM answer exact match 排到最高。

實驗結果支持的重點如下：

| Dataset | Comparison | Answer quality | Evidence / cost observation | 論文解讀 |
|---|---|---:|---|---|
| MetaQA | HRG-Proposed-triple vs BFS | F1 `0.5948` vs `0.5528` | context ratio `4.28%`, subgraph ratio約 `5.24%` | 在 relation-chain-friendly 任務上，HRG prior + executable spine 能用極小 context 維持甚至提高 answer-set F1。 |
| KQAPro | HRG-Proposed-triple vs Spine-Correction-triple | F1 `0.0854` vs `0.0245` | context ratio `20.26%`; failure count `1.75` vs BFS `45.25` | Out-of-scope stress test：HRG-Proposed 沒有解決 KQAPro QA，但比 strict spine ablation 補回更多 executable evidence coverage。 |
| MLPQ | HRG-Proposed-triple vs Spine-Correction-triple | F1 `0.2498` vs `0.2100` | retrieval recall@5 `0.5628` vs `0.4210`; context ratio `4.20%` | 在跨語言 path QA，HRG prior 提供較高 evidence recall，但 entity/relation 對齊仍限制 EM。 |
| WikiMovies | HRG-Proposed-json vs BFS | EM `0.8938` vs `0.8025` | BFS already small; HRG-json context ratio `133.41%` | WikiMovies 是 1-hop 為主，不能主打壓縮；它可作為高 accuracy / low difficulty sanity check。 |

結果敘述可採用以下版本：

```text
如果只看 EM，會把 retrieval 是否有效、context 是否被壓縮、evidence 是否可檢查全部混在一起。
本文的主問題不是讓模型背答案，而是讓 KGQA 在較小且可驗證的 evidence 上回答。
因此本文同時報告 EM/F1、retrieval recall、faithfulness、context tokens、subgraph size 與 failure counts。
在 MetaQA，HRG-Proposed 用 BFS 4.28% 的 context 取得更高 F1；
在 KQAPro 與 MLPQ，HRG-Proposed 相對 strict spine ablation 補回 hard cases 的 evidence coverage；
這些結果說明 HRG 的作用是 structural prior and fallback，不是保證所有資料集 EM 第一。
```

本文不採用以下過度 claim：

1. 本方法不宣稱在所有資料集全面優於 BFS。
2. HRG-Proposed 是本文主方法，Spine-Only / Spine-Correction / KGValidFallback / relation-prior rows 是 ablation；HRG-GrammarFirst 是新增 stronger variant，用來檢驗 HRG 是否能在不依賴 LLM hop/chain 預測的情況下產生候選路徑。
3. Token 減少必須和 answer quality、evidence coverage 一起解讀；只有在品質仍接近 baseline 時，compact context 才有意義。
4. MLPQ、KQAPro 主要用來說明 HRG prior 對 hard cases 的補強效果與目前限制，不被包裝成已完整解決的資料集。

### 1.1.1 核心貢獻

本文的貢獻不以「LLM 產生 chain 再去 KG 執行」作為單一主張，因為這和既有 executable semantic parsing / path retrieval 很接近。更準確的貢獻可以拆成四層：

1. **Executable relation-spine retrieval**：把 LLM 輸出的自然語言意圖轉成可在 KG 上驗證的 ordered relation chain，並只序列化通過 KB validation 的 evidence spine，而不是把 BFS neighborhood 全部丟給 LLM。
2. **KG-validated fallback and candidate recovery**：correction 不是每題都做，而是當 initial candidates 全部 invalid 時才啟動；每個 correction / fallback candidate 都必須再次通過 KB validation。從目前 trigger statistics 看，LLM correction 不是主要 empirical driver，真正主力是 deterministic KG-valid fallback 與 candidate recovery。
3. **HRG-like structural prior for candidate selection and ranking**：HRG-like grammar 不是裝飾，也不是完整 HRG decoder；它主要在 grammar-hit candidate selection、KG-valid fallback 與 ranking features 中提供 relation-structure prior。KQAPro 與 MLPQ 的結果顯示，當 strict spine evidence coverage 不足時，HRG-Proposed 能比 Spine-Correction 補回更多可用 evidence。
4. **Grammar-first retrieval variant**：新增 HRG-GrammarFirst，將 HRG prior 從「LLM candidate 的 rerank/recovery signal」推進到「candidate generation/search-space control」。它先由 HRG grammar 產生 relation path-bank，再由 KG 做 executable filtering，最後用 HRG / LLM rerank 選路徑；結果需等新版 rerun 後回填，不和舊 HRG-Proposed 數字混比較。

因此，本論文的主軸應寫成：

```text
HRG-Proposed is an HRG-guided compact evidence retrieval method for KGQA.
It improves the quality-cost trade-off by combining executable relation spines,
KG-validated fallback, candidate recovery, and grammar-guided candidate selection.
```

本文的 claim 邊界如下：

1. 與 neural path retriever、GraphRAG、token-budgeted BFS、relation-frequency reranker 等方法的完整比較，必須等新 rerun 完成後再決定是否放入主 claim；在結果尚未統一前只作 supplementary / diagnostic claim。
2. 本文不使用 gold entity / gold relation / gold chain oracle experiments，因此 MLPQ 與 KQAPro 的錯誤定位採用 grounding、relation canonicalization、operator / qualifier semantics 的粗粒度分類。
3. Token 統計以 final answer context tokens 為主，並以 online token proxy 補充 parse / correction / rerank / answer generation 的成本輪廓；本文不把它宣稱為精確 API billing cost。
4. Traceability 是 evidence structure 可檢查，不是人工標註過的 evidence sufficiency 結論。

### 1.1.2 Offline + Online 兩段式架構

本文架構分成兩個主部分：

> 圖：`img/paper_figures/13_offline_online_architecture.pdf`。  
> 圖說：Offline HRG-like grammar extraction provides a soft structural prior, which is then used online for candidate selection, fallback generation, and evidence retrieval.

**Part I: Offline HRG structural prior extraction**

這部分回答：

```text
HRG-like grammar 是否真的從 KG 中萃取出可量化、可分析、可作為 prior 的結構？
```

內容包含：

1. 使用 MCS ordering、triangulation、clique bags、clique tree 從 KG local subgraphs 萃取 HRG-like rules。
2. 報告 rule count、unique relation-pattern count、terminal arity、relation coverage、rule frequency distribution。
3. 報告 structural pattern compression：例如 unique relation patterns / total extracted rules。這不是無損還原原圖，而是 recurring structural pattern 的壓縮與覆蓋分析。
4. 做 KG perturbation robustness：隨機刪除 10% / 20% / 30% nodes 或 relations，觀察 extracted rules、relation vocabulary retention、exact pattern retention 如何變化。
5. 從 offline 結果導出 online 設計選擇：因為 exact high-order patterns 對 perturbation 敏感，所以 HRG 在 online 階段應作為 soft prior，而不是 hard grammar decoder。

**Part II: Online HRG-guided KGQA retrieval**

這部分回答：

```text
HRG structural prior 如何幫助 QA retrieval 取得更 compact、可執行、可檢查的 evidence？
```

內容包含：

1. **HRG-Proposed path**：LLM parse topic entity 與 relation-chain candidates。
2. **HRG-GrammarFirst path**：只抽 topic entity / entity candidates，從 KG adjacency 枚舉 1..D hop KG-valid relation chains。
3. entity grounding 與 relation normalization。
4. KB validation 確認 relation chain 是否 executable。
5. HRG prior 用於 candidate selection、grammar-aware ranking features、fallback generation 與 grammar-first candidate scoring。
6. 建立 strict spine evidence subgraph。
7. LLM 根據 evidence context 生成答案。
8. 用 answer quality、context cost、evidence coverage、failure analysis 與 KG perturbation robustness 評估。

這樣寫的好處是：HRG 不再只是 online pipeline 裡一個名字，而是從 offline grammar extraction 到 online evidence retrieval 都是主軸。

### 1.1.3 論文圖片使用索引

目前圖檔分成兩組：`img/01_*.pdf` 到 `img/12_*.pdf` 是方法說明、案例與診斷圖；`img/paper_figures/13_*.pdf` 到 `img/paper_figures/19_*.pdf` 是較適合直接放入結果章的論文圖。PDF 可直接用於 LaTeX，SVG 可作後續編修。老師建議的圖面修正原則是：全部使用向量圖、放大字體、減少框內長句、統一配色與圖例，並且每個 caption 明確寫出 dataset、model、sample size、aggregation unit。若圖中數字來自單模型或單設定診斷，正文需要搭配本文件的主結果表說明，不和四模型平均混成同一個統計口徑。

| 圖號 | 圖檔 | 放置章節 | 論文用途 | 使用狀態 |
|---|---|---|---|---|
| Fig. 01 | `img/01_method_pipeline.pdf` | 1.1、6、13、14 | 方法總覽：offline grammar prior、online retrieval、HRG signals | 可用；方法導論圖 |
| Fig. 02 | `img/02_example_bfs_vs_spine_linda_evans.pdf` | 1.2、5、21.8 | BFS 大子圖與 HRG-guided evidence 的案例對比 | 可用；案例圖 |
| Fig. 03 | `img/03_explainability_output_linda_evans.pdf` | 1.1、21.8 | selected entity、relation chain、spine edges 等可檢查輸出 | 可用；可解釋性圖 |
| Fig. 04 | `img/04_metaqa_token_subgraph_compression.pdf` | 21.3 | MetaQA context tokens 與 subgraph size 壓縮效果 | 可用；單設定診斷圖 |
| Fig. 05 | `img/05_metaqa_quality_vs_tokens.pdf` | 21.1、21.3 | answer-set F1 與 context tokens 的 quality-cost trade-off | 可用；單設定診斷圖 |
| Fig. 06 | `img/06_dataset_takeaway_matrix.pdf` | 23.1、24 | 各資料集支撐的 claim 與限制邊界 | 可用；資料集定位圖 |
| Fig. 07 | `img/07_metaqa_hop_analysis.pdf` | 21.2 | MetaQA hop-level 分析，凸顯 long-hop evidence structure | 可用；補充診斷圖 |
| Fig. 08 | `img/08_hrg_grammar_extraction.pdf` | 13.1、21.A | Offline HRG-like grammar extraction 流程 | 可用；方法圖 |
| Fig. 09 | `img/09_chain_validation_algorithm.pdf` | 9、22.2、22.3 | KB relation-label validation 與 strict spine construction | 可用；演算法圖 |
| Fig. 10 | `img/10_failure_counts_spine_correction.pdf` | 21.4、24 | failure analysis 與 hard dataset limitation | 可用；限制分析圖 |
| Fig. 11 | `img/11_dataset_semantics_examples.pdf` | 18、23 | MetaQA、WikiMovies、MLPQ、KQAPro 的語意差異 | 可用；資料集說明圖 |
| Fig. 12 | `img/12_evaluation_design.pdf` | 19、20 | 評估設計：answer quality、evidence quality、cost、failure | 可用；評估框架圖 |
| Fig. 13 | `img/paper_figures/13_offline_online_architecture.pdf` | 1.1.2、6、13、14 | Offline + online 兩段式主架構 | 可用；主方法圖 |
| Fig. 14 | `img/paper_figures/14_offline_grammar_statistics.pdf` | 21.A | Offline grammar extraction 統計 | 可用；結果圖 |
| Fig. 15 | `img/paper_figures/15_grammar_perturbation_trends.pdf` | 21.B、21.5 | KG perturbation 下的 grammar rule 與 relation retention 趨勢 | 可用；robustness 圖 |
| Fig. 16 | `img/paper_figures/16_evidence_coverage_hard_cases.pdf` | 21.6 | MLPQ / KQAPro hard cases 的 evidence coverage 改善 | 可用；HRG 存在感主圖 |
| Fig. 17 | `img/paper_figures/17_online_token_proxy.pdf` | 21.6.1 | online token proxy，補充 final context token 之外的成本 | 可用；成本分析圖 |
| Fig. 18 | `img/paper_figures/18_bootstrap_hard_case_effects.pdf` | 21.6.2 | paired bootstrap CI，檢查 HRG-Proposed 相對 Spine-Correction 的效果 | 可用；統計圖 |
| Fig. 19 | `img/paper_figures/19_prior_pilot.pdf` | 21.6.3 | HRG score 與 simple relation prior 的 candidate-level 比較 | 可用；prior 對照圖 |

圖片使用規則：

1. 方法圖與案例圖可以先保留，但 caption 要標註它們是 conceptual / case-study figure，不是四模型平均結果。
2. 所有帶數字的圖必須等新 rerun 完成後從同一個 summary dump 重新產生；不要把舊 dump、single-model diagnostic 與 four-model average 放在同一張圖或同一段落直接比較。
3. Fig. 04、05、10、14、15、16、17、18、19 是數據圖，重生時要在 caption 裡標註 `four-model average`、`single-model diagnostic`、`question-level n` 或 `question-model-pair-level n`。
4. KQAPro 圖不應用「hard-case success」語氣；caption 應寫成 stress-test evidence recovery over strict-spine ablations。

### 1.2 研究動機與方法直覺

本節用較直覺的方式說明 HRG-Proposed KG-RAG 的問題來源、方法設計與實驗解讀，作為後續正式演算法定義的背景。

#### 1.2.1 KGQA 問題設定

KGQA 的任務是：

```text
給一個問題 + 一個知識圖譜，系統要從圖裡找到答案。
```

例如：

```text
Question:
who directed the films starred by [Linda Evans]?

KG facts:
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen

Answer:
Andrew V. McLaglen
```

人類看這題會想：

```text
先找 Linda Evans 演過哪部電影
再找那部電影是誰導演
```

也就是：

```text
Linda Evans --starred_actors[r-]--> Movie --directed_by[r+]--> Director
```

這就是一條 relation chain。

#### 1.2.2 原本最直覺的做法：BFS KG-RAG

傳統 KG-RAG 很直覺：

```text
1. 從問題找 topic entity，例如 Linda Evans
2. 從 Linda Evans 在 KG 裡往外做 BFS
3. 把附近幾跳的 facts 全部拿出來
4. 丟給 LLM 回答
```

這個方法的好處是簡單，而且常常真的會把答案附近的 facts 抓進來。

但問題是：

```text
BFS 抓的是 neighborhood，不是 reasoning path。
```

它可能抓到：

```text
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen
Mitchell --starred_actors--> Martin Balsam
Mitchell --starred_actors--> Joe Don Baker
Mitchell --starred_actors--> John Saxon
Mitchell --release_year--> 1975
...
```

其中只有前兩條真的跟答案 path 有關，其他是 context noise。

所以一開始的研究動機可以理解成：

```text
BFS 很會「包答案」，但不一定很會「告訴 LLM 哪條路是答案依據」。
```

#### 1.2.3 核心想法：以 relation spine 取代大子圖

如果問題本身可以被理解成一條 relation chain，那就不一定要 BFS 抓一大包 facts。

可以先讓 LLM 做這件事：

```text
Question:
who directed the films starred by [Linda Evans]?

LLM parses:
entity = Linda Evans
chain = [starred_actors, directed_by]
```

然後系統不是直接相信 LLM，而是拿這條 chain 去 KG 裡走：

```text
Start: Linda Evans
Hop 1: starred_actors[r-] -> Mitchell
Hop 2: directed_by[r+] -> Andrew V. McLaglen
```

如果走得通，就只保留這條路上的 facts：

```text
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen
```

這就是本文所稱的 evidence spine。

因此，本方法不是：

```text
LLM 直接回答
```

也不是：

```text
BFS 抓全部附近 facts
```

而是：

```text
LLM 提出可驗證的 relation chain
KB 檢查這條 chain 能不能走通
只把走通的 evidence spine 給 LLM 回答
```

#### 1.2.4 為什麼這樣會減少 token

BFS 的 context 來源是：

```text
topic entity 周圍幾跳內的很多 facts
```

Spine 的 context 來源是：

```text
selected relation chain 真正走到的 facts
```

所以如果 BFS 抓到 300 多條邊，而 spine 只需要 9 條左右，context tokens 自然會大幅下降。

MetaQA 結果就是這樣：

```text
BFS avg context tokens: 4429.98
HRG-Proposed-triple avg context tokens: 189.55
Spine-Correction-triple ablation avg context tokens: 287.79
```

這是合併四個有效模型後的平均。HRG-Proposed-triple 在 MetaQA 上用 BFS 4.28% 的 context 取得高於 BFS 的 F1；Spine-Correction-triple 則作為 ablation，顯示 strict executable spine 本身也能大幅壓縮 context。

直覺上就是：

```text
把「大包鄰居資料」換成「一條可執行證據路徑」。
```

#### 1.2.5 為什麼這樣會增加可解釋性

本文的可解釋性不是指解釋 LLM 內部推理過程。

系統能解釋的是 retrieval evidence：

```text
這題被解析成哪個 entity？
這題被解析成哪條 relation chain？
這條 chain 在 KB 上有沒有走通？
最後給 LLM 的 facts 是哪些？
```

例如：

```text
selected_entity = Linda Evans
selected_chain = starred_actors -> directed_by

spine_edges:
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen
```

可表述為：

```text
這個問題滿足 actor-to-movie-to-director 的 KG structure。
系統不只輸出答案，也能把這個 structure 印出來。
```

#### 1.2.6 為什麼需要 KB validation

LLM 可能會猜錯 relation chain。

例如它可能猜：

```text
Linda Evans -> directed_by -> ...
```

但 Linda Evans 是演員，不是電影，這條 chain 在 KG 上可能走不通。

因此系統必須做 KB validation：

```text
從 grounded entity 出發
一個 relation 一個 relation 往下走
如果某一步 frontier 變空，這條 chain 就 invalid
```

這讓方法比「LLM 直接產生解釋」更可靠，因為 evidence 必須對得上 KB。

#### 1.2.7 為什麼需要 correction

如果 LLM 一開始給的 candidate chains 全部走不通，那系統不能直接放棄。

因此需要 fallback correction：

```text
1. LLM correction：請 LLM 根據失敗情況修正 chain
2. grammar fallback：用 grammar 裡常見 relation pattern 補候選
3. deterministic KG-valid fallback：從 KG adjacency 枚舉一定走得通的 chain
```

Correction 的觸發條件如下：

```text
Correction 不是每題都做。
只有所有 initial candidates 都 invalid 時才做。
```

#### 1.2.8 這個研究的演算法分成 offline 和 online 兩段

從演算法觀點，本文方法分成 offline 與 online 兩段。

第一段是 offline：

```text
KB triples
-> 建圖
-> 抽 HRG grammar
-> 得到常見 KG 結構規則
```

這段不看每一道問題，而是先從整個 KG 裡學「常見 relation structure」。

第二段是 online：

```text
Question
-> entity grounding
-> candidate chain generation
   (HRG-Proposed: LLM parse entity + relation chain)
   (HRG-GrammarFirst: HRG path-bank + KG executable filtering)
-> KB validation
-> correction / fallback / optional rerank
-> build spine subgraph
-> serialize context
-> LLM answer
-> evaluate answer and evidence
```

這段才是一題一題回答問題。

也就是：

```text
Offline = learn structural priors from the KG.
Online = use entity grounding, KG validation, ranking, and optionally LLM parsing / reranking
         to retrieve a compact executable evidence spine.
```

#### 1.2.9 KG 的基本資料結構是什麼

所有演算法都建立在 triples 上。

一條 triple 是：

```text
(head entity, relation, tail entity)
```

例如：

```text
(Mitchell, starred_actors, Linda Evans)
(Mitchell, directed_by, Andrew V. McLaglen)
```

演算法需要兩種圖資料結構。

第一種是 retrieval 用的 adjacency index：

```text
OutAdj[head][relation] = {tail1, tail2, ...}
InAdj[tail][relation]  = {head1, head2, ...}
```

它的用途是快速回答：

```text
從目前 entity 經過 relation r 可以到哪些鄰居？
```

第二種是 grammar extraction 用的 labeled directed multigraph：

```text
G = (V, E, R)
E contains labeled directed edges (u, r, v)
```

它保留完整 relation label，後面才能從局部子圖抽出 graph grammar rules。

#### 1.2.10 HRG grammar 是怎麼抓出來的

HRG grammar 可以先用一句話理解：

```text
從 KG 的局部子圖裡，抽出常見的 labeled graph production rules。
```

完整流程是：

```text
1. Build a labeled directed graph from triples
2. Convert it to a structural skeleton for decomposition
3. Sample local subgraphs with capped BFS
4. Triangulate each sampled graph
5. Build a clique tree
6. Convert clique tree bags into HRG production rules
7. Merge duplicate rules and count their frequency
```

下面逐步解釋。

**Step 1：把 KB triples 讀成 labeled directed MultiDiGraph**

例如：

```text
Mitchell | starred_actors | Linda Evans
Mitchell | directed_by | Andrew V. McLaglen
```

會變成：

```text
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen
```

為什麼用 `MultiDiGraph`：

```text
同一對 entity 之間可能有多種 relation，所以不能只用普通 Graph。
```

**Step 2：建立 graph skeleton**

HRG 抽結構時，先把 KG 局部子圖轉成可做 clique decomposition 的 graph skeleton：

```text
labeled KG local subgraph -> graph skeleton for clique decomposition
```

用途：

```text
後面要做 BFS sampling、MCS ordering、triangulation、clique tree。
這些圖結構操作在 structural skeleton 上做。
```

**Step 3：robust BFS sampling**

完整 KG 太大，不能直接對整張圖抽 grammar，所以先抽局部子圖 samples。

預設設定：

```text
K_SAMPLES = 4
S_SAMPLE_SIZE = 500
SEED_DEGREE_QUANTILE = 0.80
BFS_MAX_BRANCH = 30
```

意思是：

```text
抽 4 個 BFS samples
每個 sample 目標約 500 個 nodes
seed 避免選 degree 太高的 hub
每個 node BFS 展開最多看 30 個鄰居，避免 hub explosion
```

為什麼要這樣：

```text
KG 裡常有 hub node，例如年份、類別、熱門 entity。
如果 BFS 不限制，sample 會爆炸，clique extraction 也會爆炸。
因此 extractor 採用 capped BFS，而不是無限制展開局部子圖。
```

**Step 4：對 sample 做 MCS ordering**

MCS 是 Maximum Cardinality Search。

直覺上它是在 sample graph 上找一個 node elimination order，方便後面 triangulation。

可概括為：

```text
本文使用 MCS 取得穩定的 elimination ordering，避免直接找所有 maximal cliques 的高成本。
```

**Step 5：triangulation 並找 clique candidates**

HRG extraction 需要把圖分解成 clique tree。

做的事：

```text
按照 MCS order 消去 node
把 later neighbors 補成 clique
產生 maximal clique candidates
```

這一步的重點是：

```text
不用 nx.find_cliques() 暴力找 clique，而是用 elimination order 快速產生 clique candidates。
```

**Step 6：建立 clique tree**

拿到 cliques 後，建立 clique graph：

```text
每個 clique 是一個 node
兩個 clique 的交集大小是 edge weight
```

然後取 maximum spanning tree。

為什麼取 maximum spanning tree：

```text
希望 clique tree 中相鄰 bags 共享的 nodes 盡量多，保留 tree decomposition 的連接性。
```

**Step 7：binarize 和 prune**

clique tree 可能某個 node 有很多 children，不方便抽二元 derivation。

所以做：

```text
把 clique tree 轉成每個 node 最多兩個 children。
```

接著：

```text
刪掉沒有新 internal nodes 的 leaf bag，避免產生沒有意義的 rules。
```

**Step 8：從 clique tree 抽 HRG rules**

這是最核心的 grammar extraction。

每個 clique tree node，也就是每個 bag，會產生一條 rule。

對某個 bag：

```text
bag = 這個 clique 裡的 nodes
parent bag = 父節點 clique
external nodes = bag 和 parent bag 的交集
internal nodes = bag 裡扣掉 external nodes
```

如果是 root：

```text
LHS = S / 0
```

如果不是 root：

```text
LHS = N / rank
rank = external nodes 數量
```

RHS 由兩部分組成：

1. terminal edges：

```text
這個 bag 中負責的 KG edges，例如 (0, starred_actors, 1)
```

2. nonterminal children：

```text
子 bag 形成的 nonterminal，例如 N/2(attached nodes)
```

rule 大概長這樣：

```text
LHS:
N / rank

RHS:
terminal_edges = [
  (a, relation, b),
  ...
]
nonterminals = [
  N / rank with attachment nodes
]
```

抽象資料結構：

```text
Nonterminal(name, rank)
RHS(terminals, nonterms)
Rule(lhs, rhs, count)
```

**Step 9：合併重複 rules 並計數**

不同 samples 可能抽出一樣的 rule。

演算法會把 RHS canonicalize：

```text
把 terminal edges 排序
把 nonterminal attachments 排序
用 canonical signature 判斷兩條 rule 是否相同
```

最後每條 rule 有：

```text
count = 這個 rule 出現幾次
```

這個 count 後面會被當成 grammar score / frequency prior。

**Step 10：得到 grammar rule set**

一條 rule 的抽象長相大概是：

```json
{
  "lhs": {"name": "N", "rank": 2},
  "rhs": {
    "terminals": [
      {"a": 0, "rel": "starred_actors", "b": 1}
    ],
    "nonterms": [
      {"name": "N", "rank": 1, "att": [1]}
    ]
  },
  "count": 3
}
```

可以用文字理解成：

```text
N/2 count=3
  T: [(0, 'starred_actors', 1)]
  N: [('N', 1, (1,))]
```

這就是 grammar 怎麼被「抓出來」的完整過程。

#### 1.2.11 抽出的 grammar 在線上怎麼用

HRG grammar 可被視為：

```text
從 KG 裡統計出常見的 relation 結構 pattern。
```

在本文的 HRG-Proposed 設定中，它不是事後附加的裝飾，而是主方法的 structural prior：

```text
哪些 relation 常一起出現？
哪些 chain 比較像 KG 裡常見結構？
invalid 時能不能從 grammar 產生 fallback candidate？
哪些 KG-valid fallback candidate 比較有結構可信度？
```

因此本文需要把主從關係講清楚：

```text
HRG-Proposed 是本文主方法。
Spine-Only 是去除 correction 與 HRG prior 的 strict spine ablation。
Spine-Correction 是保留 correction、但不使用 HRG-guided candidate selection / fallback signals 的 ablation。
HRG-GrammarFirst 是新增 grammar-first variant，用來檢驗 HRG 是否能主動搜尋候選 chains。
```

online 階段會先把 grammar rule 轉成可快速比對的索引：

```text
每條 rule -> terminal relation labels
每條 rule -> relation label counts
每條 rule -> terminal edge count
```

這樣之後看到一條 LLM 預測的 chain 時，可以快速問：

```text
這些 relation 是否出現在某個 grammar rule 中？
是否有相同 hop 數的 rule？
是否存在 ordered path structure？
```

之後主要有三種用途。

**用途 1：match relation chain**

如果 LLM parse 出：

```text
["starred_actors", "directed_by"]
```

grammar matcher 會找 grammar rules 裡是否包含這些 relation labels。

目前主要不是完整 HRG decoding，而是：

```text
label subset match
ordered path match
same arity preference
```

因此本文將此界定為：

```text
grammar 在目前系統中是 structural prior，不是完整圖文法推導器。
```

**用途 2：candidate rerank**

candidate ranking 時會看：

```text
grammar_hit
same_arity_hit
ordered_path_hit
grammar_score
matched_count
```

這些會和 KB executability、LLM confidence、failure hop 等一起排序。

但最重要永遠是：

```text
valid chain 優先
```

**用途 3：fallback / hard-case candidate support**

當 initial candidates 都 invalid 時，可以用 grammar fallback 產生修正候選：

```text
從 matched grammar rules 中取出常見 relation labels
重新組成候選 relation chain
再交給 KB validation 檢查
```

Grammar fallback 本身不是 dominant source；更主要的是 HRG-related features 會出現在 selected candidate 的 `grammar_hit`、`matched_rules`、`grammar_score` 與 KG-valid fallback ranking 中。這些訊號在 KQAPro / MLPQ 這類 strict spine 容易失效的資料集中特別重要。

#### 1.2.12 Online 一題問題實際怎麼跑

以 Linda Evans 這題為例：

```text
Question:
who directed the films starred by [Linda Evans]
```

**Step 1：LLM parse top-k candidates**

prompt 會要求 LLM 回 JSON：

```json
[
  {
    "entity": "Linda Evans",
    "chain": ["starred_actors", "directed_by"],
    "confidence": 0.82
  }
]
```

演算法上會做：

```text
提供 relation shortlist
提供 relation alias guide
要求 bare relation names
抽取 balanced JSON
做 relation fuzzy match
丟掉不在 KG relation vocabulary 的 relation
去除重複 candidates
```

**Step 2：entity grounding**

會嘗試：

```text
exact match
lowercase
space / underscore variants
alias map
token overlap fallback
```

目的：

```text
"Linda Evans" 要對到 KG node。
```

**Step 3：KB validation**

執行：

```text
frontier = {Linda Evans}

relation = starred_actors
找到 Mitchell

relation = directed_by
找到 Andrew V. McLaglen

chain valid
```

**Step 4：candidate ranking**

排序優先考慮：

```text
valid chain
LLM rerank score
same-arity grammar hit
ordered grammar hit
grammar score
failure progress
step survival
source priority
LLM confidence
```

核心是：

```text
能在 KB 上走通的 candidate 才有資格成為 evidence。
```

**Step 5：build strict spine**

會沿著 selected chain 收集 edges：

```text
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen
```

這些就是：

```text
spine_edges
final_edges
```

**Step 6：serialize context**

JSON format：

```json
[
  {"head": "Mitchell", "relation": "starred_actors", "tail": "Linda Evans"},
  {"head": "Mitchell", "relation": "directed_by", "tail": "Andrew V. McLaglen"}
]
```

Triple format：

```text
Mitchell starred_actors Linda Evans. Mitchell directed_by Andrew V. McLaglen.
```

重要：

```text
json / triple 不改 retrieved edge set，但會改變 answer-generation interface。
因此 serialization 不是純外觀；它可能影響 LLM 是否讀懂 evidence。
```

**Step 7：LLM answer**

prompt 要求：

```text
只能根據 context 回答
只輸出 final answer
不要 reasoning
不要 markdown
多答案用 |
context 不足就回答 I don't know
```

最後輸出：

```text
Andrew V. McLaglen
```

**Step 8：輸出 evaluation diagnostics**

系統輸出以下 evaluation diagnostics：

```text
question
selected_entity
selected_chain
edges
spine_edges
answer
token_usage
retrieval_recall / precision / f1
subgraph_size
failure_stage
final_context
```

因此可解釋性不是事後補上的文字說明，而是 pipeline 每題都會產生的 evidence diagnostics。

#### 1.2.13 Baseline-BFS 的演算法

流程：

```text
1. LLM parse topic entity
2. resolve entity to KB node
3. 從 entity 做 BFS 到指定 depth
4. 每 hop 收集 edges
5. 用 max_edges_per_hop / max_frontier 防止爆炸
6. serialize 全部 edges
7. LLM answer
```

它不需要 relation chain，也不看 grammar。

所以 BFS 的優勢是：

```text
不怕 LLM chain parse 錯
只要 entity 對，附近 facts 容易被抓到
```

但代價是：

```text
context 大
reasoning path 不明確
LLM 必須從大量 facts 中自行定位答案
```

#### 1.2.14 評估演算法怎麼設計

評估不是只問「答案有沒有對」，而是同時問三件事：

```text
1. answer quality 是否貼近 BFS
2. context 是否比 BFS 小
3. evidence structure 是否能被檢查
```

因此每個方法都用同一批問題做：

```text
for each question:
    retrieve evidence
    generate answer
    compare answer with gold answers
    record retrieved edges and token cost
    record failure stage
```

主要 metrics：

```text
EM
Hits@1 / Hits@3 / Hits@5
MRR
answer_set_precision / recall / F1
avg_ctx_tokens
avg_subgraph_size
avg_retrieval_recall / precision / F1
failure_counts
answerable_rate
```

指標定義如下：

```text
avg_ctx_tokens 是 final context 的估計 token 數。
avg_subgraph_size 是最後 retrieved edges 數量。
compression_vs_bfs_ctx_ratio 是和同 backbone BFS 比例。
```

#### 1.2.15 實驗是怎麼設計的

實驗其實是在問三件事：

```text
1. 答案品質有沒有貼近 BFS？
2. context tokens 有沒有下降？
3. retrieval evidence 能不能被印出來檢查？
```

因此比較以下方法：

```text
Baseline-BFS
Degree-Capped-BFS
Token-Budgeted-BFS
Spine-Only
Spine-Correction
Spine-Correction-KGValidFallback
HRG-Proposed-NoExpansion
HRG-Proposed-Expansion
HRG-GrammarFirst-NoExpansion
HRG-GrammarFirst-Expansion
RelationUnigram / RelationBigram / RelationTrigram
```

並且看：

```text
EM / F1
avg context tokens
avg subgraph size
retrieval recall / precision
failure counts
case-level diagnostics
```

評估重點不只看 EM/F1，因為本文主題不是單純 accuracy，而是：

```text
貼近 BFS answer quality + 大幅減少 token + 提供 evidence-level explainability
```

因此結果章的主表應拆成三張，而不是只放一張 EM 排名表：

1. **Answer quality table**：EM、Hits@1、answer-set F1，用來承認最後答案表現。
2. **Evidence-cost table**：retrieval recall@5、claim faithfulness、avg context tokens、avg subgraph size、compression ratio，用來回答 HRG retrieval 是否有價值。
3. **Ablation / hard-case table**：Baseline-BFS、Spine-Only、Spine-Correction、HRG-Proposed 在 MetaQA、MLPQ、KQAPro 的比較，用來顯示 HRG prior 在 strict spine 不足時是否補回 coverage。

本文不主張「EM 不重要」，而是採用以下更完整的評估敘事：

```text
EM measures final answer exactness, but it cannot distinguish whether an error comes from entity grounding,
relation-chain parsing, evidence retrieval, context serialization, or LLM generation.
Because this thesis studies compact KG evidence retrieval, EM is reported together with evidence recall,
faithfulness, context size, subgraph size, and failure counts.
```

目前結果最能支持的結論不是「所有資料集 EM 第一」，而是：

```text
MetaQA: HRG-Proposed-triple uses 4.28% BFS context and improves F1 over BFS.
KQAPro: HRG-Proposed-triple recovers executable evidence and F1 over strict spine ablations, but both HRG and BFS remain low; treat it as an out-of-scope stress test.
MLPQ: HRG-Proposed-triple improves retrieval recall@5 from Spine-Correction-triple 0.4210 to 0.5628.
WikiMovies: dataset is mostly 1-hop; BFS is already compact, so it is a sanity-check dataset rather than the main compression evidence.
```

#### 1.2.16 為什麼結果會長這樣

結果差異主要由資料集決定。

MetaQA 很適合，因為它就是乾淨的 relation chain：

```text
actor -> movie -> director
movie -> writer -> movie -> genre
```

WikiMovies 多數是 1-hop，BFS 已經很小，所以改善空間有限。

MLPQ 雖然是 path QA，但會跨語言：

```text
English question -> English DBpedia relation -> Chinese entity -> Chinese relation
```

所以難點變成 relation/entity 對齊。

KQAPro 更難，因為很多問題不是 chain，而是 program：

```text
count
filter
compare
verify
qualifier
```

因此本文的適用邊界是：

```text
當問題能表示成 executable relation chain，方法很有效。
當問題需要跨語言對齊、operator、qualifier semantics，單純 spine retrieval 就不夠。
```

#### 1.2.17 核心演算法整理

核心演算法可整理如下：

| 演算法 | 輸入 | 輸出 | 目的 |
|---|---|---|---|
| HRG grammar extraction | KG triples | grammar rules with counts | 學 KG 常見結構 prior |
| LLM relation-chain parsing | question + relation vocabulary | top-k entity/chain candidates | 把自然語言問題轉成可驗證 retrieval intent |
| HRG-GrammarFirst candidate generation | question + entity seed + KG adjacency + grammar | 1..D hop KG-valid chain candidates | 不依賴 LLM hop/chain，直接從 KG-valid search space 產生候選 |
| Entity grounding | entity string + KG nodes | grounded node | 找 retrieval 起點 |
| Chain validation | grounded node + relation chain + KG | valid/invalid, frontier sizes | 檢查 chain 是否真的可執行 |
| Fallback correction | failed candidates + KG/grammar | revised candidates | 補救 LLM parse 失敗 |
| Candidate ranking | candidates + KB result + grammar features | ranked candidates | 選較可信且可執行的 chain |
| Strict spine retrieval | valid chain + KG | spine edges | 只取 chain 上的 evidence |
| Context serialization | final edges | JSON/triple context | 讓 LLM 可以讀 evidence |
| Answer generation | question + context | answer string | 根據 compact evidence 回答 |
| Evaluation | predicted answer + gold + evidence stats | EM/F1/tokens/failures | 驗證 answer quality、token reduction、可解釋性 |

#### 1.2.18 一句話總結

本文的一句話總結是：

```text
這篇研究不是要讓 LLM 看更多資料，而是讓 LLM 看更少但結構更清楚、可以被 KB 驗證的 evidence。
```

簡潔表述：

```text
本文將 KG-RAG 的 retrieval evidence 從 BFS 大子圖，轉為可執行 relation spine。
這讓 context tokens 大幅下降，也讓每題的 evidence structure 可以被檢查。
```

---

## 2. 系統整體目標

### 2.1 輸入

系統每次回答需要以下輸入：

1. 一個自然語言問題 `q`
2. 一個知識圖譜 `G`
3. 可選的 relation list
4. 可選的 alias mapping
5. 可選的 HRG grammar
6. 一個 LLM backbone

演算法上，這些輸入會被轉成兩類資料結構：

```text
1. KG adjacency index：支援快速 chain execution
2. Grammar rule set：提供結構 prior 和 fallback candidates
```

### 2.2 輸出

每題輸出包含：

1. 最終答案
2. selected entity
3. selected relation chain
4. retrieved evidence edges
5. parse、correction、generation token 使用量
6. retrieval recall / precision / F1
7. subgraph size
8. failure stage
9. case-level diagnostics

這些輸出用來回答三個研究問題：

```text
1. answer quality 是否貼近 BFS？
2. context tokens / subgraph size 是否下降？
3. retrieved evidence 是否能被印出來檢查？
```

---

## 3. 演算法架構對應

整個研究可以拆成四個演算法模組，而不是用檔案或執行指令理解。

| 模組 | 輸入 | 輸出 | 目的 |
|---|---|---|---|
| Offline grammar learning | KG triples | HRG-like grammar rules with counts | 學 KG 裡常見 relation structure |
| Baseline BFS retrieval | question + KG | BFS neighborhood subgraph | 強 baseline：高 recall，但 context 大 |
| Spine-Correction retrieval | question + KG + optional grammar | executable spine subgraph | 用小 context 取得可檢查 evidence |
| Evaluation | predicted answers + gold answers + evidence stats | EM/F1/tokens/failure analysis | 驗證貼近 baseline、token reduction、可解釋性 |

演算法流程：

```text
KB triples
  -> learn grammar rules from local graph structures

Question
  -> parse entity + relation chain
  -> ground entity
  -> validate chain over KG
  -> retrieve strict spine evidence
  -> generate answer from compact context
  -> compare with BFS baseline
```

---

## 4. 問題定義

令知識圖譜為：

```text
G = (V, E, R)
E subset of V x R x V
```

其中：

1. `V` 是 entity / node 集合
2. `R` 是 relation label 集合
3. `E` 是 triples，例如 `(Tom_Hanks, starred_actors, Cast_Away)`

給定問題 `q`，系統希望找出答案集合 `Aq`。

傳統 BFS baseline 的目標是：

```text
從 topic entity e 出發，展開 h hop，收集所有鄰近 triples。
```

Spine-Correction 的目標是：

```text
找出一條從 topic entity 出發、在 KG 上可執行的 ordered relation chain，
並只取回該 chain 對應的核心 evidence spine。
```

形式化表示：

```text
LLM parser output:
Cq = {(entity_i, chain_i, confidence_i)}

chain_i = [r1, r2, ..., rL]

Frontier_0 = {ground(entity_i)}
Frontier_t = {v' | exists v in Frontier_{t-1},
                  (v, rt, v') in E or (v', rt, v) in E}

direction_t(v, v') =
  rt[r+] if the executed stored KG triple is (v, rt, v')
  rt[r-] if the executed stored KG triple is (v', rt, v)

valid(entity_i, chain_i) = true
iff Frontier_t is non-empty for every t = 1 ... L
```

也就是說，LLM 預測的 relation chain 不只是一段文字，而是必須能在 KB 裡真的走通。

### 4.1 Retrieval Evaluation Objective

Retrieval evaluation 不能只說「compact evidence context」，必須明確定義什麼叫好的 retrieval。這裡的 objective 是 evaluation objective / reporting frame，不是訓練出的 differentiable surrogate，也不是 online system 實際最佳化的單一分數。Online selection 實際使用的是 Section 11 的 deterministic lexicographic ranking key。

```text
Given question q and KG G, retrieve evidence subgraph S
that maximizes expected answer utility and evidence faithfulness,
while minimizing retrieval and generation cost.

maximize:
  U(q, S) = AnswerQuality(q, S)
          + lambda_e * EvidenceExecutability(q, S)
          + lambda_g * GrammarCompatibility(q, S)

subject to:
  Cost(q, S) <= B
```

其中 `Cost(q, S)` 應包含：

1. final context tokens：送進 answer generator 的 evidence tokens
2. parse / correction / rerank tokens：online LLM calls 的額外成本
3. retrieval latency：entity grounding、KB validation、fallback enumeration 與 subgraph construction 的時間
4. evidence edge count：最後輸出的 triples / edges 數量

本文完整報告 final context tokens、latency 與 subgraph size；parse / correction / rerank 成本以 online token proxy 補充，因此 efficiency claim 限定為 **final evidence context compactness** 與 online cost proxy，而非完整 API cost 最佳化。

### 4.2 Relation-Chain-Friendly 的事前判定

為避免「做得好才說是 relation-chain-friendly」的事後解釋，本文把適用條件定義為可在執行前檢查的 dataset / question properties：

1. 問題可由 topic entity 出發，以長度 `L <= h` 的 ordered relation sequence 表示主要推理路徑。
2. gold answer 多數是 chain endpoint 或 endpoint 集合，而不是 count、argmax、comparison、negation、qualifier filtering 等 operator 的輸出。
3. KG relation label 與問題文字之間存在可辨識 lexical cue、alias 或固定模板。
4. 每一 hop 的 frontier branching 在 hub cap 內，不會讓 valid spine 退化成接近 BFS 的大子圖。
5. 若資料集提供 template、program 或 hop label，這些 metadata 可用來估計 chain-like ratio；若沒有，則用 sample-level parse-valid rate、no-candidates rate、no-valid-chain rate 作為 proxy，但必須標明這是 empirical diagnostic，不是定義本身。

依此定義，MetaQA 是最符合方法假設的資料集；WikiMovies 是 1-hop ceiling case；MLPQ 受到跨語言 canonicalization 影響；KQAPro 則大量包含 operator / qualifier semantics，因此 strict relation spine 不足，HRG-Proposed 的價值主要體現在補 evidence coverage，而不是完整解決 semantic program reasoning。

### 4.3 Relation Label Chain、Direction 與 Branching 定義

Prompt 的要求很單純：LLM 只輸出 **出現在 KB vocabulary 裡的 relation label**，例如：

```text
chain_i = [r1, r2, ..., rL]
rt in R
```

本文不要求 LLM 輸出額外的 relation operator 或不存在於 KB 的 relation name。KB validation 的任務是檢查這串 relation labels 是否能從 grounded topic entity 在 KB 上實際執行成功。

`r+ / r-` 只是一個 **derived traversal-direction notation**，不是新增的 reverse relation，也不是 parser 要輸出的 relation label。也就是說，KG vocabulary 裡仍然只有原本的 `r`；當系統從 stored triple 的 head 走到 tail 時，trace 可標成 `r+`，從 tail 走回 head 時，trace 可標成 `r-`。這個 notation 的用途是回應 relation direction 的可檢查性問題：同一個 relation label 被用於雙向 traversal 時，paper / case study / evidence table 需要標明實際走的是 stored KG triple 的哪個方向。

例子：

```text
Stored KG triples:
  Mitchell --starred_actors--> Linda Evans
  Mitchell --directed_by--> Andrew V. McLaglen

Question anchor:
  Linda Evans

Parser output relation labels:
  starred_actors -> directed_by

Derived signed execution trace:
  Linda Evans --starred_actors[r-]--> Mitchell --directed_by[r+]--> Andrew V. McLaglen
```

因此，若文件或圖中出現 `r+ / r-`，意思是「執行方向註記」，不是「資料中多了反向 relation」。主實驗仍應 report 原始 relation label；signed trace 只放在 validation log、case study 或 evidence appendix。

多答案 branching 的處理如下：

1. 每一 hop 保留所有 reachable frontier nodes，而不是只選一條路。
2. 一題的 evidence spine 是所有 valid paths 的 union；因此多答案問題可以保留多個 endpoint。
3. 為防止 frontier 爆炸，候選 ranking 會偏好較小 final frontier、較高 grammar score 與較高 textual relevance；若候選或 evidence 超過 budget，優先保留 strict spine edges。
4. 若多條 chain 均 executable，先依 KB-valid、LLM rerank、grammar hit、grammar score、step survival、final frontier size、source priority、LLM confidence 與原始順序排序。

---

## 5. 為什麼 BFS 不夠

BFS baseline 的流程是：

```text
Question
  -> LLM parse topic entity
  -> entity grounding
  -> BFS from entity up to depth h
  -> collect all edges
  -> serialize context
  -> LLM answer
```

它的優點：

1. 不需要 LLM 預測 relation chain
2. 對 1-hop 或局部很小的 KG 很強
3. answer node 容易被包含在 retrieved subgraph 中

它的問題：

1. context tokens 大
2. multi-hop noise 會快速累積
3. retrieval recall 高不代表 LLM 能用這些 context 正確回答
4. evidence path 不明確，解釋性較弱

可表述為：

```text
BFS 解決了「找得到」的問題，但沒有解決「LLM 能不能用」的問題。
```

---

## 6. Spine-Correction KG-RAG 主流程

Spine-Correction / HRG-Proposed 的 online 階段可以拆成六步。這條 path 是 **LLM-chain-guided**：先由 LLM 提出 candidate chains，再由 KB / HRG 驗證、補救與排序。

```text
1. Parse: LLM 解析 entity + relation chain candidates
2. Ground: 把 entity surface form 對齊到 KG node
3. Validate: 在 KB 上執行每條 chain，確認是否可走通
4. Correct: 如果全部 initial candidates 都 invalid，啟動 fallback correction
5. Retrieve: 對 valid chain 建立 strict spine subgraph
6. Answer: 把 subgraph 序列化後交給 LLM 產生答案
```

更完整的條件分支：

```text
Question q
  -> LLM parse top-k candidates
  -> entity grounding
  -> KB validation for each candidate
  -> if any initial candidate is valid:
       skip correction
       build subgraph from valid candidates
     else:
       generate correction candidates
       validate correction candidates
       if still no valid chain and deterministic fallback is enabled:
           enumerate KG-valid fallback chains
       if still no valid chain:
           return failure stage = no_valid_chain
  -> build subgraph for each valid candidate
  -> rank candidate subgraphs
  -> serialize best subgraph
  -> LLM answer generation
  -> write metrics and case-level diagnostics
```

### 6.1 HRG-GrammarFirst 主流程

新增的 `HRG-GrammarFirst` 則是另一條 online path。它是 **HRG path-bank first / KG-valid-chain generation**：不使用 LLM 產生的 relation chain 作為 search-space 前提，而是先從 offline HRG rules 抽出 relation-path signatures，再由 KG 驗證這些 paths 是否能從 entity seed 執行。

```text
Question q
  -> extract / ground topic entity candidates
  -> derive HRG relation path-bank from grammar rules
  -> execute path-bank candidates from each entity seed
  -> score chains by:
       relation cue coverage from the question
       relation-question relevance
       HRG label / same-arity / ordered-path hits
       grammar score / relation n-gram score
       frontier compactness
  -> optional LLM rerank over executable chains
  -> build subgraph for each valid candidate
  -> rank candidate subgraphs
  -> serialize best subgraph
  -> LLM answer generation
  -> write metrics and case-level diagnostics
```

因此，目前程式裡有兩種 HRG 使用方式：

1. `HRG-Proposed`：LLM candidate first，HRG 做 recovery / ranking / optional expansion。
2. `HRG-GrammarFirst`：entity first，HRG path-bank first，KG 只負責 executable filtering；這才是用來回答「LLM hop 猜錯時 HRG 能不能自己找」的新增 variant。

---

## 7. Step 1：LLM 解析 Relation Chain

這一步不是讓 LLM 直接回答，而是要求 LLM 輸出可驗證的查詢意圖。它適用於 `Spine-Only`、`Spine-Correction`、`HRG-Proposed` 與 relation-prior reranker rows；`HRG-GrammarFirst` 會跳過這一步的 relation-chain parsing，只保留 entity grounding / entity fallback。

輸出格式是 JSON array：

```json
[
  {
    "entity": "Cast Away",
    "chain": ["written_by", "written_by", "has_genre"],
    "confidence": 0.82
  },
  {
    "entity": "Cast Away",
    "chain": ["directed_by", "has_genre"],
    "confidence": 0.41
  }
]
```

設計理由：

1. 直接回答不可驗證
2. relation chain 可以拿到 KG 裡執行
3. top-k candidates 可以降低第一個解析錯誤造成的單點失敗
4. relation shortlist 與 alias guide 可以讓 LLM 更接近 KG vocabulary

具體做法：

1. 從 KG relation list 挑出 candidate relations
2. 根據問題文字與 entity 鄰近 relation 做 relation shortlist
3. 加入 relation alias guide
4. prompt 要求 LLM 只輸出 JSON array
5. parse LLM 回傳的 JSON
6. 對每個 relation token 做 fuzzy match
7. 移除非法 relation
8. 去除重複候選

如果 LLM 完全 parse 失敗，系統會嘗試從問題中用 `[...]` 抽 topic entity，作為 fallback seed。`HRG-GrammarFirst` 則一開始就採用這種 entity-first 設計：先解析 / 抽取 topic entity candidates，再由 KG/HRG 產生 chains。

---

## 8. Step 2：Entity Grounding

這一步把 LLM 輸出的 entity string 對到 KG 裡的 node。

例如：

```text
"Tom Hanks"
"Tom_Hanks"
"tom hanks"
```

都應該能對到同一個 KG node。

實作策略：

1. sanitize entity：移除中括號、處理空白與底線
2. exact match
3. lower-case match
4. space / underscore 互換
5. punctuation normalize
6. alias map lookup
7. token overlap fallback

這一步很重要，因為 relation chain 即使正確，只要起點 entity 對錯，後面 KB validation 仍然會失敗。

---

## 9. Step 3：KB Validation

Validation 的目的：

```text
確認 LLM 預測的 relation chain 是否能從 grounded entity 在 KG 上實際走通。
```

演算法：

```text
Input:
  entity e
  chain [r1, r2, ..., rL]

start = resolve_entity_to_kb(e)
frontier = {start}

for each relation rt in chain:
    next_frontier = {}
    for each node v in frontier:
        edges = execute_relation_step(v, rt)
        for each edge in edges:
            next_node = other_endpoint(v, edge)
            next_frontier.add(next_node)
            signed_trace.add(rt[r+] if v == edge.head else rt[r-])
    if next_frontier is empty:
        return invalid with failed_hop = t
    frontier = next_frontier

return valid with final_size = |frontier| and signed_trace
```

注意：`execute_relation_step(v, rt)` 只使用 KB vocabulary 中的 relation label `rt`。也就是說，parser 不需要產生任何額外 relation name；validation 只確認這個 label 對應的 KB step 是否存在，並把成功執行的 triples 收進 `spine_edges`。如果 validation 允許從 stored KG triple 的 tail 走回 head，系統只在 trace 裡加上 derived direction sign，例如 `rt[r-]`；這不是新增 reverse relation，也不改變 KG triple 本身。

Validation 回傳資訊包含：

1. `valid`
2. `step_sizes`
3. `final_size`
4. `failed_hop`
5. `resolved_chain`
6. `signed_execution_trace`，只作為可解釋輸出，不作為新的 relation vocabulary

這些資訊後續會用於候選排序。

---

## 10. Step 4：Fallback Correction

Correction 的觸發條件非常重要：

```text
只有當所有 initial candidates 都 invalid 時，才做 correction。
```

如果初始 top-k 裡已經有 valid chain，系統不會再浪費 token 做 correction，而是直接進入 subgraph construction。

Correction pool 來源：

1. LLM correction：把失敗候選與失敗 hop 提供給 LLM，請它產生修正版 chain
2. grammar fallback：用 HRG grammar 的 relation label match 產生候選
3. deterministic valid-chain fallback：從 KG adjacency 枚舉保證可執行的 relation chains
4. grammar-first candidate generation：只在 `HRG-GrammarFirst` row 啟用；它不是 fallback，而是主 candidate generator

Correction 不是盲目相信 LLM，而是每個 correction candidate 仍然要再次通過 KB validation。

### 10.1 Correction 與 Fallback 的可重現設定

為了讓 fallback correction 可重現，本文將流程固定為以下順序：

```text
Trigger:
  only if all initial top-k parse candidates are invalid

Candidate sources:
  1. LLM correction candidates from failed chain and failed hop
  2. HRG grammar label / ordered-path candidates
  3. deterministic KG-valid fallback chains from local adjacency

Validation:
  every candidate must be executed on KG before retrieval

Stop:
  if at least one valid candidate is found and ranked
  otherwise return failure = no_valid_chain or no_candidates
```

Deterministic KG-valid fallback 的風險是「可執行不等於語意正確」，因此它不能只靠 executability 排名。本文目前用以下訊號降低任意 path 被選中的風險：

1. relation label 是否和 question token / alias 對得上
2. 是否命中 HRG grammar rule 或 ordered relation pattern
3. path frequency / grammar score
4. frontier size 是否過大
5. candidate source priority
6. LLM confidence 或 rerank score

在 `HRG-GrammarFirst` 中，deterministic KG-valid chain enumeration 不再等到 all initial candidates invalid 才觸發，而是在最前面就觸發。這是它和 `HRG-Proposed` 的核心差異。

若候選完全沒有 textual relevance 或 grammar support，系統允許輸出 failure / abstain，而不是硬選一條任意可執行路徑。Failure counts 與 token metrics 一起報告，避免把低 token 的失敗案例誤當作成功壓縮。

---

## 11. Step 5：Candidate Ranking

所有 candidate 都會被轉成一個 ranking key。排序不是只看 LLM confidence，而是看一組可解釋因素。

排序優先順序大致如下：

1. KB executability：valid chain 優先
2. LLM rerank score：若 deterministic fallback 有啟用 LLM rerank
3. relation n-gram prior enabled / score：RelationUnigram、RelationBigram、RelationTrigram rows 使用
4. same-arity grammar hit
5. ordered path grammar hit
6. grammar label hit
7. grammar score
8. matched grammar rule count
9. low-information relation penalty
10. failed hop progress：失敗得越晚越好
11. step survival：每 hop frontier 是否還活著
12. final frontier size
13. source priority：LLM、correction、KG fallback、grammar-first 等來源不同
14. LLM confidence
15. LLM 原始排序

### 11.1 Ranking Key 的論文寫法

可重現版本可寫成 lexicographic ranking：

```text
rank(c) = (
  valid(c),
  llm_rerank_score(c),
  relation_ngram_enabled(c),
  relation_ngram_score(c),
  same_arity_grammar_hit(c),
  ordered_path_grammar_hit(c),
  grammar_label_hit(c),
  grammar_score(c),
  matched_rule_count(c),
  - low_information_chain(c),
  failed_hop_progress(c),
  step_survival(c),
  final_frontier_size(c),
  source_priority(c),
  llm_confidence(c),
  - original_index(c)
)
```

其中 `valid(c)` 是硬條件；沒有通過 KB validation 的 candidate 不能進入 final evidence retrieval。程式目前的 `final_frontier_size` 使用 capped positive signal，表示完全走通且 final frontier 非空的候選較穩；frontier 爆炸則主要由 fallback beam / branch cap、subgraph size、context edge cap 與 subgraph ranking 控制。若所有 valid candidates 排名分數相同，使用 deterministic original index tie-break，保證同一輸入可重現。

這個 ranking key 是可重現的工程排序規則，不是學到的 multi-objective optimizer。論文中應把它稱為 lexicographic candidate ranking，而不是宣稱系統直接最佳化 Section 4.1 的 objective。新增 ablation 應把 `KG-valid fallback`、`HRG grammar features`、`LLM rerank`、`serialization` 分開，避免把 fallback 帶來的 recovery 都說成 LLM correction 或 HRG decoding 的效果。

這樣設計的原因：

```text
KGQA retrieval 不能只依賴 LLM 的語意判斷，
也不能只依賴 grammar frequency。
最重要的是這條 chain 能不能在 KB 上真的執行。
```

---

## 12. Step 6：Spine Subgraph Construction

對每條 valid chain，系統會建立 strict spine。

Strict spine 是指：

```text
只收集沿著 selected relation chain 實際走到的 edges。
```

演算法：

```text
Input:
  start_entity
  relation_chain [r1, r2, ..., rL]

frontier = {start_entity}
edge_counts = {}

for relation rt in chain:
    next_frontier = {}
    for node in frontier:
        edges = neighbors(node, rt)
        for edge in edges:
            edge_counts[edge] += 1
            next_frontier.add(other endpoint)
    if next_frontier empty:
        break
    frontier = next_frontier

return edges in edge_counts
```

Spine 的優點：

1. context 小
2. relation path 明確
3. evidence 可解釋
4. 適合 multi-hop chain reasoning

---

## 13. HRG Grammar 的角色

HRG grammar 是 offline 從 KB 裡抽出的結構 prior。

### 13.1 Offline Grammar Extraction

抽取流程大致如下：

```text
KB triples
  -> parse triple line
  -> build labeled directed MultiDiGraph
  -> build structural skeleton
  -> robust BFS sampling
  -> triangulation / clique tree
  -> extract production rules
  -> count / score rules
  -> save hrg_grammar.json and hrg_grammar.txt
```

為了避免 hub node 造成 clique explosion，演算法採用：

1. seed degree quantile
2. capped BFS branching
3. random sampling
4. fast clique candidate extraction

### 13.2 Online Grammar Matching

grammar matching 目前支援：

1. label subset matching
2. ordered path matching
3. exact-size matching

目前主設定中，grammar matching 主要是 relation label subset prior，而不是完整精確的 HRG decoding。

因此結果表不能只 report 一個混合的 `grammar_hit`。新的 summary 應至少拆成四層：

1. `candidate_weak_label_hit_rate` / label-subset hit：candidate 的 relation labels 出現在某個 grammar rule 中。這是最寬鬆的 signal。
2. `candidate_ordered_path_hit_rate`：candidate 的 ordered relation path 被 grammar / extracted path pattern 支持。
3. `same_arity_grammar_hit`：candidate relation count 與 matched rule terminal arity 相容。
4. `full structural hit`：方向、角色、attachment order、external/internal node role 都相容；目前尚未作為主結果 claim，若未實作只能列為 future work 或 supplementary diagnostic。

這樣寫可以避免把寬鬆 relation-label match 包裝成完整 HRG structural recognition。本文目前最保守的說法是：HRG-like grammar 提供 soft structural prior；完整 direction/role-aware structural signature 是下一版要補強的 grammar stability / specificity 分析。

這點在論文中需要講清楚：

```text
在 HRG-Proposed 中，HRG grammar 主要扮演 retrieval prior / candidate-selection prior / fallback prior，
不是用 grammar 完全取代 LLM parser，也不是保證每條 chain 都由 grammar 生成。
在 HRG-GrammarFirst 中，HRG grammar 進一步參與 candidate generation / search-space ranking：
系統先枚舉 KG-valid chains，再用 grammar features 排序，而不是依賴 LLM 先猜 hop 數。
```

---

## 14. HRG-Proposed 與 Ablations 的差別

本文正式主軸以 **HRG-Proposed KG-RAG** 為主。

`HRG-Proposed` 是本文 main method family，它在 executable relation spine core 之外加入：

1. grammar-aware candidate selection features
2. grammar-hit / matched-rule diagnostics
3. deterministic KG-valid chain fallback
4. optional LLM rerank over KG-valid fallback chains

HRG-Proposed main-method 流程：

```text
parse top-k candidates
  -> validate
  -> correction if all invalid
  -> deterministic valid-chain fallback if still invalid
  -> rank candidates with grammar features
  -> build strict spine
  -> subgraph ranking
  -> answer generation
```

`HRG-GrammarFirst` 不是 `HRG-Proposed` 的改名，而是新增 variant。兩者差別如下：

| Method family | Candidate source | 是否依賴 LLM hop/chain | HRG 角色 | 目前程式 row |
|---|---|---|---|---|
| Spine-Only | LLM parse top-k chains | 是 | 不使用 | `Spine-Only-json/triple` |
| Spine-Correction | LLM parse + LLM correction | 是 | 不使用 | `Spine-Correction-json/triple` |
| Spine-Correction-KGValidFallback | LLM parse + deterministic KG-valid fallback | fallback 時不完全依賴 | 不使用 grammar | `Spine-Correction-KGValidFallback-json/triple` |
| HRG-Proposed | LLM parse + correction + KG-valid fallback | 主要仍依賴 LLM initial search space；fallback 可補一部分 | grammar-aware ranking / fallback scoring / optional expansion | `HRG-Proposed-*`、`HRG-Proposed-NoExpansion-*`、`HRG-Proposed-Expansion-*` |
| HRG-GrammarFirst | entity seed + HRG relation path-bank + KG executable filtering | 否，不依賴 LLM chain/hop 先驗 | grammar controls candidate search space + optional LLM rerank | `HRG-GrammarFirst-NoExpansion-*`、`HRG-GrammarFirst-Expansion-*` |

正式表述：

```text
HRG-Proposed 是本文主方法，強調 HRG structural prior-guided executable retrieval。
Spine-Only 與 Spine-Correction 是 ablation，用來分離 strict spine、correction fallback 與 HRG prior 的貢獻。
HRG-GrammarFirst 是新增 stronger variant，用來檢驗 HRG 是否能主動產生 / 搜尋候選 chains，而不是只 rerank LLM candidates。
```

---

## 15. Subgraph Ranking

系統不是找到第一條 valid chain 就直接回答，而是：

```text
對所有 valid candidates 建 subgraph
-> 對 subgraph 排序
-> 選一個最佳 subgraph
```

排序考量：

1. 是否有 edges
2. same-arity grammar hit
3. ordered path grammar hit
4. grammar hit
5. grammar score
6. candidate ranking key
7. subgraph support size
8. compactness

設計理由：

```text
一條 chain valid 只代表它在 KG 上走得通，
但不一定代表它最能回答問題。
因此需要再對 candidate subgraph 做第二層排序。
```

---

## 16. Context Serialization 與 Answer Generation

系統支援兩種 serialization：

### 16.1 JSON format

```json
[
  {"head": "Cast Away", "relation": "written_by", "tail": "William Broyles Jr."},
  {"head": "William Broyles Jr.", "relation": "written_by", "tail": "Apollo 13"}
]
```

### 16.2 Triple format

```text
Cast Away written_by William Broyles Jr. William Broyles Jr. written_by Apollo 13.
```

重要：`json` 與 `triple` 不改變 retrieved edge set，但它們不是純外觀差異，而是 answer-generation interface 的一部分。相同 evidence 用不同 serialization 交給 LLM，可能會改變模型是否讀懂 head / relation / tail、是否能處理多答案，以及是否誤把 relation label 當答案。因此主表必須把 method 與 serialization 分開列，不能只把 serialization 當成附註。

保守解讀：

1. `triple` 通常更省 token，適合 MetaQA 這種 schema 小、relation label 直覺的資料集。
2. `json` 明確保留欄位名，對 relation label 抽象、跨語言或需要 metadata 的資料集可能更穩，但 token 成本較高。
3. 若同一 retrieval method 的 JSON / triple 結果差很多，應解讀為 answer interface 效果，而不是 retrieval evidence 改變。

Answer prompt 明確要求：

1. 只能使用 context
2. 只輸出 final answer
3. 不輸出 reasoning
4. 不輸出 markdown
5. 多答案用 ` | ` 連接
6. context 不足時輸出 `I don't know`

---

## 17. Baseline 與 Ablation 方法

### 17.1 Baseline-BFS

流程：

```text
LLM parse entity
-> entity grounding
-> BFS up to hop depth
-> collect edges
-> serialize context
-> LLM answer
```

Baseline 的目的：

```text
代表傳統 KG-RAG / KAG retrieval：不使用 relation chain，不使用 grammar，只靠 entity neighborhood。
```

### 17.2 Spine-Only

演算法開關：

```text
不用 grammar rerank
不用 fallback correction
```

流程：

```text
LLM parse chain
-> KB validation
-> valid chain 建 strict spine
-> answer
```

它用來測試：

```text
只靠 LLM 預測 relation chain 是否足夠。
```

### 17.3 Spine-Correction

演算法開關：

```text
不用 grammar rerank
啟用 fallback correction
```

流程：

```text
LLM parse chain
-> KB validation
-> if all invalid, correction
-> strict spine
-> answer
```

它用來測試：

```text
KB validation + fallback correction 是否能改善 chain parse failure。
```

### 17.4 HRG-Proposed

演算法開關：

```text
啟用 grammar-aware rerank
啟用 grammar-hit candidate features
啟用 fallback correction
啟用 deterministic KG-valid chain fallback
啟用 valid-chain LLM rerank
```

它用來測試：

```text
grammar prior + KG-valid fallback + grammar-aware candidate selection 是否能在仍控制 token 的情況下，提供更多結構命中與可檢查 evidence。
```

### 17.5 HRG-GrammarFirst

演算法開關：

```text
啟用 use_grammar_first_retrieval
啟用 grammar-aware rerank
跳過 LLM relation-chain parsing
從 HRG grammar 抽 relation path-bank，再從 entity seed 驗證可執行 chains
啟用 optional valid-chain LLM rerank
可分 NoExpansion / Expansion
```

流程：

```text
entity extraction / grounding
-> derive HRG relation path-bank from grammar rules
-> filter to KG-valid relation chains from each entity seed
-> score by question-relation relevance + grammar features + compactness
-> optional LLM rerank executable candidates
-> build strict spine
-> answer
```

它用來測試：

```text
HRG prior 是否能主動改變 search space，而不是只在 LLM 已提出的 candidate set 裡 rerank。
```

這個方法會比 HRG-Proposed 更有「retrieval engine」味道，但成本與錯誤風險也更高：HRG path-bank 產生的 chain 都會再經 KG executable filtering，但可執行不代表語意正確；因此必須報 candidate count、context tokens、LLM rerank tokens、failure/OOM 與 retrieval coverage。

### 17.6 Relation-prior 與 fair BFS ablations

目前 full rerun 也包含：

1. `RelationUnigram`、`RelationBigram`、`RelationTrigram`：固定使用 candidate / fallback machinery，但把 HRG grammar scoring 改成 relation n-gram prior，用來檢查 HRG 是否只是 relation frequency prior。
2. `Degree-Capped-BFS-{50,100,200,500}`：限制 BFS 每 hop degree，建立更公平的 compact BFS baseline。
3. `Token-Budgeted-BFS-{200,500,1000}`：限制 BFS context token budget，檢查同等 token 下 BFS 是否能追上 HRG。

### 17.7 可延伸但目前不作主 claim 的消融

若時間有限，應優先補強的是 HRG prior 與 simpler relation-prior / token-budgeted graph retrieval baseline 的比較。這能直接回答「HRG 是否真的比普通 relation prior 有用」。

Expansion 相關 row 必須特別標清楚。本文主方法可以採用 no-expansion strict spine 作為主要敘事；`HRG-Proposed-Expansion`、`Spine-GrammarExpansion`、`RandomExpansion`、`FrequencyExpansion` 則作為老師要求的 controlled ablation，用來檢查「額外擴邊」是否真的帶來 evidence coverage，或只是增加 context / OOM。若主文不想把 expansion 包進方法，就不要把 expansion row 混名為 `HRG-Proposed`，而是固定寫成 `HRG-Proposed-NoExpansion` 與 `HRG-Proposed-Expansion`。

另外新增 `HRG-GrammarFirst` 作為更強的 grammar-first retrieval ablation。它不使用 LLM 產生的 relation chain / hop 數作為 search-space 前提，而是：

```text
Question
-> entity grounding
-> derive HRG relation path-bank from offline grammar
-> validate path-bank chains against KG from entity seeds
-> rank by grammar ordered-path hit / same-arity hit / grammar score / question-relation relevance
-> optional LLM rerank
-> KB validation and strict spine retrieval
```

這個 row 專門回答「如果 LLM 一開始 hop 猜錯，HRG 能不能自己找候選路徑？」。主表應把 `HRG-GrammarFirst-NoExpansion` 與 `HRG-GrammarFirst-Expansion` 和原本 `HRG-Proposed-*` 分開列，不要把它包裝成原方法已經具備的能力。

---

## 18. Dataset 說明

目前 pipeline 支援：

1. MetaQA
2. WikiMovies
3. MLPQ
4. KQAPro
5. WQSP
6. CWQ
7. Mintaka
8. custom dataset

主實驗資料集聚焦以下四類：

### 18.1 MetaQA

特性：

1. movie domain
2. 1-hop / 2-hop / 3-hop 清楚
3. relation-chain-friendly
4. 最適合檢驗 executable spine 與 HRG-Proposed compact retrieval 的核心假設

適合強調：

```text
如果問題真的能表示成 ordered relation chain，HRG-Proposed 可以用 HRG structural prior 與很小 context 保留可檢查的推理 evidence。
```

### 18.2 WikiMovies

特性：

1. 多數問題偏 1-hop
2. BFS 本身就很強
3. 原始 KB 可能有 composite tail，需要 normalization

適合強調：

```text
當 KG neighborhood 本來就小，BFS 是合理強 baseline，Spine 方法主要展現 context compactness。
```

### 18.3 MLPQ

特性：

1. multilingual KGQA
2. question language 與 KB relation language 可能不同
3. grounding 與 relation normalization 更困難

適合強調 limitation：

```text
Spine-Correction 目前對跨語言 relation canonicalization 仍不完整。
```

### 18.4 KQAPro

特性：

1. compositional reasoning 更複雜
2. statement / qualifier semantics 更重要
3. 不一定能用普通 relation chain 完整表示

適合強調 limitation：

```text
一般 relation spine 對 operator-heavy 或 qualifier-heavy 問題不一定足夠。
```

---

## 19. Evaluation Metrics

主要分成四類。

### 19.1 Answer Metrics

1. EM
2. Hits@1
3. Hits@3
4. Hits@5
5. MRR
6. Answer-set precision
7. Answer-set recall
8. Answer-set F1

### 19.2 Retrieval Metrics

1. avg retrieval recall
2. avg retrieval precision
3. avg retrieval F1
4. retrieval recall@k
5. retrieval nDCG@k

### 19.3 Efficiency Metrics

1. avg context tokens
2. avg subgraph size
3. compression vs BFS context ratio
4. compression vs BFS subgraph ratio
5. avg latency
6. parse latency
7. retrieval latency
8. generation latency

### 19.4 Failure / Faithfulness Metrics

1. failure counts
2. generation failure count
3. answerable rate
4. claim faithfulness
5. claim hallucination
6. evidence precision / recall / F1
7. citation correctness

注意：

```text
claim faithfulness 與 hallucination 是 heuristic 版本，
不是人工標註的完整 factuality evaluation。
```

---

## 20. 實驗設計流程

本節說明實驗如何驗證本文 claim。

### 20.1 每個 dataset 都做同樣比較

對每個 dataset，固定同一批問題，分別跑：

```text
Baseline-BFS
Degree-Capped-BFS
Token-Budgeted-BFS
Spine-Only
Spine-Correction
Spine-Correction-KGValidFallback
HRG-Proposed-NoExpansion
HRG-Proposed-Expansion
HRG-GrammarFirst-NoExpansion
HRG-GrammarFirst-Expansion
RelationUnigram / RelationBigram / RelationTrigram
```

### 20.2 每題都記錄三類資訊

```text
1. Answer: predicted answer vs gold answer
2. Evidence: selected chain, retrieved edges, subgraph size
3. Cost / failure: context tokens, latency, failure stage
```

### 20.3 最後聚合成 dataset-level 指標

```text
answer quality:
  EM, F1, Hits@k, MRR

efficiency:
  avg context tokens
  avg subgraph size
  compression ratio vs BFS

retrieval:
  retrieval recall / precision / F1

failure:
  no_candidates
  no_valid_chain
  retrieval_empty
  generation failure
```

### 20.4 可重現設定與程式對應

新 rerun 的設定要和程式碼對齊，不能只在文字中描述方法。論文或 supplementary 應集中列出：

1. 執行入口：`run_everything_sequential.sh` 先做 syntax / py_compile、Qwen3.5 diagnosis、reachability audit，再呼叫 `run_full_rerun.sh`。
2. 完整 rerun matrix：`run_full_rerun.sh` 預設開啟 `EXPERIMENT_SUITE=full`、`ENABLE_NEW_ABLATION_SPECS=1`、`ENABLE_RELATION_NGRAM_SPECS=1`、`ENABLE_BFS_CAP_SPECS=1`，並跑四個 dataset config。
3. 控制預算：新增 ablation 在 `benchmark.py` 使用相同 controlled budget，例如 `num_candidates`、`valid_chain_fallback_topk`、fallback beam / branch budget 與 `max_total_context_edges`。
4. GrammarFirst row：`HRG-GrammarFirst-NoExpansion/Expansion` 對應 `use_grammar_first_retrieval=True`、`use_valid_chain_llm_rerank=True`；它的 relation chains 來自 HRG relation path-bank + KG executable filtering，而不是 LLM relation-chain parser。
5. Perturbation：`run_full_rerun.sh` 預設 `KB_ABLATION_SEEDS=0 1 2 3 4`、`drop_nodes/drop_relations`、`0.1/0.2/0.3`；若主文只使用 single-model artifact，caption 必須標成 diagnostic。
6. 模型與 prompt：checkpoint、temperature、max tokens、parse prompt、answer prompt 與 relation vocabulary source 應以 `benchmark.py`、`knowledgegraph_agent.py` 與各 `configs/*.env` 的實際值為準，附錄需列完整 prompt。

### 20.5 為什麼這樣設計

因為本文 claim 不是單純：

```text
HRG-Proposed 的 EM 比 BFS 高。
```

而是：

```text
在 answer quality 貼近 BFS 的情況下，
context tokens 和 subgraph size 明顯下降，
且每題 evidence structure 可以被印出來檢查。
```

---

## 21. 資料與結果補充區

本節保留舊 dump 的四個有效模型主結果，並新增 latest MetaQA rerun 的模型別狀態表。有效模型集合仍是 `gpt-oss`、`gemma4`、`llama3.1`、`qwen2.5`；`qwen3.5` 的實驗未納入平均。數據解讀採用本文主軸：**HRG-Proposed 是否能在 context 明顯小於 BFS 的條件下，使 answer quality 接近 BFS，並輸出可檢查的 HRG-guided evidence structure**。

注意：latest MetaQA rerun 中 `gemma4`、`llama3.1`、`qwen2.5` 已有 `HRG-GrammarFirst-NoExpansion` 與 `HRG-GrammarFirst-Expansion` 結果，但 `gpt-oss` 在同一個 run 尚未完成。因此新 row 目前只能作為 interim model-level result，不應和後面舊四模型平均混成同一個 aggregate。

`qwen3.5` 的排除規則必須明確寫在實驗設定中，而不是只在表格裡標成一般失敗。目前環境可用 GPU 是 A40，compute capability 為 8.6；`Qwen/Qwen3.5-35B-A3B-FP8` 在 Transformers 中需要 compute capability >= 8.9 才支援 FP8 runtime。實際載入時 Transformers 會退回 bf16 dequantization，導致 load-time OOM。因此 `qwen3.5` 是 checkpoint / hardware incompatibility，不是 HRG-Proposed 方法失敗，也不納入 main aggregation。

本節所有 `n` 都要標明 evaluation unit。若表格沒有另行註明，主結果採用 question-model-pair level aggregate：

| Dataset | Evaluated questions in current dump | Active models | Nominal question-model pairs | Exclusion rule |
|---|---:|---:|---:|---|
| MetaQA | 600 | 4 | 2400 | method-specific OOM / empty / format failures may reduce row-level n |
| WikiMovies | 200 | 4 | 800 | method-specific OOM / empty / format failures may reduce row-level n |
| MLPQ | 400 | 4 | 1600 | method-specific OOM / empty / format failures may reduce row-level n |
| KQAPro | 600 | 4 | 2400 | method-specific OOM / empty / format failures may reduce row-level n |

如果後續圖表使用 single-model perturbation artifacts，caption 必須寫成 `single-model diagnostic`；如果使用四模型平均，caption 必須寫成 `four-model average`。不要再使用容易誤導的 `full-data` 字眼，除非它真的是官方完整 test set；較安全的說法是 `current evaluated dump`。

新 rerun 的完整 method matrix 應至少包含：

| Method row | Serialization | Purpose |
|---|---|---|
| `Baseline-BFS` | baseline context | strong high-recall graph baseline |
| `Degree-Capped-BFS-*` | baseline context | fairer BFS under degree cap |
| `Token-Budgeted-BFS-*` | baseline context | fairer BFS under token budget |
| `Spine-Only` | json / triple | LLM chain + KB validation only |
| `Spine-Correction` | json / triple | strict spine + correction |
| `Spine-Correction-KGValidFallback` | json / triple | isolates deterministic KG-valid fallback without HRG grammar |
| `HRG-Proposed-NoExpansion` | json / triple | LLM-chain-guided HRG recovery/ranking without expansion |
| `HRG-Proposed-Expansion` | json / triple | same as above with gated expansion |
| `HRG-GrammarFirst-NoExpansion` | json / triple | HRG path-bank first + KG executable filtering + optional LLM rerank |
| `HRG-GrammarFirst-Expansion` | json / triple | GrammarFirst path-bank with gated expansion |
| `RelationUnigram/Bigram/Trigram` | json / triple | simple relation-frequency prior comparison |

### 21.0 Latest MetaQA rerun status

以下表格來自 `artifacts_full/metaqa-vanilla-test-full-20260629-002637/results/benchmark_results.json`。這是目前最新 MetaQA full rerun 的模型別結果，cell 格式為：

```text
EM / answer-set F1 / avg final context tokens
```

截至目前，`gemma4`、`llama3.1`、`qwen2.5` 都已有完整 36-row 結果；`gpt-oss` 在此 run 尚未出現在 `benchmark_results.json`，因此先標為 `執行中`。`qwen3.5` 仍依前述硬體 / FP8 checkpoint 不相容規則排除，不納入有效模型平均。

注意：下表是 **latest MetaQA rerun status**，不是舊四模型平均。等 `gpt-oss` 在同一個 run 完成後，才能把它重新聚合成新的 four-model average，並取代後面舊 dump 的 MetaQA 主表。

注意：下表中的 `HRG-GrammarFirst-*` 數字來自修正前的 GrammarFirst 實作，當時 candidate generation 主要是 KG-valid adjacency enumeration + grammar scoring。程式已改成 HRG relation path-bank first：先由 offline grammar 產生 relation-path signatures，再由 KG 驗證可執行 chains，並加入 MetaQA relation alias map 與 stopword-filtered relation cue coverage。修正後的 GrammarFirst rows 需要重新跑，不能用下表舊數字判斷新版 path-bank 方法。

| Method | gpt-oss | gemma4 | llama3.1 | qwen2.5 |
|---|---:|---:|---:|---:|
| Baseline-BFS | 執行中 | 0.3917 / 0.4542 / 4449.7 | 0.4483 / 0.5850 / 4449.7 | 0.4083 / 0.5349 / 4449.7 |
| Degree-Capped-BFS-50 | 執行中 | 0.3917 / 0.4772 / 594.6 | 0.4283 / 0.5639 / 594.6 | 0.4000 / 0.5253 / 594.6 |
| Degree-Capped-BFS-100 | 執行中 | 0.3950 / 0.4840 / 1058.7 | 0.4467 / 0.5755 / 1058.7 | 0.3983 / 0.5275 / 1058.7 |
| Degree-Capped-BFS-200 | 執行中 | 0.3983 / 0.4885 / 1950.0 | 0.4517 / 0.5811 / 1950.0 | 0.4017 / 0.5263 / 1950.0 |
| Degree-Capped-BFS-500 | 執行中 | 0.3917 / 0.4542 / 4449.7 | 0.4483 / 0.5850 / 4449.7 | 0.4083 / 0.5349 / 4449.7 |
| Token-Budgeted-BFS-200 | 執行中 | 0.3550 / 0.4311 / 135.2 | 0.3983 / 0.5177 / 135.2 | 0.3867 / 0.4876 / 135.2 |
| Token-Budgeted-BFS-500 | 執行中 | 0.3883 / 0.4664 / 285.4 | 0.4283 / 0.5483 / 285.4 | 0.4017 / 0.5148 / 285.4 |
| Token-Budgeted-BFS-1000 | 執行中 | 0.3917 / 0.4786 / 510.3 | 0.4383 / 0.5578 / 510.3 | 0.4017 / 0.5163 / 510.3 |
| Spine-Only-json | 執行中 | 0.5667 / 0.6290 / 197.2 | 0.5167 / 0.7338 / 1660.9 | 0.4900 / 0.5416 / 130.4 |
| Spine-Only-triple | 執行中 | 0.5483 / 0.6149 / 88.8 | 0.4633 / 0.6719 / 812.5 | 0.4450 / 0.5192 / 59.7 |
| Spine-Correction-json | 執行中 | 0.5683 / 0.6307 / 198.1 | 0.5250 / 0.7439 / 1661.8 | 0.5017 / 0.5647 / 147.5 |
| Spine-Correction-triple | 執行中 | 0.5500 / 0.6166 / 89.2 | 0.4700 / 0.6810 / 812.8 | 0.4550 / 0.5469 / 67.3 |
| Spine-GrammarExpansion-json | 執行中 | 0.4867 / 0.5668 / 500.2 | 0.4583 / 0.6308 / 1144.8 | 0.4467 / 0.5372 / 410.0 |
| Spine-GrammarExpansion-triple | 執行中 | 0.4600 / 0.5593 / 272.2 | 0.3817 / 0.5443 / 823.8 | 0.3883 / 0.5027 / 187.5 |
| Spine-RandomExpansion-json | 執行中 | 0.5100 / 0.5929 / 680.4 | 0.4617 / 0.6844 / 3599.1 | 0.4167 / 0.5155 / 563.1 |
| Spine-RandomExpansion-triple | 執行中 | 0.4717 / 0.5666 / 378.8 | 0.4333 / 0.6619 / 3399.7 | 0.3717 / 0.4917 / 251.1 |
| Spine-FrequencyExpansion-json | 執行中 | 0.5150 / 0.6001 / 704.5 | 0.4650 / 0.6788 / 3631.4 | 0.4283 / 0.5170 / 569.1 |
| Spine-FrequencyExpansion-triple | 執行中 | 0.4750 / 0.5716 / 386.0 | 0.4383 / 0.6464 / 3479.9 | 0.3700 / 0.4827 / 257.0 |
| Spine-Correction-KGValidFallback-json | 執行中 | 0.5700 / 0.6352 / 560.6 | 0.5083 / 0.7304 / 1996.1 | 0.5050 / 0.5758 / 977.3 |
| Spine-Correction-KGValidFallback-triple | 執行中 | 0.5500 / 0.6201 / 334.1 | 0.4583 / 0.6724 / 1332.3 | 0.4583 / 0.5613 / 420.0 |
| HRG-Proposed-NoExpansion-json | 執行中 | 0.5650 / 0.6291 / 290.1 | 0.4750 / 0.6558 / 795.6 | 0.5067 / 0.5892 / 444.4 |
| HRG-Proposed-NoExpansion-triple | 執行中 | 0.5500 / 0.6158 / 144.1 | 0.4000 / 0.5912 / 341.7 | 0.4533 / 0.5718 / 198.4 |
| HRG-Proposed-Expansion-json | 執行中 | 0.5650 / 0.6291 / 290.1 | 0.4750 / 0.6558 / 795.6 | 0.5067 / 0.5892 / 444.4 |
| HRG-Proposed-Expansion-triple | 執行中 | 0.5500 / 0.6158 / 144.1 | 0.4000 / 0.5912 / 341.7 | 0.4533 / 0.5718 / 198.4 |
| HRG-GrammarFirst-NoExpansion-json | 執行中 | 0.4483 / 0.4855 / 1018.7 | 0.3550 / 0.4850 / 1376.2 | 0.3917 / 0.4624 / 1436.7 |
| HRG-GrammarFirst-NoExpansion-triple | 執行中 | 0.4417 / 0.4856 / 609.0 | 0.3450 / 0.4740 / 623.4 | 0.3367 / 0.4272 / 653.1 |
| HRG-GrammarFirst-Expansion-json | 執行中 | 0.4483 / 0.4855 / 1018.7 | 0.3550 / 0.4850 / 1376.2 | 0.3917 / 0.4624 / 1436.7 |
| HRG-GrammarFirst-Expansion-triple | 執行中 | 0.4417 / 0.4856 / 609.0 | 0.3450 / 0.4740 / 623.4 | 0.3367 / 0.4272 / 653.1 |
| HRG-Proposed-json | 執行中 | 0.5650 / 0.6291 / 290.1 | 0.4750 / 0.6558 / 795.6 | 0.5067 / 0.5892 / 444.4 |
| HRG-Proposed-triple | 執行中 | 0.5500 / 0.6158 / 144.1 | 0.4000 / 0.5912 / 341.7 | 0.4533 / 0.5718 / 198.4 |
| RelationUnigram-json | 執行中 | 0.5667 / 0.6331 / 545.5 | 0.5083 / 0.6875 / 1639.8 | 0.5000 / 0.5779 / 1273.5 |
| RelationUnigram-triple | 執行中 | 0.5483 / 0.6202 / 310.1 | 0.4317 / 0.6270 / 919.5 | 0.4550 / 0.5594 / 617.5 |
| RelationBigram-json | 執行中 | 0.5683 / 0.6350 / 550.3 | 0.5067 / 0.7216 / 1887.6 | 0.5067 / 0.5855 / 1121.0 |
| RelationBigram-triple | 執行中 | 0.5483 / 0.6191 / 312.4 | 0.4283 / 0.6452 / 1274.4 | 0.4600 / 0.5642 / 466.5 |
| RelationTrigram-json | 執行中 | 0.5683 / 0.6350 / 550.3 | 0.4933 / 0.7061 / 1599.9 | 0.5017 / 0.5794 / 1154.2 |
| RelationTrigram-triple | 執行中 | 0.5483 / 0.6191 / 312.4 | 0.4217 / 0.6322 / 972.2 | 0.4550 / 0.5565 / 479.8 |

目前這張表可先得到幾個暫定觀察：

1. `gemma4`、`llama3.1`、`qwen2.5` 的 `Spine-Correction` / `Spine-Correction-KGValidFallback` 仍是 MetaQA 上很強的 strict-spine family。
2. `HRG-Proposed-Expansion` 與 `HRG-Proposed-NoExpansion` 在三個已完成模型上數字完全相同，表示目前 MetaQA latest run 中 gated expansion 沒有實質加入額外有效 evidence。
3. `HRG-GrammarFirst` 在三個已完成模型上整體低於 LLM-chain-guided rows，尤其 `llama3.1` 的 R@5 先前診斷只有約 `0.4171`；因此它目前應定位為 diagnostic / negative result，而不是新的成功主張。
4. 在 `gpt-oss` 完成前，不應重新宣稱 four-model average；目前只能寫成 three completed models + one running model 的 interim result。

### 21.A Offline grammar extraction 統計

這一節回答「HRG-like grammar 到底有沒有被抽出來、抽出多少、是否穩定」的問題。這裡的 `unique patterns` 是以 rule RHS 中的 relation multiset 當作 pattern signature；它不是完整圖同構檢查，但可作為 relation-structure prior 的可重現統計。

> 圖：`img/paper_figures/14_offline_grammar_statistics.pdf`。  
> 圖說：Offline grammar statistics across datasets. MetaQA and WikiMovies have compact relation vocabularies, while MLPQ and KQAPro contain more heterogeneous relation structures.

| Dataset | Rules | Unique relation patterns | Pattern ratio | Avg terminal arity | Max terminal arity | Unique relations | Terminal edges in rules | Top relations |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MetaQA | 304 | 247 | 0.812 | 10.82 | 134 | 9 | 3288 | has_genre, has_tags, starred_actors |
| WikiMovies | 335 | 248 | 0.740 | 8.91 | 93 | 10 | 2984 | starred_actors, has_genre, has_tags |
| MLPQ | 398 | 353 | 0.887 | 9.22 | 392 | 262 | 3670 | subdivisionType, subdivisionName, stadium |
| KQAPro | 187 | 185 | 0.989 | 44.74 | 1377 | 300 | 8366 | occupation, instanceOf, twinned administrative body |

解讀：

1. MetaQA / WikiMovies 的 relation vocabulary 很小，rule patterns 較集中，符合 relation-chain-friendly KGQA 的設定。
2. MLPQ / KQAPro 的 unique relations 多很多，pattern ratio 也更高，表示圖結構更稀疏、更異質；這能解釋為什麼 online relation canonicalization 與 operator / qualifier semantics 更困難。
3. KQAPro 的 max arity 很高，代表部分 rule 對應 hub / statement-like structures；這也是 KG-valid fallback 與 candidate ranking 容易造成 context 與 OOM 上升的原因。

這張表目前還不足以單獨支撐「grammar 很穩健」。新 rerun 後應補一張 Grammar Extraction Configuration and Stability 表，至少包含：

1. grammar sample count、sample depth / size、branch cap、hub cap、random seed。
2. sampled nodes / edges / relations 與 sampled edges / total KG edges。
3. terminal arity 的 median、P90、P95、max，而不是只報 average / max。
4. rule frequency distribution，避免 high-degree hub 造成的 rule 被誤解成有意義 local structure。
5. coarse relation-multiset signature 與 direction/role-aware structural signature 的差異；目前主結果只能保守 claim coarse structural prior。

### 21.B Offline grammar robustness under KG perturbation

這一節回答 offline HRG prior 在 KG 隨機缺失時是否仍保留結構訊號。

> 圖：`img/paper_figures/15_grammar_perturbation_trends.pdf`。  
> 圖說：Under random node deletion, extracted rule counts remain available while relation vocabulary retention differs by dataset complexity. This supports using HRG as a soft structural prior rather than a hard decoder.

#### 21.B.1 Rule count under perturbation

| Dataset | Clean rules | Node drop 10% | Node drop 20% | Node drop 30% | Relation drop 10% | Relation drop 20% | Relation drop 30% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 304 | 326 | 368 | 433 | 336 | 371 | 401 |
| WikiMovies | 335 | 314 | 360 | 376 | 314 | 328 | 383 |
| MLPQ | 398 | 467 | 453 | 526 | 364 | 451 | 379 |
| KQAPro | 187 | 343 | 350 | 288 | 299 | - | - |

Rule count 在 perturbation 後不一定下降，因為 MCS / triangulation 後的 clique decomposition 可能把缺失後的圖切成更多局部 pattern。這不能解讀成「刪除越多越好」，而應解讀成：KG perturbation 改變了局部結構分解，HRG extraction 仍能產生可用的 structural prior。

因此 perturbation 不能只看 rule count。新分析應同步報告：

1. gold answer 是否仍可從 perturbed KG 到達。
2. original correct path 是否仍存在。
3. retained rules 中有多少和 query-relevant path 有關。
4. deletion 後新增的 rules 是否可能只是 fragmentation noise。
5. 在 valid perturbed questions 上的 answer quality / evidence coverage，而不是把已不可達題目一起當成 method failure。

#### 21.B.2 Relation vocabulary retention

| Dataset | Clean relations | Node 10% | Node 20% | Node 30% | Relation 10% | Relation 20% | Relation 30% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 9 | 9 / 1.000 | 9 / 1.000 | 8 / 0.889 | 8 / 0.889 | 7 / 0.778 | 6 / 0.667 |
| WikiMovies | 10 | 9 / 0.900 | 10 / 1.000 | 10 / 1.000 | 8 / 0.800 | 8 / 0.800 | 7 / 0.700 |
| MLPQ | 262 | 116 / 0.443 | 120 / 0.458 | 137 / 0.523 | 131 / 0.500 | 119 / 0.454 | 85 / 0.324 |
| KQAPro | 300 | 179 / 0.597 | 165 / 0.550 | 146 / 0.487 | 165 / 0.550 | - | - |

表格中每格為 `retained clean relations / retention ratio`。MetaQA / WikiMovies 的 relation vocabulary 在 node drop 下大多保留，表示 domain schema 穩定；MLPQ / KQAPro 的 retention 較低，表示它們的 schema 更長尾，也更依賴 relation normalization。

#### 21.B.3 Exact relation-pattern retention

以完整 relation multiset 當作 rule signature 時，clean grammar 與 perturbation grammar 的 exact pattern overlap 偏低。例如 MetaQA node-drop 10% 只保留 `39 / 247 = 0.158` clean patterns，MLPQ node-drop 10% 只保留 `22 / 353 = 0.062`，KQAPro node-drop 10% 只保留 `4 / 185 = 0.022`。

這個結果不表示 HRG 不穩定，而是表示高階 clique-level pattern 對隨機刪圖很敏感；因此 robustness claim 分成兩層：

```text
Relation-level grammar vocabulary is relatively stable in schema-clean datasets.
Exact high-order HRG rule patterns are sensitive to KG perturbation,
so online QA should rely on grammar as a soft structural prior rather than a hard decoder.
```

這也支撐本文目前的設計：HRG-Proposed 沒有把 grammar 當作唯一合法推導器，而是把它用在 candidate selection、fallback 與 ranking features。

#### 21.B.4 論文圖趨勢

輔助圖如下：

1. **Offline rule count vs perturbation ratio**  
   x 軸為 clean / 10% / 20% / 30%，y 軸為 extracted rule count。每個 dataset 一條線。這張圖展示 KG 被刪除後 grammar extraction 仍能產生 structural prior。

2. **Relation vocabulary retention vs perturbation ratio**  
   x 軸為 drop ratio，y 軸為 retained clean relation ratio。MetaQA / WikiMovies 通常較穩定，MLPQ / KQAPro 較低，對應 dataset schema complexity。

3. **Exact pattern retention vs perturbation ratio**  
   x 軸為 drop ratio，y 軸為 clean rule pattern retained ratio。這張圖會顯示高階 HRG patterns 對 perturbation 敏感，因此 HRG 在本文中作為 soft prior，而不是 hard decoder。

4. **Online evidence coverage vs method**  
   x 軸為 Baseline-BFS、Spine-Correction、HRG-Proposed，y 軸為 answer-in-spine 或 retrieval F1。這張圖最適合用 MLPQ / KQAPro，因為能直接展示 HRG-Proposed 補回 strict spine 缺失的 evidence coverage。

5. **Quality-cost trade-off scatter**  
   x 軸為 context tokens 或 total stage tokens，y 軸為 answer F1 或 answer-in-spine。點的顏色代表 dataset，形狀代表 method。這張圖回應 token reduction 的核心問題：HRG-Proposed 不是單純少拿資料，而是在成本與 evidence coverage 間做 trade-off。

圖說：

```text
Offline grammar perturbation analysis shows that exact high-order rule patterns are sensitive to random KG deletion,
while relation-level structural vocabulary remains more stable in schema-clean datasets.
This motivates using HRG-like grammar as a soft structural prior for online retrieval rather than as a complete hard decoder.
```

### 21.0 結果總結

摘要或結果總結可採用 HRG-Proposed 導向。四個有效模型平均如下：

```text
MetaQA:
Baseline-BFS:             EM 0.4504 / F1 0.5528 / 4429.98 ctx tokens / 369.08 edges
HRG-Proposed-triple:      EM 0.4933 / F1 0.5948 / 189.55 ctx tokens / 19.33 edges

KQAPro:
Baseline-BFS:             EM 0.0804 / F1 0.0842 / 3595.48 ctx tokens / 231.01 edges
Spine-Correction-triple:  EM 0.0225 / F1 0.0245 / 3.89 ctx tokens / 0.27 edges
HRG-Proposed-triple:      EM 0.0754 / F1 0.0854 / 783.54 ctx tokens / 63.85 edges

MLPQ:
Baseline-BFS:             EM 0.2381 / F1 0.3078 / 4983.34 ctx tokens / 333.59 edges
Spine-Correction-json:    EM 0.1306 / F1 0.2185 / 101.48 ctx tokens / 4.22 edges
HRG-Proposed-json:        EM 0.1544 / F1 0.2514 / 359.79 ctx tokens / 15.15 edges
HRG-Proposed-triple:      EM 0.1781 / F1 0.2498 / 209.30 ctx tokens / 18.02 edges
```

這代表 HRG-Proposed 的主要 claim 不是「全面超越 BFS」，而是「以明顯小於 BFS 的 context 取得貼近 BFS、甚至在部分資料集高於 BFS 的 F1」。MetaQA 上 HRG-Proposed-triple 的 context 只有 BFS 的 `4.28%`，F1 高於 BFS；MLPQ 上 HRG-Proposed 比 Spine-Correction 更接近 BFS，但仍低於 BFS，顯示跨語言 relation canonicalization 仍是瓶頸。KQAPro 上 HRG-Proposed-triple 的 F1 `0.0854` 接近 BFS `0.0842`，但兩者都很低，因此只能解讀為 executable evidence recovery over strict-spine ablations，不能寫成 KQAPro 成功案例。

Spine-Only 與 Spine-Correction 仍然必須放在論文中，但定位是 ablation：Spine-Only 回答「只做 LLM chain parse + KB validation + strict spine retrieval 時，效果如何」；Spine-Correction 回答「當 initial chain 失敗時，fallback correction 是否帶來額外改善」。它們不是本文主方法。

因此，本文將 HRG-Proposed 定位為 main method，Spine-Only / Spine-Correction 定位為 ablation。數據顯示，在 clean relation-chain dataset 上 strict spine 已經很強；在 KQAPro / MLPQ 這類 strict spine 較弱的 hard cases，HRG prior 能補回部分 evidence coverage。

WikiMovies 可作為反例或 ceiling case：BFS 在 1-hop 任務已很小，HRG-Proposed 不一定最接近 BFS。這不是 HRG 失敗，而是說明 HRG prior 的主要價值不在 1-hop 小子圖，而在 strict spine 不足或需要補 evidence coverage 的情境。

MLPQ 與 KQAPro 不被包裝成完全解決，而是用來支持 HRG 的 recovery 作用：strict Spine-Correction 在這兩個資料集上明顯不足，而 HRG-Proposed 更接近 BFS。KQAPro 的 F1 接近 BFS 但絕對值很低，所以它是 stress-test evidence recovery，不是 KQAPro QA 解法。

### 21.1 主結果表

| Dataset | Method | EM | F1 | Avg ctx tokens | Avg subgraph size | Context vs BFS | Subgraph vs BFS | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MetaQA | Baseline-BFS | 0.4504 | 0.5528 | 4429.98 | 369.08 | 100.00% | 100.00% | 強 baseline，但 context 大 |
| MetaQA | Spine-Only-json | 0.5521 | 0.6401 | 601.70 | 27.34 | 13.58% | 7.41% | strict spine ablation；不用 correction 已能支撐主假設 |
| MetaQA | Spine-Correction-json | 0.5583 | 0.6501 | 606.54 | 27.56 | 13.69% | 7.47% | ablation：strict spine + correction 很強 |
| MetaQA | Spine-Correction-triple | 0.5221 | 0.6257 | 287.79 | 30.57 | 6.50% | 8.28% | ablation：token 更少，品質仍高於 BFS |
| MetaQA | HRG-Proposed-triple | 0.4933 | 0.5948 | 189.55 | 19.33 | 4.28% | 5.24% | main method：比 BFS 更小 context，F1 高於 BFS，且最接近 BFS EM/F1 |
| WikiMovies | Baseline-BFS | 0.8025 | 0.8574 | 50.12 | 3.30 | 100.00% | 100.00% | 1-hop 任務，BFS 子圖本來很小 |
| WikiMovies | Spine-Only-json | 0.8875 | 0.9093 | 78.84 | 3.39 | 157.31% | 102.76% | 品質高，但 JSON tokens 不適合支撐壓縮 claim |
| WikiMovies | Spine-Correction-json | 0.8988 | 0.9210 | 79.43 | 3.42 | 158.48% | 103.44% | 品質高於 BFS；compression comparison 以 triple serialization 更清楚 |
| WikiMovies | Spine-Correction-triple | 0.7575 | 0.8199 | 36.48 | 3.42 | 72.78% | 103.44% | triple 可降 token，但品質略低於 BFS |
| WikiMovies | HRG-Proposed-triple | 0.7150 | 0.7843 | 31.32 | 2.82 | 62.49% | 85.47% | 1-hop ceiling case；HRG 不是此資料集主證據 |
| MLPQ | Baseline-BFS | 0.2381 | 0.3078 | 4983.34 | 333.59 | 100.00% | 100.00% | BFS 答案較好但 context 大 |
| MLPQ | Spine-Only-json | 0.1263 | 0.2119 | 98.02 | 4.07 | 1.97% | 1.22% | token 降很多，但 answer quality 未貼近 BFS |
| MLPQ | Spine-Correction-json | 0.1306 | 0.2185 | 101.48 | 4.22 | 2.04% | 1.26% | correction 小幅改善，仍是 limitation |
| MLPQ | HRG-Proposed-triple | 0.1781 | 0.2498 | 209.30 | 18.02 | 4.20% | 5.40% | main method：比 Spine-Correction 更接近 BFS，但跨語言仍是瓶頸 |
| KQAPro | Baseline-BFS | 0.0804 | 0.0842 | 3595.48 | 231.01 | 100.00% | 100.00% | 整體任務困難，BFS 也低 |
| KQAPro | Spine-Only-json | 0.0225 | 0.0232 | 5.95 | 0.22 | 0.17% | 0.10% | 低 token 主要來自 retrieval failure |
| KQAPro | Spine-Correction-json | 0.0229 | 0.0237 | 7.33 | 0.27 | 0.20% | 0.12% | 普通 relation chain 不足以處理 program semantics |
| KQAPro | HRG-Proposed-triple | 0.0754 | 0.0854 | 783.54 | 63.85 | 21.79% | 27.64% | stress test：補回 executable evidence coverage，但 BFS 本身也低 |

結果解讀：

```text
主證據應拆成兩類：MetaQA 證明 HRG-Proposed 能在 clean relation-chain task 中用極小 context 貼近或高於 BFS；MLPQ / KQAPro 則顯示當 strict spine 不足時，HRG-Proposed 能補回 executable evidence coverage。

WikiMovies 顯示 BFS 對 1-hop 小子圖非常強，因此不是 HRG-Proposed 的主要展示資料集。

MLPQ 與 KQAPro 顯示方法限制：當任務需要跨語言 relation canonicalization、operator 或 qualifier reasoning 時，
單純 relation spine 不足以穩定貼近 BFS。KQAPro 中 HRG-Proposed-triple 的 F1 接近 BFS，但兩者都很低；這是 evidence recovery diagnostic，不是完整 operator-aware solution。
```

### 21.2 MetaQA hop-level 分析

以下 hop-level 表保留原先 `gpt-oss` 主結果，因為它最清楚展示 3-hop 中「大 BFS context 不等於可用 evidence」的現象。跨模型平均請以前一節主結果表為準。

| Hop | Baseline-BFS EM/F1 | Spine-Correction-json EM/F1 | Spine-Correction-triple EM/F1 | Interpretation |
|---|---:|---:|---:|---|
| 1-hop | 0.9950 / 0.9950 | 0.9850 / 0.9850 | 0.9550 / 0.9603 | BFS 在簡單 1-hop 幾乎滿分；Spine 仍貼近 |
| 2-hop | 0.7250 / 0.8326 | 0.6950 / 0.7405 | 0.6900 / 0.7262 | Spine 略低於 BFS，但 context 小很多 |
| 3-hop | 0.0050 / 0.0050 | 0.1600 / 0.2612 | 0.1400 / 0.2480 | 長 hop 中，BFS 大 context 不等於可用 evidence；spine 結構反而更利於 generation |

MetaQA 3-hop 是關鍵分析：

```text
BFS 在 3-hop 的 retrieval recall 高，但 answer EM 只有 0.0050。
Spine-Correction-json 的 3-hop EM 達 0.1600，F1 達 0.2612。
這表示長 hop 問題中，LLM 不只需要答案附近的 facts，更需要明確的 relation-chain evidence structure。
```

### 21.3 Compression 分析

| Dataset | Method | Context ratio vs BFS | Subgraph ratio vs BFS | Interpretation |
|---|---:|---:|---:|---|
| MetaQA | HRG-Proposed-triple | 4.28% | 5.24% | main method：context 最小，F1 高於 BFS，EM/F1 最接近 BFS |
| KQAPro | HRG-Proposed-triple | 21.79% | 27.64% | stress test：BFS 本身 F1 很低；HRG prior 補回 executable evidence coverage，但不是解決 KQAPro |
| MLPQ | HRG-Proposed-triple | 4.20% | 5.40% | main method：比 strict spine 更接近 BFS，但跨語言仍未完全解決 |
| WikiMovies | HRG-Proposed-triple | 62.49% | 85.47% | 1-hop ceiling case；context 小但品質不是主證據 |
| MetaQA | Spine-Correction-triple | 6.50% | 8.28% | ablation：strict spine + correction 已經很強 |
| KQAPro | Spine-Correction-json | 0.20% | 0.12% | ablation：低 token 主要來自 retrieval failure，不是成功壓縮 |

這張表要避免錯誤解讀：

```text
token reduction 必須和 answer quality 一起看。
HRG-Proposed 在 MetaQA / MLPQ 的重點是：context 明顯小於 BFS，同時比 strict spine 更接近 BFS。KQAPro 的重點則是 strict spine failure 時的 executable evidence recovery，不是 high-quality QA success。
KQAPro 中 Spine-Correction 的低 token 是 retrieval 沒取到 evidence，不能當成功壓縮；HRG-Proposed 補回 evidence 後才是可解釋的 compact retrieval。
```

### 21.4 Failure analysis

| Dataset | Method | ok | no_candidates | no_valid_chain | Other | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| MetaQA | Spine-Correction-json | 435 | 160 | 5 | 0 | parser 仍有 no_candidates，但 valid 後品質佳 |
| WikiMovies | Spine-Correction-json | 197 | 3 | 0 | 0 | 1-hop 問題穩定，失敗少 |
| MLPQ | Spine-Correction-json | 207 | 128 | 65 | 0 | 跨語言 grounding / relation normalization 造成 chain failure |
| MLPQ | HRG-Proposed-json | 389 | 0 | 0 | 11 OOM | fallback 大幅降低 no_candidates，但仍未貼近 BFS |
| KQAPro | Spine-Correction-json | 7 | 353 | 240 | 0 | 普通 relation chain 無法覆蓋 operator/qualifier semantics |
| KQAPro | HRG-Proposed-json | 272 | 317 | 9 | 2 OOM | deterministic fallback 增加可執行 chain，但答案品質仍不足 |

### 21.5 KG Perturbation Robustness 消融

本小節的 robustness 分析回答：當 KG 被刪除 10% / 20% nodes 或 relations 時，HRG-Proposed 是否仍能提供比 strict spine 更多 evidence coverage。現有表格主要是 diagnostic / selected artifacts；若不是四模型平均，caption 必須寫明使用哪個 model、哪個 seed、哪個 perturbation dump。它不能和四模型主結果放在同一個 claim 強度下。

這張表不表示「HRG 在所有 perturbation 都最好」。正確解讀是：

1. MetaQA / WikiMovies 屬於 chain-friendly 或 1-hop ceiling case，strict spine 已經很強；HRG-Proposed 的價值主要是維持 compact evidence，並在 perturbation 後透過 grammar-aware candidate selection / fallback 補一些 recall。
2. MLPQ 中 HRG-Proposed 在 clean 與 perturbation 下都比 Spine-Correction 更接近 BFS，支持 HRG prior 對跨語言 path evidence 有補強效果，但仍低於 BFS。
3. KQAPro 中 Spine-Correction 幾乎取不到 evidence；HRG-Proposed 在 node-drop 10% / 20% 下提高 R@5 與 F1，但代價是 context 與 OOM 上升。這只能作為 stress-test diagnostic：它支持 KG-valid fallback / HRG ranking 能補 evidence，但不等於完整證明 KQAPro robustness。

#### 21.5.1 MetaQA robustness

| Perturbation | Method | EM | F1 | R@5 | Avg ctx tokens | Avg subgraph size | Failure summary |
|---|---|---:|---:|---:|---:|---:|---|
| clean | Baseline-BFS | 0.5750 | 0.6109 | 0.9783 | 4415.8 | 369.08 | ok=600 |
| clean | Spine-Correction-triple | 0.5950 | 0.6448 | 0.7250 | 96.9 | 9.20 | ok=435; no_candidates=160; no_valid_chain=5 |
| clean | HRG-Proposed-triple | 0.5350 | 0.5718 | 0.6575 | 77.3 | 7.37 | ok=397; no_candidates=203 |
| drop nodes 10% | Baseline-BFS | 0.5717 | 0.6049 | 0.9733 | 4445.4 | 369.08 | ok=600 |
| drop nodes 10% | HRG-Proposed-triple | 0.5700 | 0.6262 | 0.8096 | 178.2 | 16.97 | ok=599; oom=1 |
| drop nodes 20% | Baseline-BFS | 0.5633 | 0.6020 | 0.9850 | 4429.2 | 369.08 | ok=600 |
| drop nodes 20% | Spine-Correction-triple | 0.5950 | 0.6428 | 0.7267 | 96.9 | 9.21 | ok=436; no_candidates=160; no_valid_chain=4 |
| drop nodes 20% | HRG-Proposed-triple | 0.5517 | 0.5992 | 0.7686 | 403.2 | 43.66 | ok=591; oom=9 |
| drop relations 10% | Baseline-BFS | 0.5683 | 0.6079 | 0.9750 | 4424.6 | 369.08 | ok=600 |
| drop relations 10% | Spine-Correction-triple | 0.5933 | 0.6467 | 0.7267 | 96.9 | 9.21 | ok=436; no_candidates=160; no_valid_chain=4 |
| drop relations 10% | HRG-Proposed-triple | 0.5650 | 0.6142 | 0.8064 | 417.5 | 44.48 | ok=589; oom=11 |
| drop relations 20% | Baseline-BFS | 0.5733 | 0.6041 | 0.9800 | 4446.7 | 369.08 | ok=600 |
| drop relations 20% | Spine-Correction-triple | 0.5933 | 0.6421 | 0.7267 | 96.9 | 9.21 | ok=436; no_candidates=160; no_valid_chain=4 |
| drop relations 20% | HRG-Proposed-triple | 0.5667 | 0.6193 | 0.8166 | 332.7 | 31.80 | ok=600 |

MetaQA 的解讀需要區分 answer quality 與 quality-cost trade-off：clean gpt-oss 下 Spine-Correction-triple 比 HRG-Proposed-triple F1 高，因此 HRG-Proposed 在 MetaQA 不是 answer quality 最佳，而是 quality-cost / robustness trade-off。perturbation 後 HRG-Proposed 的 R@5 上升，代表 grammar-aware candidate selection / fallback 補回更多 answer-containing evidence，但 context 也明顯增加。

#### 21.5.2 MLPQ robustness

| Perturbation | Method | EM | F1 | R@5 | Avg ctx tokens | Avg subgraph size | Failure summary |
|---|---|---:|---:|---:|---:|---:|---|
| clean | Baseline-BFS | 0.2550 | 0.3693 | 0.8550 | 4992.3 | 333.59 | ok=400 |
| clean | Spine-Correction-triple | 0.1550 | 0.2242 | 0.3625 | 28.6 | 2.52 | no_candidates=128; ok=207; no_valid_chain=65 |
| clean | HRG-Proposed-triple | 0.1950 | 0.2771 | 0.5297 | 358.4 | 29.34 | ok=398; oom=2 |
| drop nodes 10% | Baseline-BFS | 0.2425 | 0.3627 | 0.8650 | 4920.3 | 333.59 | ok=400 |
| drop nodes 10% | Spine-Correction-triple | 0.1500 | 0.2218 | 0.3625 | 28.6 | 2.52 | no_candidates=128; ok=207; no_valid_chain=65 |
| drop nodes 10% | HRG-Proposed-triple | 0.1700 | 0.2472 | 0.5028 | 410.5 | 33.38 | ok=398; oom=2 |
| drop nodes 20% | Baseline-BFS | 0.2325 | 0.3520 | 0.9125 | 4942.0 | 333.59 | ok=400 |
| drop nodes 20% | Spine-Correction-triple | 0.1625 | 0.2285 | 0.3650 | 28.7 | 2.52 | no_candidates=128; ok=208; no_valid_chain=64 |
| drop nodes 20% | HRG-Proposed-triple | 0.1800 | 0.2586 | 0.4976 | 217.6 | 19.32 | ok=396; oom=4 |
| drop relations 10% | Baseline-BFS | 0.2425 | 0.3569 | 0.8825 | 4914.6 | 333.59 | ok=400 |
| drop relations 10% | Spine-Correction-triple | 0.1525 | 0.2227 | 0.3625 | 28.6 | 2.52 | no_candidates=128; ok=207; no_valid_chain=65 |
| drop relations 10% | HRG-Proposed-triple | 0.1775 | 0.2586 | 0.5083 | 187.1 | 15.83 | ok=399; oom=1 |
| drop relations 20% | Baseline-BFS | 0.2475 | 0.3559 | 0.9075 | 4927.7 | 333.59 | ok=400 |
| drop relations 20% | Spine-Correction-triple | 0.1575 | 0.2301 | 0.3650 | 28.7 | 2.52 | no_candidates=128; ok=208; no_valid_chain=64 |
| drop relations 20% | HRG-Proposed-triple | 0.2150 | 0.2903 | 0.5165 | 237.1 | 19.19 | ok=397; oom=3 |

MLPQ 的重點是：HRG-Proposed-triple 在 clean 與 10% / 20% perturbation 下都比 Spine-Correction-triple 更高 F1 與 R@5，表示 HRG-guided candidate selection / fallback 能補回 strict spine 缺少的 evidence coverage。但 BFS 仍最高，代表主要瓶頸仍是 multilingual grounding 與 relation canonicalization。

#### 21.5.3 KQAPro robustness

KQAPro robustness 的主要比較使用 clean、drop nodes 10% 與 drop nodes 20%，因為 drop relations 10% 缺少同條件下可比的 Spine-Correction / HRG-Proposed 列。

| Perturbation | Method | EM | F1 | R@5 | Avg ctx tokens | Avg subgraph size | Failure summary |
|---|---|---:|---:|---:|---:|---:|---|
| clean | Baseline-BFS | 0.0700 | 0.0700 | 0.3283 | 2985.0 | 191.92 | ok=486; retrieval_empty=63; entity_parse_failure=51 |
| clean | Spine-Correction-triple | 0.0083 | 0.0083 | 0.0067 | 0.5 | 0.03 | no_candidates=353; no_valid_chain=240; ok=7 |
| clean | HRG-Proposed-triple | 0.0400 | 0.0400 | 0.0820 | 32.8 | 3.57 | no_candidates=317; ok=273; no_valid_chain=9; oom=1 |
| drop nodes 10% | Baseline-BFS | 0.0683 | 0.0683 | 0.3217 | 2938.7 | 192.26 | ok=483; retrieval_empty=66; entity_parse_failure=51 |
| drop nodes 10% | Spine-Correction-triple | 0.0067 | 0.0067 | 0.0050 | 0.4 | 0.03 | no_candidates=353; no_valid_chain=240; ok=7 |
| drop nodes 10% | HRG-Proposed-triple | 0.0633 | 0.0633 | 0.1825 | 800.5 | 58.22 | ok=480; oom=110; no_valid_chain=9; no_candidates=1 |
| drop nodes 20% | Baseline-BFS | 0.0783 | 0.0783 | 0.3333 | 2888.4 | 196.02 | ok=491; retrieval_empty=58; entity_parse_failure=51 |
| drop nodes 20% | Spine-Correction-triple | 0.0067 | 0.0067 | 0.0050 | 0.4 | 0.03 | no_candidates=353; no_valid_chain=240; ok=7 |
| drop nodes 20% | HRG-Proposed-triple | 0.0467 | 0.0467 | 0.1693 | 907.7 | 65.56 | ok=471; oom=122; no_valid_chain=6; no_candidates=1 |
| drop relations 10% | Baseline-BFS | 0.0750 | 0.0750 | 0.3267 | 2987.7 | 193.42 | ok=489; retrieval_empty=60; entity_parse_failure=51 |

KQAPro 的消融應以 stress-test 語氣解讀。在 clean gpt-oss 下，Spine-Correction-triple 幾乎沒有 evidence (`R@5=0.0067`, avg ctx `0.5`)，HRG-Proposed-triple 雖然仍低於 BFS，但將 F1 提升到 `0.0400`、R@5 提升到 `0.0820`。在 drop nodes 10% 時，HRG-Proposed-triple 進一步達到 F1 `0.0633`、R@5 `0.1825`，明顯高於 Spine-Correction。這表示 HRG-guided candidate selection / KG-valid fallback 不是可有可無，而是在 strict relation spine 失效時提供可執行 evidence coverage；但因為 BFS 本身也很低，不能把它寫成 KQAPro 已被解決。

同時，KQAPro 的 HRG-Proposed 在 perturbation 下 OOM 數上升，表示 KG-valid fallback 與 candidate ranking 都需要更嚴格的 budget 控制。這不是否定 HRG，而是指出下一步要把 HRG prior 從「能補 evidence」推進到「穩定且成本可控地補 evidence」。

### 21.6 Evidence coverage 指標

本文以 automatic evidence-level diagnostics 補充 answer quality 指標：

> 圖：`img/paper_figures/16_evidence_coverage_hard_cases.pdf`。  
> 圖說：In MLPQ and KQAPro stress-test settings, HRG-Proposed improves executable evidence coverage over strict-spine ablations; this is not a claim that KQAPro QA is solved.

1. `retrieval_recall / retrieval_precision / retrieval_f1`：目前實作以 gold answer 是否被 retrieved subgraph 覆蓋為基礎。
2. `answer in edges`：gold answer string 是否出現在 final retrieved edges 的 head / tail。
3. `answer in spine`：gold answer string 是否出現在 strict spine edges 的 head / tail。
4. `grammar hit rate`：selected candidate 是否命中 grammar prior。
5. `selected chain nonempty rate`：系統是否選出非空 relation chain。
6. `parse1 / correction / context tokens`：可作為 end-to-end token cost 的初步拆解。

#### 21.6.0 HRG-Proposed 機制觸發率

為了確認 HRG 在系統中的實際作用，統計 HRG-Proposed-triple 的機制觸發率：

| Dataset | n | grammar_hit | kg_valid_fallback source | grammar_fallback source | llm_correction source | nonzero LLM rerank |
|---|---:|---:|---:|---:|---:|---:|
| MetaQA | 2400 | 2196 / 91.5% | 245 / 10.2% | 53 / 2.2% | 29 / 1.2% | 241 / 10.0% |
| WikiMovies | 800 | 795 / 99.4% | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| MLPQ | 1598 | 857 / 53.6% | 629 / 39.4% | 35 / 2.2% | 16 / 1.0% | 513 / 32.1% |
| KQAPro | 2396 | 1784 / 74.5% | 1723 / 71.9% | 93 / 3.9% | 8 / 0.3% | 1498 / 62.5% |

這張表是 HRG 存在感的主要證據之一。它顯示：

1. HRG 的主要實際作用是 grammar_hit、matched_rules、KG-valid fallback 與 hard-case reranking signals。
2. grammar_fallback 與 LLM correction 有發生，但比例低，不能寫成主要改善來源。
3. KQAPro 中 `kg_valid_fallback` 與 `nonzero LLM rerank` 比例很高，說明 hard cases 的改善主要來自 HRG-guided candidate/fallback retrieval，而不是單純 strict spine。

因此貢獻敘事要跟 trigger statistics 對齊：LLM correction 是 fallback module 的 minor component；deterministic KG-valid fallback、grammar-aware candidate selection 與 ranking 才是目前主要 recovery mechanism。

新 summary 會把單一 `grammar_hit` 拆成多個欄位：`candidate_weak_label_hit_rate`、`candidate_ordered_path_hit_rate`、`candidate_weak_label_only_rate`、`avg_matched_rule_count`，並搭配 same-arity / ordered-path hit。若沒有 full structural hit，就不能把高 grammar hit 解讀成完整 HRG clique-tree match。

以下表格使用四個有效模型的所有題目。這不是人工 gold evidence sufficiency，但已經比「能印出 evidence」更接近可量化的 evidence-level analysis。

| Dataset | Method | n | Ret R | Ret P | Ret F1 | Answer in edges | Answer in spine | Grammar hit | Chain nonempty | Spine edges | Parse1 tok | Corr tok | Context tok | OK rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | Baseline-BFS | 2400 | 0.903 | 0.252 | 0.329 | 0.810 | 0.000 | 0.000 | 0.000 | 0.00 | 0.0 | 0.0 | 4430.0 | 1.000 |
| MetaQA | Spine-Correction-triple | 2400 | 0.757 | 0.367 | 0.471 | 0.755 | 0.755 | 0.000 | 0.933 | 30.57 | 585.9 | 61.3 | 287.8 | 0.822 |
| MetaQA | HRG-Proposed-triple | 2400 | 0.723 | 0.379 | 0.475 | 0.718 | 0.718 | 0.915 | 0.915 | 19.33 | 816.0 | 133.4 | 189.6 | 0.915 |
| WikiMovies | Baseline-BFS | 800 | 0.998 | 0.599 | 0.733 | 0.994 | 0.000 | 0.000 | 0.000 | 0.00 | 78.1 | 0.0 | 50.1 | 1.000 |
| WikiMovies | Spine-Correction-triple | 800 | 0.990 | 0.576 | 0.713 | 0.987 | 0.987 | 0.000 | 0.995 | 3.42 | 509.9 | 8.7 | 36.5 | 0.994 |
| WikiMovies | HRG-Proposed-triple | 800 | 0.991 | 0.619 | 0.751 | 0.988 | 0.988 | 0.994 | 0.994 | 2.82 | 738.0 | 0.0 | 31.3 | 0.994 |
| MLPQ | Baseline-BFS | 1600 | 0.849 | 0.039 | 0.071 | 0.434 | 0.000 | 0.000 | 0.000 | 0.00 | 0.0 | 0.0 | 4983.3 | 1.000 |
| MLPQ | Spine-Correction-triple | 1600 | 0.398 | 0.088 | 0.141 | 0.225 | 0.225 | 0.000 | 0.918 | 4.22 | 992.4 | 344.8 | 48.6 | 0.568 |
| MLPQ | HRG-Proposed-triple | 1598 | 0.521 | 0.104 | 0.167 | 0.283 | 0.283 | 0.536 | 0.999 | 18.00 | 1146.3 | 563.1 | 209.1 | 0.999 |
| KQAPro | Baseline-BFS | 2400 | 0.359 | 0.048 | 0.069 | 0.133 | 0.000 | 0.000 | 0.000 | 0.00 | 124.1 | 0.0 | 3595.5 | 0.899 |
| KQAPro | Spine-Correction-triple | 2400 | 0.032 | 0.014 | 0.018 | 0.026 | 0.026 | 0.000 | 0.787 | 0.27 | 1240.4 | 956.6 | 3.9 | 0.058 |
| KQAPro | HRG-Proposed-triple | 2396 | 0.214 | 0.085 | 0.105 | 0.152 | 0.152 | 0.745 | 0.818 | 63.88 | 1613.9 | 1558.5 | 784.2 | 0.799 |

這張表可以支撐三個重要論點：

1. **HRG 不是沒差**：MLPQ 中 HRG-Proposed-triple 將 answer-in-spine 從 Spine-Correction 的 `0.225` 提升到 `0.283`，KQAPro 從 `0.026` 提升到 `0.152`；這是 strict spine 失效時 HRG prior 補 evidence coverage 的直接證據。
2. **HRG 是有成本的 soft prior**：KQAPro 中 HRG-Proposed 的 context tokens 與 correction tokens 明顯上升，表示它用成本換取 evidence coverage；因此本文主張 quality-cost trade-off，而不是單純 token 最少。
3. **MetaQA / WikiMovies 已接近 chain-friendly ceiling**：HRG-Proposed 的 grammar hit 高，但 strict spine 已能覆蓋大多答案；因此 HRG 的主要價值在 hard cases，而不是每個 clean chain task 都要壓過 Spine-Correction。

#### 21.6.1 End-to-end online token proxy

本文不只報 final context tokens，也加入 online token proxy：

> 圖：`img/paper_figures/17_online_token_proxy.pdf`。  
> 圖說：End-to-end online token proxy ratio shows that HRG-Proposed often remains below BFS-scale total online cost, but pays extra parse/correction cost over strict spine; therefore the correct claim is quality-cost trade-off.

```text
total online token proxy = parse1_tokens + correction_tokens + parse2_tokens + context_tokens
```

這不是精確 API billing cost，因為不同模型 tokenizer 不同，且部分 baseline 的 entity parsing / generation token 記錄方式不同；但它比只看 final context tokens 更接近 end-to-end online cost。

| Dataset | Method | n | Parse1 | Correction | Parse2 | Context | Total proxy | Correction share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | Baseline-BFS | 2400 | 0.0 | 0.0 | 4676.0 | 4430.0 | 9105.9 | 0.000 |
| MetaQA | Spine-Correction-triple | 2400 | 585.9 | 61.3 | 495.3 | 287.8 | 1430.3 | 0.043 |
| MetaQA | HRG-Proposed-triple | 2400 | 816.0 | 133.4 | 413.2 | 189.6 | 1552.2 | 0.086 |
| WikiMovies | Baseline-BFS | 800 | 78.1 | 0.0 | 289.4 | 50.1 | 417.6 | 0.000 |
| WikiMovies | Spine-Correction-triple | 800 | 509.9 | 8.7 | 264.6 | 36.5 | 819.7 | 0.011 |
| WikiMovies | HRG-Proposed-triple | 800 | 738.0 | 0.0 | 283.4 | 31.3 | 1052.7 | 0.000 |
| MLPQ | Baseline-BFS | 1600 | 0.0 | 0.0 | 5233.9 | 4983.3 | 10217.2 | 0.000 |
| MLPQ | Spine-Correction-triple | 1600 | 992.4 | 344.8 | 186.9 | 48.6 | 1572.7 | 0.219 |
| MLPQ | HRG-Proposed-triple | 1598 | 1146.3 | 563.1 | 467.1 | 209.1 | 2385.7 | 0.236 |
| KQAPro | Baseline-BFS | 2400 | 124.1 | 0.0 | 3806.6 | 3595.5 | 7526.2 | 0.000 |
| KQAPro | Spine-Correction-triple | 2400 | 1240.4 | 956.6 | 17.6 | 3.9 | 2218.5 | 0.431 |
| KQAPro | HRG-Proposed-triple | 2396 | 1613.9 | 1558.5 | 972.7 | 784.2 | 4929.2 | 0.316 |

解讀：

1. MetaQA / MLPQ / KQAPro 中，HRG-Proposed 的 total proxy 仍低於 BFS，但高於 Spine-Correction，表示它不是單純最低成本，而是用額外 parsing、correction、fallback/ranking 成本換取 evidence coverage。
2. WikiMovies 是 1-hop 小子圖，BFS 的 end-to-end proxy 最低，因此 WikiMovies 不作為 HRG efficiency 主證據。
3. KQAPro 中 Spine-Correction token 很低主要來自 retrieval failure；HRG-Proposed 的成本上升伴隨 answer-in-spine 與 retrieval F1 上升，因此應解讀為 hard-case coverage trade-off。

#### 21.6.2 Paired bootstrap significance

為了避免只報平均值，本文對 `HRG-Proposed-triple - Spine-Correction-triple` 做 paired bootstrap，抽樣 1000 次，報 95% CI。

> 圖：`img/paper_figures/18_bootstrap_hard_case_effects.pdf`。  
> 圖說：Paired bootstrap confidence intervals show that HRG-Proposed has positive evidence-coverage effects on MLPQ and KQAPro, while MetaQA/WikiMovies do not support a universal improvement claim.

| Dataset | Metric | n pairs | Mean diff | 95% CI |
|---|---|---:|---:|---|
| MetaQA | EM | 2400 | -0.0013 | [-0.0096, 0.0079] |
| MetaQA | Answer in spine | 2400 | -0.0370 | [-0.0512, -0.0226] |
| MetaQA | Retrieval F1 | 2400 | 0.0042 | [-0.0054, 0.0139] |
| WikiMovies | EM | 800 | -0.0312 | [-0.0462, -0.0163] |
| WikiMovies | Answer in spine | 800 | 0.0013 | [-0.0063, 0.0088] |
| WikiMovies | Retrieval F1 | 800 | 0.0383 | [0.0303, 0.0464] |
| MLPQ | EM | 1598 | 0.0250 | [0.0150, 0.0363] |
| MLPQ | Answer in spine | 1598 | 0.0576 | [0.0438, 0.0726] |
| MLPQ | Retrieval F1 | 1598 | 0.0255 | [0.0189, 0.0324] |
| KQAPro | EM | 2396 | 0.0530 | [0.0430, 0.0630] |
| KQAPro | Answer in spine | 2396 | 0.1256 | [0.1119, 0.1398] |
| KQAPro | Retrieval F1 | 2396 | 0.0867 | [0.0772, 0.0961] |

這張表是目前最適合回應「HRG 到底有沒有差」的統計證據：MLPQ 與 KQAPro 的三個指標 95% CI 都為正，代表 HRG-Proposed 在 hard cases 中確實比 Spine-Correction 補回更多 answer/evidence coverage。MetaQA 與 WikiMovies 則不支持「HRG 全面提升」，因此它們應被定位為 clean-chain / 1-hop ceiling cases。

#### 21.6.3 HRG prior vs simple relation n-gram priors

為了檢查 HRG 是否只是簡單 relation-pattern prior，本研究加入 candidate-level unigram / bigram / trigram prior ablation。比較對象如下：

1. `HRG-Proposed-gpt-oss-triple` candidate pools。
2. `HRG-Proposed-gpt-oss-json` candidate pools。

這個 ablation 固定同一批 candidates，只改 candidate ranking policy。所有 simple priors 都只使用 offline grammar 中出現過的 KB relation labels，不新增 relation，也不宣稱是完整 HRG decoding。

> 圖：`img/paper_figures/19_prior_pilot.pdf`。  
> 圖說：Within the same HRG-Proposed candidate pools, relation bigram/trigram priors provide a stronger simple-prior comparison than unigram alone, but they do not replace the full HRG-Proposed retrieval pipeline.

`HRG-Proposed-gpt-oss-triple` candidate-level 結果：

| Dataset | Policy | n | Valid | Same as original | Candidate Ret F1 | Answer in candidate edges | Candidate subgraph size |
|---|---|---:|---:|---:|---:|---:|---:|
| MetaQA | hrg_score | 397 | 1.000 | 0.975 | 0.6482 | 0.9632 | 10.92 |
| MetaQA | relation_unigram | 397 | 1.000 | 0.924 | 0.6484 | 0.9698 | 15.05 |
| MetaQA | relation_bigram | 397 | 1.000 | 0.955 | 0.6449 | 0.9582 | 11.01 |
| MetaQA | relation_trigram | 397 | 1.000 | 0.955 | 0.6461 | 0.9598 | 10.97 |
| WikiMovies | hrg_score | 196 | 1.000 | 1.000 | 0.7580 | 1.0000 | 2.82 |
| WikiMovies | relation_unigram | 196 | 1.000 | 1.000 | 0.7580 | 1.0000 | 2.82 |
| WikiMovies | relation_bigram | 196 | 1.000 | 1.000 | 0.7580 | 1.0000 | 2.82 |
| WikiMovies | relation_trigram | 196 | 1.000 | 1.000 | 0.7580 | 1.0000 | 2.82 |
| MLPQ | hrg_score | 398 | 1.000 | 0.910 | 0.1440 | 0.5000 | 31.82 |
| MLPQ | relation_unigram | 398 | 1.000 | 0.628 | 0.1491 | 0.4899 | 38.24 |
| MLPQ | relation_bigram | 398 | 1.000 | 0.688 | 0.1458 | 0.5000 | 27.50 |
| MLPQ | relation_trigram | 398 | 1.000 | 0.721 | 0.1444 | 0.4874 | 26.03 |
| KQAPro | hrg_score | 282 | 0.968 | 0.876 | 0.1117 | 0.1383 | 9.14 |
| KQAPro | relation_unigram | 282 | 0.968 | 0.755 | 0.1064 | 0.1418 | 7.78 |
| KQAPro | relation_bigram | 282 | 0.968 | 0.660 | 0.1093 | 0.1418 | 8.70 |
| KQAPro | relation_trigram | 282 | 0.968 | 0.660 | 0.1093 | 0.1418 | 8.82 |

這張表的解讀應保守：

```text
Bigram/trigram priors make the simple-prior comparison stronger than unigram alone.
They are competitive in some candidate pools, but they do not consistently dominate HRG score or LLM confidence.
Therefore HRG-Proposed should be argued as a full retrieval pipeline with grammar-hit, KG-valid fallback,
and ranking signals, not merely as a relation n-gram frequency model.
```

因此，本節可用來回應「是否只是 relation frequency prior」：不是。Relation unigram / bigram / trigram 可以解釋部分候選排序效果，但無法取代 HRG-Proposed 的完整機制。上述結果定位為 candidate-level reranking analysis，支撐 HRG prior 不只是 relation frequency 的論點。

#### 21.6.4 Evidence causality controls

老師建議的 no-context、wrong-context、masked-answer controls 應作為 answer generator 依賴 retrieval evidence 的檢查：

1. `NoContext`：不給 KG evidence，只給問題。若分數仍高，代表模型可能靠 parametric memory。
2. `WrongContext`：給同資料集其他題目的 retrieved context。若分數接近主方法，代表 evidence 沒有被有效使用。
3. `MaskedAnswerContext`：保留 relation structure 但遮掉 answer entity。若答案仍被猜中，需要小心解讀 evidence faithfulness。

這些 controls 不改變 retrieval pipeline；它們是 answer-generation interface 的 causality sanity check。新 rerun 後若時間不足，至少在 MetaQA / KQAPro 各抽樣一批做 single-model diagnostic，caption 要標為 diagnostic 而不是主結果。

### 21.7 Gold evidence evaluation scope

本文報告以下 automatic evidence-level diagnostics：

1. **Endpoint answer coverage**：gold answer 是否出現在 retrieved edges / spine edges。
2. **Answer-conditioned retrieval recall / precision / F1**：以 gold answer 是否被 retrieved evidence 覆蓋為基礎。
3. **Grammar-hit conditioned success**：可以比較 grammar_hit=true / false 的 answer F1、answer-in-spine、failure rate。
4. **Failure taxonomy**：把題目分成 no_candidates、no_valid_chain、KG-valid fallback、LLM correction、grammar fallback 等類別。
5. **Case analysis**：展示 Spine-Correction 失敗但 HRG-Proposed 成功的代表題目。

本文不宣稱以下項目已完整解決：

1. **Gold edge precision / recall**：除非 metadata 中有 `gold_path_parts`，否則無法知道哪些 retrieved edges 是唯一正確 evidence。
2. **Gold program match**：KQAPro 的 operator / qualifier program match 需要額外的 program-level evaluator。
3. **Human evidence sufficiency**：目前沒有人工標註，不能說輸出的 evidence 一定足以讓人驗證答案，只能說 evidence structure 可檢查且 answer coverage 可量化。

因此本文採用兩層寫法：

```text
We report endpoint evidence coverage as automatic evidence-level diagnostics.
Gold path / program evidence precision and human sufficiency evaluation are left as future work,
because they require gold path/program annotations or human evidence labels.
```

#### 21.7.1 Answer normalization and scoring

主表中的 EM / Hits@1 / answer-set F1 需要固定 normalization，否則多答案 KGQA 的數字很容易被質疑。本文採用以下規則：

1. 對 predicted answer 與 gold answer 做 lowercase、trim、whitespace normalization，並移除常見標點。
2. 多答案輸出以 `|`、換行或明確分隔符切成 answer set；set matching 不看順序。
3. EM 表示 normalized predicted set 與 gold set 完全相同。
4. Hits@1 表示第一個 predicted answer 命中任一 gold answer。
5. Answer-set precision = `|predicted set intersect gold set| / |predicted set|`，recall = `|predicted set intersect gold set| / |gold set|`，F1 為 harmonic mean。
6. 空輸出、format error、`I don't know` 在 answer quality 中算錯；可以另報 abstain / failure rate。
7. 若資料集有官方 evaluator 或 alias table，應優先使用官方規則；否則必須明確說明 alias 不自動合併，避免把 normalization 當成隱藏增益。

### 21.8 Evidence-level 可解釋性案例

以下用 MetaQA 2-hop 案例展示本研究的可解釋性。

成功案例：

```text
Question:
who directed the films starred by [Linda Evans]

Selected entity:
Linda Evans

Selected relation chain:
starred_actors -> directed_by

Derived signed execution trace:
starred_actors[r-] -> directed_by[r+]

Spine edges:
Mitchell --starred_actors--> Linda Evans
Mitchell --directed_by--> Andrew V. McLaglen

Answer:
Andrew V. McLaglen

Context tokens:
44

Subgraph size:
2
```

這個案例可以用來說明：

1. 系統不是只輸出答案，而是輸出問題對應的 KG 結構。
2. `starred_actors -> directed_by` 是 parser 輸出的原始 relation labels；`starred_actors[r-] -> directed_by[r+]` 是 validation 後推導出的 traversal direction trace。
3. `starred_actors[r-]` 不代表資料中有反向 relation，只代表從 `Linda Evans` 沿 stored triple `Mitchell --starred_actors--> Linda Evans` 反向走到 `Mitchell`。
4. 每條 spine edge 都能對應到 KB triple。
5. 最終答案 `Andrew V. McLaglen` 來自第二條 edge 的 tail。
6. 對比 BFS，這種 evidence 不是大鄰域，而是明確 path。

論文放法：

```text
Question -> Parsed structure -> Executable spine -> Answer
```

其中 parsed structure 與 executable spine 就是本研究的可解釋性來源。

---

## 22. 演算法偽代碼

### 22.1 LLM-chain-guided KG-RAG

```text
Algorithm: LLM-chain-guided KG-RAG

Input:
  question q
  knowledge graph G
  relation vocabulary R
  optional alias map A
  LLM M
  optional HRG grammar prior H

Output:
  answer y
  evidence subgraph S

1. C = LLM_Parse_TopK(q, R)
2. for each candidate c in C:
       c.entity_node = Ground(c.entity, A, G)
       c.kb_result = ExecuteChain(G, c.entity_node, c.chain)
       c.score = ScoreCandidate(c)

3. if no candidate in C is valid:
       C_corr = GenerateCorrectionCandidates(q, C)
       for each candidate c in C_corr:
           c.entity_node = Ground(c.entity, A, G)
           c.kb_result = ExecuteChain(G, c.entity_node, c.chain)
           c.score = ScoreCandidate(c)
       C = C union C_corr

4. if no candidate in C is valid and KG-valid fallback is enabled:
       C_kg = DeterministicKGValidFallback(q, C, H)
       optionally rerank C_kg with LLM
       C = C union C_kg

5. V = {c in C | c.kb_result.valid = true}

6. if V is empty:
       return failure "no_valid_chain"

7. for each valid candidate v in V:
       S_v = BuildStrictSpine(G, v.entity_node, v.chain)
       subgraph_score_v = ScoreSubgraph(v, S_v)

8. S = argmax_v subgraph_score_v

9. context = Serialize(S)

10. y = LLM_Answer(q, context)

11. return y, S
```

這個 pseudo-code 對應 `Spine-Only`、`Spine-Correction`、`Spine-Correction-KGValidFallback` 與 `HRG-Proposed` family。差別在於是否啟用 correction、KG-valid fallback、HRG grammar features、expansion 與 LLM rerank。

### 22.2 Chain Execution

```text
Algorithm: ExecuteChain

Input:
  graph G
  start node e
  chain [r1, r2, ..., rL]

Output:
  valid flag
  step sizes
  final frontier
  failed hop
  signed execution trace

1. frontier = {e}
2. step_sizes = []

3. for t = 1 to L:
       next_frontier = {}
       for each node v in frontier:
          for each KB triple edge = (h, rt, t_node) whose relation label matches rt
               and whose endpoint set contains v:
               if v == h:
                   next_frontier.add(t_node)
                   trace.add(rt[r+])
               else:
                   next_frontier.add(h)
                   trace.add(rt[r-])

       step_sizes.append(|next_frontier|)

       if |next_frontier| = 0:
           return invalid, step_sizes, empty, t

       frontier = next_frontier

4. return valid, step_sizes, frontier, None, trace
```

### 22.3 Strict Spine Construction

```text
Algorithm: BuildStrictSpine

Input:
  graph G
  start node e
  valid chain [r1, r2, ..., rL]

Output:
  edge set S

1. frontier = {e}
2. S = {}

3. for each relation rt in chain:
       next_frontier = {}
       for each node v in frontier:
           for each KB triple edge = (h, rt, t_node) whose relation label matches rt
               and whose endpoint set contains v:
               S.add(edge)
               next_frontier.add(other_endpoint(v, edge))
       frontier = next_frontier

4. return S
```

Relation chain 只儲存 KB 中已存在的 relation label；輸出的 evidence edge 保留 KB triple 原本的 `(head, relation, tail)` 方向。
若要在論文中呈現 direction，請從 executed triples 推導 `r+ / r-` trace；不要把 `r+ / r-` 放進 parser vocabulary 或 KG relation vocabulary。

### 22.4 Deterministic KG-Valid Fallback

```text
Algorithm: Deterministic KG-Valid Chain Fallback

Input:
  question q
  seed entity e
  max depth D
  beam width B
  branch limit K

Output:
  executable candidate chains

1. if seed candidates contain non-empty chains:
       target_depths = lengths of seed chains within [1, D]
   else:
       target_depths = {1, 2, ..., D}

2. frontier_beams = [(score=0, chain=[], nodes={e})]
3. completed = []

4. for depth = 1 to max(target_depths):
       next_beams = []
       for each beam in frontier_beams:
           options = actual relations reachable from beam.nodes
           rank options by question overlap, preferred relations, grammar score
           keep top K options

           for each relation r in options:
               next_nodes = ExecuteOneHop(beam.nodes, r)
               new_chain = beam.chain + [r]
               new_score = beam.score + relation_score + grammar_bonus - compactness_penalty
               next_beams.add((new_score, new_chain, next_nodes))

       frontier_beams = top B next_beams
       if depth in target_depths:
           completed.add(frontier_beams)

5. return top executable chains from completed
```

### 22.5 HRG-GrammarFirst

```text
Algorithm: HRG-GrammarFirst Retrieval

Input:
  question q
  knowledge graph G
  optional alias map A
  HRG grammar prior H
  LLM M for optional rerank

Output:
  answer y
  evidence subgraph S

1. E = ExtractEntityCandidates(q, A, G)
   if E is empty:
       return failure "no_candidates"

2. C = {}
   for each entity e in E:
       P = ExtractRelationPathBank(H)
       C_e = {p in P | ExecuteChain(G, e, p) is valid}
       add C_e to C with source = "grammar_first"

3. if C is empty:
       return failure "no_candidates"

4. score C by relation cue coverage, question-relation relevance,
   HRG compatibility features, grammar score, and KG frontier size

5. optionally rerank executable chains C with LLM M

6. for each candidate c in C:
       c.kb_result = ExecuteChain(G, c.entity_node, c.chain)
       c.score = ScoreCandidate(c)

7. V = {c in C | c.kb_result.valid = true}
   if V is empty:
       return failure "no_valid_chain"

8. for each valid candidate v in V:
       S_v = BuildStrictSpine(G, v.entity_node, v.chain)
       subgraph_score_v = ScoreSubgraph(v, S_v)

9. S = argmax_v subgraph_score_v
10. context = Serialize(S)
11. y = LLM_Answer(q, context)
12. return y, S
```

這個 algorithm 對應目前程式中的 `use_grammar_first_retrieval=True`。它和 HRG-Proposed 的核心差異是：candidate chain 不是由 LLM 先提出，而是由 HRG relation path-bank 先產生，再由 KG 從 grounded entity 做 executable filtering；HRG grammar 直接控制 candidate search space。

---

## 23. 為什麼各資料集結果不同

這一節說明：結果差異不是隨機，而是由資料集語意和方法假設的貼合程度決定。

HRG-Proposed 的核心假設是：

```text
KGQA question ≈ topic entity + ordered relation chain
answer can be reached by executing this chain on the KG
HRG grammar can provide structural priors when strict relation spine is insufficient
```

更精確地說，HRG-Proposed 仍假設 LLM top-k parse、correction 或 KG-valid fallback 至少能把候選空間帶到合理 relation-chain 附近；它主要是 LLM-chain-guided retrieval。新增的 HRG-GrammarFirst 放寬這個假設：它不使用 LLM 預測 hop 數或 relation chain 作為 search-space 前提，而是從 grounded entity 枚舉 1..D hop KG-valid chains，再用 grammar / relevance / compactness / optional LLM rerank 排序。不過 GrammarFirst 仍然假設答案可由 single-anchor relation-chain spine 表示，不能直接解決 count、comparison、qualifier program。

因此，資料集越像乾淨的 relation-chain traversal，結果越好；越需要跨語言對齊、operator、filter、count、comparison、qualifier，結果越困難。GrammarFirst 的新結果需等 rerun 完成後回填，不能和本節既有 HRG-Proposed 數字混成同一批 evidence。

### 23.1 Dataset Fit Summary

| Dataset | 結果現象 | 主要原因 | 報告定位 |
|---|---|---|---|
| MetaQA | HRG-Proposed-triple 用 BFS 4.28% context 且 F1 高於 BFS | 問題天然是 movie-domain relation chain，strict spine 已很強 | main-method compactness 證據 |
| WikiMovies | HRG-Proposed 不是最強；BFS/Spine 已接近 ceiling | 幾乎都是 1-hop，BFS 子圖本來就很小 | ceiling case |
| MLPQ | HRG-Proposed 比 Spine-Correction 更接近 BFS，但仍低於 BFS | 跨語言 entity/relation 對齊困難 | HRG 補強 + limitation |
| KQAPro | Spine-Correction 幾乎失敗；HRG-Proposed-triple 補回 executable evidence coverage，但 BFS 本身也低 | 問題是 semantic program，不是單純 chain | out-of-scope stress test |

摘要句：

```text
方法成功與否主要取決於 dataset semantics 是否能對齊到 executable relation chain。
MetaQA 最符合這個假設，所以能在貼近 BFS answer quality 的同時大幅減少 token。
WikiMovies 多數是 1-hop，BFS 本來就很小，所以改善空間有限。
MLPQ 雖然也是 path QA，但跨語言 entity/relation 對齊是主要瓶頸。
KQAPro 則包含 count、filter、compare、verify、qualifier 等 program semantics，單純 relation spine 不足以表示完整問題。
```

### 23.2 MetaQA 為什麼最適合

MetaQA 的 KG schema 很小、很乾淨，relation 只有 9 種：

```text
directed_by
written_by
starred_actors
release_year
in_language
has_genre
has_tags
has_imdb_rating
has_imdb_votes
```

KB triple 例子：

```text
Kismet | directed_by | William Dieterle
Kismet | written_by | Edward Knoblock
Kismet | starred_actors | Marlene Dietrich
Kismet | in_language | English
```

問題也天然接近 relation chain：

```text
who directed the films starred by [Linda Evans]
```

可轉成：

```text
Linda Evans --starred_actors[r-]--> Mitchell --directed_by[r+]--> Andrew V. McLaglen
```

這裡的 `r- / r+` 只是 validation 後從 executed triples 推導出的 traversal direction trace。原始 KG relation 仍然是 `starred_actors` 與 `directed_by`，系統沒有新增 reverse relation label，也沒有要求 parser 產生 `starred_actors^-1` 這種 relation vocabulary。

因此 executable spine 很容易發揮作用，而 HRG-Proposed 可以進一步用更小 context 維持接近 BFS 的品質：

```text
MetaQA BFS, four valid models average:
EM 0.4504 / F1 0.5528 / ctx tokens 4429.98 / subgraph 369.08

MetaQA Spine-Only-json, four valid models average:
EM 0.5521 / F1 0.6401 / ctx tokens 601.70 / subgraph 27.34

MetaQA Spine-Correction-json, four valid models average:
EM 0.5583 / F1 0.6501 / ctx tokens 606.54 / subgraph 27.56

MetaQA Spine-Correction-triple, four valid models average:
EM 0.5221 / F1 0.6257 / ctx tokens 287.79 / subgraph 30.57

MetaQA HRG-Proposed-triple, four valid models average:
EM 0.4933 / F1 0.5948 / ctx tokens 189.55 / subgraph 19.33
```

解讀：

```text
MetaQA 上，HRG-Proposed-triple 不是單純少拿資料，而是在 F1 高於 BFS 的同時，
把 evidence 壓成可執行、可印出的 HRG-guided relation spine，context 只有 BFS 的 4.28%。
```

MetaQA 3-hop 是關鍵細節：

```text
gpt-oss main result:
BFS 3-hop EM = 0.0050
Spine-Correction-json 3-hop EM = 0.1600
```

這表示 BFS 雖然 retrieval recall 高，但 LLM 面對大量 context 時未必能找到正確 reasoning path；spine 明確化 path有助於 long-hop answer generation。這個 hop-level 表是 gpt-oss 單模型補充分析；主結論仍以 HRG-Proposed 的跨模型平均為準。

### 23.3 WikiMovies 為什麼接近但不突出

WikiMovies 多數問題是 1-hop：

```text
what films did Michelle Trachtenberg star in?
Joe Thomas appears in which movies?
```

這類問題 BFS 很容易直接抓到答案，而且原始 BFS 子圖本來就小。

結果：

```text
WikiMovies BFS, four valid models average:
EM 0.8025 / F1 0.8574 / ctx tokens 50.12 / subgraph 3.30

Spine-Correction-json, four valid models average:
EM 0.8988 / F1 0.9210 / ctx tokens 79.43 / subgraph 3.42

HRG-Proposed-triple, four valid models average:
EM 0.7150 / F1 0.7843 / ctx tokens 31.32 / subgraph 2.82
```

解讀：

```text
WikiMovies 是 BFS ceiling case。
因為問題多為 1-hop，BFS context 本來就小，所以 token reduction 空間有限。
Spine-Correction-json 在此資料集品質高，但 JSON serialization token 較多。
HRG-Proposed-triple 能減少 context，但不是最接近 BFS 的品質；因此 WikiMovies 應作為 ceiling case，而不是 HRG-Proposed 的主要成功證據。
```

### 23.4 MLPQ 為什麼困難

MLPQ 雖然是 path QA，但它是 multilingual KG path QA。問題可能是英文，但 gold path 會跨英文 DBpedia 和中文 DBpedia。

例子：

```text
English question
-> English DBpedia relation
-> English entity
-> Chinese equivalent entity
-> Chinese DBpedia relation
-> Chinese answer
```

實際 gold path 形態：

```text
dbpedia.org/resource/Russian_destroyer_Marshal_Shaposhnikov
dbpedia.org/property/shipNamesake
dbpedia.org/resource/Boris_Shaposhnikov
zh.dbpedia.org/resource/鮑里斯·米哈伊洛維奇·沙波什尼科夫
zh.dbpedia.org/property/allegiance
zh.dbpedia.org/resource/蘇聯
```

因此它的困難不是「有沒有 path」，而是：

```text
英文問題文字 -> 英文 relation token -> 中文 equivalent entity -> 中文 relation token -> 中文答案
```

結果：

```text
MLPQ BFS, four valid models average:
EM 0.2381 / F1 0.3078 / ctx tokens 4983.34

Spine-Correction-json, four valid models average:
EM 0.1306 / F1 0.2185 / ctx tokens 101.48

HRG-Proposed-json, four valid models average:
EM 0.1544 / F1 0.2514 / ctx tokens 359.79

HRG-Proposed-triple, four valid models average:
EM 0.1781 / F1 0.2498 / ctx tokens 209.30
```

解讀：

```text
BFS 不理解語意，但大範圍掃描容易把答案附近 facts 包進 context。
Spine-Correction 很省 token，但 chain parse / relation alignment 容易失敗。
HRG-Proposed 透過 deterministic fallback、grammar prior 與 candidate ranking features 改善 retrieval，使結果比 Spine-Correction 更接近 BFS，但仍未完全貼近 BFS。
```

因此 MLPQ 的主要瓶頸是 cross-lingual grounding 和 relation canonicalization，不是 strict spine retrieval 本身。

### 23.5 KQAPro 為什麼最難

KQAPro 不是單純 relation chain。它的問題常是 semantic program。

常見 query families：

```text
qualifier: 2806
comparison: 2152
relation_query: 1786
attribute: 1294
count: 1291
verify: 1235
relation_path: 1233
```

常見 program functions：

```text
Find
Relate
FilterConcept
FilterNum
Count
VerifyStr
SelectBetween
QueryRelationQualifier
QueryAttrQualifier
```

例子：

```text
How many Pennsylvania counties have a population greater than 7800 or a population less than 40000000?
```

這需要：

```text
FindAll -> FilterNum -> Or -> Count
```

不是一條 `[relation1, relation2, relation3]` 可以完整表示。

另一類 qualifier 問題：

```text
Who was the prize winner when Mrs. Miniver got the Academy Award for Best Writing, Adapted Screenplay?
```

這需要 statement node / qualifier traversal，而不是普通 entity-to-entity chain。

KQAPro KB 也比 MetaQA 複雜很多：

```text
1,609,139 triples
1,527 relations
relations include:
ASIN
ASIN::statement
Alexa rank
Alexa rank::statement
...
```

結果：

```text
KQAPro BFS, four valid models average:
EM 0.0804 / F1 0.0842 / ctx tokens 3595.48

Spine-Correction-json, four valid models average:
EM 0.0229 / F1 0.0237 / ctx tokens 7.33

HRG-Proposed-triple, four valid models average:
EM 0.0754 / F1 0.0854 / ctx tokens 783.54
```

解讀：

```text
Spine-Correction 幾乎抓不到 valid chain。
HRG-Proposed 能補回 executable evidence coverage，使 F1 接近同樣很低的 BFS，而且 context 只有 BFS 的 21.79%。
KQAPro 應定位為 out-of-scope stress test；完整解法需要 operator-aware parser 和 statement-aware traversal。
```

### 23.6 方法局限可以如何分類

1. **Structural fit limitation**

適合：

```text
Who directed movies starred by X?
X --starred_actors[r-]--> Movie --directed_by[r+]--> Director
```

不適合：

```text
How many entities satisfy condition A or condition B?
Which entity has the largest value?
Is value X equal to attribute Y?
```

2. **Grounding limitation**

如果 topic entity 或 relation token 對不到 KG，chain 再合理也會 invalid。這在 MLPQ 特別明顯。

3. **Cross-lingual canonicalization limitation**

MLPQ 需要：

```text
English question phrase -> English DBpedia relation
English entity -> Chinese equivalent entity
Chinese relation -> answer
```

目前方法沒有完整 relation canonicalization layer。

4. **Operator / qualifier limitation**

KQAPro 需要：

```text
count
filter
compare
verify
statement qualifier
```

目前 ordered relation chain 表達不了這些 operation。

總結句：

```text
這些結果不是互相矛盾，而是定義了 HRG-Proposed 的適用邊界。
當 KGQA 問題能被表示成 executable relation chain，本方法能用更少 token 提供更可解釋的 evidence。
當主要瓶頸變成跨語言對齊、operator、或 qualifier semantics 時，單純 spine retrieval 就不夠；HRG prior 可以補回部分 evidence coverage，但仍需要更完整的 semantic parser。
```

---

## 24. Limitation

目前方法的限制：

1. HRG-Proposed 對 relation-chain-friendly 任務可有效壓縮 context；MetaQA 是 compactness 主證據。
2. WikiMovies 是 1-hop 小子圖任務，BFS 本身已接近飽和；HRG-Proposed 不一定最強，應定位為 ceiling case。
3. 對跨語言 relation canonicalization 仍不完整；MLPQ 顯示 HRG prior 能補強 Spine-Correction，但仍未完全貼近 BFS。
4. 對 KQAPro 這類 qualifier-heavy / operator-heavy 問題，HRG-Proposed 能補回部分 evidence coverage，但仍不是完整 operator-aware solution。
5. grammar matching 目前主要是 label subset prior，不是完整圖文法推導；因此 HRG-Proposed 是 HRG-guided retrieval main method，而非完整 HRG decoder。
6. claim faithfulness 等指標目前是 heuristic，不是人工驗證。
7. correction 增益需要看觸發率與有效率，不能只看總分；合併四個有效模型後，MetaQA correction 平均增益約為 json `+0.0063 EM / +0.0100 F1`、triple `+0.0054 EM / +0.0102 F1`，屬於穩定但不大的增益。從 trigger statistics 看，LLM correction 不是主要 empirical driver，應降級為 fallback module 的一部分。
8. 若 topic entity grounding 錯誤，後面 relation chain 再正確也會失敗。
9. `qwen3.5` 不應只標成一般失敗。目前可用 A40 GPU compute capability 8.6 不支援 `Qwen/Qwen3.5-35B-A3B-FP8` 在 Transformers 的 FP8 runtime；模型會退回 bf16 dequantization 並 load-time OOM。因此它是硬體 / checkpoint 不相容，排除於 main aggregation。
10. 主結果與 robustness / ablation 結果使用不同模型集合；兩者不混成同一個平均，圖表 caption 標明模型集合與 aggregation method。
11. 10% / 20% KG perturbation 顯示 HRG-Proposed 能在 MLPQ / KQAPro 補回 strict spine 缺少的 evidence coverage，但也帶來 context 與 OOM 上升；因此 KG-valid fallback 與 candidate ranking budget 屬於後續優化方向。
12. MetaQA clean gpt-oss 下 HRG-Proposed-triple 低於 Spine-Correction-triple，因此不能宣稱主方法在 MetaQA answer quality 最佳；較準確說法是 HRG-Proposed 在多模型平均與 perturbation 情境提供 quality-cost / robustness trade-off。
13. `r+ / r-` 是 derived execution trace，不是新增 relation label。若論文使用 signed relation notation，必須同時保留原始 KG triple，以免讀者誤以為資料被擴充了 reverse relations。
14. Expansion 不是主方法必要條件；若新實驗包含 expansion，必須以 `NoExpansion` / `Expansion` 明確分 row，避免把擴邊效果混進 strict-spine claim。
15. HRG-GrammarFirst 能降低對 LLM hop/chain 預測的依賴，但它會擴大 KG-valid candidate search space；若 relation relevance、answer type、frontier compactness 或 context budget 控制不足，可能產生「可執行但不對題」的 chains。新 rerun 完成前只能把它寫成新增 variant / pending experiment，不能寫成已證明優於 HRG-Proposed。
16. 目前主流程主要是 failure-triggered fallback；程式已有 `use_low_confidence_valid_chain_fallback` 這類低信心觸發開關，但固定 ablation 預設未宣稱其效果。若要主張 correction 能處理「valid but semantically wrong」candidate，必須另跑 low-confidence-triggered ablation。

### 24.1 Additional Experimental Scope

以下項目屬於後續實驗範圍；在本文中不作為主要實證 claim：

1. **Additional baseline scope**：token-budgeted BFS、degree-capped BFS、relation-similarity-pruned BFS、relation-frequency reranker、beam-search relation path、symbolic endpoint answer 等方法可作為後續比較；新 rerun 完成前不把它們的結果混進主表。
2. **HRG vs simpler priors**：unigram / bigram / trigram relation-prior ablation 顯示簡單 n-gram prior 能解釋部分 candidate ranking，但不能取代 HRG-Proposed 的 grammar-hit、KG-valid fallback 與 ranking signals；本文不宣稱 HRG 已全面優於所有簡單 relation-pattern prior。
3. **Oracle error decomposition**：gold entity + predicted relation、predicted entity + gold relation、gold entity + gold relation、all predicted 等 oracle 設定可作為後續錯誤分解，使 MLPQ / KQAPro 的失敗比例能更精細地拆成 grounding、relation linking、retrieval 或 generation。
4. **End-to-end cost**：本文報告 parse1、correction、parse2、context 的 online token proxy；它用於比較方法成本輪廓，不等同於精確 API billing cost。
5. **Evidence correctness**：本文可以輸出 chain、spine edges 與 matched rule；gold path / gold program 與人工 evidence sufficiency 不作為本文主 claim。
6. **Statistical significance**：本文報告 paired bootstrap confidence interval；多次 decoding repeat 與完整 significance matrix 可作為後續統計分析。

本文已完成的是 HRG-guided executable compact evidence retrieval 的系統、主要結果、evidence diagnostics 與初步 robustness 分析；更廣泛的 graph retrieval / relation-prior baseline 比較需要等新 full rerun 後再決定主文或 supplementary 放法。

---

## 25. Future Work

後續可以延伸：

1. Relation canonicalization：建立跨資料集、跨語言的 relation normalization layer
2. Statement-aware traversal：針對 KQAPro qualifier / statement node 設計專門 traversal
3. Operator-aware parsing：讓 parser 輸出 filter、count、compare、argmax 等 operation
4. Better correction analysis：統計 correction trigger rate、success rate、failure type
5. Learned reranker：用訓練資料學 candidate / subgraph ranking
6. Evidence supervision：加入 gold path 或人工 evidence label
7. Adaptive retrieval：根據問題類型決定 strict spine、BFS 或 hybrid retrieval
8. Multi-anchor retrieval：支援多 topic entities、intersection / join / constraint-style questions
9. Answer-type constraints：用 KG domain / range、entity type 或 expected answer type 過濾 fallback candidates
10. Schema retriever：在 LLM parsing 前先取 top-k relation schema / aliases / bilingual labels，降低 relation vocabulary 太大造成的 no_candidates 或錯 relation
11. Low-confidence-triggered correction：除了 no valid chain，也在 textual relevance 低、frontier 過大、answer type 不合理時觸發 correction / fallback。
12. Top-m path evidence：把目前「所有 valid paths union」改成 per-path scoring，只保留 top-m endpoints 或 top-m supporting paths，降低 generic relation 的 frontier explosion。
13. GrammarFirst semantic filters：在 HRG-GrammarFirst enumeration 後加入 relation description similarity、domain/range、answer type 與 operator-aware filters，避免可執行但不對題的候選。

---

## 26. Conclusion

本研究提出 HRG-Proposed KG-RAG，將 KGQA retrieval 從無結構 BFS 子圖展開轉為 HRG structural prior-guided executable relation-chain retrieval。合併四個有效模型後，MetaQA 顯示 HRG-Proposed-triple 能以 BFS `4.28%` 的 final evidence context 取得高於 BFS 的 F1；MLPQ 顯示 HRG prior 比 Spine-Correction 更接近 BFS，但跨語言 relation canonicalization 仍是瓶頸；KQAPro 則應定位為 out-of-scope stress test，HRG-Proposed 能比 strict spine ablation 補回 executable evidence coverage，但不代表已解決 operator-heavy KGQA。此外，系統能輸出 selected entity、relation chain、derived signed execution trace、matched grammar prior、spine edges 與 final context，因此提供 evidence-level explainability。Spine-Only、Spine-Correction、KGValidFallback、relation-prior rows、NoExpansion/Expansion rows 與 HRG-GrammarFirst 在本文中作為 ablation / variant，用來拆解 strict spine、KG-validated candidate recovery、HRG prior、expansion 與 grammar-first search 的貢獻。HRG-GrammarFirst 已和目前程式碼及 rerun matrix 對齊，但其數據需等新 full rerun 完成後再回填。對於跨語言、operator-heavy 或 qualifier-heavy 資料集，目前結果顯示仍需進一步加入 relation canonicalization、operator-aware parsing、answer-type constraints、low-confidence correction 與 statement-aware reasoning。
