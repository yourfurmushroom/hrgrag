# HRG-Proposed KG-RAG 完整方法說明稿

> 用途：這份文件是 HRG-Proposed KG-RAG 的完整技術說明。  
> 報告主軸：**以 HRG structural prior 引導可執行 relation spine retrieval，在答案品質貼近 BFS baseline 的條件下減少 context tokens，並輸出可檢查的 KG evidence structure。**

---

## 1. 研究主題與一句話摘要

本研究處理的是 Knowledge Graph Question Answering，也就是給定一個自然語言問題與一個知識圖譜，系統必須從 KG 中找出支持答案的 evidence，並讓 LLM 根據 evidence 輸出答案。

傳統 KG-RAG 常用做法是從題目中的 topic entity 出發做 BFS，取回附近多跳子圖後交給 LLM。這種方法通常召回率高，但缺點是 context 很容易爆炸，尤其在 multi-hop 問題中，LLM 會看到大量與答案無關的邊，最後不一定能沿著正確推理路徑回答。

本研究的主方法是 **HRG-Proposed KG-RAG**：offline 階段先從 KG 局部子圖抽取 HRG-like grammar rules，形成 relation structure prior；online 階段讓 LLM 把問題解析成 topic entity 與 ordered relation chain，再以 KB validation 確認這條 chain 是否可執行。如果 initial chain 走不通，系統會使用 correction、grammar prior 與 deterministic KG-valid fallback 產生修正候選；若 chain 走通，則以 strict evidence spine 為核心。HRG 的主要作用是 grammar-hit candidate selection、KG-valid fallback 與 ranking features。

目前程式另外加入 **HRG-GrammarFirst** 作為 diagnostic ablation。它不是把 `HRG-Proposed` 改名，而是新增一條 retrieval variant：只先做 entity grounding，不要求 LLM 先猜 relation chain / hop 數；系統會先從 offline HRG rules 抽 relation path-bank，再由 KG 驗證哪些 paths 可從 entity seed 執行，最後用 HRG grammar features、question-relation relevance、frontier compactness 與 optional LLM rerank 排序。這個 row 專門回答「LLM 一開始 hop 猜錯時，HRG 能不能自己產生候選路徑」。latest ablation 顯示 GrammarFirst 能產生 executable path-bank，但 semantic relevance 與 answer-type filtering 仍是瓶頸；因此它應被寫成 HRG retrieval-engine extension 的診斷結果，而不是取代 HRG-Proposed 的主方法。

目前程式版本已同步到 `RANKING_POLICY=lax-hrg-prior-v1`：candidate ranking 以 KB-valid 為 hard gate，並將 same-arity grammar hit、ordered-path grammar hit、grammar label hit、grammar score 與 matched rule count 放在 LLM rerank score 之前。這是為了讓 HRG structural prior 在 candidate selection 中有明確位置。本文現有數據表先沿用已完成 artifacts；新的 `artifacts_laxhrg` rerun 完成前，不把新 ranking policy 的結果數字混入主表。

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

1. **HRG-Proposed 是主方法，GrammarFirst 是 diagnostic ablation**：本文主方法不是單純 Spine-Correction，而是結合 executable relation spine、HRG structural prior、KG-valid fallback 與 grammar-aware ranking signals 的 HRG-Proposed KG-RAG。Spine-Only、Spine-Correction、Spine-Correction-KGValidFallback、HRG-Proposed-NoExpansion/Expansion、HRG-GrammarFirst-NoExpansion/Expansion 是 ablation / variant，用來拆解 strict spine、fallback、HRG prior、expansion 與 grammar-first search 的貢獻。
2. **Compact retrieval**：MetaQA 主敘事保留 200-per-hop legacy four-model dump。四模型平均下，Baseline-BFS 的 F1 / context tokens 是 `0.5528 / 4430.0`；HRG-Proposed-triple 是 `0.5948 / 189.6`，context 為 BFS 的 `4.28%`。`gpt-oss` 單模型中，BFS 是 `0.6109 / 4415.8`，HRG-Proposed-triple 是 `0.5718 / 77.3`，context 為 BFS 的 `1.75%`。因此本文主軸是 final evidence context compression + traceable HRG-guided evidence，不是只看單一 EM/F1 排名。
3. **HRG claim boundary**：latest 50-per-hop final-metrics 用來補 fair BFS、relation n-gram、GrammarFirst 與 evidence diagnostics。這些新增消融顯示 strict spine 與 simple relation priors 在 MetaQA 很強；因此 HRG 應定位為 structural prior、grammar decomposition、KG-valid fallback / ranking signals 與 evidence traceability，而不是保證在每個 clean small-schema ablation 上都最高分。
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
| MetaQA main 200-per-hop | HRG-Proposed-triple vs BFS | F1 `0.5948` vs `0.5528` | context ratio `4.28%`, subgraph ratio `5.24%` | 四模型主結果：HRG-guided evidence 以極小 final context 保持並提升 answer-set F1。 |
| MetaQA gpt-oss 200-per-hop | HRG-Proposed-triple vs BFS | F1 `0.5718` vs `0.6109` | context ratio `1.75%`, subgraph ratio `2.00%` | 單模型圖表來源：品質略降但壓縮非常強，支撐 quality-cost trade-off 圖。 |
| MetaQA latest ablation | HRG-Proposed-triple vs BFS | F1 `0.6114` vs `0.5277` | context ratio `13.49%` | latest 50-per-hop rerun 補上 evidence diagnostics；HRG-Proposed 仍高於 BFS。 |
| MetaQA latest ablation | RelationBigram/Trigram control | F1 約 `0.657` | context 約 `754-758` | simple priors 是強 baseline；這是 HRG claim boundary，不是主方法無效。 |

結果敘述可採用以下版本：

```text
如果只看 EM，會把 retrieval 是否有效、context 是否被壓縮、evidence 是否可檢查全部混在一起。
本文的主問題不是讓模型背答案，而是讓 KGQA 在較小且可驗證的 evidence 上回答。
因此本文同時報告 EM/F1、retrieval recall、faithfulness、context tokens、subgraph size 與 failure counts。
在 MetaQA 200-per-hop 主結果中，HRG-Proposed 用 BFS 4.28% 的 final evidence context 取得更高 F1；
在 latest 50-per-hop 消融中，HRG-Proposed 仍高於 BFS，並額外提供 answer-in-context、grammar decomposition 與 truncation diagnostics。
因此本文主張 HRG 的價值在 structural prior、fallback/ranking signal、evidence traceability 與 final context compression，而不是保證所有消融 row 的 EM/F1 第一。
```

本文不採用以下過度 claim：

1. 本方法不宣稱在所有資料集全面優於 BFS。
2. HRG-Proposed 是本文主方法，Spine-Only / Spine-Correction / KGValidFallback / relation-prior rows 是 ablation；HRG-GrammarFirst 是 diagnostic variant，用來檢驗 HRG 是否能在不依賴 LLM hop/chain 預測的情況下產生候選路徑。它目前揭示的是 semantic filtering bottleneck，可作為 future-work motivation。
3. Token 減少必須和 answer quality、evidence coverage 一起解讀；只有在品質仍接近 baseline 時，compact context 才有意義。
4. MLPQ、KQAPro 主要用來說明 HRG prior 對 hard cases 的補強效果與目前限制，不被包裝成已完整解決的資料集。

### 1.1.1 核心貢獻

本文的貢獻不以「LLM 產生 chain 再去 KG 執行」作為單一主張，因為這和既有 executable semantic parsing / path retrieval 很接近。更準確的貢獻可以拆成四層：

1. **Executable relation-spine retrieval**：把 LLM 輸出的自然語言意圖轉成可在 KG 上驗證的 ordered relation chain，並只序列化通過 KB validation 的 evidence spine，而不是把 BFS neighborhood 全部丟給 LLM。
2. **KG-validated fallback and candidate recovery**：correction 不是每題都做，而是當 initial candidates 全部 invalid 時才啟動；每個 correction / fallback candidate 都必須再次通過 KB validation。從目前 trigger statistics 看，可量化增益主要來自 deterministic KG-valid fallback、candidate recovery 與後續 ranking signals；LLM correction 是其中一個保守啟動的補救模組。
3. **HRG-like structural prior for candidate selection and ranking**：HRG-like grammar 不是裝飾，也不是完整 HRG decoder；它主要在 grammar-hit candidate selection、KG-valid fallback 與 ranking features 中提供 relation-structure prior。KQAPro 與 MLPQ 的結果顯示，當 strict spine evidence coverage 不足時，HRG-Proposed 能比 Spine-Correction 補回更多可用 evidence。
4. **Grammar-first retrieval diagnostic**：新增 HRG-GrammarFirst，將 HRG prior 從「LLM candidate 的 rerank/recovery signal」推進到「candidate generation/search-space control」。latest ablation 顯示它能把 HRG 推到 candidate generation 層，但若缺少 semantic relation filtering，executable path-bank 會帶來 off-target candidates；因此它在本文中應作為 HRG extension / bottleneck analysis，而不是主方法替代品。

因此，本論文的主軸應寫成：

```text
HRG-Proposed is an HRG-guided compact evidence retrieval method for KGQA.
It improves the quality-cost trade-off by combining executable relation spines,
KG-validated fallback, candidate recovery, and grammar-guided candidate selection.
```

本文的 claim 邊界如下：

1. 最新 MetaQA final-metrics 已納入 token-budgeted BFS、degree-capped BFS 與 relation n-gram reranker。與 neural path retriever、GraphRAG、relation-similarity-pruned BFS 等更廣泛 baseline 的比較仍屬 supplementary / future work，不能和本次 MetaQA artifact 混成同一個主平均。
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

目前圖檔分成兩組：`img/01_*.pdf` 到 `img/12_*.pdf` 是方法說明、案例與診斷圖；`img/paper_figures/13_*.pdf` 到 `img/paper_figures/19_*.pdf` 是較適合直接放入結果章的論文圖。PDF 可直接用於 LaTeX，SVG 可作後續編修。老師建議的圖面修正原則是：全部使用向量圖、放大字體、減少框內長句、統一配色與圖例，並且每個 caption 明確寫出 dataset、model、sample size、aggregation unit。若圖中數字來自單模型、legacy dump 或單設定診斷，正文需要搭配本文件的主結果表說明，不和 200-per-hop 主表或 latest 3-model supplement 混成同一個統計口徑。

| 圖號 | 圖檔 | 放置章節 | 論文用途 | 使用狀態 |
|---|---|---|---|---|
| Fig. 01 | `img/01_method_pipeline.pdf` | 1.1、6、13、14 | 方法總覽：offline grammar prior、online retrieval、HRG signals | 可用；方法導論圖 |
| Fig. 02 | `img/02_example_bfs_vs_spine_linda_evans.pdf` | 1.2、5、21.8 | BFS 大子圖與 HRG-guided evidence 的案例對比 | 可用；案例圖 |
| Fig. 03 | `img/03_explainability_output_linda_evans.pdf` | 1.1、21.8 | selected entity、relation chain、spine edges 等可檢查輸出 | 可用；可解釋性圖 |
| Fig. 04 | `img/04_metaqa_token_subgraph_compression.pdf` | 21.0、21.3 | MetaQA context tokens 與 subgraph size 壓縮效果 | 可用；gpt-oss 200-per-hop 單模型圖 |
| Fig. 05 | `img/05_metaqa_quality_vs_tokens.pdf` | 21.0、21.3 | answer-set F1 與 context tokens 的 quality-cost trade-off | 可用；gpt-oss 200-per-hop 單模型圖 |
| Fig. 06 | `img/06_dataset_takeaway_matrix.pdf` | 23.1、24 | 各資料集支撐的 claim 與限制邊界 | 可用；資料集定位圖 |
| Fig. 07 | `img/07_metaqa_hop_analysis.pdf` | 23.2 | MetaQA hop-level 分析，凸顯 long-hop evidence structure | 可用；補充診斷圖 |
| Fig. 08 | `img/08_hrg_grammar_extraction.pdf` | 13.1、21.5 | Offline HRG-like grammar extraction 流程 | 可用；方法圖 |
| Fig. 09 | `img/09_chain_validation_algorithm.pdf` | 9、22.2、22.3 | KB relation-label validation 與 strict spine construction | 可用；演算法圖 |
| Fig. 10 | `img/10_failure_counts_spine_correction.pdf` | 21.4、24 | failure analysis 與 hard dataset limitation | 可用；限制分析圖 |
| Fig. 11 | `img/11_dataset_semantics_examples.pdf` | 18、23 | MetaQA、WikiMovies、MLPQ、KQAPro 的語意差異 | 可用；資料集說明圖 |
| Fig. 12 | `img/12_evaluation_design.pdf` | 19、20 | 評估設計：answer quality、evidence quality、cost、failure | 可用；評估框架圖 |
| Fig. 13 | `img/paper_figures/13_offline_online_architecture.pdf` | 1.1.2、6、13、14 | Offline + online 兩段式主架構 | 可用；主方法圖 |
| Fig. 14 | `img/paper_figures/14_offline_grammar_statistics.pdf` | 21.5 | Offline grammar extraction 統計 | 可用；結果圖 |
| Fig. 15 | `img/paper_figures/15_grammar_perturbation_trends.pdf` | 21.6 | KG perturbation 下的 grammar rule 與 relation retention 趨勢 | 可用；robustness 圖 |
| Fig. 16 | `img/paper_figures/16_evidence_coverage_hard_cases.pdf` | 23、24 / supplement | MLPQ / KQAPro hard cases 的 evidence coverage 改善 | 可用但須標 legacy stress-test |
| Fig. 17 | `img/paper_figures/17_online_token_proxy.pdf` | 21.2 / supplement | online token proxy，補充 final context token 之外的成本 | 可用但須標來源 artifact |
| Fig. 18 | `img/paper_figures/18_bootstrap_hard_case_effects.pdf` | supplement | paired bootstrap CI，檢查 HRG-Proposed 相對 Spine-Correction 的效果 | 可用但須標 legacy artifact |
| Fig. 19 | `img/paper_figures/19_prior_pilot.pdf` | 21.3 / supplement | HRG score 與 simple relation prior 的 candidate-level 比較 | 可用但須標來源 artifact |
| Fig. 20 | `img/20_bfs_vs_hrg_process.pdf` | 1.3、5、6 | 經典 BFS KG-RAG 與本文 HRG-guided retrieval 的流程對比 | 新增；回應老師流程圖建議 |
| Fig. 21 | `img/21_mcs_triangulation_clique_tree.pdf` | 1.2.10、13.1 | MCS、triangulation、clique tree 與 HRG rule extraction 直觀圖 | 新增；解釋 clique tree |
| Fig. 22 | `img/22_fallback_sources_examples.pdf` | 10、21.4 | fallback 來源、失敗形式與補回方式 | 新增；支援 Step 4 解釋 |

圖片使用規則：

1. 方法圖與案例圖可以先保留，但 caption 要標註它們是 conceptual / case-study figure，不是四模型平均結果。
2. 所有帶數字的圖必須標註來源 artifact 與 aggregation unit；不要把舊 dump、single-model diagnostic、200-per-hop main table 與 latest 3-model supplement 放在同一張圖或同一段落直接比較。
3. Fig. 04、05、10、14、15、16、17、18、19 是數據圖，重生時要在 caption 裡標註 `latest 3-model MetaQA final-metrics`、`legacy stress-test`、`single-model diagnostic`、`question-level n` 或 `question-model-pair-level n`。
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
MetaQA main 200-per-hop, four-model average:
BFS avg context tokens: 4430.0
HRG-Proposed-triple avg context tokens: 189.6

MetaQA legacy gpt-oss, 200-per-hop figure source:
BFS avg context tokens: 4415.8
HRG-Proposed-triple avg context tokens: 77.3
```

這是論文主敘事保留的 200-per-hop 口徑。四模型平均下，HRG-Proposed-triple 用 BFS `4.28%` 的 final context 取得高於 BFS 的 F1；`gpt-oss` 單模型圖表則使用 `1.75%` 這個更容易直觀看出壓縮幅度的例子。latest 50-per-hop run 放在補充消融，用來報告 fair BFS、relation-prior、GrammarFirst 與 evidence diagnostics。

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

**MCS、triangulation、clique tree 的具體長相**

> 圖：`img/21_mcs_triangulation_clique_tree.pdf`。
> 圖說：MCS supplies an elimination order, triangulation adds fill edges only for decomposition, and the resulting clique tree provides local bags for HRG-like rule extraction.

可以用一個簡化 local graph 理解：

```text
M = movie
D = director
C = country
G = genre

Original skeleton:

M -- D
|    |
G -- C
```

這是一個 4-cycle。如果沒有 chord，它不是 chordal graph，不方便直接形成 clique tree。MCS 會先給一個 elimination order，例如：

```text
MCS order example:
M, D, G, C
```

triangulation 做的是：

```text
按照 elimination order 消去 node。
如果某個 node 的 later neighbors 彼此還不是 clique，就補 fill edge。
```

例如補一條 decomposition-only fill edge：

```text
D -- G
```

得到 triangulated skeleton：

```text
M -- D
|  / |
G -- C
```

重要：這條 `D -- G` 只是為了 clique decomposition 加的 skeleton fill edge，不是 KG 裡新增的 triple，也不是新增 relation label。論文中要避免讓讀者誤會 HRG extraction 改寫了 KG。

triangulation 後可以得到 clique bags：

```text
Bag 1 = {M, D, G}
Bag 2 = {D, C, G}
```

clique tree 則是：

```text
{M, D, G}
    |
    | separator = {D, G}
    |
{D, C, G}
```

抽 HRG rule 時：

```text
parent bag = {M, D, G}
child bag  = {D, C, G}
external nodes = child ∩ parent = {D, G}
internal nodes = child - parent = {C}
```

因此 child bag 會形成一條 rank=2 的 nonterminal rule，代表「在 attachment nodes D/G 上，可以引入一個 local structure」。這就是 clique tree 如何變成 HRG-like production rules 的直觀意義。

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
endpoint_coverage_final / spine / expanded
answer_in_final_context / answer_in_spine / answer_in_expanded_edges
gold_only_in_expansion
context_truncated / context_truncation_ratio
conditional answer F1 when answer is / is not in context
grammar_label_subset_hit / ordered_path_hit / arity_compatible_hit / structural_proxy_hit
```

指標定義如下：

```text
avg_ctx_tokens 是 final context 的估計 token 數。
avg_subgraph_size 是最後 retrieved edges 數量。
compression_vs_bfs_ctx_ratio 是和同 backbone BFS 比例。
endpoint_coverage_final 是 gold answer entity 是否出現在 final retrieved edge endpoints 中的 macro coverage。
endpoint_coverage_spine / expanded 分別只看 spine edges 與 expanded edges。
answer_in_final_context / answer_in_spine / answer_in_expanded_edges 是 endpoint coverage 是否大於 0 的 binary rate。
gold_only_in_expansion 表示 gold answer 不在 spine，但出現在 expanded edges；這是 HRG expansion 是否真正補回答案的直接診斷。
context_truncated 表示 raw retrieved evidence 曾被 token / edge budget 截斷；context_truncation_ratio 是被截掉的 edge 比例。
answer_f1_when_answer_in_context 與 answer_f1_when_answer_not_in_context 用來檢查 LLM 在 evidence 已含答案時是否仍失敗，以及沒有答案 evidence 時是否靠 parametric memory 猜中。
grammar_label_subset_hit、grammar_ordered_path_hit、grammar_arity_compatible_hit 將原本混合的 grammar_hit 拆開；grammar_structural_proxy_hit 表示 ordered path 與 arity-compatible 同時成立。
grammar_full_structural_hit 目前保留為 null / not implemented，因為現行 online matcher 尚未做完整 clique-tree structural derivation match；本文不能把它宣稱為已完成指標。
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
MetaQA main 200-per-hop: HRG-Proposed-triple uses 4.28% BFS context and improves F1 over BFS in the four-model average.
MetaQA latest supplement: HRG-Proposed remains above BFS, while strict spine and relation-prior rows define the clean-schema claim boundary.
GrammarFirst diagnostic: KG-valid grammar paths are executable, but semantic relation filtering and answer-type constraints are the current bottlenecks.
Legacy stress-test analysis: MLPQ / KQAPro are useful for discussing cross-lingual and operator-heavy limitations, but they must not be mixed with the latest MetaQA artifact average.
WikiMovies diagnostic: dataset is mostly 1-hop; BFS is already compact, so it is a sanity-check dataset rather than the main compression evidence.
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

### 1.3 Related Work and Challenge Mapping

這一節不是單純列文獻，而是回答老師指出的問題：相關工作各自解決了什麼 challenge，還有哪些 challenge 沒有處理，而本文的 HRG-guided executable retrieval 補在哪裡。

#### 1.3.1 Classical KGQA and question-specific subgraph retrieval

早期 KGQA / open-domain QA over KG 常見作法，是先建一個 question-specific subgraph，再在子圖上做 reasoning 或 answer extraction。GRAFT-Net 從 KB 與 entity-linked text 建 question-specific subgraph，並用 graph neural network 抽答案；PullNet 進一步用 iterative retrieval 決定下一步要從 KB 或 text 拉哪些資訊。這類方法很接近「先 retrieval，再 reasoning」的 KG-RAG 思路。

代表工作：

1. [GRAFT-Net: Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text](https://arxiv.org/abs/1809.00782)
2. [PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text](https://arxiv.org/abs/1904.09537)

它們和本文的關係：

| Challenge | Related work 做法 | 還沒有解決的點 | 本文對應 |
|---|---|---|---|
| 找到 relevant KG evidence | 建 question-specific subgraph，用 GNN 或 iterative retrieval 找答案 | 子圖對 LLM 來說仍可能太大；不一定輸出一條可檢查 relation spine | 用 KB validation 建 executable spine，只序列化通過 chain 的 evidence |
| multi-hop reasoning | 在子圖上傳播訊息或反覆 retrieval | reasoning path 對讀者未必是 explicit relation-chain trace | 每題輸出 selected entity、relation chain、signed execution trace、spine edges |
| training / supervision | 需要 QA supervision 或 learned graph retriever | 本文情境主要是 LLM + KG pipeline，不訓練新的 GNN | 用 deterministic KB validation、fallback、grammar prior 與 lexicographic ranking |

因此，本文不是說 GRAFT-Net / PullNet 沒有 retrieval，而是說它們的 retrieval 輸出通常不是「給 LLM 使用的 compact, executable, printable evidence spine」。

#### 1.3.2 GraphRAG and LLM-on-KG path search

GraphRAG 類方法把文本或 entity 建成 graph index，再用 graph community / graph retrieval 幫助 LLM answer。Microsoft GraphRAG 的重點是從 text corpus 建 entity graph 與 community summaries，適合 global sensemaking / query-focused summarization。ToG、RoG、GCR 這類 LLM-on-KG reasoning 則更接近 KGQA：讓 LLM 或 constrained decoding 在 KG 上搜尋 reasoning paths。

代表工作：

1. [GraphRAG: From Local to Global](https://arxiv.org/abs/2404.16130)
2. [Think-on-Graph](https://arxiv.org/abs/2307.07697)
3. [Reasoning on Graphs](https://arxiv.org/abs/2310.01061)
4. [Graph-constrained Reasoning](https://arxiv.org/abs/2410.13080)

它們和本文的差異：

| Challenge | Related work 強項 | 本文仍要補的點 |
|---|---|---|
| LLM hallucination | ToG / RoG / GCR 都強調 KG-grounded path reasoning | 本文把 grounding 拆成 parse、entity grounding、KB validation、fallback、spine construction，每題可回報 failed hop / frontier size |
| path retrieval | ToG / RoG 能搜尋或生成 KG reasoning paths | 本文額外加入 offline HRG-like grammar prior，讓 relation structures 可被統計、匹配與診斷 |
| GraphRAG indexing | GraphRAG 對 text corpus global summarization 很強 | 本文處理的是 structured KG triples 的 executable relation-chain retrieval，不是 community summary |
| evidence explainability | 有些方法能輸出 path | 本文輸出 original KG triples、derived r+/r- trace、grammar-hit decomposition、answer-in-context diagnostics |

安全說法：

```text
Existing LLM-on-KG methods already show the value of grounded path reasoning.
This thesis focuses on a narrower but inspectable setting: converting KG-RAG evidence from large BFS neighborhoods into KB-validated relation spines, and using HRG-like grammar as a structural prior and diagnostic signal.
```

#### 1.3.3 Semantic parsing and executable logical forms

Semantic parsing-based KBQA 會把問題轉成 SPARQL、lambda-DCS 或其他 executable logical form。這類方法的優點是可驗證、可執行、精確；KBQA survey 通常把 complex KBQA 分成 semantic parsing-based 與 information retrieval-based 兩大類。

代表參考：

1. [A Survey on Complex Knowledge Base Question Answering](https://arxiv.org/abs/2105.11644)
2. [Uni-Parser](https://arxiv.org/abs/2211.05165)
3. [How Proficient Are LLMs in Formal Languages for KBQA](https://arxiv.org/abs/2401.05777)

和本文的關係：

| Challenge | Semantic parsing 解法 | 本文選擇 |
|---|---|---|
| 可驗證性 | 生成 executable logical form，執行結果可驗證 | 不要求完整 SPARQL/program，只要求 entity + relation chain 可在 KG 執行 |
| complex operators | 可表達 count、filter、argmax、comparison | 本文目前主要處理 relation-chain-friendly 問題；KQAPro 這類 operator-heavy task 是 stress test |
| schema sensitivity | relation / operator / formal language 輸出很敏感 | 用 relation shortlist、alias、KB validation、fallback 與 HRG prior 降低 relation label 錯誤造成的單點失敗 |

因此本文比較像 executable semantic parsing 與 IR-based KG-RAG 之間的折衷：比 BFS 更可驗證，比完整 SPARQL parser 更輕量，但也承認 count / filter / qualifier program 不是目前主 claim。

#### 1.3.4 HRG, clique tree, and structural graph priors

Hyperedge replacement grammar / graph grammar 的核心思想是用 production rules 表示局部 graph structure。既有 HRG graph-generation 工作會從 clique tree 抽 grammar rules，再用 grammar 生成保留局部結構的 synthetic graphs。Tree decomposition / clique tree 則是圖論中把 graph 分解成 bags 的經典工具；chordal graph / triangulation 讓 clique-tree decomposition 更容易建立。

代表參考：

1. [Growing Graphs with Hyperedge Replacement Graph Grammars](https://arxiv.org/abs/1608.03192)
2. [Tree decomposition](https://en.wikipedia.org/wiki/Tree_decomposition)
3. [Chordal graph and triangulation](https://en.wikipedia.org/wiki/Chordal_graph)

本文和傳統 HRG graph-generation 的差異：

| Challenge | HRG graph generation | 本文 HRG-guided KG-RAG |
|---|---|---|
| 目標 | 生成或建模 graph | retrieval 時當 structural prior / diagnostic signal |
| 使用方式 | grammar decoder 生成 graph | soft prior：grammar hit、ordered path hit、arity-compatible hit、fallback/ranking score |
| 輸出 | synthetic graph 或 production rules | 每題 evidence trace、matched rules、grammar decomposition metrics |
| 限制 | 不直接回答自然語言問題 | 需要 LLM parse / entity grounding / KB validation 搭配 |

所以本文不能宣稱「完整 HRG decoder 解決 KGQA」，更準確是：

```text
We use HRG-like rules as an offline structural prior for online executable evidence retrieval.
```

#### 1.3.5 Challenge-to-contribution matrix

| Thesis challenge | 原本 KG-RAG / related work 的不足 | 本文解法 | 仍保留的限制 |
|---|---|---|---|
| Token explosion | BFS 取大鄰域，multi-hop 時 edge 數快速上升 | executable relation spine，只取 selected chain 上 evidence | 若 valid frontier 本身很大，仍需 cap / top-m path |
| Unverified evidence | BFS context 只保證鄰近，不保證是回答路徑 | 每個 candidate chain 必須 KB validation；invalid 不進 final retrieval | LLM final answer 是否真的使用 evidence 仍需 answer-in-context diagnostics |
| Black-box retrieval | LLM 看大 context 後回答，難知道用了哪條 path | 輸出 selected entity、relation chain、signed trace、spine edges、matched grammar prior | 不解釋 LLM hidden reasoning，只解釋 retrieval evidence |
| LLM relation parse 錯 | 一次 parse 錯可能導致 no valid chain | top-k candidates、LLM correction、KG-valid fallback、HRG-supported ranking | valid 不等於 semantically correct |
| LLM hop 數猜錯 | LLM-chain-guided retrieval 會受 initial candidate 限制 | HRG-GrammarFirst diagnostic 從 entity + HRG path-bank 產生 candidates | 目前 semantic filtering 不夠，分數低於主方法 |
| Simple relation priors 很強 | n-gram 可在 MetaQA 小 schema 上競爭 | 最新消融把 unigram/bigram/trigram 納入 strong controls | HRG claim 需定位為 structural prior + diagnostics，不是單純 F1 winner |

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

輸入範例：

```text
q:
  who directed the films starred by [Linda Evans]

KG triples:
  Mitchell | starred_actors | Linda Evans
  Mitchell | directed_by | Andrew V. McLaglen
  Mitchell | written_by | John Michael Hayes
  Mitchell | has_genre | Western

relation list:
  starred_actors, directed_by, written_by, has_genre, release_year, ...

alias / lexical cue:
  "starred by" -> starred_actors
  "directed" -> directed_by

HRG grammar signal:
  repeated movie-domain path pattern:
  actor --starred_actors[r-]--> movie --directed_by[r+]--> director
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

輸出範例：

```text
answer:
  Andrew V. McLaglen

selected_entity:
  Linda Evans

selected_relation_chain:
  starred_actors -> directed_by

signed_execution_trace:
  starred_actors[r-] -> directed_by[r+]

retrieved evidence edges:
  Mitchell | starred_actors | Linda Evans
  Mitchell | directed_by | Andrew V. McLaglen

diagnostics:
  valid = true
  failed_hop = none
  final_frontier_size = 1
  grammar_ordered_path_hit = true
  answer_in_final_context = true
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

End-to-end miniature example:

```text
Offline:
  KG samples repeatedly contain movie -> actor and movie -> director structures.
  HRG extraction stores this as a reusable relation-structure prior.

Online:
  q = who directed the films starred by [Linda Evans]
  LLM candidate = Linda Evans + [starred_actors, directed_by]
  KB validation = executable
  HRG signal = ordered-path hit
  final context = two evidence triples
  answer = Andrew V. McLaglen

Evaluation:
  compare this answer and evidence cost against Baseline-BFS on the same question.
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
3. 為防止 frontier 爆炸，候選 ranking 會使用 capped final frontier signal、grammar score、source priority 與 subgraph-level budget；若候選或 evidence 超過 budget，優先保留 strict spine edges。
4. 若多條 chain 均 executable，目前程式使用 `lax-hrg-prior-v1`：先依 KB-valid、same-arity grammar hit、ordered-path grammar hit、grammar label hit、grammar score、matched rule count 排序，再看 LLM rerank、failed-hop progress、step survival、final frontier size、source priority、LLM confidence 與原始順序。

---

## 5. 為什麼 BFS 不夠

> 圖：`img/20_bfs_vs_hrg_process.pdf`。
> 圖說：Canonical BFS KG-RAG retrieves a broad neighborhood from the topic entity, while HRG-guided executable retrieval first validates relation-chain candidates and then serializes only compact evidence spines.

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

以 MetaQA 問題為例：

```text
Question:
who directed the films starred by [Linda Evans]

Canonical BFS KG-RAG:
1. Ground topic entity: Linda Evans
2. BFS hop 1: 找到 Linda Evans 參與的所有 movie facts
3. BFS hop 2: 對每部 movie 再取 director、writer、genre、tag、rating、language、year 等所有鄰近 facts
4. Serialize all triples
5. LLM 從大 context 中自行判斷哪幾條 edge 是 supporting evidence
```

這裡的 failed case 長這樣：

```text
BFS context 可能同時包含：
  Mitchell | starred_actors | Linda Evans
  Mitchell | directed_by | Andrew V. McLaglen
  Mitchell | written_by | John Michael Hayes
  Mitchell | has_genre | Western
  Mitchell | has_tags | based on novel
  ...

真正需要的 path 只有：
  Linda Evans --starred_actors[r-]--> Mitchell --directed_by[r+]--> Andrew V. McLaglen
```

所以 BFS 的問題不是完全找不到答案，而是把答案和大量非必要 facts 混在一起。這會造成兩種論文裡要明確定義的問題：

1. **Unverified evidence**：retrieved triples 只是 topic neighborhood，不等於已驗證的 reasoning path。
2. **Black-box evidence use**：LLM 看到大 context 後直接回答，研究者很難知道它到底用了哪幾條 triples，還是靠 parametric memory 猜答案。

本文流程相對 BFS 多了：

1. relation-chain parsing：先要求 LLM 提出可驗證 retrieval intent，而不是直接回答。
2. KB validation：每個 relation step 都要在 KG 上走通。
3. HRG structural prior：用 grammar hit、ordered path、arity compatibility、grammar score 輔助 fallback / ranking。
4. strict spine construction：只輸出實際走過的 evidence triples。
5. case-level diagnostics：記錄 failed hop、frontier size、answer-in-context、grammar hit decomposition。

本文流程相對 BFS 少了：

1. 不取完整 h-hop neighborhood。
2. 不把所有 relation 的鄰居都交給 LLM。
3. 不把 invalid candidate 放進 final context。
4. 不把 reverse traversal 當成新增 relation；`r+ / r-` 只是 derived execution trace。

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

### 6.0 Running Example：每一步實際長什麼樣子

本文建議主文用同一個 MetaQA 例子貫穿方法章，避免每步驟看起來像抽象 pipeline。

```text
Question:
who directed the films starred by [Linda Evans]

Gold-style evidence:
Mitchell | starred_actors | Linda Evans
Mitchell | directed_by | Andrew V. McLaglen
```

各步驟對應如下：

| Step | 成功形式 | 常見失敗形式 | 本文補救 |
|---|---|---|---|
| Step 1 LLM parse | `entity=Linda Evans`, `chain=[starred_actors, directed_by]` | LLM 輸出 `acted_in -> director`，不是 KG relation vocabulary | relation shortlist / alias normalization / fuzzy relation match |
| Step 2 grounding | `Linda Evans` 對到 KG node `Linda Evans` | entity 寫成 `Linda_Evans`、大小寫不同、括號殘留 | sanitize、space/underscore normalize、alias map |
| Step 3 KB validation | `starred_actors` 從 Linda Evans 反向走到 movie，再 `directed_by` 正向走到 director | 第一 hop 或第二 hop frontier 為空 | 記錄 failed hop，進入 correction / fallback |
| Step 4 fallback | 找到另一條 KG-valid candidate | LLM correction 仍輸出不存在 relation，或 deterministic fallback 找到太多 executable paths | 每個 fallback candidate 再 KB validation，並用 HRG / relevance / frontier compactness 排序 |
| Step 5 ranking | 選 `starred_actors -> directed_by` | 多條 chain 都 valid，例如 `starred_actors -> written_by` 也可能走通但語意不對 | question-relation relevance、ordered-path grammar hit、answer-type signal |
| Step 6 spine retrieval | 只取兩條 evidence triples | frontier 太大或 relation 太 generic | context edge cap、strict spine priority、failure diagnostics |
| Step 7 answer generation | LLM 根據 compact triples 回答 `Andrew V. McLaglen` | evidence 有答案但 LLM 格式錯 / 少答多答案 | answer normalization、answer-in-context-but-wrong metric |

這個例子能直接說明本文與 BFS 的差異：BFS 會把 Linda Evans 附近所有 movie facts 都序列化；本文先把問題壓成可驗證 relation intent，再只輸出真的走過的 evidence spine。

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

Step 1 的具體例子：

```text
Question:
who directed the films starred by [Linda Evans]

Good parse:
[
  {"entity": "Linda Evans", "chain": ["starred_actors", "directed_by"], "confidence": 0.91}
]

Recoverable bad parse:
[
  {"entity": "Linda Evans", "chain": ["acted_in", "director"], "confidence": 0.78}
]

Failure form:
acted_in / director are natural-language aliases, but not exact KG relation labels.

Correction route:
relation shortlist + alias guide normalize them toward starred_actors / directed_by,
then KB validation decides whether the corrected chain is executable.
```

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

Step 2 的具體例子：

```text
Question surface:
[Linda Evans]

Possible parser outputs:
Linda Evans
Linda_Evans
linda evans
 Linda Evans

Grounded KG node:
Linda Evans

Failure form:
If the entity is grounded to a different Linda / Evans-like node,
then even the correct chain [starred_actors, directed_by] will fail or retrieve the wrong frontier.
```

因此 entity grounding error 不能算成 HRG grammar error；它是 retrieval 起點錯誤，後面的 KB validation 只會忠實反映「從錯起點走不到正確答案」。

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

Step 3 的具體例子：

```text
Stored KG triples:
Mitchell | starred_actors | Linda Evans
Mitchell | directed_by | Andrew V. McLaglen

Candidate:
entity = Linda Evans
chain = [starred_actors, directed_by]

Execution:
frontier_0 = {Linda Evans}
frontier_1 = {Mitchell}
trace_1 = starred_actors[r-]
frontier_2 = {Andrew V. McLaglen}
trace_2 = directed_by[r+]

Result:
valid = true
final_size = 1
spine_edges = the two stored KG triples above
```

Invalid example:

```text
Candidate:
entity = Linda Evans
chain = [directed_by]

Execution:
frontier_0 = {Linda Evans}
frontier_1 = {}

Result:
valid = false
failed_hop = 1
```

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

### 10.2 Step 4 各種 fallback source 的失敗形式與補回方式

> 圖：`img/22_fallback_sources_examples.pdf`。
> 圖說：Fallback sources are activated only after initial candidates fail, and every recovered candidate must be revalidated against the KG before retrieval.

Step 4 不能只寫「fallback correction」，因為不同 source 解決的是不同 failure mode。論文中應拆成以下四種。

#### 10.2.1 LLM correction

適用情況：LLM 初始 chain 接近正確，但 relation label、順序或 hop 數有小錯。

失敗例子：

```text
Question:
who directed the films starred by [Linda Evans]

Initial candidate:
entity = Linda Evans
chain = [acted_in, director]

Validation result:
failed_hop = 1
reason = relation acted_in is not in KG vocabulary
```

補回方式：

```text
System gives failed candidate + failed hop + valid relation vocabulary to LLM.
LLM correction returns:
chain = [starred_actors, directed_by]

Then the corrected chain is executed again on KG:
Linda Evans --starred_actors[r-]--> Mitchell --directed_by[r+]--> Andrew V. McLaglen
```

論文寫法重點：

```text
LLM correction is not trusted directly.
It only proposes revised candidates; KB validation decides whether they can enter retrieval.
```

#### 10.2.2 Deterministic KG-valid fallback

適用情況：LLM correction 仍失敗，或所有 parsed candidates 都走不通，但 entity 已經 grounded。

失敗例子：

```text
Initial candidates:
1. [acted_in, director]       invalid relation labels
2. [starred_actors, genre]    valid but wrong answer type for "who directed"
3. [directed_by]              failed from actor entity

Grounded entity:
Linda Evans
```

補回方式：

```text
Enumerate executable local chains from Linda Evans:
1-hop:
  starred_actors

2-hop:
  starred_actors -> directed_by
  starred_actors -> written_by
  starred_actors -> has_genre
  starred_actors -> release_year

Keep only chains that execute successfully on KG.
Then rank by question-relation relevance, frontier size, source priority, and optional grammar signals.
```

這個 fallback 的價值是保證 candidate 至少可執行；它的風險是「可執行不等於對題」。因此本文不能把 deterministic fallback 單獨包裝成語意理解，而是要和 ranking / HRG prior / answer-type filter 一起說。

#### 10.2.3 HRG-supported fallback / ranking

這是老師問「valid fallback from HRG」時最適合使用的例子。嚴格說，現行 `HRG-Proposed` 不是完整 HRG decoder；HRG 主要提供 fallback/ranking signal。可以這樣寫：

```text
Offline HRG rules repeatedly observe movie-domain patterns such as:

actor --starred_actors[r-]--> movie --directed_by[r+]--> director
actor --starred_actors[r-]--> movie --written_by[r+]--> writer
movie --has_genre[r+]--> genre
```

當 deterministic fallback 產生多條 KG-valid chains 時：

```text
candidate A: starred_actors -> directed_by
candidate B: starred_actors -> written_by
candidate C: starred_actors -> has_genre
candidate D: starred_actors -> release_year
```

HRG-supported ranking 會檢查：

1. chain 的 relation labels 是否命中 grammar rule
2. ordered path 是否和 extracted grammar path 相容
3. arity / attachment pattern 是否和常見 rule 相容
4. grammar score / matched rule count 是否較高
5. question token 是否支持 relation，例如 `directed` 對 `directed_by`

因此對問題：

```text
who directed the films starred by [Linda Evans]
```

`starred_actors -> directed_by` 會同時得到：

```text
KB-valid = true
question relation relevance = high
HRG ordered-path hit = true
frontier size = compact
answer type = person-like endpoint
```

這就是「from HRG 的 valid fallback」在論文中最安全的說法：HRG 不憑空創造 relation，也不新增 reverse edge；它在 KG-valid candidate pool 中提供 structural prior，使系統偏好更像 domain grammar 且可執行的 chain。

#### 10.2.4 HRG-GrammarFirst candidate generation

`HRG-GrammarFirst` 則更進一步：它不是等 initial candidates 失敗才啟動，而是一開始就跳過 LLM relation-chain prior。

例子：

```text
Question:
who directed the films starred by [Linda Evans]

Only use LLM / heuristic to find entity seed:
entity = Linda Evans

Then enumerate relation path-bank from HRG:
  starred_actors -> directed_by
  starred_actors -> written_by
  starred_actors -> has_genre
  ...

Execute each path from Linda Evans on KG.
Rank executable chains.
```

這可以回答「如果 LLM hop 猜錯，HRG 能不能自己找？」：程式上做得到，而且已經納入 latest supplement；但目前結果顯示 semantic filtering 仍不足，因此它是 diagnostic extension，不是主方法替代品。

---

## 11. Step 5：Candidate Ranking

所有 candidate 都會被轉成一個 ranking key。排序不是只看 LLM confidence，而是看一組可解釋因素。

目前程式採用 **lax HRG-prior lexicographic ranking**。`valid(c)` 是 hard gate；在可執行候選之間，先看 HRG grammar compatibility，再看 LLM rerank。

排序優先順序如下：

1. KB executability：valid chain 優先
2. same-arity grammar hit
3. ordered path grammar hit
4. grammar label hit
5. grammar score
6. matched grammar rule count
7. LLM rerank score：若 deterministic fallback 有啟用 LLM rerank
8. failed hop progress：失敗得越晚越好
9. step survival：每 hop frontier 是否還活著
10. final frontier size
11. source priority：LLM、correction、KG fallback、grammar-first 等來源不同
12. LLM confidence
13. LLM 原始排序

### 11.1 Ranking Key 的論文寫法

可重現版本可寫成 lax grammar-prior lexicographic ranking：

```text
rank(c) = (
  valid(c),
  same_arity_grammar_hit(c),
  ordered_path_grammar_hit(c),
  grammar_label_hit(c),
  grammar_score(c),
  matched_rule_count(c),
  llm_rerank_score(c),
  failed_hop_progress(c),
  step_survival(c),
  final_frontier_size(c),
  source_priority(c),
  llm_confidence(c),
  - original_index(c)
)
```

其中 `valid(c)` 是硬條件；沒有通過 KB validation 的 candidate 不能進入 final evidence retrieval。這個 lax order 的設計重點是：在 KG-valid candidates 裡，HRG structural compatibility 先於 LLM rerank score。這能避免 LLM reranker 把 grammar-compatible、可執行且可檢查的 evidence path 壓到後面。程式目前的 `final_frontier_size` 使用 capped positive signal，表示完全走通且 final frontier 非空的候選較穩；frontier 爆炸則主要由 fallback beam / branch cap、subgraph size、context edge cap 與 subgraph ranking 控制。若所有 valid candidates 排名分數相同，使用 deterministic original index tie-break，保證同一輸入可重現。

這個 ranking key 是可重現的工程排序規則，不是學到的 multi-objective optimizer。論文中應把它稱為 lax HRG-prior lexicographic candidate ranking，而不是宣稱系統直接最佳化 Section 4.1 的 objective。新增 ablation 應把 `KG-valid fallback`、`HRG grammar features`、`LLM rerank`、`serialization` 分開，避免把 fallback 帶來的 recovery 都說成 LLM correction 或 HRG decoding 的效果。Relation n-gram rows 仍作為 simple-prior controls 報告，但在這個 lax HRG-prior ranking key 中不放在 HRG grammar signals 前面。

這樣設計的原因：

```text
KGQA retrieval 不能只依賴 LLM 的語意判斷，
也不能只依賴 grammar frequency。
最重要的是這條 chain 能不能在 KB 上真的執行。
```

Step 5 的具體例子：

```text
Question:
who directed the films starred by [Linda Evans]

Candidate A:
chain = starred_actors -> directed_by
KB-valid = true
question relevance = high, because "directed" maps to directed_by
HRG ordered-path hit = true
final frontier = director entities

Candidate B:
chain = starred_actors -> written_by
KB-valid = true
question relevance = lower, because question asks director, not writer
HRG ordered-path hit = true
final frontier = writer entities

Candidate C:
chain = starred_actors -> has_genre
KB-valid = true
question relevance = lower
answer type = genre, not person
```

在這種情況下，ranking 不只是「哪條 chain 能走通」，而是：

```text
能走通 + 問題文字支持 + grammar pattern 支持 + frontier 不爆炸 + answer type 合理
```

這也是為什麼本文要同時報告 relation-prior ablation 與 HRG diagnostics：MetaQA 中 simple relation priors 很強，但 HRG 提供的是更完整的 structural-prior decomposition 與可檢查 ranking signal。

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

Step 6 的具體例子：

```text
Selected chain:
Linda Evans --starred_actors[r-]--> Mitchell --directed_by[r+]--> Andrew V. McLaglen

Strict spine evidence:
1. Mitchell | starred_actors | Linda Evans
2. Mitchell | directed_by | Andrew V. McLaglen
```

對比 BFS context：

```text
BFS also includes:
Mitchell | written_by | ...
Mitchell | has_genre | ...
Mitchell | has_tags | ...
Mitchell | release_year | ...
other movies near Linda Evans | ...
```

因此 grammar / relation-chain retrieval 解決 token explosion 的方式不是「刪掉隨機邊」，而是把 retrieval 約束成：

```text
Question relation cue -> candidate relation chain -> KB executable path -> compact evidence spine
```

也就是說，只有和 selected relation-chain execution 相關的 edges 會進 final answer context；其他 BFS 鄰居即使在 KG 中真實存在，也不會被序列化給 LLM。

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
HRG-GrammarFirst 是新增 diagnostic variant，用來檢驗 HRG 是否能主動產生 / 搜尋候選 chains，而不是只 rerank LLM candidates。最新 MetaQA 結果顯示它目前的瓶頸在 semantic relation filtering 與 answer-type constraints，因此應作為 retrieval-engine extension 的診斷，而不是取代 HRG-Proposed 的主方法。
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

同一組 Linda Evans evidence 的 serialization 例子：

```json
[
  {"head": "Mitchell", "relation": "starred_actors", "tail": "Linda Evans"},
  {"head": "Mitchell", "relation": "directed_by", "tail": "Andrew V. McLaglen"}
]
```

對應 triple format：

```text
Mitchell starred_actors Linda Evans. Mitchell directed_by Andrew V. McLaglen.
```

兩者 retrieved edge set 完全相同，但給 LLM 的介面不同：

| Serialization | 優點 | 風險 |
|---|---|---|
| JSON | head / relation / tail 欄位清楚，較不容易把 relation label 當 entity | token 較多，欄位名稱本身也消耗 context |
| Triple text | token 較少，MetaQA 這類 relation label 直觀的 dataset 通常足夠 | 若 relation label 抽象、跨語言或多答案，LLM 可能誤讀 head/tail 角色 |

因此如果同一 retrieval method 在 JSON / triple 下分數不同，最可能的原因是 answer-generation interface 差異，而不是 retrieval evidence 改變。

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

Linda Evans 例子中，Baseline-BFS 會：

```text
ground Linda Evans
retrieve all nearby facts up to hop depth
serialize movie、director、writer、genre、tag、year 等混合 facts
let LLM decide which facts answer "who directed"
```

它的典型失敗形式是：gold answer 出現在 context 裡，但 context 太大、干擾 facts 太多，LLM 沒有抽出正確 director。

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

Linda Evans 例子中，若 LLM 初始輸出：

```text
starred_actors -> directed_by
```

Spine-Only 可以成功；但若 LLM 輸出：

```text
acted_in -> director
```

Spine-Only 不會 correction，也不會 deterministic fallback，這題會停在 invalid candidate。

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

Linda Evans 例子中，Spine-Correction 會把 failed chain 和 failed hop 交給 correction prompt，例如：

```text
failed chain = acted_in -> director
failed_hop = 1
valid relation vocabulary includes starred_actors, directed_by
```

若 correction 回傳 `starred_actors -> directed_by` 且 KB validation 成功，就能補回這題。

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

Linda Evans 例子中，HRG-Proposed 和 Spine-Correction 最大差別不是 evidence edge 一定不同，而是 candidate selection 會多看 grammar signals：

```text
candidate A = starred_actors -> directed_by
candidate B = starred_actors -> written_by
candidate C = starred_actors -> has_genre
```

三者都可能 KG-valid，但 HRG-Proposed 可以用 ordered-path hit、grammar score、question-relation relevance 和 frontier compactness，偏好更符合 `who directed` 與 movie-domain grammar 的 candidate。

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

Linda Evans 例子中，HRG-GrammarFirst 只先找 entity：

```text
entity = Linda Evans
```

然後從 HRG relation path-bank 產生 candidates：

```text
starred_actors -> directed_by
starred_actors -> written_by
starred_actors -> has_genre
```

再由 KG validation 過濾可執行 paths。這能避開「LLM hop 數一開始猜錯」的問題，但也會產生更多可執行但不對題的 chains，所以目前應寫成 diagnostic bottleneck。

這個方法會比 HRG-Proposed 更有「retrieval engine」味道，但成本與錯誤風險也更高：HRG path-bank 產生的 chain 都會再經 KG executable filtering，但可執行不代表語意正確；因此必須報 candidate count、context tokens、LLM rerank tokens、failure/OOM 與 retrieval coverage。

### 17.6 Relation-prior 與 fair BFS ablations

最新 MetaQA final-metrics 也包含：

1. `RelationUnigram`、`RelationBigram`、`RelationTrigram`：固定使用 candidate / fallback machinery，但把 HRG grammar scoring 改成 relation n-gram prior，用來檢查 HRG 是否只是 relation frequency prior。
2. `Degree-Capped-BFS-{50,100,200,500}`：限制 BFS 每 hop degree，建立更公平的 compact BFS baseline。
3. `Token-Budgeted-BFS-{200,500,1000}`：限制 BFS context token budget，檢查同等 token 下 BFS 是否能追上 HRG。

Linda Evans 例子中：

```text
RelationBigram 可能學到 starred_actors -> directed_by 在 MetaQA 很常見。
Token-Budgeted-BFS-200 可能只保留 Linda Evans 附近前 200 tokens 的 facts。
Degree-Capped-BFS-50 會限制每個 node 最多展開 50 條邊。
```

這些 ablations 的意義是建立更公平的對照：如果 simple prior 或小 BFS 就能做到，HRG 的 claim 就必須定位為 structural decomposition / traceability / fallback diagnostics，而不是單純 F1 winner。

### 17.7 可延伸但目前不作主 claim 的消融

目前 MetaQA 已補上 HRG prior 與 simpler relation-prior / token-budgeted graph retrieval baseline 的比較。它直接回答「HRG 是否真的比普通 relation prior 有用」：在 MetaQA 小 schema 上，relation n-gram 很強，因此 HRG claim 必須保守。

Expansion 相關 row 必須特別標清楚。本文主方法可以採用 no-expansion strict spine 作為主要敘事；`HRG-Proposed-Expansion`、`Spine-GrammarExpansion`、`RandomExpansion`、`FrequencyExpansion` 則作為老師要求的 controlled ablation，用來檢查「額外擴邊」是否真的帶來 evidence coverage，或只是增加 context / OOM。若主文不想把 expansion 包進方法，就不要把 expansion row 混名為 `HRG-Proposed`，而是固定寫成 `HRG-Proposed-NoExpansion` 與 `HRG-Proposed-Expansion`。

另外新增 `HRG-GrammarFirst` 作為 grammar-first retrieval diagnostic。它不使用 LLM 產生的 relation chain / hop 數作為 search-space 前提，而是：

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

Answer metric toy example：

```text
Gold answer set:
{Andrew V. McLaglen, John Smith}

Predicted answer set:
{Andrew V. McLaglen, Jane Doe}

Intersection:
{Andrew V. McLaglen}
```

計算：

```text
EM = 0
因為 predicted set 不等於 gold set

Hits@1 = 1
如果第一個 predicted answer 是 Andrew V. McLaglen

Precision = 1 / 2 = 0.5
Recall = 1 / 2 = 0.5
Answer-set F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
```

這就是為什麼多答案 KGQA 不能只看 EM：部分答對時，F1 能反映 partial credit。

### 19.2 Retrieval Metrics

1. avg retrieval recall
2. avg retrieval precision
3. avg retrieval F1
4. retrieval recall@k
5. retrieval nDCG@k

Retrieval metric toy example：

```text
Gold endpoint answers:
{Andrew V. McLaglen, Director B}

Retrieved evidence endpoints:
{Andrew V. McLaglen, Writer C, Genre D}
```

計算：

```text
retrieval precision = 1 / 3
retrieval recall = 1 / 2
retrieval F1 = 2 * (1/3) * (1/2) / ((1/3) + (1/2)) = 0.4
```

注意這裡衡量的是 endpoint answer coverage，不等於完整 gold path precision；若 dataset 沒有 gold path annotation，就不能宣稱 retrieved edges 都是人工標註的 supporting evidence。

### 19.3 Efficiency Metrics

1. avg context tokens
2. avg subgraph size
3. compression vs BFS context ratio
4. compression vs BFS subgraph ratio
5. avg latency
6. parse latency
7. retrieval latency
8. generation latency

Efficiency toy example：

```text
BFS:
context tokens = 4415.8
subgraph edges = 369.1

HRG-Proposed-triple:
context tokens = 77.3
subgraph edges = 7.4
```

計算：

```text
compression_vs_bfs_ctx_ratio = 77.3 / 4415.8 = 1.75%
compression_vs_bfs_subgraph_ratio = 7.4 / 369.1 = 2.00%
```

這個 ratio 只能和同 dataset、同 model、同 evaluation setup 的 BFS 比；不能跨 artifact 混算。

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

Failure metric toy example：

```text
Case A:
no_candidates = true
原因：LLM JSON parse 失敗，且 entity fallback 找不到 topic entity。

Case B:
no_valid_chain = true
原因：entity 有 grounded，但所有 chains 在 KB validation 的某一 hop frontier 為空。

Case C:
answer_in_context_but_wrong = true
原因：Andrew V. McLaglen 出現在 final context，但 LLM 最後輸出 John Michael Hayes。
```

這些 failure flags 的目的，是把「retrieval 沒找到」、「retrieval 找到但 LLM 沒用好」、「問題本身超出 relation-chain assumption」分開。

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

實驗設定要和程式碼對齊，不能只在文字中描述方法。論文或 supplementary 應集中列出：

1. 執行入口：目前建議用 `run_laxhrg_clean_all.sh` 跑最新 clean rerun。它會設定 `RANKING_POLICY=lax-hrg-prior-v1`、`EXPERIMENT_SUITE=full`、`SAMPLE_LIMIT=50`，輸出到 `artifacts_laxhrg`，但不硬指定 GPU；實際 GPU / sharding 以 shell 環境或 `configs/*.env` 為準。
2. 完整 sequential rerun：`run_everything_sequential.sh` 先做 syntax / py_compile、Qwen3.5 diagnosis、reachability audit，再呼叫 `run_full_rerun.sh`。目前預設同樣使用 `RANKING_POLICY=lax-hrg-prior-v1` 與 `artifacts_laxhrg`。
3. 完整 rerun matrix：`run_full_rerun.sh` 預設開啟 `EXPERIMENT_SUITE=full`、`ENABLE_NEW_ABLATION_SPECS=1`、`ENABLE_RELATION_NGRAM_SPECS=1`、`ENABLE_BFS_CAP_SPECS=1`，並跑四個 dataset config。
4. 控制預算：新增 ablation 在 `benchmark.py` 使用相同 controlled budget，例如 `num_candidates`、`valid_chain_fallback_topk`、fallback beam / branch budget 與 `max_total_context_edges`。
5. Candidate ranking：`knowledgegraph_agent.py` 的 `_score_candidate()` 使用 `lax-hrg-prior-v1`，即 `valid -> same_arity_grammar_hit -> ordered_path_grammar_hit -> grammar_label_hit -> grammar_score -> matched_rule_count -> llm_rerank_score -> failed_hop_progress -> step_survival -> final_frontier_size -> source_priority -> llm_confidence -> -original_index`。`benchmark.py` 會把 `ranking_policy` 與 `candidate_ranking_key` 寫入 result payload。
6. GrammarFirst row：`HRG-GrammarFirst-NoExpansion/Expansion` 對應 `use_grammar_first_retrieval=True`、`use_valid_chain_llm_rerank=True`；它的 relation chains 來自 HRG relation path-bank + KG executable filtering，而不是 LLM relation-chain parser。
7. Perturbation：`run_laxhrg_clean_all.sh` 預設 `RUN_PERTURBATION=0`，只跑 clean rerun；若要 robustness，可用 `RUN_PERTURBATION=1 bash run_laxhrg_clean_all.sh`。`run_full_rerun.sh` 中 perturbation 使用 `KB_ABLATION_SEEDS=0 1 2 3 4`、`drop_nodes/drop_relations`、`0.1/0.2/0.3`。
8. 模型與 prompt：checkpoint、temperature、max tokens、parse prompt、answer prompt 與 relation vocabulary source 應以 `benchmark.py`、`knowledgegraph_agent.py` 與各 `configs/*.env` 的實際值為準，附錄需列完整 prompt。

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

本節採用 **兩層結果口徑**，以符合目前實際完成的實驗與 `new_suggest.docx` 的要求：

1. **MetaQA main 200-per-hop table**：使用 `artifect_all/metaqa-vanilla-test/results/benchmark_results.json`。這是論文主敘事表，保留 `gpt-oss`、`gemma4`、`llama3.1`、`qwen2.5` 四個有效模型；每個模型是 MetaQA 1-hop / 2-hop / 3-hop 各 `200` 題，總計每模型 `600` 題、四模型 `2400` question-model pairs。`img/01`、`img/04`、`img/05` 的 `gpt-oss` 數字均來自這個 artifact。
2. **Latest ablation / metric supplement**：使用 `artifacts_full/metaqa-vanilla-test-final-metrics-gpu1-20260701-012816/results/benchmark_results.json`。這個 run 有 `llama3.1`、`gemma4`、`qwen2.5` 三個模型，`36` 個 method rows，1-hop / 2-hop / 3-hop 各 `50` 題，用來補 fair BFS、relation n-gram、GrammarFirst、answer-in-context、grammar decomposition、truncation 等老師要求的指標。
3. **`gpt-oss` ablation source**：若 latest ablation 沒有 `gpt-oss`，優先沿用 `artifect_all` 中同名 core method；若只存在於 `artifacts_full/metaqa-vanilla-test-full-20260627-134548`，標為 legacy ablation；若某 row 在 `gpt-oss` 舊 artifact 也不存在，標 `NA`。

注意：以下數據先沿用已完成 artifacts。`lax-hrg-prior-v1` 是目前程式碼與下一輪 rerun 的 ranking policy；在 `artifacts_laxhrg` 完成前，本節不把新 policy 的未完成結果混入表格。

`qwen3.5` 不納入平均：目前環境 GPU 為 A40，compute capability `8.6`，而 `Qwen/Qwen3.5-35B-A3B-FP8` 在 Transformers 中需要 compute capability >= `8.9` 才能直接使用 FP8 runtime；實際載入時會退回 bf16 dequantization 並導致 load-time OOM。這是 checkpoint / hardware incompatibility，不是 HRG-Proposed 方法失敗。

| Artifact result file | Entries | Models | 用途 |
|---|---:|---|---|
| `artifect_all/metaqa-vanilla-test/results/benchmark_results.json` | 35 | `gpt-oss`, `gemma4`, `llama3.1`, `qwen2.5`, `qwen3.5` | MetaQA 200-per-hop main table；`qwen3.5` 不納入有效模型平均 |
| `artifacts_full/metaqa-vanilla-test-final-metrics-gpu1-20260701-012816/results/benchmark_results.json` | 108 | `llama3.1`, `gemma4`, `qwen2.5` | latest ablation / evidence metrics supplement |
| `artifacts_full/metaqa-vanilla-test-full-20260627-134548/results/benchmark_results.json` | 19 | `gpt-oss` | `gpt-oss` fair BFS / expansion legacy ablation source |
| `wikimovies-wiki_entities-test-full-20260629-002637` | 4 | `llama3.1` | WikiMovies BFS / capped-BFS diagnostic |
| `_qwen35_probe/*` | 3 nonempty probes | `qwen3.5` | qwen3.5 載入失敗診斷 |

### 21.0 MetaQA main 200-per-hop 四模型主表

下表 cell 格式為：

```text
EM / answer-set F1 / avg final context tokens / avg subgraph size
```

| Method | gpt-oss | gemma4 | llama3.1 | qwen2.5 | 4-model avg |
|---|---:|---:|---:|---:|---:|
| Baseline-BFS | 0.5750 / 0.6109 / 4415.8 / 369.1 | 0.3917 / 0.4872 / 4434.7 / 369.1 | 0.4367 / 0.5837 / 4434.7 / 369.1 | 0.3983 / 0.5295 / 4434.7 / 369.1 | 0.4504 / 0.5528 / 4430.0 / 369.1 |
| Spine-Only-json | 0.6133 / 0.6622 / 212.0 / 9.2 | 0.5783 / 0.6313 / 201.4 / 8.8 | 0.5250 / 0.7225 / 1865.4 / 85.8 | 0.4917 / 0.5444 / 127.9 / 5.5 | 0.5521 / 0.6401 / 601.7 / 27.3 |
| Spine-Only-triple | 0.5950 / 0.6448 / 96.9 / 9.2 | 0.5467 / 0.6188 / 90.6 / 8.8 | 0.4633 / 0.6727 / 896.5 / 97.9 | 0.4617 / 0.5256 / 58.5 / 5.5 | 0.5167 / 0.6155 / 285.6 / 30.4 |
| Spine-Correction-json | 0.6133 / 0.6622 / 212.0 / 9.2 | 0.5783 / 0.6313 / 201.4 / 8.8 | 0.5350 / 0.7331 / 1866.8 / 85.9 | 0.5067 / 0.5739 / 145.9 / 6.3 | 0.5583 / 0.6501 / 606.5 / 27.6 |
| Spine-Correction-triple | 0.5950 / 0.6448 / 96.9 / 9.2 | 0.5467 / 0.6188 / 90.6 / 8.8 | 0.4733 / 0.6835 / 897.2 / 97.9 | 0.4733 / 0.5557 / 66.5 / 6.3 | 0.5221 / 0.6257 / 287.8 / 30.6 |
| HRG-Proposed-json | 0.5517 / 0.5881 / 169.6 / 7.4 | 0.5767 / 0.6310 / 272.6 / 11.8 | 0.5150 / 0.6865 / 516.5 / 23.2 | 0.5017 / 0.5823 / 732.8 / 33.2 | 0.5363 / 0.6220 / 422.9 / 18.9 |
| HRG-Proposed-triple | 0.5350 / 0.5718 / 77.3 / 7.4 | 0.5500 / 0.6202 / 138.1 / 13.5 | 0.4200 / 0.6226 / 225.7 / 23.2 | 0.4683 / 0.5648 / 317.2 / 33.2 | 0.4933 / 0.5948 / 189.6 / 19.3 |

這張表是論文主結果的安全版本：四模型平均下，`HRG-Proposed-triple` 用 BFS `4.28%` 的 final context tokens 和 `5.24%` 的 retrieved edges，F1 從 `0.5528` 到 `0.5948`。`gpt-oss` 單模型則顯示更強壓縮：`77.3 / 4415.8 = 1.75%`，這正是 `img/01`、`img/04`、`img/05` 的圖內數字來源。表格為一位小數，圖檔保留兩位小數；兩者來自同一個 `artifect_all/metaqa-vanilla-test` artifact。

### 21.1 Latest ablation / metric supplement

下表補上 latest final-metrics 的完整消融矩陣，並保留 `gpt-oss` 舊值。`*` 代表舊版只有 `HRG-Proposed`，沒有 NoExpansion / Expansion 分 row，因此沿用舊版 HRG-Proposed 數值；`NA` 代表 `gpt-oss` 舊 artifact 沒有該 row。

| Method | gpt-oss source value | llama3.1 latest | gemma4 latest | qwen2.5 latest | latest 3-model avg | gpt-oss source |
|---|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.5750 / 0.6109 / 4415.8 / 369.1 | 0.4933 / 0.5978 / 4341.9 / 360.9 | 0.3933 / 0.4877 / 4341.9 / 360.9 | 0.4000 / 0.4976 / 4341.9 / 360.9 | 0.4289 / 0.5277 / 4341.9 / 360.9 | legacy core |
| Degree-Capped-BFS-50 | 0.5667 / 0.6249 / 586.0 / 48.5 | 0.4667 / 0.6072 / 604.0 / 49.6 | 0.3867 / 0.4886 / 604.0 / 49.6 | 0.3800 / 0.5004 / 604.0 / 49.6 | 0.4111 / 0.5321 / 604.0 / 49.6 | legacy ablation |
| Degree-Capped-BFS-100 | 0.5733 / 0.6314 / 1045.7 / 86.9 | 0.4733 / 0.6117 / 1064.9 / 88.1 | 0.3800 / 0.4822 / 1064.9 / 88.1 | 0.3867 / 0.5067 / 1064.9 / 88.1 | 0.4133 / 0.5335 / 1064.9 / 88.1 | legacy ablation |
| Degree-Capped-BFS-200 | 0.5767 / 0.6315 / 1906.5 / 160.5 | 0.4933 / 0.6141 / 1943.5 / 160.5 | 0.3867 / 0.4826 / 1943.5 / 160.5 | 0.3933 / 0.5019 / 1943.5 / 160.5 | 0.4244 / 0.5329 / 1943.5 / 160.5 | legacy ablation |
| Degree-Capped-BFS-500 | 0.5717 / 0.6079 / 4382.9 / 369.1 | 0.4933 / 0.5978 / 4341.9 / 360.9 | 0.3933 / 0.4877 / 4341.9 / 360.9 | 0.4000 / 0.4976 / 4341.9 / 360.9 | 0.4289 / 0.5277 / 4341.9 / 360.9 | legacy ablation |
| Token-Budgeted-BFS-200 | 0.4883 / 0.5536 / 135.3 / 10.6 | 0.4333 / 0.5273 / 135.7 / 10.6 | 0.3400 / 0.4016 / 135.7 / 10.6 | 0.3667 / 0.4481 / 135.7 / 10.6 | 0.3800 / 0.4590 / 135.7 / 10.6 | legacy ablation |
| Token-Budgeted-BFS-500 | 0.5633 / 0.6129 / 285.2 / 23.7 | 0.4800 / 0.6042 / 291.7 / 23.9 | 0.3800 / 0.4576 / 291.7 / 23.9 | 0.3867 / 0.4927 / 291.7 / 23.9 | 0.4156 / 0.5182 / 291.7 / 23.9 | legacy ablation |
| Token-Budgeted-BFS-1000 | 0.5717 / 0.6215 / 510.1 / 43.4 | 0.4867 / 0.6057 / 521.4 / 44.4 | 0.3800 / 0.4663 / 521.4 / 44.4 | 0.3867 / 0.5097 / 521.4 / 44.4 | 0.4178 / 0.5272 / 521.4 / 44.4 | legacy ablation |
| Spine-Only-json | 0.6133 / 0.6622 / 212.0 / 9.2 | 0.5267 / 0.7478 / 1227.4 / 56.2 | 0.5600 / 0.6658 / 246.1 / 10.7 | 0.5267 / 0.6113 / 204.1 / 8.9 | 0.5378 / 0.6750 / 559.2 / 25.3 | legacy core |
| Spine-Only-triple | 0.5950 / 0.6448 / 96.9 / 9.2 | 0.4800 / 0.6856 / 524.2 / 56.2 | 0.5600 / 0.6514 / 111.5 / 10.7 | 0.4800 / 0.5726 / 92.8 / 8.9 | 0.5067 / 0.6365 / 242.8 / 25.3 | legacy core |
| Spine-Correction-json | 0.6133 / 0.6622 / 212.0 / 9.2 | 0.5267 / 0.7478 / 1227.4 / 56.2 | 0.5600 / 0.6658 / 246.1 / 10.7 | 0.5267 / 0.6166 / 219.0 / 9.5 | 0.5378 / 0.6767 / 564.2 / 25.5 | legacy core |
| Spine-Correction-triple | 0.5950 / 0.6448 / 96.9 / 9.2 | 0.4800 / 0.6856 / 524.2 / 56.2 | 0.5600 / 0.6514 / 111.5 / 10.7 | 0.4867 / 0.5887 / 99.9 / 9.5 | 0.5089 / 0.6419 / 245.2 / 25.5 | legacy core |
| Spine-GrammarExpansion-json | 0.5267 / 0.5709 / 342.0 / 15.0 | 0.3467 / 0.4864 / 1314.8 / 58.3 | 0.5133 / 0.6022 / 639.9 / 28.1 | 0.4400 / 0.5632 / 626.7 / 27.4 | 0.4333 / 0.5506 / 860.5 / 37.9 | legacy ablation |
| Spine-GrammarExpansion-triple | 0.4950 / 0.5485 / 270.7 / 26.1 | 0.3267 / 0.4652 / 586.1 / 58.3 | 0.5133 / 0.6116 / 332.5 / 32.5 | 0.3800 / 0.5165 / 283.9 / 27.4 | 0.4067 / 0.5311 / 400.8 / 39.4 | legacy ablation |
| Spine-RandomExpansion-json | 0.5550 / 0.6164 / 451.5 / 20.1 | 0.4933 / 0.7232 / 4517.7 / 202.0 | 0.5200 / 0.6351 / 955.8 / 42.6 | 0.4467 / 0.5643 / 917.4 / 40.7 | 0.4867 / 0.6409 / 2130.3 / 95.1 | legacy ablation |
| Spine-RandomExpansion-triple | 0.5450 / 0.6050 / 400.0 / 39.8 | 0.4533 / 0.6622 / 1992.1 / 202.0 | 0.5267 / 0.6293 / 467.9 / 47.1 | 0.3800 / 0.5182 / 408.0 / 40.7 | 0.4533 / 0.6032 / 956.0 / 96.6 | legacy ablation |
| Spine-FrequencyExpansion-json | 0.5733 / 0.6364 / 444.7 / 19.6 | 0.4867 / 0.6952 / 4568.8 / 202.2 | 0.5333 / 0.6290 / 964.6 / 42.6 | 0.4600 / 0.5821 / 926.8 / 40.8 | 0.4933 / 0.6354 / 2153.4 / 95.2 | legacy ablation |
| Spine-FrequencyExpansion-triple | 0.5483 / 0.6038 / 407.2 / 39.8 | 0.4333 / 0.6419 / 2041.0 / 202.2 | 0.5400 / 0.6391 / 476.4 / 47.1 | 0.3800 / 0.5110 / 416.7 / 40.8 | 0.4511 / 0.5973 / 978.1 / 96.7 | legacy ablation |
| Spine-Correction-KGValidFallback-json | 0.6150 / 0.6765 / 646.3 / 28.0 | 0.5533 / 0.7517 / 2565.0 / 118.9 | 0.5600 / 0.6663 / 420.0 / 18.5 | 0.5267 / 0.6341 / 1082.9 / 49.5 | 0.5467 / 0.6840 / 1355.9 / 62.3 | legacy ablation |
| Spine-Correction-KGValidFallback-triple | NA | 0.5067 / 0.7088 / 1362.2 / 144.9 | 0.5600 / 0.6570 / 283.1 / 29.5 | 0.4867 / 0.6088 / 463.3 / 49.5 | 0.5178 / 0.6582 / 702.9 / 74.7 | NA |
| HRG-Proposed-NoExpansion-json | 0.5517 / 0.5881 / 169.6 / 7.4* | 0.4400 / 0.6067 / 2023.7 / 92.4 | 0.5600 / 0.6541 / 439.9 / 20.1 | 0.5333 / 0.6149 / 1577.2 / 70.1 | 0.5111 / 0.6252 / 1346.9 / 60.9 | legacy mapped |
| HRG-Proposed-NoExpansion-triple | 0.5350 / 0.5718 / 77.3 / 7.4* | 0.4267 / 0.6032 / 868.1 / 92.4 | 0.5600 / 0.6439 / 188.8 / 20.1 | 0.4867 / 0.5872 / 700.6 / 70.1 | 0.4911 / 0.6114 / 585.9 / 60.9 | legacy mapped |
| HRG-Proposed-Expansion-json | 0.5517 / 0.5881 / 169.6 / 7.4* | 0.4400 / 0.6067 / 2023.7 / 92.4 | 0.5600 / 0.6541 / 439.9 / 20.1 | 0.5333 / 0.6149 / 1577.2 / 70.1 | 0.5111 / 0.6252 / 1346.9 / 60.9 | legacy mapped |
| HRG-Proposed-Expansion-triple | 0.5350 / 0.5718 / 77.3 / 7.4* | 0.4267 / 0.6032 / 868.1 / 92.4 | 0.5600 / 0.6439 / 188.8 / 20.1 | 0.4867 / 0.5872 / 700.6 / 70.1 | 0.4911 / 0.6114 / 585.9 / 60.9 | legacy mapped |
| HRG-GrammarFirst-NoExpansion-json | NA | 0.4067 / 0.5170 / 1700.9 / 75.9 | 0.3800 / 0.4290 / 435.0 / 19.2 | 0.3600 / 0.4332 / 861.8 / 38.3 | 0.3822 / 0.4597 / 999.2 / 44.4 | NA |
| HRG-GrammarFirst-NoExpansion-triple | NA | 0.3200 / 0.4673 / 752.4 / 75.9 | 0.3667 / 0.4299 / 611.3 / 60.5 | 0.3067 / 0.3850 / 382.9 / 38.3 | 0.3311 / 0.4274 / 582.2 / 58.2 | NA |
| HRG-GrammarFirst-Expansion-json | NA | 0.4067 / 0.5170 / 1700.9 / 75.9 | 0.3800 / 0.4290 / 435.0 / 19.2 | 0.3600 / 0.4332 / 861.8 / 38.3 | 0.3822 / 0.4597 / 999.2 / 44.4 | NA |
| HRG-GrammarFirst-Expansion-triple | NA | 0.3200 / 0.4673 / 752.4 / 75.9 | 0.3667 / 0.4299 / 611.3 / 60.5 | 0.3067 / 0.3850 / 382.9 / 38.3 | 0.3311 / 0.4274 / 582.2 / 58.2 | NA |
| HRG-Proposed-json | 0.5517 / 0.5881 / 169.6 / 7.4 | 0.4400 / 0.6067 / 2023.7 / 92.4 | 0.5600 / 0.6541 / 439.9 / 20.1 | 0.5333 / 0.6149 / 1577.2 / 70.1 | 0.5111 / 0.6252 / 1346.9 / 60.9 | legacy core |
| HRG-Proposed-triple | 0.5350 / 0.5718 / 77.3 / 7.4 | 0.4267 / 0.6032 / 868.1 / 92.4 | 0.5600 / 0.6439 / 188.8 / 20.1 | 0.4867 / 0.5872 / 700.6 / 70.1 | 0.4911 / 0.6114 / 585.9 / 60.9 | legacy core |
| RelationUnigram-json | NA | 0.5533 / 0.7490 / 3069.2 / 141.8 | 0.5600 / 0.6678 / 556.1 / 25.2 | 0.5333 / 0.6401 / 1388.7 / 63.7 | 0.5489 / 0.6856 / 1671.3 / 76.9 | NA |
| RelationUnigram-triple | NA | 0.5067 / 0.7086 / 1578.8 / 167.7 | 0.5600 / 0.6584 / 297.0 / 31.3 | 0.4867 / 0.5980 / 591.8 / 63.7 | 0.5178 / 0.6550 / 822.5 / 87.6 | NA |
| RelationBigram-json | NA | 0.5467 / 0.7381 / 2854.9 / 132.3 | 0.5600 / 0.6678 / 556.1 / 25.2 | 0.5400 / 0.6440 / 1157.8 / 53.1 | 0.5489 / 0.6833 / 1523.0 / 70.2 | NA |
| RelationBigram-triple | NA | 0.5067 / 0.7075 / 1484.4 / 158.2 | 0.5600 / 0.6584 / 297.0 / 31.3 | 0.4933 / 0.6060 / 493.2 / 53.1 | 0.5200 / 0.6573 / 758.2 / 80.9 | NA |
| RelationTrigram-json | NA | 0.5467 / 0.7381 / 2826.4 / 130.8 | 0.5600 / 0.6678 / 556.1 / 25.2 | 0.5400 / 0.6449 / 1149.8 / 52.8 | 0.5489 / 0.6836 / 1510.8 / 69.6 | NA |
| RelationTrigram-triple | NA | 0.5067 / 0.7075 / 1474.0 / 156.8 | 0.5600 / 0.6584 / 297.0 / 31.3 | 0.4933 / 0.6064 / 489.4 / 52.8 | 0.5200 / 0.6574 / 753.5 / 80.3 | NA |

### 21.2 Evidence diagnostics and token cost

下表聚焦和老師建議最相關的 evidence / cost 指標。`Total proxy` 是 `parse1_tokens + correction_tokens + parse2_tokens + context_tokens`，不是精確 API billing cost，但可以反映 online pipeline 成本。`Grammar L/O/A` 分別是 label-subset hit、ordered-path hit、arity-compatible hit。

| Method | F1 | Ctx tok | Total proxy | Ret R | Ret F1 | Ans in final | Ans in spine | Grammar L/O/A |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline-BFS | 0.5277 | 4341.9 | 8919.2 | 0.9013 | 0.3288 | 0.9800 | 0.9800 | - |
| Spine-Correction-triple | 0.6419 | 245.2 | 1465.6 | 0.8067 | 0.4750 | 0.8311 | 0.8311 | - |
| Spine-Correction-KGValidFallback-triple | 0.6582 | 702.9 | 2434.8 | 0.8252 | 0.4681 | 0.8778 | 0.8778 | - |
| HRG-Proposed-triple | 0.6114 | 585.9 | 2503.2 | 0.7575 | 0.4375 | 0.8400 | 0.8400 | 1.0000 / 0.9978 / 0.6778 |
| HRG-GrammarFirst-NoExpansion-triple | 0.4274 | 582.2 | 3854.7 | 0.5355 | 0.3288 | 0.6322 | 0.6322 | 1.0000 / 1.0000 / 1.0000 |
| RelationBigram-triple | 0.6573 | 758.2 | 2543.7 | 0.8148 | 0.4634 | 0.8711 | 0.8711 | - |
| RelationTrigram-triple | 0.6574 | 753.5 | 2534.7 | 0.8170 | 0.4646 | 0.8733 | 0.8733 | - |

補充觀察：

1. `context_truncated` 在上述主要 rows 中都是 `0.0`，表示這次 MetaQA final-metrics 的差異不是因為某個方法被截斷造成。
2. `gold_only_in_expansion` 在 HRG-Proposed / GrammarFirst rows 中都是 `0.0`。因此這次資料不能宣稱 expansion 補進了 gold answer；`NoExpansion` 與 `Expansion` 數字相同，應解讀為 expansion 在此設定下沒有實質貢獻。
3. BFS 的 `answer_in_final_context = 0.9800`，但 answer-set F1 只有 `0.5277`。這是很重要的反擊點：答案在大 context 裡不代表 LLM 能正確抽出答案，raw recall 不等於 answer quality。
4. Spine-Correction / relation-prior rows 的 answer-in-context 低於 BFS，但 F1 更高，表示更小、更結構化的 evidence 對 generation 有幫助。
5. HRG-Proposed 的 grammar label/order hit 幾乎滿分，但 arity-compatible hit 只有 `0.6778`；所以不能把 grammar hit 說成完整 HRG structural derivation，只能說它是 decomposed grammar prior signal。

### 21.3 最新結果解讀

latest 50-per-hop final-metrics 是補充消融，不取代 200-per-hop 主表。它支持的結論是：HRG-Proposed 在 fair BFS / evidence diagnostics 口徑下仍優於 BFS，同時也揭示 MetaQA clean schema 上的 simple-prior 強度。

1. **HRG-Proposed-triple 仍比 BFS 有更好的 quality-cost trade-off**：BFS F1 `0.5277`、context `4341.9`；HRG-Proposed-triple F1 `0.6114`、context `585.9`，context 只有 BFS 的 `13.49%`。因此 HRG-Proposed 不是無效，它確實把 evidence 壓小並提升 F1。
2. **HRG-Proposed 是主方法，不是唯一高分消融 row**：Spine-Correction-triple F1 `0.6419`、context `245.2`；KGValidFallback-triple F1 `0.6582`；RelationBigram/Trigram-triple 約 `0.657`。論文應把這些列為 ablation / strong simple-prior controls，用來界定 HRG claim，而不是把它們寫成主方法被否定。
3. **MetaQA 的可量化增益來自 executable evidence construction，HRG 讓它成為可解釋的 structural-prior pipeline**：strict spine family 很強，表示 MetaQA 非常符合 relation-chain assumption；HRG-Proposed 在此基礎上加入 grammar decomposition、KG-valid fallback/ranking signals 與可檢查 evidence trace。
4. **Relation n-gram baseline 很強，必須正面處理**：RelationBigram/Trigram 在 MetaQA 接近或超過 KGValidFallback。這不代表 HRG 無效，而是說 MetaQA schema 小、relation transition regularity 高，簡單 relation prior 本來就很吃香。論文應把它當成 strong simple-prior ablation。
5. **GrammarFirst 是 diagnostic bottleneck result**：GrammarFirst-triple F1 `0.4274`，低於 BFS、Spine-Correction 與 HRG-Proposed。它證明「只要找到 entity，再由 HRG path-bank / KG-valid filtering 產生可執行 chain」還不夠；可執行不等於對題，semantic relation relevance 與 answer-type filtering 才是下一步關鍵。
6. **Expansion 在此 run 沒有幫助**：HRG-Proposed-NoExpansion、HRG-Proposed-Expansion、HRG-Proposed 三者完全相同；GrammarFirst-NoExpansion / Expansion 也完全相同。這次不能把 F1 提升歸因於 expansion。
7. **BFS 的高 coverage 反而暴露 context overload**：BFS 幾乎把答案放進 context，但 F1 明顯低於 spine rows。這可以回應「BFS 不是已經拿到答案了嗎？」：拿到答案只是 retrieval recall，還要讓 answer generator 在可讀、低干擾的 evidence 上正確回答。

最適合寫進論文的 MetaQA 主結論是：

```text
On the MetaQA 200-per-hop main table, HRG-Proposed-triple improves over BFS while using 4.28% of BFS final evidence context in the four-model average.
The latest 50-per-hop supplement confirms the same quality-cost direction and adds evidence diagnostics, while strict spine and relation-prior ablations define a strong clean-schema control.
Therefore MetaQA supports HRG-guided compact executable evidence retrieval, with HRG contributing structural-prior decomposition, fallback/ranking signals, and traceable evidence rather than a claim of universal dominance over every simple prior.
```

### 21.4 推薦反擊說法

如果口委問「這樣 HRG 還需要存在嗎？」可以回答：

```text
在 MetaQA 這種 clean relation-chain dataset 上，strict executable spine 和 simple relation priors 已經很強，所以 HRG-Proposed 的價值不應只用單一 answer-quality 排名定義。
HRG 的定位是 structural prior and recovery signal：它讓系統能輸出 grammar-compatible evidence trace、在 hard cases 中提供 fallback/ranking features，並讓我們能拆解 grammar label/order/arity compatibility。
最新結果把 HRG 的角色說得更清楚：MetaQA 上 executable spine 是基礎，HRG 則提供 structural prior、candidate recovery/ranking signals 與可檢查 evidence decomposition。
```

如果口委問「是不是只是 relation frequency / bigram？」可以回答：

```text
MetaQA 的 relation vocabulary 很小，relation transition regularity 很高，因此 bigram/trigram 是很強的 simple-prior baseline。
我把它放進實驗正是為了避免把 HRG 的效果誤講成單純 relation frequency。
結果顯示 MetaQA 上 simple priors 很強，這限制了 HRG-Proposed 的 claim；HRG 的論文價值應放在完整 pipeline、grammar decomposition、KG-valid fallback 與 evidence diagnostics，而不是聲稱在所有 clean-chain cases 都贏過 n-gram prior。
```

如果口委問「BFS 已經 98% 把答案放進 context，為什麼還需要你的方法？」可以回答：

```text
Answer-in-context 不是 answer quality。最新 MetaQA 中 BFS 的 answer-in-context 是 0.9800，但 F1 只有 0.5277。
Spine-Correction / HRG-Proposed 的 context 更小、answer-in-context 較低，F1 卻更高，表示大 BFS context 會造成干擾；我的方法處理的是 evidence selection and presentation，不只是 retrieval recall。
```

如果口委問「grammar hit 是不是太鬆？」可以回答：

```text
是，所以我已經把 grammar hit 拆成 label subset、ordered path、arity compatible 與 structural proxy。
本文不宣稱目前已有完整 HRG derivation matcher；目前 HRG 是 soft structural prior，用於 ranking / fallback / diagnostics。
這比只報一個 grammar_hit 更誠實，也能避免把 relation label overlap 誤講成完整圖文法匹配。
```

如果口委問「GrammarFirst 為什麼目前分數較低？」可以回答：

```text
GrammarFirst 目前分數較低，不是因為 KG-valid path 產生不出來，而是因為它產生太多可執行但不一定對題的 path。
也就是說，可執行性解決 syntax，沒有解決 semantic relevance。
這個 diagnostic result 說明 HRG path-bank 已經能把 search space 推到 candidate generation 層，但要成為主方法還需要 question-relation alignment、answer type constraints 或 learned reranker。
```

### 21.5 Offline grammar extraction 統計

這一節回答「HRG-like grammar 到底有沒有被抽出來、抽出多少、是否穩定」的問題。這裡的 `unique patterns` 是以 rule RHS 中的 relation multiset 當作 pattern signature；它不是完整圖同構檢查，但可作為 relation-structure prior 的可重現統計。

> 圖：`img/paper_figures/14_offline_grammar_statistics.pdf`。
> 圖說：Offline grammar statistics across datasets. MetaQA and WikiMovies have compact relation vocabularies, while MLPQ and KQAPro contain more heterogeneous relation structures.

| Dataset | Rules | Unique relation patterns | Pattern ratio | Avg terminal arity | Max terminal arity | Unique relations | Terminal edges in rules | Top relations |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MetaQA | 304 | 247 | 0.812 | 10.82 | 134 | 9 | 3288 | has_genre, has_tags, starred_actors |
| WikiMovies | 335 | 248 | 0.740 | 8.91 | 93 | 10 | 2984 | starred_actors, has_genre, has_tags |
| MLPQ | 398 | 353 | 0.887 | 9.22 | 392 | 262 | 3670 | subdivisionType, subdivisionName, stadium |
| KQAPro | 187 | 185 | 0.989 | 44.74 | 1377 | 300 | 8366 | occupation, instanceOf, twinned administrative body |

解讀：MetaQA / WikiMovies 的 relation vocabulary 小、rule patterns 集中，適合 relation-chain retrieval；MLPQ / KQAPro 的 relation vocabulary 大且 pattern ratio 高，表示 schema 更異質，online relation canonicalization 與 operator / qualifier semantics 更困難。KQAPro 的 terminal arity 特別高，代表 hub / statement-like structures 很多，這也是 fallback 容易造成 context 上升的原因。

### 21.6 Offline grammar robustness under KG perturbation

這一節回答 offline HRG prior 在 KG 隨機缺失時是否仍保留結構訊號。

> 圖：`img/paper_figures/15_grammar_perturbation_trends.pdf`。
> 圖說：Under random node deletion, extracted rule counts remain available while relation vocabulary retention differs by dataset complexity. This supports using HRG as a soft structural prior rather than a hard decoder.

| Dataset | Clean rules | Node drop 10% | Node drop 20% | Node drop 30% | Relation drop 10% | Relation drop 20% | Relation drop 30% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 304 | 326 | 368 | 433 | 336 | 371 | 401 |
| WikiMovies | 335 | 314 | 360 | 376 | 314 | 328 | 383 |
| MLPQ | 398 | 467 | 453 | 526 | 364 | 451 | 379 |
| KQAPro | 187 | 343 | 350 | 288 | 299 | - | - |

Rule count 在 perturbation 後不一定下降，因為 MCS / triangulation 後的 clique decomposition 可能把缺失後的圖切成更多局部 pattern。這不能解讀成「刪除越多越好」，而應解讀成：KG perturbation 改變了局部結構分解，HRG extraction 仍能產生可用的 structural prior。

因此 perturbation 不能只看 rule count。論文中應同步說明：目前 perturbation robustness 是 grammar availability / structural-prior stability 的 diagnostic，不是 gold evidence robustness 的完整證明；若要更完整，需要另外報告 gold answer 是否仍可達、original correct path 是否仍存在，以及 retained rules 中有多少和 query-relevant path overlap。

### 21.7 Gold evidence evaluation scope

本文報告以下 automatic evidence-level diagnostics：

1. **Endpoint answer coverage**：gold answer 是否出現在 retrieved edges / spine edges。
2. **Answer-conditioned retrieval recall / precision / F1**：以 gold answer 是否被 retrieved evidence 覆蓋為基礎。
3. **Grammar-hit conditioned success**：可以比較 grammar_hit=true / false 的 answer F1、answer-in-spine、failure rate。
4. **Failure taxonomy**：把題目分成 no_candidates、no_valid_chain、KG-valid fallback、LLM correction、grammar fallback 等類別。
5. **Case analysis**：展示 Spine-Correction 失敗但 HRG-Proposed 成功的代表題目。

目前 benchmark 已在 aggregated result 中輸出以下欄位，對應老師建議的 evidence sanity check：

| 類別 | 欄位 | 用途 |
|---|---|---|
| Endpoint coverage | `endpoint_coverage_final`, `endpoint_coverage_spine`, `endpoint_coverage_expanded` | 分別檢查 gold answer entity 是否出現在 final context、strict spine、expanded evidence。 |
| Binary coverage | `answer_in_final_context`, `answer_in_spine`, `answer_in_expanded_edges` | 將 coverage 轉成 rate，方便主表報告。 |
| Expansion contribution | `gold_only_in_expansion` | 若為 1，代表答案只由 expansion 補進來，是 HRG expansion 有實質貢獻的 case。 |
| Conditional generation | `answer_f1_when_answer_in_context`, `answer_f1_when_answer_not_in_context`, `answer_in_context_but_wrong`, `answer_correct_without_context` | 檢查 LLM 是否真的能使用 retrieved evidence；也能診斷 parametric memory 或 context overload。 |
| Context truncation | `raw_retrieved_edge_count`, `final_edge_count`, `context_truncated`, `context_truncation_ratio` | 回答 BFS / HRG 是否因 budget 被截斷，特別用於 MetaQA 3-hop BFS sanity check。 |
| Evidence composition | `spine_edge_count`, `expanded_edge_count` | 區分 strict spine 與 expansion 的 evidence 成本。 |
| Grammar decomposition | `grammar_label_subset_hit`, `grammar_ordered_path_hit`, `grammar_arity_compatible_hit`, `grammar_structural_proxy_hit`, `grammar_full_structural_hit` | 把原本的 grammar hit 拆開，避免把 relation label overlap 誤講成完整 HRG structural match。 |

這些 endpoint coverage 指標的 gold set 是 answer entity set，不是完整 gold path / gold program。因此它們可以用來說明「答案是否進入 evidence context」，但不能直接宣稱 retrieved edges 全部都是人工正確 supporting evidence。

Evidence diagnostic toy examples：

```text
Example 1: answer in context but wrong

Gold answer:
Andrew V. McLaglen

Final context:
Mitchell | starred_actors | Linda Evans
Mitchell | directed_by | Andrew V. McLaglen

LLM output:
John Michael Hayes

Metrics:
answer_in_final_context = true
answer_in_context_but_wrong = true
answer_f1 = 0
```

這代表 retrieval 已經把答案放進 context，但 answer generator 仍然失敗；不能把錯誤歸因於 KG retrieval。

```text
Example 2: answer not in context but model guesses correctly

Gold answer:
Andrew V. McLaglen

Final context:
Mitchell | starred_actors | Linda Evans
Mitchell | written_by | John Michael Hayes

LLM output:
Andrew V. McLaglen

Metrics:
answer_in_final_context = false
answer_correct_without_context = true
```

這代表模型可能靠 parametric memory 或 dataset prior 猜對；這種 case 不能當成 evidence retrieval 成功。

```text
Example 3: gold only in expansion

Strict spine:
Mitchell | starred_actors | Linda Evans

Expanded edge:
Mitchell | directed_by | Andrew V. McLaglen

Gold answer:
Andrew V. McLaglen

Metrics:
answer_in_spine = false
answer_in_expanded_edges = true
gold_only_in_expansion = true
```

這種 case 才能支持「expansion 補進了必要 gold evidence」。目前 MetaQA latest supplement 中 `gold_only_in_expansion = 0.0`，所以不能把該 run 的 F1 提升歸因於 expansion。

每個 model report 另保留 `evaluation_unit="question"`、`evaluated_question_count`、`sample_limit`，每個 dataset row 保留 `n_questions`。主文若把多模型結果合併，必須明確寫 aggregation unit 是 latest 3-model average、legacy four-model dump，還是 question-model pairs，不能再用模糊的 full-data version。

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

#### 21.7.2 老師建議與目前程式指標對照

| 老師建議 | 本次對應欄位 / 狀態 |
|---|---|
| gold answer 是否在 BFS / HRG context | `answer_in_final_context`, `endpoint_coverage_final` |
| answer-in-spine 與 expansion 貢獻 | `answer_in_spine`, `answer_in_expanded_edges`, `gold_only_in_expansion` |
| context truncation / context overload | `raw_retrieved_edge_count`, `final_edge_count`, `context_truncated`, `context_truncation_ratio` |
| answer 已在 context 時 LLM 是否仍失敗 | `answer_f1_when_answer_in_context`, `answer_in_context_but_wrong` |
| 沒有答案 evidence 時是否仍答對 | `answer_f1_when_answer_not_in_context`, `answer_correct_without_context` |
| grammar hit 拆解 | `grammar_label_subset_hit`, `grammar_ordered_path_hit`, `grammar_arity_compatible_hit`, `grammar_structural_proxy_hit` |
| full structural HRG match | `grammar_full_structural_hit` 保留為 null；現行程式沒有完整 structural derivation matcher，不能宣稱已完成。 |
| 每個模型與樣本數 | `n_questions`, `evaluated_question_count`, `sample_limit`；完整 model × method 表由 `benchmark_results.json` 和 wide CSV 產生。 |
| final context tokens vs total online cost | 既有 `avg_ctx_tokens`, `avg_parse1_tokens`, `avg_correction_tokens`, `avg_parse2_tokens`, `avg_total_online_token_proxy`。 |
| gold edge precision / recall | 既有 `evidence_precision`, `evidence_recall`, `evidence_f1`，但只在 metadata 有 `gold_path_parts` 時有效。 |
| no-context / wrong-context / masked-answer controls | 屬於另跑 answer-generation control，不是本次主 benchmark 欄位。 |
| 多 seed perturbation / hyperparameter sensitivity | 屬於另跑實驗；本次只確保主 benchmark 能報 truncation、coverage、failure 與 token 成本。 |
| KQAPro gold-program upper bound / official program match | 需要 KQAPro program evaluator；目前不宣稱已完成。 |

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

Linda Evans 例子的 pseudo-code trace：

```text
q =
  who directed the films starred by [Linda Evans]

R contains:
  starred_actors, directed_by, written_by, has_genre, ...

Step 1:
C = [
  c1 = {entity: Linda Evans, chain: [starred_actors, directed_by], confidence: 0.91},
  c2 = {entity: Linda Evans, chain: [starred_actors, written_by], confidence: 0.43}
]

Step 2:
c1.entity_node = Linda Evans
c1.kb_result.valid = true
c1.kb_result.signed_trace = [starred_actors[r-], directed_by[r+]]
c1.kb_result.final_frontier = {Andrew V. McLaglen}

c2.entity_node = Linda Evans
c2.kb_result.valid = true
c2.kb_result.final_frontier = {writer candidates}

Step 3-4:
skip correction / KG-valid fallback because at least one initial candidate is valid.

Step 5:
V = {c1, c2}

Step 7:
S_c1 =
  Mitchell | starred_actors | Linda Evans
  Mitchell | directed_by | Andrew V. McLaglen

S_c2 =
  Mitchell | starred_actors | Linda Evans
  Mitchell | written_by | John Michael Hayes

Step 8:
S = S_c1 because question relevance and answer type support directed_by.

Step 9:
context =
  Mitchell starred_actors Linda Evans.
  Mitchell directed_by Andrew V. McLaglen.

Step 10:
y = Andrew V. McLaglen
```

如果 `c1` 一開始不存在，且所有 initial candidates invalid，才會進入 Step 3 / Step 4 的 correction 與 fallback；這是本文和「每題都讓 LLM 修正」不同的地方。

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

更精確地說，HRG-Proposed 仍假設 LLM top-k parse、correction 或 KG-valid fallback 至少能把候選空間帶到合理 relation-chain 附近；它主要是 LLM-chain-guided retrieval。HRG-GrammarFirst 放寬這個假設：它不使用 LLM 預測 hop 數或 relation chain 作為 search-space 前提，而是從 grounded entity 枚舉 1..D hop KG-valid chains，再用 grammar / relevance / compactness / optional LLM rerank 排序。不過 latest MetaQA supplement 顯示，GrammarFirst 目前的瓶頸是可執行 chain 不一定 question-relevant；它仍然需要更強的 semantic relation filtering、answer-type constraints 與 operator-aware parsing。

因此，資料集越像乾淨的 relation-chain traversal，executable spine 越有效；越需要跨語言對齊、operator、filter、count、comparison、qualifier，結果越困難。MetaQA 200-per-hop 主表支持 compact HRG-guided executable evidence retrieval；latest supplement 則補上 simple relation-prior controls，界定 clean small-schema dataset 下的 claim boundary。

### 23.1 Dataset Fit Summary

| Dataset | 結果現象 | 主要原因 | 報告定位 |
|---|---|---|---|
| MetaQA main 200-per-hop | HRG-Proposed-triple 用 BFS 4.28% context，四模型平均 F1 高於 BFS | 問題天然是 movie-domain relation chain，HRG-guided executable evidence 可大幅壓縮 final context | compact executable evidence 主證據 |
| MetaQA latest supplement | HRG-Proposed 仍高於 BFS；Spine-Correction、KGValidFallback 與 relation-prior rows 是 strong controls | clean schema 讓 strict spine 與 relation transitions 很有效 | evidence diagnostics + claim boundary |
| MetaQA GrammarFirst diagnostic | F1 低於 BFS / HRG-Proposed | KG-valid grammar path 可執行但不一定對題 | semantic filtering bottleneck / future work |
| WikiMovies diagnostic | HRG-Proposed 不是最強；BFS/Spine 已接近 ceiling | 幾乎都是 1-hop，BFS 子圖本來就很小 | ceiling case |
| MLPQ legacy stress test | HRG-Proposed 比 Spine-Correction 更接近 BFS，但仍低於 BFS | 跨語言 entity/relation 對齊困難 | HRG 補強 + limitation |
| KQAPro legacy stress test | Spine-Correction 幾乎失敗；HRG-Proposed-triple 補回 executable evidence coverage，但 BFS 本身也低 | 問題是 semantic program，不是單純 chain | out-of-scope stress test |

摘要句：

```text
方法成功與否主要取決於 dataset semantics 是否能對齊到 executable relation chain。
MetaQA 最符合這個假設，所以 HRG-guided executable evidence 能在高於 BFS answer quality 的同時大幅減少 token；latest supplement 也顯示 relation-prior baseline 必須被認真比較。
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

因此 executable spine 很容易發揮作用，而 HRG-Proposed 可以用更小 context 高於 BFS F1。200-per-hop 主表是本文主結果；latest supplement 則用來展示 strict spine、KG-valid fallback 與 relation-prior controls：

```text
MetaQA Baseline-BFS, latest 3-model supplement:
EM 0.4289 / F1 0.5277 / ctx tokens 4341.9 / subgraph 360.9

MetaQA Spine-Correction-triple, latest 3-model supplement:
EM 0.5089 / F1 0.6419 / ctx tokens 245.2 / subgraph 25.5

MetaQA Spine-Correction-KGValidFallback-triple, latest 3-model supplement:
EM 0.5178 / F1 0.6582 / ctx tokens 702.9 / subgraph 74.7

MetaQA HRG-Proposed-triple, latest 3-model supplement:
EM 0.4911 / F1 0.6114 / ctx tokens 585.9 / subgraph 60.9

MetaQA RelationTrigram-triple, latest 3-model supplement:
EM 0.5200 / F1 0.6574 / ctx tokens 753.5 / subgraph 80.3
```

解讀：

```text
MetaQA 200-per-hop 主表中，HRG-Proposed-triple 比 BFS F1 更高且 context 只有 BFS 的 4.28%。
latest supplement 進一步顯示 executable spine、KG-valid recovery 與 relation n-gram priors
在小 schema MetaQA 上也很有競爭力，因此它們應作為 strong controls 一起報告。
```

MetaQA 3-hop 是關鍵細節：

```text
gpt-oss main result:
BFS 3-hop EM = 0.0050
Spine-Correction-json 3-hop EM = 0.1600
```

這表示 BFS 雖然 retrieval recall 高，但 LLM 面對大量 context 時未必能找到正確 reasoning path；spine 明確化 path 有助於 long-hop answer generation。這個 hop-level 表是 gpt-oss 單模型補充分析；主文以 200-per-hop 四模型表作主結果，`20260701-012816` 三模型 final-metrics 作補充消融與 evidence diagnostics。

### 23.3 WikiMovies 為什麼接近但不突出

WikiMovies 多數問題是 1-hop：

```text
what films did Michelle Trachtenberg star in?
Joe Thomas appears in which movies?
```

這類問題 BFS 很容易直接抓到答案，而且原始 BFS 子圖本來就小。

結果：

```text
WikiMovies BFS, legacy four-model dump:
EM 0.8025 / F1 0.8574 / ctx tokens 50.12 / subgraph 3.30

Spine-Correction-json, legacy four-model dump:
EM 0.8988 / F1 0.9210 / ctx tokens 79.43 / subgraph 3.42

HRG-Proposed-triple, legacy four-model dump:
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
MLPQ BFS, legacy four-model dump:
EM 0.2381 / F1 0.3078 / ctx tokens 4983.34

Spine-Correction-json, legacy four-model dump:
EM 0.1306 / F1 0.2185 / ctx tokens 101.48

HRG-Proposed-json, legacy four-model dump:
EM 0.1544 / F1 0.2514 / ctx tokens 359.79

HRG-Proposed-triple, legacy four-model dump:
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
KQAPro BFS, legacy four-model dump:
EM 0.0804 / F1 0.0842 / ctx tokens 3595.48

Spine-Correction-json, legacy four-model dump:
EM 0.0229 / F1 0.0237 / ctx tokens 7.33

HRG-Proposed-triple, legacy four-model dump:
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
7. correction 增益需要看觸發率與有效率，不能只看總分；latest MetaQA supplement 顯示 Spine-Correction 的可量化增益主要來自 executable spine / KG-valid recovery。LLM correction 應寫成保守啟動的 fallback module，而不是單獨承擔整個方法效果。
8. 若 topic entity grounding 錯誤，後面 relation chain 再正確也會失敗。
9. `qwen3.5` 不應只標成一般失敗。目前可用 A40 GPU compute capability 8.6 不支援 `Qwen/Qwen3.5-35B-A3B-FP8` 在 Transformers 的 FP8 runtime；模型會退回 bf16 dequantization 並 load-time OOM。因此它是硬體 / checkpoint 不相容，排除於 main aggregation。
10. 主結果與 robustness / ablation 結果使用不同模型集合；兩者不混成同一個平均，圖表 caption 標明模型集合與 aggregation method。
11. 10% / 20% KG perturbation 顯示 HRG-Proposed 能在 MLPQ / KQAPro 補回 strict spine 缺少的 evidence coverage，但也帶來 context 與 OOM 上升；因此 KG-valid fallback 與 candidate ranking budget 屬於後續優化方向。
12. latest MetaQA supplement 中，Spine-Correction-triple、KGValidFallback-triple 與 RelationBigram/Trigram-triple 是強消融 / 強 baseline，因此不能宣稱 HRG-Proposed 是所有 row 的 answer-quality 最高點；較準確說法是 HRG-Proposed 在主表比 BFS 有更好的 quality-cost trade-off，並提供 grammar decomposition、fallback/ranking signals 與可檢查 evidence trace。
13. `r+ / r-` 是 derived execution trace，不是新增 relation label。若論文使用 signed relation notation，必須同時保留原始 KG triple，以免讀者誤以為資料被擴充了 reverse relations。
14. Expansion 不是主方法必要條件；若新實驗包含 expansion，必須以 `NoExpansion` / `Expansion` 明確分 row，避免把擴邊效果混進 strict-spine claim。
15. HRG-GrammarFirst 能降低對 LLM hop/chain 預測的依賴，但 latest MetaQA supplement 顯示它目前低於 HRG-Proposed 與 strict spine rows。原因是它會擴大 KG-valid candidate search space；若 relation relevance、answer type、frontier compactness 或 context budget 控制不足，會產生「可執行但不對題」的 chains。因此它應寫成 diagnostic bottleneck / future-work motivation，不能寫成已證明優於 HRG-Proposed。
16. 目前主流程主要是 failure-triggered fallback；程式已有 `use_low_confidence_valid_chain_fallback` 這類低信心觸發開關，但固定 ablation 預設未宣稱其效果。若要主張 correction 能處理「valid but semantically wrong」candidate，必須另跑 low-confidence-triggered ablation。

### 24.1 Additional Experimental Scope

以下項目屬於後續實驗範圍；在本文中不作為主要實證 claim：

1. **Additional baseline scope**：最新 MetaQA final-metrics 已納入 token-budgeted BFS、degree-capped BFS、relation-frequency reranker 等 baseline；relation-similarity-pruned BFS、beam-search relation path、symbolic endpoint answer 仍可作為後續比較。
2. **HRG vs simpler priors**：unigram / bigram / trigram relation-prior ablation 顯示簡單 n-gram prior 能解釋部分 candidate ranking，但不能取代 HRG-Proposed 的 grammar-hit、KG-valid fallback 與 ranking signals；本文不宣稱 HRG 已全面優於所有簡單 relation-pattern prior。
3. **Oracle error decomposition**：gold entity + predicted relation、predicted entity + gold relation、gold entity + gold relation、all predicted 等 oracle 設定可作為後續錯誤分解，使 MLPQ / KQAPro 的失敗比例能更精細地拆成 grounding、relation linking、retrieval 或 generation。
4. **End-to-end cost**：本文報告 parse1、correction、parse2、context 的 online token proxy；它用於比較方法成本輪廓，不等同於精確 API billing cost。
5. **Evidence correctness**：本文可以輸出 chain、spine edges 與 matched rule；gold path / gold program 與人工 evidence sufficiency 不作為本文主 claim。
6. **Statistical significance**：本文報告 paired bootstrap confidence interval；多次 decoding repeat 與完整 significance matrix 可作為後續統計分析。

本文已完成的是 HRG-guided executable compact evidence retrieval 的系統、MetaQA final-metrics、evidence diagnostics 與初步 robustness 分析；更廣泛的 graph retrieval / cross-dataset rerun 可以放在 future work 或 supplementary。

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

本研究提出 HRG-Proposed KG-RAG，將 KGQA retrieval 從無結構 BFS 子圖展開轉為 HRG structural prior-guided executable relation-chain retrieval。MetaQA 200-per-hop 四模型主表顯示，HRG-Proposed-triple 能以 BFS `4.28%` 的 final evidence context 取得高於 BFS 的 F1；`gpt-oss` 單模型也保留 `1.75%` context 的直觀壓縮例子。latest MetaQA supplement 進一步補上 fair BFS、relation-prior controls、GrammarFirst 與 evidence diagnostics，顯示 HRG-Proposed 仍高於 BFS，同時 strict spine、KGValidFallback 與 relation n-gram rows 是必須正面比較的 strong controls。系統能輸出 selected entity、relation chain、derived signed execution trace、matched grammar prior、spine edges 與 final context，因此提供 evidence-level explainability。HRG-GrammarFirst 已完成實作並納入 latest supplement；它目前揭示 semantic relation filtering 與 answer-type constraints 的 bottleneck，作為 HRG retrieval-engine extension 的後續方向。對於跨語言、operator-heavy 或 qualifier-heavy 資料集，目前結果仍指向後續需要 relation canonicalization、operator-aware parsing、answer-type constraints、low-confidence correction 與 statement-aware reasoning。
