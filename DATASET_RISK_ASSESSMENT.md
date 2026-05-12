# DATASET_RISK_ASSESSMENT

## Purpose

This note separates dataset-side risk from method-side risk for the datasets currently supported in this repo.  
The goal is to avoid over-attributing weak benchmark results to the HRG retrieval method when some failures are actually caused by KB encoding, entity surface-form mismatch, or dataset structure.

The assessment is based on:

1. The current dataset loaders in [LLM_inference_benchmark/dataset_utils.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py)
2. The current benchmark pipeline in [LLM_inference_benchmark/benchmark.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/benchmark.py)
3. The observed small-sample debug runs on `WikiMovies`
4. The stored benchmark outputs already discussed in `COMPLETE_WORKFLOW_DETAILED.md`

---

## Overall View

### Most aligned with the proposed method

1. `MetaQA`

Reason:

1. KB triples are clean and atomic
2. Questions are relation-chain friendly
3. Topic entity identification is relatively easy
4. Multi-hop structure matches the intended HRG / chain-based workflow

### Highest dataset-side risk

1. `WikiMovies`

Reason:

1. The KB format needed a parser fix
2. The `wiki_entities` KB uses composite tail values for some relations such as `starred_actors`
3. Many actor names are not stored as standalone entity nodes
4. Entity lookup can fail even when the information is semantically present in the KB

### More likely method-stress datasets than parser-bug datasets

1. `KQAPro`
2. `MLPQ`

Reason:

1. Their loaders and record formats are comparatively clean
2. Their questions are harder semantically
3. Their retrieval depends more on robust entity grounding and executable relation chains
4. Weak results there are more likely to reflect method limitations than a simple loader error

---

## Dataset-by-Dataset Assessment

## 1. MetaQA

### Current data path

1. KB: [Datasets/MetaQA/kb.txt](/home/zihui/projects/masterPaperRemake/portable_runner/Datasets/MetaQA/kb.txt)
2. Loader: [load_metaqa_dataset()](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py:242)

### Structural characteristics

1. KB triples are atomic: `head|relation|tail`
2. Entities and relations are comparatively regular
3. Question styles are close to explicit hop/path reasoning
4. Topic entity is usually easy to identify

### Dataset-side risk

Low.

### Why it fits your method well

1. Your workflow assumes that LLM can parse an entity and a relation chain
2. It also assumes that executing the chain over the KB is meaningful
3. MetaQA is one of the closest datasets to that assumption

### Likely source of weak results if they appear

If MetaQA underperforms, it is more reasonable to suspect:

1. chain parsing quality
2. chain ranking quality
3. grammar matching / expansion design
4. generation from retrieved context

rather than dataset corruption.

### Oral-defense framing

The safest claim is:

`MetaQA is the cleanest environment for validating the intended HRG-style retrieval hypothesis, because the dataset structure closely matches the method assumptions.`

---

## 2. WikiMovies

### Current data path

1. KB: [Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt](/home/zihui/projects/masterPaperRemake/portable_runner/Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt)
2. Questions: [Datasets/WikiMovies/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt](/home/zihui/projects/masterPaperRemake/portable_runner/Datasets/WikiMovies/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt)
3. Loader: [load_wikimovies_dataset()](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py:265)

### What was already confirmed

Two independent dataset-side issues were confirmed.

#### A. Parser issue existed and was fixed

Previously, the KB line parser could mis-split WikiMovies lines and contaminate relation extraction.  
That caused grammar labels to include title fragments rather than canonical relations.

This was fixed by detecting the canonical WikiMovies relation token first and then splitting:

1. left side as head
2. relation token in the middle
3. right side as tail

Files changed:

1. [LLM_inference_benchmark/dataset_utils.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py)
2. [hrg_grammar/hrg_extract.py](/home/zihui/projects/masterPaperRemake/portable_runner/hrg_grammar/hrg_extract.py)

#### B. Composite-tail encoding exists in the KB

Example raw rows:

1. `The Inbetweeners 2 starred_actors James Buckley, Simon Bird, Blake Harrison, Joe Thomas`
2. `Chance Pe Dance starred_actors Shahid Kapoor, Genelia D'Souza`
3. `Michael Jordan to the Max starred_actors Michael Jordan, Doug Collins, Phil Jackson`

This means:

1. the actor is often not a standalone node
2. the actor string is embedded inside a multi-person tail value
3. reverse traversal from actor to movie is not naturally represented as a clean atomic node-edge-node pattern

### Why this matters for your method

Your retrieval pipeline expects:

1. parse an entity surface form
2. resolve that entity to a KB node
3. execute a chain like `starred_actors^-1`

But in `WikiMovies`, many actor questions actually require:

1. resolving the actor name to a composite tail node
2. then traversing backward to the movie

Without that alias support, many questions fail before grammar can help.

### What was fixed

Additional alias support was added:

1. accent folding
2. punctuation normalization
3. whitespace / underscore normalization
4. composite-tail alias splitting for `WikiMovies`

Files changed:

1. [LLM_inference_benchmark/dataset_utils.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py)
2. [LLM_inference_benchmark/knowledgegraph_agent.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py)
3. [LLM_inference_benchmark/baseline.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/baseline.py)

### Measured effect of the alias fix

Small-sample `WikiMovies` debug run with `qwen2.5`:

Before alias fix:

1. `Baseline-BFS`: `Answer-Set F1 = 0.0641`, `Hits@1 = 0.2000`
2. `Spine-Only-json`: `Answer-Set F1 = 0.0641`, `Hits@1 = 0.2000`

After alias fix:

1. `Baseline-BFS`: `Answer-Set F1 = 0.2641`, `Hits@1 = 0.4000`
2. `Spine-Only-json`: `Answer-Set F1 = 0.2641`, `Hits@1 = 0.4000`

Result file:

1. [artifacts_debug_qwen25_alias/wikimovies-wiki_entities-test/results/benchmark_results.json](/home/zihui/projects/masterPaperRemake/portable_runner/artifacts_debug_qwen25_alias/wikimovies-wiki_entities-test/results/benchmark_results.json)

### Dataset-side risk

High.

### What weak WikiMovies results do and do not mean

Weak `WikiMovies` results do not automatically mean:

1. the HRG idea is wrong
2. the chain parser is the only problem

They may instead reflect:

1. KB encoding mismatch
2. entity-node granularity mismatch
3. insufficient aliasing for composite values

### Oral-defense framing

The safest claim is:

`WikiMovies is not a pure method benchmark in this repo, because KB encoding choices materially affect whether a question is even executable as a graph query.`

---

## 3. MLPQ

### Current data path

1. Questions: loaded by [load_mlpq_dataset()](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py:308)
2. KB path resolution: [resolve_mlpq_kb_path()](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py:301)

### Structural characteristics

1. Bilingual / cross-lingual KG fusion
2. Evaluation focuses on `2-hop` and `3-hop`
3. The loader injects topic entity text into the question
4. KB is more complex than MetaQA due to merged multilingual structure

### Why MLPQ is risky for your method

Even if the loader itself is fine, MLPQ introduces several method-level stresses:

1. cross-lingual entity naming mismatch
2. cross-lingual relation interpretation mismatch
3. higher dependence on exact chain executability
4. more opportunities for alias or surface-form mismatch

### Why MLPQ is not obviously a WikiMovies-style parser bug

I do not currently see evidence that MLPQ has the same kind of parser corruption as WikiMovies.

Reasons:

1. the loader is more structured
2. the topic entity is explicitly injected into the question
3. the failure pattern previously discussed looked more like trade-off and execution weakness than malformed KB parsing

### Dataset-side risk

Medium.

### Method-side risk

High.

### Interpretation guideline

If MLPQ performs weakly, the first suspects should be:

1. multilingual entity grounding
2. path execution fragility
3. candidate ranking and chain validity

before blaming raw dataset corruption.

### Oral-defense framing

The safest claim is:

`MLPQ is structurally cleaner than WikiMovies in this pipeline, but it is much harsher on multilingual entity grounding and multi-hop chain execution.`

---

## 4. KQAPro

### Current data path

1. Questions: [Datasets/KQAPro/normalized/validation.jsonl](/home/zihui/projects/masterPaperRemake/portable_runner/Datasets/KQAPro/normalized/validation.jsonl)
2. Loader: [load_normalized_jsonl_dataset()](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/dataset_utils.py:416)
3. KB: [Datasets/KQAPro/kqapro_kb_triples.tsv](/home/zihui/projects/masterPaperRemake/portable_runner/Datasets/KQAPro/kqapro_kb_triples.tsv)

### Structural characteristics

1. The normalized QA file format is clean JSONL
2. Questions can be semantically complex even at low hop count
3. Answers include entities, counts, dates, and comparison-style outcomes
4. The benchmark path does not rely on the same kind of brittle WikiMovies row parsing

### Why KQAPro is hard for your method

KQAPro stresses:

1. semantic parsing quality
2. abstract relation selection
3. compositional reasoning
4. robustness of chain validity checking

So even with a clean loader, your method can still fail often.

### Why KQAPro is less suspicious as a dataset-loader bug

Compared with WikiMovies:

1. the question file format is straightforward
2. the normalization path is simple
3. there is no obvious evidence yet of relation labels being corrupted by parsing

### Dataset-side risk

Low to medium.

### Method-side risk

High.

### Interpretation guideline

Weak KQAPro results should more likely be interpreted as:

1. semantic parse difficulty
2. insufficiently constrained retrieval
3. fragile chain ranking
4. grammar matching being too weak structurally

than as raw dataset formatting failure.

### Oral-defense framing

The safest claim is:

`KQAPro is a cleaner benchmark for method stress than WikiMovies, because poor results there are more plausibly caused by compositional reasoning difficulty than by KB parsing artifacts.`

---

## Risk Summary Table

| Dataset | Data Format Risk | KB Structure Risk | Entity Lookup Risk | Method Stress | Overall Note |
|---|---:|---:|---:|---:|---|
| MetaQA | Low | Low | Low | Medium | Best match to method assumptions |
| WikiMovies | High | High | High | Medium | Dataset structure can dominate failure |
| MLPQ | Low-Medium | Medium | Medium-High | High | More multilingual stress than parser stress |
| KQAPro | Low | Medium | Medium | High | Cleaner data, harder reasoning |

---

## Practical Takeaways

## What you can safely claim now

1. `MetaQA` is the cleanest validation environment for your proposed workflow
2. `WikiMovies` results were materially affected by dataset-side representation issues
3. `KQAPro` and `MLPQ` are more likely to reveal genuine method bottlenecks

## What you should not claim

1. `All weak results are caused by the method`
2. `All datasets are equally fair to a chain-based HRG workflow`
3. `WikiMovies is directly comparable to MetaQA without qualification`

## Recommended next checks

1. Run the same alias-aware debug on `2-hop` or more complex datasets to see whether entity grounding or chain validity dominates
2. Add oracle analysis:
   - gold entity + predicted chain
   - gold entity + gold chain
   - predicted entity + gold chain
3. Keep `WikiMovies` conclusions explicitly qualified in the thesis and oral defense

---

## Bottom-Line Statement

If you need one concise conclusion:

`MetaQA is the dataset most aligned with the proposed HRG workflow; WikiMovies has confirmed dataset-side representation issues; KQAPro and MLPQ are less suspicious as loader bugs and more useful for diagnosing true method limitations.`
