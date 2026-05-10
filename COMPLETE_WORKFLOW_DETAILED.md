# Complete Workflow and Ablation Notes

## 1. Scope and Positioning

This document consolidates the current `portable_runner` workflow into a thesis-safe, implementation-grounded description. It covers:

1. environment and dataset preparation
2. offline HRG grammar induction
3. online LLM-based KGQA inference
4. benchmark execution flow
5. ablation design
6. evaluation metrics
7. output artifacts
8. practical constraints and interpretation boundaries

The current system is best described as a **training-free, model-agnostic KGQA framework**. It does not fine-tune the LLM. Instead, it:

1. extracts structural priors from the KG offline
2. uses the LLM online for semantic parsing and answer generation
3. uses HRG as a retrieval constraint and context compression mechanism

This framing is important. The goal is not only answer quality, but also the trade-off among:

1. answer correctness
2. context size
3. token cost
4. execution time

## 2. End-to-End Execution Entry Points

### 2.1 Docker workflow

The default Docker entry point is:

```bash
cd portable_runner
docker compose up --build
```

This currently runs the four datasets that can be closed end-to-end inside the portable workflow:

1. `metaqa`
2. `wikimovies`
3. `mlpq`
4. `kqapro`

The default batch list is defined in [run_all_benchmarks.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_all_benchmarks.sh).

### 2.2 Non-Docker workflow

The local one-command entry point is:

```bash
cd portable_runner
bash run_local_all.sh
```

This script performs:

1. environment setup
2. dataset download / normalization
3. config generation
4. sequential benchmark execution

### 2.3 Main orchestration scripts

The main scripts are:

1. [setup_env.sh](/home/zihui/projects/masterPaperRemake/portable_runner/setup_env.sh)
2. [download_datasets.sh](/home/zihui/projects/masterPaperRemake/portable_runner/download_datasets.sh)
3. [download_datasets.py](/home/zihui/projects/masterPaperRemake/portable_runner/download_datasets.py)
4. [generate_configs.py](/home/zihui/projects/masterPaperRemake/portable_runner/generate_configs.py)
5. [resolve_kb.py](/home/zihui/projects/masterPaperRemake/portable_runner/resolve_kb.py)
6. [run_pipeline.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_pipeline.sh)
7. [auto_benchmark.sh](/home/zihui/projects/masterPaperRemake/portable_runner/auto_benchmark.sh)
8. [run_all_benchmarks.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_all_benchmarks.sh)
9. [run_local_all.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_local_all.sh)

## 3. Dataset Preparation and Current Coverage

### 3.1 Datasets that currently run end-to-end

The current portable workflow is wired for:

1. `MetaQA`
2. `WikiMovies`
3. `MLPQ`
4. `KQAPro`

Their fixed roots are under `portable_runner/Datasets/`.

### 3.2 Dataset-specific notes

#### MetaQA

Files are fixed to:

1. dataset root: `Datasets/MetaQA`
2. KB: `Datasets/MetaQA/kb.txt`
3. relations: `Datasets/MetaQA/relations.json`
4. benchmark split: `test`
5. variant: `vanilla`

This is the cleanest fit to the current pipeline.

#### WikiMovies

Files are fixed to:

1. dataset root: `Datasets/WikiMovies`
2. KB: `Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt`
3. question file: `Datasets/WikiMovies/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt`
4. benchmark split: `test`

This also fits the current line-based triple loader directly.

#### MLPQ

Files are fixed to:

1. dataset root: `Datasets/MLPQ`
2. KB: `Datasets/MLPQ/datasets/KGs/fusion_bilingual_KGs/ILLs_fusion/merged_ILLs_KG_en_zh.txt`
3. default question setting: `en-zh / en / ills`

This is a multilingual setting and is more sensitive to answer normalization than MetaQA or WikiMovies.

#### KQAPro

KQAPro required an adaptation layer. The original dataset snapshot includes `hf_snapshot/kb.json`, which is not directly consumable by the line-based triple parser used by the benchmark. The portable workflow therefore:

1. downloads the snapshot
2. converts `kb.json` into `Datasets/KQAPro/kqapro_kb_triples.tsv`
3. uses `validation` instead of `test`, because the official `test.json` does not contain gold answers

This makes KQAPro runnable, but it should be described as an **adapted compatibility setting**, not a fully original benchmark setting.

### 3.3 Datasets not currently closed end-to-end

The portable workflow also downloads:

1. `WQSP`
2. `CWQ`
3. `Mintaka`

However, these currently provide question data only. They do not ship with benchmark-ready KG triples in the same format used by this pipeline, so they are not included in the default runnable batch.

## 4. Offline Stage: HRG Grammar Generation

### 4.1 Role of the offline stage

The offline stage converts the original KG into a reusable set of structural rules. The grammar is not used to generate answers. Instead, it serves as:

1. a structural prior
2. a relation co-occurrence model
3. a retrieval-space constraint during online QA

### 4.2 Main implementation

Offline grammar extraction is implemented in:

1. [portable_runner/hrg_grammar/hrg_extract.py](/home/zihui/projects/masterPaperRemake/portable_runner/hrg_grammar/hrg_extract.py)

The high-level procedure is:

1. load the KG as a labeled `MultiDiGraph`
2. convert it to an undirected skeleton for decomposition
3. perform robust BFS node-induced sampling
4. triangulate sampled subgraphs via MCS ordering
5. build clique candidates and a clique tree
6. binarize and prune the clique tree
7. convert each bag into an HRG rule
8. merge duplicate rules and save the grammar

### 4.3 Why sampling is used

The code does not induce grammar from the full KG directly. Instead, it samples subgraphs first. This is a deliberate engineering choice to avoid:

1. clique explosion
2. high-degree hub domination
3. intractable intermediate structures

The current code uses several control parameters:

1. `K_SAMPLES = 4`
2. `S_SAMPLE_SIZE = 500`
3. `SEED_DEGREE_QUANTILE = 0.80`
4. `BFS_MAX_BRANCH = 30`
5. `RANDOM_SEED = 0`

These are partly resource-driven and partly heuristic. In a thesis, it is safer to describe them as **computation-control parameters** and not as theoretically optimal constants.

### 4.4 Clique-tree-to-HRG conversion

Each clique-tree bag becomes one local rule unit:

1. the left-hand side `lhs` is either `S/rank` or `N/rank`
2. terminal edges become `rhs.terminals`
3. child attachments become `rhs.nonterms`

After extraction, duplicate `(lhs, rhs)` rules are merged and their counts accumulated.

### 4.5 Output locations

For a given run tag, the offline stage writes to:

1. `artifacts/<run_tag>/grammar/hrg_grammar.json`
2. `artifacts/<run_tag>/grammar/hrg_grammar.txt`

The exact output path is coordinated through [experiment_naming.py](/home/zihui/projects/masterPaperRemake/experiment_naming.py) and [run_pipeline.sh](/home/zihui/projects/masterPaperRemake/portable_runner/run_pipeline.sh).

### 4.6 Example grammar rule

An HRG rule stored in JSON looks like:

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

Interpretation:

1. `lhs` is the nonterminal being expanded
2. `rhs.terminals` are explicit relation edges in the local pattern
3. `rhs.nonterms` are expandable structural slots
4. `count` records how often the pattern appears in sampled structures

The numbers in `a`, `b`, and `att` are **local node indices within the rule**, not original KG entity IDs.

### 4.7 What the offline stage is really learning

The grammar captures repeated local structural motifs such as:

1. a movie node linked to director, writer, and tags
2. a core entity linked to several recurring relation bundles

This is why the grammar is later used to decide which local neighborhoods around a spine are structurally plausible.

## 5. Online Stage: LLM Inference and Retrieval

### 5.1 Main implementation

The online benchmark is centered on:

1. [portable_runner/LLM_inference_benchmark/benchmark.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/benchmark.py)
2. [portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/knowledgegraph_agent.py)
3. [portable_runner/LLM_inference_benchmark/baseline.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/baseline.py)
4. [portable_runner/LLM_inference_benchmark/LLM_stratiges.py](/home/zihui/projects/masterPaperRemake/portable_runner/LLM_inference_benchmark/LLM_stratiges.py)

### 5.2 LLM prompt formatting

The Hugging Face inference path now uses `tokenizer.apply_chat_template(...)` when available. This matters because modern instruct checkpoints such as:

1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`

perform significantly better with their native chat template than with a manually concatenated `System / User / Assistant` text prompt.

### 5.3 Online retrieval flow

For each question, the online stage performs:

1. semantic parsing by the LLM
2. candidate relation-chain generation
3. candidate executability checking against the KG
4. optional correction if all first-round candidates fail
5. strict spine retrieval
6. optional local expansion around the spine
7. serialization of the final subgraph
8. answer generation conditioned on that serialized context

### 5.4 Spine

The `spine` is the strict evidence path obtained by following the selected relation chain over the KG. It is intentionally narrow:

1. only relations on the predicted chain are used
2. no unconstrained BFS expansion is allowed at this stage

This makes the spine the cleanest representation of the model’s first semantic hypothesis.

### 5.5 Correction

Correction is only triggered if all initial candidates fail. Current correction sources include:

1. direction flip candidates
2. grammar fallback candidates
3. LLM correction candidates

This keeps correction as a recovery mechanism rather than the primary parser.

### 5.6 Expansion

Once a spine is found, the system can optionally add local edges around spine nodes.

There are currently two expansion families:

1. **grammar-guided expansion**
   - only relations supported by matched HRG rules are allowed
   - expansion is locally constrained
2. **random expansion**
   - same expansion budget
   - no grammar guidance
   - serves as a control condition

### 5.7 Serialization

The final subgraph can be serialized in two formats:

1. `triples`
2. `json`

The current benchmark now evaluates both, because representation cost and answer quality can differ across datasets and models.

## 6. Current Benchmark Matrix

### 6.1 Backbone models

The current backbone set is:

1. `Qwen/Qwen2.5-7B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`

### 6.2 Baseline

The non-HRG baseline is:

1. `Baseline-BFS-{llama3.1,qwen2.5}`

It uses BFS-style retrieval with oracle hop depth supplied by the benchmark.

### 6.3 Core ablations

For each backbone, the benchmark now runs:

1. `Spine-Only-{backbone}-{json,triple}`
2. `Spine-Correction-{backbone}-{json,triple}`
3. `Spine-GrammarExpansion-{backbone}-{json,triple}`
4. `Spine-RandomExpansion-{backbone}-{json,triple}`
5. `HRG-Proposed-{backbone}-{json,triple}`

This gives:

1. method comparison
2. serialization comparison
3. backbone comparison

### 6.4 Meaning of each method

#### Spine-Only

Components enabled:

1. LLM semantic parsing
2. strict spine retrieval

Components disabled:

1. correction
2. grammar expansion
3. random expansion

Purpose:

1. lower-bound clean chain-first baseline

#### Spine-Correction

Components enabled:

1. LLM semantic parsing
2. correction
3. strict spine retrieval

Components disabled:

1. grammar expansion
2. random expansion

Purpose:

1. isolate the contribution of correction

#### Spine-GrammarExpansion

Components enabled:

1. LLM semantic parsing
2. strict spine retrieval
3. grammar-guided expansion

Components disabled:

1. correction
2. random expansion

Purpose:

1. isolate the contribution of grammar-guided local context completion

#### Spine-RandomExpansion

Components enabled:

1. LLM semantic parsing
2. strict spine retrieval
3. random local expansion with the same edge budget

Components disabled:

1. correction
2. grammar-guided expansion

Purpose:

1. test whether HRG is better than naive neighborhood growth

#### HRG-Proposed

Components enabled:

1. LLM semantic parsing
2. correction
3. strict spine retrieval
4. grammar-guided expansion

Purpose:

1. evaluate the full proposed method

### 6.5 Why Random Expansion matters

Without `Spine-RandomExpansion`, a reviewer can always argue that improvement comes only from adding more edges. The random variant addresses that concern directly:

1. same local expansion budget
2. no HRG guidance
3. therefore a cleaner control for “extra context” vs “structured extra context”

## 7. Evaluation Metrics

### 7.1 Main answer metrics

The benchmark now reports:

1. `em`
2. `answer_set_f1`
3. `answer_recall`
4. `hit_at_1_any`

Interpretation:

1. `em`
   - full exact set match
   - for single-answer questions, equivalent to exact answer correctness
2. `answer_set_f1`
   - main balanced score for multi-answer settings
3. `answer_recall`
   - partial credit over gold answers
   - if gold has 4 answers and the model gets 3, this becomes `0.75`
4. `hit_at_1_any`
   - whether at least one gold answer appears in the predicted answer set

### 7.2 Auxiliary answer metrics

The benchmark also keeps:

1. `answer_set_precision`
2. `answer_set_recall`
3. `bleu`
4. `contains_hit`

`BLEU` is retained for completeness, but it should not be the main metric for KGQA. `contains_hit` remains in JSON outputs for compatibility, but it is no longer treated as a main displayed metric.

### 7.3 Efficiency metrics

The benchmark also records:

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

### 7.4 Retrieval metrics

For runs where reference answers are available, the agent also tracks:

1. `avg_retrieval_recall`
2. `avg_retrieval_precision`
3. `avg_retrieval_f1`

These are useful because they help separate:

1. retrieval quality
2. answer-generation quality

## 8. Artifact Layout

The portable workflow uses a normalized artifact layout:

1. datasets live under `portable_runner/Datasets/...`
2. generated outputs live under `portable_runner/artifacts/<run_tag>/...`

Per run tag:

1. grammar: `artifacts/<run_tag>/grammar/`
2. result JSON: `artifacts/<run_tag>/results/benchmark_results.json`
3. detail CSV: `artifacts/<run_tag>/results/all_models_outputs_wide.csv`
4. dumps and shared retrieval caches: `artifacts/<run_tag>/dumps/`

The batch summary is written to:

1. `artifacts/_batch/run_all_summary.txt`

## 9. Example End-to-End Sequence

For one dataset run, the practical sequence is:

1. prepare Python environment
2. download and normalize dataset
3. resolve fixed dataset root and KB path
4. build `run_tag`
5. generate grammar JSON/TXT from the dataset KB
6. load benchmark split
7. initialize baseline and KG agents
8. for each question:
   - parse candidate chain
   - validate against KG
   - optionally correct
   - retrieve strict spine
   - optionally expand
   - serialize as `json` or `triples`
   - answer with the LLM
   - compute metrics
9. aggregate results and write run artifacts

## 10. Current Interpretation Boundaries

### 10.1 Safe claims

The current workflow supports these claims reasonably well:

1. HRG can serve as an offline structural prior for KGQA retrieval
2. chain-first retrieval plus local structural expansion can improve the trade-off between answer quality and context cost
3. grammar-guided expansion can be compared against random expansion under the same budget
4. `json` and `triples` can be compared as LLM-facing serialization choices

### 10.2 Claims that should be made carefully

These should be stated conservatively:

1. KQAPro is an adapted setting because `kb.json` is converted into triples
2. cross-dataset averages should not be overemphasized; dataset-wise analysis is safer
3. `json` vs `triples` compares LLM-facing prompt serialization, not abstract formal superiority of one KG representation
4. some offline parameters are heuristic and should be presented as computation-control choices rather than universal optima

### 10.3 Parameters that may look heuristic

The most likely “magic-number” suspects are:

1. `K_SAMPLES = 4`
2. `S_SAMPLE_SIZE = 500`
3. `SEED_DEGREE_QUANTILE = 0.80`
4. `BFS_MAX_BRANCH = 30`
5. `topk_expansion_rules = 1`
6. `expansion_min_prob = 0.005`
7. `expansion_per_node_cap = 5`

These are not necessarily wrong, but they should be described as:

1. computation-budget controls
2. context-budget controls
3. conservative defaults

and not as theoretically optimal values.

## 11. Recommended Result Presentation Style

A safer thesis structure is to compare methods **within each dataset**, not to force a single pooled ranking across very different KGQA datasets.

Recommended per-dataset presentation:

1. answer metrics
   - `EM`
   - `Answer-Set F1`
   - `Answer Recall`
2. efficiency metrics
   - `Avg Context Tokens`
   - `Avg Latency`
   - `Avg Subgraph Size`
3. retrieval metrics
   - `Subgraph Recall`
   - `Subgraph Precision`
   - `Subgraph F1`
4. serialization comparison
   - `json` vs `triples`

This keeps the conclusions dataset-specific and easier to defend.

## 12. Practical Summary

The current portable workflow is no longer just a demo runner. It now provides:

1. fixed-path dataset handling
2. offline grammar generation
3. adapted KQAPro support
4. training-free online KGQA inference
5. backbone comparison
6. method ablations
7. serialization comparison
8. answer, retrieval, token, and latency metrics

In short, the system can now support a thesis narrative built around:

1. offline graph structure induction
2. online chain-first constrained retrieval
3. explicit ablations for correction and expansion
4. controlled comparison between grammar-guided and random expansion
5. explicit efficiency–effectiveness trade-off analysis
