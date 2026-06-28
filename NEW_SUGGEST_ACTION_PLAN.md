# New Suggestion Action Plan

This plan maps `new_suggest.docx` to the current code/configuration and to the thesis text in `LLM_PROFESSOR_METHOD_FULL_EXPLANATION.md`.

## Current Configuration Facts

- Active model backbones: `gpt-oss`, `llama3.1`, `gemma4`, `qwen2.5`.
- `qwen3.5` is intentionally excluded from `MODEL_BACKBONES`.
- Reason: `Qwen/Qwen3.5-35B-A3B-FP8` requires GPU compute capability >= 8.9 for FP8 in Transformers; A40 is 8.6, so Transformers dequantizes to bf16 and causes load-time OOM.
- The main rerun already includes:
  - `Spine-Only`
  - `Spine-Correction`
  - `HRG-Proposed`
  - `Spine-Correction-KGValidFallback`
  - `HRG-Proposed-NoExpansion`
  - `HRG-Proposed-Expansion`
  - `HRG-GrammarFirst-NoExpansion`
  - `HRG-GrammarFirst-Expansion`
  - `RelationUnigram`, `RelationBigram`, `RelationTrigram`
  - `Degree-Capped-BFS`
  - `Token-Budgeted-BFS`
  - perturbation with 5 seeds
- The current scripts already summarize:
  - per-model rows
  - perturbation mean/std
  - failure summary
  - context tokens and online token proxy
  - grammar hit / same-arity hit / expanded edge rates

## Highest Priority: Do These Before the Next Full Rerun

### 1. Fix Qwen3.5 Handling

Status: implemented in config.

Action:

- Keep `qwen3.5` excluded from the main matrix.
- Treat it as a hardware-incompatible checkpoint, not a normal failed method.
- In the thesis, replace "qwen3.5 FAILED" with:

```text
Qwen3.5-FP8 was excluded from the main aggregation because the available A40 GPUs
do not support FP8 runtime for this checkpoint in Transformers. The model is
therefore not a method failure, but an environment/hardware incompatibility.
```

### 2. Stop Overclaiming KQAPro

Status: thesis text needs revision.

Current risky claim:

```text
KQAPro 顯示 HRG-Proposed-triple 能將 F1 拉回 BFS 等級...
```

Replace with:

```text
KQAPro is treated as an out-of-scope stress test. HRG-Proposed does not solve
KQAPro QA; instead, it improves executable evidence recovery over strict-spine
ablations while remaining far from a full operator-aware KGQA solution.
```

Also split experiments into:

- `in-scope evaluation`: MetaQA, and any clearly relation-chain-friendly subsets.
- `out-of-scope stress test`: KQAPro, plus hard MLPQ cases.

### 3. Make the Main Table Complete and Consistent

Status: rerun matrix mostly supports this; thesis/table generation must enforce it.

Every dataset table should include the same rows:

- `Baseline-BFS`
- `Spine-Only-json`
- `Spine-Only-triple`
- `Spine-Correction-json`
- `Spine-Correction-triple`
- `Spine-Correction-KGValidFallback-json`
- `Spine-Correction-KGValidFallback-triple`
- `HRG-Proposed-NoExpansion-json`
- `HRG-Proposed-NoExpansion-triple`
- `HRG-Proposed-Expansion-json`
- `HRG-Proposed-Expansion-triple`
- `HRG-GrammarFirst-NoExpansion-json`
- `HRG-GrammarFirst-NoExpansion-triple`
- `HRG-GrammarFirst-Expansion-json`
- `HRG-GrammarFirst-Expansion-triple`
- `HRG-Proposed-json`
- `HRG-Proposed-triple`
- `RelationUnigram/Bigram/Trigram`
- `Degree-Capped-BFS`
- `Token-Budgeted-BFS`

Do not mix rows from older dumps with rows from a new dump in the same table.

### 4. Report Per-Model Results

Status: `summarize_experiment_runs.py` already writes `run_model_summary.csv`.

Action:

- Use `run_model_summary.csv` as the model x method source table.
- Add a derived table for method-level mean/std across active backbones.
- Make captions explicit:
  - `four-model average`
  - `single-model diagnostic`
  - `question-level n`
  - `question-model-pair-level n`

### 5. Define Metrics Exactly

Status: code has definitions; thesis text must quote them.

Add a metrics subsection:

- Answer normalization:
  - lowercase
  - punctuation removal
  - underscore/space normalization
  - KQAPro number normalization
- Multi-answer scoring:
  - split on `|`, `;`, newline
  - set precision/recall/F1
  - order ignored
- Empty / `I don't know` / format errors:
  - score as zero answer quality
  - separately counted through failure stage when available
- Retrieval diagnostics:
  - answer-conditioned coverage, not human gold-evidence sufficiency
  - distinguish endpoint answer coverage from gold edge/program correctness

### 6. Formalize Relation Direction

Status: method text and evidence tables need revision; full signed parser is optional.

Important wording:

```text
The KG relation label remains the original bare label r. The signs r+ and r-
are not new KG relations; they denote traversal direction over an existing KG
edge during chain execution.
```

Use this formal definition:

```text
r+ : traverse an edge (v, r, u) from current node v to u
r- : traverse an edge (u, r, v) from current node v to u
```

Actions:

- In method text, replace the ambiguous bidirectional step definition with signed traversal.
- In case studies, show both:
  - bare relation chain, for compatibility
  - signed relation chain, for direction
- In evidence tables, keep original triples as `(head, relation, tail)` and add a traversal-direction column.
- Do not say the KG contains reverse relation labels.
- Full signed parser output can be future work; the minimal requirement is to make hidden direction explicit in validation/case analysis.

## Additional Experiments to Add

### A. MetaQA 3-Hop BFS Sanity Check

Why: teacher flagged suspicious 3-hop BFS EM/F1.

Status: implemented as `analyze_metaqa_bfs_sanity.py`.

Add a diagnostic script that reports, by hop:

- gold answer in raw BFS context
- gold answer in budgeted/truncated BFS context
- context truncation rate
- context token distribution
- conditional accuracy when gold answer is in context
- sample failed cases

Interpretation:

- If answer is in context but LLM fails, this supports explicit spine evidence.
- If answer is not in context, fix BFS baseline before using it as a main comparator.

### B. Symbolic Endpoint Baseline

Why: separates retrieval quality from answer generation quality.

Status: implemented as post-hoc diagnostic `analyze_symbolic_endpoints.py`.

Add baselines:

- `Symbolic-Endpoint-Spine-Only`
- `Symbolic-Endpoint-Spine-Correction`
- `Symbolic-Endpoint-HRG-Proposed-NoExpansion`
- optionally `Symbolic-Endpoint-BFS`

Output the endpoint entities after traversal without asking the LLM to generate the final answer.

### C. Evidence Causal Controls

Why: teacher noted LLM may answer from parametric knowledge.

Status: implemented as optional model-loading script `run_context_controls_from_dumps.py`.

Run a small control on MetaQA and KQAPro:

- `NoContext`: question only
- `WrongContext`: same-format evidence from another question
- `MaskedAnswerContext`: replace gold answer entity in evidence with `[MASKED_ANSWER]`
- `CorrectContext`: normal HRG/BFS evidence

Only claim evidence causality if `CorrectContext` clearly beats the other three.

### D. Grammar Hit Breakdown

Why: current `grammar_hit` is too broad.

Status: implemented in `recompute_from_dumps.py` and `summarize_experiment_runs.py`; output file is `_summary/grammar_hit_breakdown.csv`.

The code already has candidate fields:

- `grammar_hit`
- `grammar_same_arity_hit`
- `grammar_ordered_path_hit`
- `grammar_weak_label_match`
- `grammar_matched_count`

Add summary columns:

- label-subset hit
- same-arity hit
- ordered-path hit
- weak-label-only hit
- full structural hit: not currently implemented; report as future work unless implemented.

### E. Hyperparameter Sensitivity

Keep this small.

Status: runner implemented as `run_hyperparameter_sensitivity.sh`; benchmark HRG parameters can be overridden through env vars.

Datasets:

- MetaQA
- KQAPro

Parameters:

- `num_candidates`: 1, 3, 5
- `valid_chain_fallback_beam_width`: 5, 10, 20
- `valid_chain_fallback_branch`: 5, 10, 20
- `max_total_context_edges`: 10, 30, 50
- grammar mode:
  - label-subset
  - ordered-path required

Metrics:

- EM
- answer-set F1
- answer-in-spine
- retrieval F1
- context tokens
- total online token proxy
- failure/OOM rate

### F. Grammar Extraction Stability

Status: `grammar_stability_analysis.py` already exists.

Run at least MetaQA and KQAPro with seeds 0-4 and report:

- rule count mean/std
- top rule signature Jaccard
- top relation label Jaccard
- top label count

If time permits, also run one downstream HRG-Proposed setting per seed.

### G. Perturbation Reachability Audit

Status: perturbation seeds are supported; reachability audit exists only for clean.

Implemented runner: `run_perturbation_reachability_audits.sh`.

For each ablated KG, report:

- topic entity deleted rate
- gold answer reachable rate
- original correct path retained rate where metadata permits
- effective test question count
- failure/OOM rate

This is needed because deletion can make F1 improve by removing noise or by changing the valid test set.

## Text-Only Revisions That Should Happen Even Without More Experiments

### Contribution Rewording

Do not present LLM correction as the main empirical driver.

Use:

1. executable relation-spine retrieval
2. KG-validated fallback and candidate recovery
3. HRG-like structural prior for fallback/ranking

Then state:

```text
In the current runs, LLM correction is not the main empirical driver; deterministic
KG-valid fallback and grammar-aware ranking are the main recovery mechanisms.
```

### Token Claim Rewording

Use two separate claims:

- final evidence context compression
- end-to-end online token proxy

Avoid saying generic "input tokens are reduced" unless the table is specifically final context tokens.

### Serialization Claim Rewording

Do not say JSON/triple is only appearance.

Use:

```text
Retrieval evidence is shared, but serialization is part of the answer-generation
interface and affects how easily the LLM can use the evidence.
```

### Grammar Claim Rewording

Current grammar is best described as:

```text
HRG-like structural prior based mainly on relation-label and arity/order compatibility.
It is not a full HRG decoder.
```

### Scope Rewording

Use:

```text
The method targets single-anchor, relation-chain-friendly KGQA. Counting,
comparison, argmax, verification, qualifier filtering, statement-node reasoning,
and multi-anchor joins are out of scope.
```

## Lower Priority / Future Work

These are good ideas but too large for the immediate rerun:

- full signed relation parser that asks the LLM to output `r+` / `r-` directly
- multi-anchor parser and join/intersection execution
- answer-type module
- schema retriever with relation aliases/descriptions/domain/range
- learned reranker or LambdaMART
- top-m path set instead of all-path union
- fine-grained role/direction-aware HRG rule signature
- gold-program/oracle-program KQAPro upper bound

Mention them as limitations/future work unless there is enough time to implement them properly.

## Recommended Execution Order

1. Run a dry-run with the current four-model matrix:

```bash
SAMPLE_LIMIT=5 RUN_PERTURBATION=0 ./run_everything_sequential.sh
```

2. If dry-run is clean, run full clean matrix:

```bash
RUN_PERTURBATION=0 ./run_everything_sequential.sh
```

3. Run perturbation matrix with 5 seeds:

```bash
RUN_PERTURBATION=1 KB_ABLATION_SEEDS="0 1 2 3 4" ./run_full_rerun.sh
```

4. Run grammar stability:

```bash
.venv/bin/python grammar_stability_analysis.py \
  --kb-path Datasets/MetaQA/kb.txt \
  --seeds 0 1 2 3 4 \
  --out-file artifacts_full/_summary/grammar_stability_metaqa.json
```

5. Add/run the small diagnostic scripts:

- MetaQA 3-hop BFS sanity
- symbolic endpoint baseline
- no/wrong/masked context controls
- grammar-hit breakdown table
- compact hyperparameter sensitivity

6. Regenerate summaries and thesis tables from one artifact root only.

## Final Claim After Revisions

Use this safer core claim:

```text
HRG-Proposed is an HRG-like structural-prior retrieval pipeline for single-anchor
relation-chain KGQA. On in-scope chain-friendly questions, it compresses final
evidence context while retaining competitive answer quality. On out-of-scope hard
datasets such as KQAPro, it does not solve the full QA task, but improves
executable evidence recovery over strict-spine ablations. The empirical driver is
mainly KG-valid fallback and grammar-aware candidate ranking, not LLM correction
or optional expansion alone.
```
