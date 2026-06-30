# Professor Suggestion Experiment Plan

This plan is based on:

- `LLM_PROFESSOR_METHOD_FULL_EXPLANATION.md`
- `new_suggest.docx`
- current code in `LLM_inference_benchmark/benchmark.py`, `knowledgegraph_agent.py`, and `baseline.py`

## 0. Immediate Diagnosis

### Why `r+` / `r-` can appear even though there is no reverse relation label

Your KG does not need explicit reverse relation names such as `starred_actors_reverse` for direction to matter.

The current implementation stores and asks the parser for bare relation tokens only:

- parser prompt says: `Use only bare relation names; do not add direction suffixes.`
- `_make_direction_flip_candidates()` returns `[]`.
- there are no active direction-flip correction candidates.

However, `_neighbors_for_token(ent, rel)` traverses both:

- outgoing edges: `ent --rel--> tail`
- incoming edges: `head --rel--> ent`

So the current semantics is effectively:

```text
step(v, r) = {u | (v, r, u) in E} union {u | (u, r, v) in E}
```

That means there is no reverse relation token in the data, but the algorithm still allows inverse traversal of the same relation. The professor's `r+` / `r-` suggestion is a formal way to make this hidden traversal direction explicit:

```text
r+ : follow (v, r, u)
r- : follow (u, r, v)
```

Example:

```text
Question: who directed the films starred by [Linda Evans]?
Current chain: [starred_actors, directed_by]
Signed chain: [starred_actors-, directed_by+]
```

Starting from `Linda Evans`, the first hop must traverse the KG edge `Movie --starred_actors--> Linda Evans` backward. That is why `r-` is conceptually present even if the relation label is still only `starred_actors`.

### Why expansion still appears in method discussion

In the current run artifacts I sampled 5000 dumps from both `artifacts/` and `artifect_all/`; all had:

```text
expanded_edges = 0
```

So for these results, "no actual expanded evidence edges" is true.

But the current `HRG-Proposed` model spec still sets:

```text
use_grammar_expansion = True
```

The gates usually block it (`no_grammar_label_subset_match`, `grammar_score_below_min`, `single_hop_spine_floor`, etc.), so no edges are added. This must be stated carefully:

```text
The current reported results contain no expanded edges, but HRG-Proposed still has gated optional expansion enabled in the method configuration.
```

If you want the thesis to say "no expansion", create an explicit `HRG-Proposed-NoExpansion` spec and make that the main method. Keep `+Expansion` only as an ablation.

## 1. Highest Priority Code Alignment

These changes should happen before rerunning expensive experiments.

### 1.1 Add explicit signed relation support

Goal: remove ambiguity in relation direction.

Implement internally:

```text
{"rel": "starred_actors", "dir": "-"}
{"rel": "directed_by", "dir": "+"}
```

or compact strings:

```text
starred_actors-
directed_by+
```

Recommended implementation:

1. Parser may still output bare relations for compatibility.
2. Normalize each hop to a signed hop during validation.
3. If parser output has no direction, infer candidate signed variants during validation.
4. Store selected signed chain in dumps:
   - `selected_chain`: bare relation labels, for backward compatibility
   - `selected_signed_chain`: direction-aware chain
   - each evidence edge includes `traversal_direction`
5. Update prompts, case studies, and tables to show signed chains.

Run two variants:

```text
Bare-Bidirectional-Spine
Signed-Spine
```

This directly answers the professor's direction critique.

### 1.2 Separate `HRG-Proposed-NoExpansion` from `HRG-Proposed+Expansion`

Current `benchmark.py` has `Spine-GrammarExpansion`, `Spine-RandomExpansion`, and `Spine-FrequencyExpansion` commented out.

Add explicit specs:

```text
Spine-Only
Spine-Correction
Spine-Correction+KGValidFallback
Spine-Correction+KGValidFallback+RelationFrequencyPrior
Spine-Correction+KGValidFallback+HRGPrior-NoExpansion
Spine-Correction+KGValidFallback+HRGPrior+Expansion
```

Use the same:

- model backbone
- sample limit
- candidate count
- deterministic fallback top-k
- beam width
- branch limit
- per-entity cap
- serialization

This is the most important experiment because the professor explicitly said the current `HRG-Proposed` mixes too many improvements.

### 1.3 Disable dataset-specific budget overrides for ablation runs

Current code increases search budgets for MLPQ and KQAPro.

For the professor-facing ablation, add a flag like:

```text
--fixed-ablation-budget
```

When enabled, bypass dataset-specific overrides so all methods use the same candidate pool and search budget.

Keep the old tuned setting as a secondary "best effort" run, not the main ablation table.

## 2. Core Rerun Matrix

Run this first on one backbone and 200 samples per dataset. Recommended first backbone: `gpt-oss`, because it is already active in `MODEL_BACKBONES`.

Datasets:

```text
MetaQA test
WikiMovies wiki_entities test
MLPQ en-zh en ills test
KQAPro validation
```

Serializations:

```text
json
triples
```

Main methods:

```text
Baseline-BFS
Degree-Capped-BFS
Token-Budgeted-BFS
Spine-Only
Spine-Correction
Spine-Correction+KGValidFallback
Spine-Correction+KGValidFallback+RelationUnigram
Spine-Correction+KGValidFallback+RelationBigram
Spine-Correction+KGValidFallback+RelationTrigram
HRG-Proposed-NoExpansion
HRG-Proposed+Expansion
```

If runtime is too high, use this minimum defensible subset:

```text
Baseline-BFS
Token-Budgeted-BFS
Spine-Only
Spine-Correction
Spine-Correction+KGValidFallback
HRG-Proposed-NoExpansion
HRG-Proposed+Expansion
```

## 3. Fair BFS Baselines

The professor's concern is valid: full BFS versus highly pruned spine retrieval can look unfair.

Add two fair BFS variants.

### 3.1 Degree-Capped BFS

Current BFS already has:

```text
max_edges_per_hop = 500
max_frontier = 5000
```

Make these explicit experiment configs:

```text
Degree-Capped-BFS-50
Degree-Capped-BFS-100
Degree-Capped-BFS-200
Degree-Capped-BFS-500
```

Report answer quality and context tokens.

### 3.2 Token-Budgeted BFS

For each question, cap BFS context to approximately the same final context tokens as the corresponding HRG method.

Recommended budgets:

```text
Budget = avg HRG-Proposed-NoExpansion context tokens per dataset/method
Budget = per-question HRG context tokens, when available
```

Use a deterministic edge ordering:

```text
hop distance, entity degree, relation frequency, lexical overlap with question, stable tie-break
```

Report whether BFS under the same token budget can match HRG's answer F1 / evidence coverage.

## 4. HRG Contribution Ablation

This is the table the professor is asking for.

All rows must use the same initial candidate pool and same search budget.

Suggested rows:

| Row | Correction | KG-valid fallback | Relation prior | HRG prior | Expansion |
|---|---:|---:|---:|---:|---:|
| A | no | no | no | no | no |
| B | yes | no | no | no | no |
| C | yes | yes | no | no | no |
| D | yes | yes | unigram | no | no |
| E | yes | yes | bigram/trigram | no | no |
| F | yes | yes | no | yes | no |
| G | yes | yes | no | yes | yes |

Primary interpretation:

- B - A = correction contribution
- C - B = deterministic KG-valid fallback contribution
- D/E - C = simple relation prior contribution
- F - C = HRG prior contribution
- G - F = optional expansion contribution

Metrics:

```text
EM
Hits@1
Answer-set F1
retrieval recall / precision / F1
answer-in-spine
avg final context tokens
avg total online token proxy
failure counts
candidate validity rate
grammar hit rate
selected candidate source counts
```

## 5. Direction Experiments

Run after signed relation support is implemented.

Variants:

```text
Bare-Bidirectional
Signed-Inferred
Signed-Parser
Signed-Parser+Fallback
```

Definitions:

- `Bare-Bidirectional`: current behavior; each bare relation can traverse out or in.
- `Signed-Inferred`: parser gives bare relation, validator enumerates signed directions.
- `Signed-Parser`: parser must output direction.
- `Signed-Parser+Fallback`: parser output plus deterministic direction fallback.

Report:

```text
direction accuracy when gold path is available
valid chain rate
answer-in-spine
retrieval F1
answer F1
failure by hop
```

This directly resolves the `r+ / r-` issue instead of arguing about notation.

## 6. KG Perturbation Robustness

The professor asked for multiple random seeds and validity controls.

Run:

```text
drop_nodes: 10%, 20%, 30%
drop_relations: 10%, 20%, 30%
seeds: 0, 1, 2, 3, 4
```

For each run, report:

```text
mean +/- std
valid test question count
topic entity deleted count
gold answer unreachable count
generation failure count
OOM / runtime failure count
avg context tokens
answer F1
retrieval F1
grammar rule count
relation vocabulary retention
exact pattern retention
```

Important: do not interpret improved F1 after deletion as robustness unless gold reachability and effective test set size are reported. It may simply mean noisy edges were removed.

## 7. Multi-Objective Formulation Fix

Do not claim the online system optimizes answer quality `Q(q, S_q)`, because the answer is unknown online.

Use one of these two thesis-safe formulations:

### Option A: evaluation objective only

Say the multi-objective expression is used to evaluate trade-offs after the fact.

### Option B: actual surrogate score

Define the actual lexicographic ranking score from code:

```text
KG executability
LLM rerank score
relation n-gram score
same-arity HRG hit
ordered-path HRG hit
grammar hit
grammar score
matched rule count
low-information penalty
failure progress
step survival
final frontier size
candidate source priority
LLM confidence
```

Then report it as the retrieval surrogate, not as an answer-quality objective.

## 8. Figure And Terminology Cleanup

Use these terms consistently:

```text
KG
KG validation
final context tokens
total online token proxy
relation chain
signed relation chain
frontier
candidate set
search budget
```

Avoid mixing:

```text
KB validation / KG validation
input tokens / context tokens / evidence tokens
expand / expend
```

For figures:

- redraw as vector figures
- increase font size
- remove long text inside boxes
- put dataset, model, sample size, and aggregation in captions
- distinguish single-model diagnostics from multi-model averaged results

## 9. Recommended Execution Order

### Phase 1: one-week defensible rerun

1. Add `HRG-Proposed-NoExpansion` and explicit ablation specs.
2. Add fixed-budget mode to disable dataset-specific budget overrides.
3. Add token-budgeted BFS.
4. Rerun 4 datasets x 1 backbone x 200 samples.
5. Produce the HRG contribution ablation table.
6. Update thesis text to say current expansion produced zero expanded edges, if that remains true.

### Phase 2: direction cleanup

1. Add signed relation representation.
2. Rerun MetaQA and WikiMovies first.
3. Then rerun MLPQ and KQAPro.
4. Add direction examples and signed-chain evidence tables.

### Phase 3: robustness

1. Run perturbation with 5 seeds.
2. Add gold reachability / topic deletion checks.
3. Report mean +/- std.

### Phase 4: final paper polish

1. Replace ambiguous objective with evaluation objective or actual surrogate score.
2. Redraw figures.
3. Normalize terminology and symbols.
4. Add limitation paragraph: signed relation chains help relation direction but do not solve operator-heavy KQAPro semantics by themselves.

## 10. Thesis Claim After These Experiments

The safer final claim should be:

```text
HRG-Proposed is not just BFS with fewer tokens, and not just correction with a larger fallback.
Under a controlled candidate/search budget, HRG prior can be isolated from deterministic fallback and simple relation-frequency priors.
Signed relation chains make the retrieval direction explicit without adding artificial reverse relation labels to the KG.
On chain-friendly datasets, compact signed spines preserve answer quality with much smaller context.
On hard datasets, HRG mainly improves evidence coverage over strict spine, while remaining limited by grounding, relation canonicalization, and operator/qualifier semantics.
```

