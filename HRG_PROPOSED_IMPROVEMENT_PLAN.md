# HRG-Proposed Improvement Plan

## Purpose

This note records the recommended changes for improving `HRG-Proposed` across the currently runnable datasets:

1. `MetaQA`
2. `WikiMovies`
3. `MLPQ`
4. `KQAPro`

The main conclusion is that weak non-MetaQA results should not be treated as one shared QA dataloader bug. The current pipeline already normalizes multiple dataset formats into a shared triple-based adjacency representation, but that is not the same as preserving each dataset's question semantics in an HRG-compatible reasoning graph.

The recommended direction is:

1. Fix dataset-to-graph semantic mismatches that prevent fair HRG retrieval.
2. Add audits that separate data adaptation failures from method failures.
3. Change `HRG-Proposed` from a mostly strict-chain method into an adaptive HRG-guided retrieval method with controlled backoff.

## Implemented First Slice

This pass implements the low-risk adapter and audit work that should precede broader HRG tuning:

1. `audit_dataset_reachability.py` reports current KB answer reachability for `MetaQA`, `WikiMovies`, `MLPQ`, and KQAPro program families.
2. WikiMovies default KB resolution now prefers the atomicized `wiki_entities_kb_normalized.txt` graph when it exists, and dataset download ensures that normalized file is generated from the raw wiki-entities KB.
3. KQAPro normalized QA rows now expose program functions, query-family buckets, and graph-depth estimates that account for statement reification in qualifier questions.
4. KQAPro KB conversion emits predicate-specific `::statement` anchor edges for qualifier-bearing facts.
5. HRG retrieval validity automatically tries a statement-anchor chain when a parsed KQAPro chain uses an ordinary fact predicate followed by a qualifier predicate.

This is not full KQAPro operator support or full MLPQ semantic canonicalization. Those remain separate changes because they alter the reasoning abstraction more broadly.

---

## Current Diagnosis

## What Already Works

The current preprocessing and loaders already provide:

1. A shared triple adjacency representation through KB parsing and adjacency loading.
2. Entity token normalization, lookup normalization, and alias indexing.
3. WikiMovies-specific relation parsing and composite-tail alias support.
4. KQAPro QA normalization and KQAPro KB serialization into TSV triples.

This is enough to run all four datasets through the same benchmark pipeline.

## What Is Still Missing

The current representation does not fully normalize dataset-specific semantic structures into graph abstractions that match the HRG relation-chain assumption.

| Dataset | Current HRG Compatibility | Main Remaining Issue |
|---|---|---|
| `MetaQA` | High | HRG must avoid regressions versus simpler spine/correction variants |
| `WikiMovies` | Partial | Composite KB values and non-atomic tails remain a graph representation risk |
| `MLPQ` | Partial | Multilingual relation and entity surface variants weaken grammar and chain matching |
| `KQAPro` | Partial to low | Statement qualifiers and operator-heavy questions do not map cleanly to ordinary relation chains |

---

## Evidence From Current Artifacts

The current artifacts show different failure patterns rather than one shared loader failure.

## MetaQA

`MetaQA` is the cleanest control case because its questions and KB largely match the relation-chain abstraction. Current `HRG-Proposed` runs are competitive, but they are not consistently better than `Spine-Correction`. Some runs also show runtime or memory failures that should not be present in the proposed method.

## WikiMovies

`WikiMovies` already shows cases where `HRG-Proposed` can lead, especially with `llama3.1`, but it is a one-hop benchmark in the current pipeline. Its main weakness is not multi-hop grammar inference. It is the graph representation of entities and relation tails.

## MLPQ

`MLPQ` shows the clearest coverage gap:

1. BFS retrieval has broader recall.
2. Spine methods lose many valid chains.
3. `HRG-Proposed` improves over the narrow spine variants, but it still often trails BFS.

This suggests that HRG needs a controlled way to widen retrieval when strict predicted chains are under-covered.

## KQAPro

`KQAPro` has a concrete semantic mismatch. A question such as the first validation item asks for a qualifier value attached to a relation fact. The normalized QA record is correct, but the current TSV graph encodes qualifier information through statement nodes:

1. `fact_h`
2. `fact_r`
3. `fact_t`
4. qualifier relations such as `statement is subject of`

That encoding is still a triple graph, but it is not equivalent to a simple one-hop or two-hop entity relation chain. The current KQAPro hop bucket is also estimated from program length, which can disagree with the executable depth of the converted graph.

---

## Recommended Scope

## Do Not Start With Full Semantic Unification

A full semantic graph abstraction for all KQAPro operator types is high cost. It would require support for statement qualifiers, attributes, counts, comparisons, verification questions, and other operator semantics. That is closer to building a query intermediate representation than to adding a small dataset adapter.

The recommended implementation scope is a smaller semantics-preserving adapter layer:

1. Repair clear graph representation mismatches.
2. Measure gold reachability after every adapter.
3. Extend HRG retrieval only where the graph abstraction is well-defined.
4. Explicitly separate unsupported or operator-heavy cases when needed.

---

## Priority Changes

## P0: Add Gold Reachability Audits

Add an audit before major algorithm tuning. The audit should report, per dataset and hop bucket:

1. Whether the gold answer appears in the current KB representation.
2. Whether the gold topic entity resolves to a KB node.
3. Whether a gold or oracle path can reach the answer.
4. Whether the answer is reachable within the benchmark hop/depth budget.
5. Which failures are caused by entity lookup, graph conversion, relation mismatch, or unsupported query operators.

Recommended dataset-specific checks:

1. `MetaQA`: verify direct answer reachability for sampled one-hop to three-hop records.
2. `WikiMovies`: report answers stored as atomic nodes versus composite tail values.
3. `MLPQ`: execute the loader-preserved `gold_path_parts` against the current KB.
4. `KQAPro`: bucket questions by program/operator family and test graph reachability separately for ordinary relation paths and qualifier-driven questions.

This audit is the main guardrail against tuning HRG on graph representations that cannot answer the question correctly.

---

## P1: Fix KQAPro Graph Semantics

KQAPro is the highest-priority semantic adapter because the current graph conversion can change the effective traversal shape.

## Required Changes

1. Stop relying only on the current rough program-length hop estimate for KQAPro.
2. Add KQAPro query-type buckets, at minimum:
   - ordinary relation path questions
   - attribute questions
   - qualifier / statement questions
   - count / comparison / verification questions
3. Add an executable representation for common qualifier cases.

## Recommended Qualifier Strategy

Prefer one of these strategies:

### Option A: Virtual Qualifier Relations

Collapse common statement qualifier patterns into semantics-preserving virtual edges that HRG can execute.

Example goal:

```text
entity --relation.qualifier--> answer
```

This keeps HRG relation-chain retrieval usable, but the generated virtual relation vocabulary must be documented and audited.

### Option B: Statement-Aware Traversal

Teach retrieval and chain validity checking that KQAPro statement nodes represent relation facts with qualifiers, not ordinary entity nodes.

This preserves KQAPro structure more directly, but it is a larger change to the execution model.

## Recommended First Step

Implement the smaller auditable subset first:

1. Ordinary relation-path KQAPro questions.
2. Common qualifier question patterns.
3. Separate reporting for unsupported operator-heavy cases until the execution model supports them.

---

## P2: Atomicize WikiMovies KB Values

WikiMovies should not rely only on alias recovery for composite relation tails.

## Recommended Changes

Split safe composite-tail relations into atomic triples during KB normalization. Candidate relations include:

1. `starred_actors`
2. `directed_by`
3. `written_by`
4. `has_tags`

Do not blindly split text-like fields such as plots.

## Intended Effect

Convert graph patterns like:

```text
Movie --starred_actors--> "Actor A, Actor B, Actor C"
```

into:

```text
Movie --starred_actors--> Actor A
Movie --starred_actors--> Actor B
Movie --starred_actors--> Actor C
```

This should improve:

1. Reverse traversal.
2. Entity lookup consistency.
3. Grammar extraction over canonical relation edges.
4. Comparability with MetaQA-style movie relations.

---

## P3: Canonicalize MLPQ Relation Families

MLPQ should keep its multilingual content, but HRG should not have to treat relation surface variants as unrelated structures when they represent the same reasoning role.

## Recommended Changes

1. Add a relation canonicalization layer for high-frequency relation families.
2. Keep the original relation label for debugging and serialization if needed.
3. Extract grammar on canonical relation families.
4. Execute chains using canonical-to-raw relation expansion or canonicalized adjacency.

Start from failures revealed by the MLPQ gold-path audit instead of trying to map every relation in the corpus up front.

## Intended Effect

Reduce fragmentation across relation variants such as namespace, language, or surface-form differences. This should improve:

1. Grammar support counts.
2. Candidate chain matching.
3. Fallback search ranking.
4. Cross-lingual retrieval coverage.

---

## P4: Make HRG-Proposed Adaptive

The current proposed method is still too close to:

```text
predicted strict chain
plus grammar reranking
plus grammar expansion
plus fallback only after hard failure
```

That is too narrow for datasets where a chain is executable but under-covers the answer.

## Recommended Retrieval Policy

For each question, build and score multiple retrieval candidates:

1. Strict spine context.
2. HRG-expanded spine context.
3. HRG-guided constrained BFS context.
4. Bounded BFS backoff context when HRG confidence is low.

The method should widen retrieval under low-confidence signals such as:

1. No candidate entity or chain.
2. No valid chain.
3. Weak grammar match.
4. No same-arity grammar match.
5. Suspiciously tiny subgraph for a nontrivial question.
6. Excessive chain ambiguity after reranking.

The backoff should remain controlled by HRG or explicit risk policy. It should not silently turn the proposed method into plain BFS.

## Why This Matters

1. `MLPQ` needs more recall than strict spine retrieval currently provides.
2. `KQAPro` needs retrieval policies that recognize graph semantics outside a clean narrow chain.
3. `MetaQA` needs a guard that prevents grammar expansion from hurting a simpler successful chain.

---

## P5: Add MetaQA Regression Guards

`MetaQA` is the best aligned benchmark. The proposed method should not lose to simpler variants there because of avoidable expansion or failure behavior.

## Recommended Guards

1. Preserve a correction-only candidate alongside the HRG-expanded candidate.
2. Skip grammar expansion when the grammar signal is weak.
3. Cap expansion context size before generation.
4. Keep a fallback answer path if HRG expansion raises runtime or memory failures.
5. Treat runtime errors and OOMs as benchmark regressions, not expected proposed-method behavior.

---

## Implementation Order

Use this order to keep the work measurable:

1. Add gold reachability audits.
2. Add WikiMovies atomic normalization and rerun the audit.
3. Add KQAPro query buckets and qualifier reachability reporting.
4. Implement the first KQAPro qualifier adapter or statement-aware traversal subset.
5. Add MLPQ high-frequency relation canonicalization based on audit failures.
6. Implement adaptive HRG retrieval with controlled backoff.
7. Add MetaQA regression guards and rerun the full benchmark suite.

---

## Evaluation Plan

## Required Reports

For every major change, report:

1. Answer metrics:
   - `EM`
   - `Hits@1`
   - `Answer-Set F1`
2. Retrieval metrics:
   - gold answer reachability
   - retrieval recall
   - retrieval precision
   - retrieval F1
3. Failure counts:
   - `no_candidates`
   - `no_valid_chain`
   - `retrieval_empty`
   - `runtime_error`
   - `oom`
4. Efficiency:
   - average subgraph size
   - context token count
   - latency

## Required Ablations

Keep ablations that separate:

1. Spine-only retrieval.
2. Correction fallback.
3. HRG reranking only.
4. HRG expansion.
5. HRG-guided constrained BFS.
6. Adaptive backoff.
7. Dataset semantic adapters.

Without these ablations it will be hard to show whether gains come from HRG structure, larger search, or dataset repair.

---

## Research Framing

The recommended claim is not:

`All datasets should be forced into the same plain triple graph.`

The stronger and safer claim is:

`HRG retrieval should operate on a semantics-preserving reasoning graph. For datasets whose raw KB encodings contain composite values, multilingual relation variants, or reified statement qualifiers, dataset adapters and reachability audits are needed before relation-chain grammar results are interpreted.`

This framing keeps the method claim honest while explaining why `MetaQA`, `WikiMovies`, `MLPQ`, and `KQAPro` do not stress the same part of the pipeline.

---

## Expected Outcome

The expected result of this plan is not only higher `HRG-Proposed` scores. It should also produce clearer diagnostics:

1. If the gold answer is unreachable after graph conversion, the issue is data adaptation.
2. If the gold answer is reachable but predicted chains fail, the issue is entity grounding, semantic parsing, or ranking.
3. If retrieval contains the answer but generation fails, the issue is answer extraction or generation behavior.

That separation is necessary before tuning `HRG-Proposed` for best-in-suite results across all current datasets.
