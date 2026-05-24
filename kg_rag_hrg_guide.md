# HRG-Guided Executable Retrieval for KG-RAG

## Introduction

Knowledge Graph Question Answering (KGQA) aims to answer natural language questions using structured knowledge graphs (KGs). Recently, Retrieval-Augmented Generation (RAG) combined with Large Language Models (LLMs) has demonstrated strong performance in knowledge-intensive tasks. However, directly applying RAG to KGQA introduces several challenges. Traditional entity-centered retrieval methods, such as Breadth-First Search (BFS), often retrieve excessively large subgraphs containing many irrelevant edges. This leads to large context windows, high token costs, noisy evidence, and unstable answer generation.

The problem becomes more severe in multi-hop reasoning settings, where the retrieval space grows rapidly with hop depth.

## Background

Knowledge Graph Question Answering (KGQA) aims to answer natural language questions using structured knowledge graphs (KGs). Recently, systems combining Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) have demonstrated strong capabilities in knowledge-intensive tasks. However, when RAG is directly applied to KGQA, several new problems arise.

Traditional entity-centered retrieval approaches, such as Breadth-First Search (BFS), typically expand a large number of neighboring nodes outward from the topic entity, resulting in oversized retrieval subgraphs containing many triples irrelevant to the question. This causes:

- Large context token usage
- Increased generation latency
- Increased retrieval noise
- Unstable answer generation

In multi-hop QA tasks, this issue becomes even more severe because the retrieval space grows rapidly as hop depth increases.

## Motivation

Most existing KG-RAG systems focus on maximizing retrieval coverage. However, in practical KGQA systems, retrieval correctness is often more important than retrieval size. A semantically plausible relation chain predicted by an LLM may still fail during actual graph traversal.

This motivates the following question:

> Can KG-RAG retrieval be guided by executable relation chains and lightweight structural priors to construct compact yet answer-preserving subgraphs?

At the same time, Hyperedge Replacement Grammar (HRG) provides a way to summarize recurring local graph structures. Instead of using HRG as a full graph derivation engine, this work investigates whether HRG can serve as a structural retrieval prior for:

- Correction
- Reranking
- Retrieval regularization
- Compact retrieval construction

## Application Scenarios

This framework can be used in:

- Multi-hop KGQA
- Compact KG-RAG systems
- Knowledge-intensive LLM retrieval
- Cross-lingual KGQA
- Low-token-budget retrieval systems

Potential applications include enterprise knowledge assistants, multilingual QA systems, domain-specific QA systems, and large-scale graph retrieval systems.

## Challenges

The main challenges are:

- BFS retrieval introduces noisy and oversized subgraphs
- LLM-predicted relation chains are frequently invalid on the KG
- Multi-hop retrieval requires both semantic and structural correctness
- Cross-lingual datasets introduce grounding ambiguity
- Exact HRG decoding is difficult to integrate directly into LLM-RAG pipelines

## Research Goals

This work aims to:

- Build compact executable subgraphs for KG-RAG
- Reduce context tokens while maintaining answer quality
- Use KG validation to filter invalid relation chains
- Use correction mechanisms to recover from failed LLM parsing
- Use HRG as a retrieval structural prior rather than exact graph decoding

## Approach Overview

### Offline Stage

1. Read KG triples
2. Construct graph representations
3. Extract HRG rules from sampled subgraphs
4. Build grammar statistics

### Online Stage

1. LLM predicts top-k `(entity, relation chain)` candidates
2. Entity grounding
3. KG executability validation
4. Correction if all chains fail
5. Strict spine retrieval
6. Grammar-aware reranking
7. Optional grammar-guided expansion
8. Subgraph serialization
9. LLM answer generation

## Contribution

The main contributions are:

- We propose an executable chain-guided retrieval framework for KG-RAG
- We design KG-validity verification for LLM-predicted relation chains
- We introduce correction mechanisms for recovering invalid chains
- We integrate HRG as a retrieval structural prior
- We perform HRG extraction parameter analysis
- We evaluate the framework on MetaQA, WikiMovies, MLPQ, and KQAPro

## Related Work

### Hyperedge Replacement Grammar

HRG methods summarize recurring graph structures through clique decomposition and grammar extraction. Traditional HRG methods are mainly designed for:

- Graph generation
- Graph compression

In contrast, this work uses HRG as:

- Correction guidance
- Reranking prior
- Retrieval regularization

rather than exact graph decoding.

### Complex KGQA Benchmarks

#### MetaQA

A clean movie-domain multi-hop KGQA benchmark.

#### WikiMovies

Contains KB normalization challenges and composite tails.

#### MLPQ

Evaluates multilingual grounding and chain executability.

#### KQAPro

A compositional reasoning benchmark with explicit program supervision.

## Problem Statement

Let the knowledge graph be defined as:

- $V$: entity set
- $R$: relation set
- $T \subseteq V \times R \times V$: KG triple set

Given a natural language question $q$, the goal is to retrieve a compact evidence subgraph $S_q \subseteq G$ and generate the answer set $A_q \subseteq V$.

The system first predicts:

- A topic entity $e_0 \in V$
- A relation chain $c = [r_1, r_2, \ldots, r_k]$

A chain is considered valid only if it can be executed on the KG from $e_0$.

The retrieval objective is to find a subgraph that is both:

- Compact
- Answer-preserving

## Objective

The ideal retrieval system should maximize answer correctness and evidence coverage while minimizing context size.

## Evaluation

The benchmark evaluates:

- Answer quality
- Retrieval quality
- Compression
- Latency
- Failure cases

Current metrics include:

- EM (Exact Match)
- Hits@1 / Hits@3 / Hits@5
- MRR
- Answer-set Precision / Recall / F1
- Retrieval Precision / Recall / F1
- Coverage
- Average context tokens
- Average subgraph size
- Compression ratio
- Latency
- Failure counts

## Methodology

### Model Overview

#### Offline

- HRG extraction from sampled KG subgraphs

#### Online

- LLM parsing
- Validation
- Correction
- Spine retrieval
- Grammar-aware retrieval refinement

### Offline HRG Extraction

The KG is first converted into a labeled MultiDiGraph. Then an undirected skeleton is constructed for clique-based processing.

To avoid hub explosion, the extractor samples subgraphs using capped BFS.

Each sampled graph is processed through:

1. Maximum cardinality search
2. Triangulation
3. Clique tree construction
4. HRG rule extraction

Duplicate rules are merged and their counts are accumulated.

An HRG rule represents relations that frequently co-occur within local graph structures.

### Online: LLM Top-k Chain Parsing

Given a question, the LLM predicts multiple candidate relation chains instead of only one.

### Entity Grounding

The predicted entity is mapped to a KG node using:

- Exact lookup
- Normalized lookup
- Token-overlap fallback

Example:

```text
Gregoire Colin
```

may be grounded to:

```text
Grégoire Colin
```

This is important for:

- Aliases
- Accents
- Composite names
- Multilingual surface forms

### Relation Fuzzy Matching

Predicted relation strings are normalized into valid KG relation tokens.

Example:

```text
directed by
```

and

```text
directed_by
```

are both aligned to:

```text
directed_by
```

### KG Chain Validation

Each candidate relation chain is executed on the KG.

If the frontier becomes empty at any hop, the chain is considered invalid.

This step ensures that the predicted relation chain is not only semantically plausible but also executable on the actual KG.

### Correction

Correction is only performed when all initial candidates are invalid.

The correction pool includes:

- Grammar fallback
- Direction flip interface, currently disabled because the function returns an empty list

The current Spine-Correction setting enables fallback correction but does not enable grammar rerank or grammar expansion.

However, the correction stage itself may still include grammar fallback and LLM correction.

### Grammar Fallback

When all initial chains fail validation, the framework performs grammar-guided correction.

The invalid relation chain is first normalized into a relation-label set, and the system searches HRG rules whose terminal relation labels overlap with the predicted chain.

These matched grammar rules are treated as structural priors for candidate reconstruction.

Example:

```json
["acted_in", "director"]
```

may first be normalized into:

```json
{"starred_actors", "directed_by"}
```

The framework then searches HRG rules containing similar relation co-occurrence patterns, such as:

```json
{"terminals": ["starred_actors", "directed_by", "has_genre"]}
```

New candidate chains are reconstructed from these matched grammar rules and re-validated on the KG.

The first executable corrected chain is selected for spine retrieval.

Importantly, the grammar fallback mechanism does not perform exact HRG derivation or attachment-preserving graph decoding.

Instead, it uses HRG relation co-occurrence statistics as a lightweight structural prior for retrieval stabilization and relation-chain correction.

### Grammar Matching

Grammar matching is not exact ordered HRG derivation.

The current implementation converts a relation chain into a relation-label set and checks whether it is included within the terminal labels of a grammar rule.

Example:

```python
chain = ["starred_actors", "directed_by"]
```

becomes:

```json
{"starred_actors", "directed_by"}
```

If a rule contains:

```json
{"starred_actors", "directed_by", "has_genre", "written_by"}
```

then it is considered a match.

Afterward, same-arity rules are preferred.

Arity can roughly be interpreted as hop count.

### Grammar-aware Expansion

After grammar matching, the framework optionally performs controlled grammar-aware expansion around the executable spine subgraph.

Instead of unrestricted BFS expansion, the expansion process is constrained by relation labels extracted from matched HRG rules.

Only relations appearing in grammar-matched terminal sets are allowed during expansion.

For example, if a matched grammar rule contains:

```json
{"starred_actors", "directed_by", "has_genre"}
```

then expansion is restricted to these relation types.

Expansion begins from the executable spine.

This design attempts to improve retrieval coverage while preserving structural compactness.

Importantly, the current implementation does not perform exact HRG decoding or attachment-preserving graph expansion.

Grammar rules are instead used as lightweight structural priors that constrain allowable expansion relations.

Current experiments suggest that grammar-aware expansion may improve retrieval coverage in some cases, but can also introduce larger contexts and additional retrieval noise.

### Spine Retrieval

Once a valid chain is found, the system retrieves strict spine edges by walking the chain on the KG.

The spine is not necessarily a single path.

If one hop reaches multiple nodes, all executable edges along the chain are included.

### Grammar-aware Reranking

Multiple valid subgraphs are ranked using:

- Valid chain
- Grammar hit
- Grammar score
- Same-arity match
- Compactness

The selected subgraph is serialized into JSON or triple format and passed to the LLM.

The prompt instructs the LLM to answer only using the provided context.

Example JSON format:

```json
[{"head": "Cast Away", "relation": "directed_by", "tail": "Robert Zemeckis"}]
```

Example triple format:

```text
Cast Away directed_by Robert Zemeckis
```

## Experiment

### Dataset Insights

| Dataset | Insight |
|---|---|
| MetaQA | Clean atomic triples; best for testing chain traversal |
| WikiMovies | Requires KB normalization due to composite tails |
| MLPQ | Tests multilingual grounding and chain executability |
| KQAPro | Hardest compositional reasoning setting |

### PreProcess

| Dataset | Process |
|---|---|
| MetaQA | Loaded from QA files and KB triples |
| WikiMovies | Composite tails are normalized into atomic triples |
| MLPQ | Only 2-hop and 3-hop questions are loaded |
| KQAPro | Uses normalized JSONL files |

### Evaluation Settings

| Group | Metrics |
|---|---|
| Answer Quality | EM, Hits@1/3/5, MRR, Answer-set Precision/Recall/F1 |
| Retrieval Quality | Retrieval Precision/Recall/F1, Coverage |
| Compression | Avg Context Tokens, Avg Subgraph Size, Compression Ratio |
| Efficiency | Avg Latency, Parse Latency, Retrieval Latency, Generation Latency |
| Failure Analysis | Failure Counts, Generation Failure Count, Answerable Rate |

### Baseline and Variants

| Method | Description |
|---|---|
| Baseline-BFS | Entity-centered BFS, no chain, no grammar |
| Spine-Only | LLM chain parsing + strict spine retrieval |
| Spine-Correction | Spine-Only + fallback correction |
| Grammar-Guided Spine Retrieval | Spine + correction + grammar rerank + grammar expansion |

### LLM Backbone

Current experiments use:

- Llama-3.1-8B-Instruct
- gemma-4-E4B-it
- Qwen2.5-7B-Instruct
- Qwen3.5-35B-A3B-FP8
- Gpt-Oss-20b

## Main Results

### Best Result on Each Dataset

| Dataset | Best Experiment | EM | Hits@1 | MRR | Ans-F1 | Avg Ctx | Gen Fail |
|---|---|---:|---:|---:|---:|---:|---:|
| MetaQA | Spine-Correction-llama3.1-json | 0.5200 | 0.6967 | 0.7558 | 0.7222 | 1638.2 | 0 |
| WikiMovies | Grammar-Guided Spine Retrieval-llama3.1-json | 0.2900 | 0.4900 | 0.4900 | 0.3763 | 46.1 | 0 |
| MLPQ | Baseline-BFS-gemma4 | 0.2750 | 0.2850 | 0.2917 | 0.2900 | 6363.2 | 40 |
| KQAPro | Grammar-Guided Spine Retrieval-llama3.1-triple | 0.1433 | 0.1600 | 0.1659 | 0.1601 | 3813.1 | 1 |

### Average Comparison

| Method | Avg EM | Avg Ans-F1 | Avg Ctx | Gen Fail |
|---|---:|---:|---:|---:|
| Baseline-BFS | 0.2286 | 0.2991 | 3591.7 | 57 |
| Spine-Only | 0.2151 | 0.2757 | 133.5 | 2 |
| Spine-Correction | 0.2192 | 0.2824 | 134.4 | 2 |
| Grammar-Guided Spine Retrieval | 0.2271 | 0.2989 | 2160.0 | 72 |

## Current Observations

### Observation 1

Spine-Correction is a very strong compact baseline.

It achieves competitive answer quality while using far smaller contexts than BFS.

This suggests that executable relation-chain retrieval is the key component.

### Observation 2

Grammar-Guided Spine Retrieval improves the overall tradeoff but increases context size.

HRG expansion improves recall but may also introduce additional retrieval noise.

### Observation 3

BFS remains strong but expensive.

Baseline-BFS still achieves strong average accuracy but uses the largest contexts and produces many generation failures.

### Observation 4

HRG is better framed as a retrieval prior.

Current evidence suggests that HRG is most useful for:

- Correction
- Reranking
- Controlled expansion

rather than exact graph decoding.

## HRG Extraction Parameter Analysis

| Parameter | Description |
|---|---|
| sample_size | BFS sampled subgraph size |
| k_samples | Number of sampled subgraphs |
| bfs_max_branch | BFS branch cap |
| seed_degree_quantile | Seed selection threshold |
| bfs_shuffle_neighbors | Neighbor order shuffle |

## Current HRG Observations

### Observation 1

`sample_size` dominates runtime and grammar size.

Increasing `sample_size` from 100 to 1000:

- Runtime increases from 0.94s to 122.38s
- Rule count increases from 47.0 to 643.3
- Compression drops from 99.60% to 93.94%

### Observation 2

`k_samples` scales grammar size approximately linearly.

Increasing `k_samples` from 1 to 8:

- Runtime increases from 3.94s to 27.81s
- Rule count increases from 86.7 to 637.7

### Observation 3

Smaller `bfs_max_branch` tends to increase grammar fragmentation.

### Observation 4

High `seed_degree_quantile` values are unstable.

Very large values such as 0.95 or 1.0 are more likely to sample hub nodes and increase runtime.

### Observation 5

`bfs_shuffle_neighbors` has relatively minor impact.

## Conclusions

This work investigates HRG-guided executable retrieval for KG-RAG.

The main finding is that compact KG retrieval should be built around executable relation chains rather than unguided BFS expansion.

The proposed framework combines:

- LLM parsing
- KG validation
- Correction
- Strict spine retrieval
- HRG-based structural priors

to construct compact evidence subgraphs.

Current results suggest:

- Spine-Correction is a strong compact retrieval method
- Grammar-Guided Spine Retrieval provides better recall-oriented retrieval but may increase context
- BFS remains competitive in accuracy but is expensive and noisy
- HRG is more suitable as a retrieval prior than as a full graph decoding engine

## Limitations

- Grammar matching is currently relation-label subset matching rather than exact ordered HRG derivation
- Grammar expansion uses rule label unions instead of attachment-preserving HRG decoding
- Current faithfulness and evidence metrics are approximate rather than fully supervised
- MLPQ remains difficult because of multilingual grounding
- KQAPro remains difficult because of compositional reasoning
- Grammar-Guided Spine Retrieval may increase context size compared with Spine-Correction
