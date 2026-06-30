# Thesis Support Figures

Generated SVG figures for the HRG-guided KG-RAG thesis.

Recommended use:

1. `01_method_pipeline.svg` - method overview.
2. `02_example_bfs_vs_spine_linda_evans.svg` - show exactly how BFS vs HRG-Proposed retrieves evidence.
3. `03_explainability_output_linda_evans.svg` - show the printable explanation fields.
4. `04_metaqa_token_subgraph_compression.svg` - token and edge compression.
5. `05_metaqa_quality_vs_tokens.svg` - comparable answer quality with fewer tokens.
6. `06_dataset_takeaway_matrix.svg` - which datasets support the claim and which are limitations.
7. `07_metaqa_hop_analysis.svg` - why the 3-hop result matters.
8. `08_hrg_grammar_extraction.svg` - paper-style figure for offline grammar learning.
9. `09_chain_validation_algorithm.svg` - paper-style figure for executable chain validation.
10. `10_failure_counts_spine_correction.svg` - failure analysis across datasets.
11. `11_dataset_semantics_examples.svg` - why datasets differ.
12. `12_evaluation_design.svg` - evaluation is quality + efficiency + HRG-supported evidence.

Notes:

- Figures 01-12 are thesis-support figures for method explanation, dataset diagnosis, and evaluation design.
- Some numeric figures use single-model diagnostic settings; the main thesis table remains the four-model average in the method document.
- Before final thesis use, numeric figures must be regenerated from one consistent summary dump.
- Captions must state dataset, model or model set, sample size, and aggregation unit (`single-model diagnostic`, `four-model average`, `question-level n`, or `question-model-pair-level n`).
- Keep figures vector-based, use larger fonts, reduce long text inside boxes, and keep colors/legends consistent across result figures.
- KQAPro figures should be phrased as stress-test evidence recovery, not as a solved QA benchmark.
