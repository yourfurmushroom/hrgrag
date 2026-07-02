# Thesis Support Figures

Generated SVG figures for the HRG-guided KG-RAG thesis.

Recommended use:

1. `01_method_pipeline.svg` - method overview.
2. `02_example_bfs_vs_spine_linda_evans.svg` - show exactly how BFS vs HRG-Proposed retrieves evidence.
3. `03_explainability_output_linda_evans.svg` - show the printable explanation fields.
4. `04_metaqa_token_subgraph_compression.svg` - gpt-oss legacy 200-per-hop token and edge compression.
5. `05_metaqa_quality_vs_tokens.svg` - gpt-oss legacy 200-per-hop answer quality with fewer context tokens.
6. `06_dataset_takeaway_matrix.svg` - which datasets support the claim and which are limitations.
7. `07_metaqa_hop_analysis.svg` - why the 3-hop result matters.
8. `08_hrg_grammar_extraction.svg` - paper-style figure for offline grammar learning.
9. `09_chain_validation_algorithm.svg` - paper-style figure for executable chain validation.
10. `10_failure_counts_spine_correction.svg` - failure analysis across datasets.
11. `11_dataset_semantics_examples.svg` - why datasets differ.
12. `12_evaluation_design.svg` - evaluation is quality + efficiency + HRG-supported evidence.
20. `20_bfs_vs_hrg_process.svg` - canonical BFS KG-RAG compared with HRG-guided executable retrieval.
21. `21_mcs_triangulation_clique_tree.svg` - MCS, triangulation, clique tree, and HRG rule extraction intuition.
22. `22_fallback_sources_examples.svg` - failure forms and fallback recovery sources.

Notes:

- Figures 01-12 are thesis-support figures for method explanation, dataset diagnosis, and evaluation design.
- Numeric figures must state their source. Figures 04 and 05 use the legacy gpt-oss 200-per-hop MetaQA artifact and match the gpt-oss column in the method document.
