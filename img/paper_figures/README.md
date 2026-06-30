# Paper Figures

| File | Use in thesis | Purpose |
|---|---|---|
| `13_offline_online_architecture.pdf` | Sections 1.1.2, Method overview | Shows the two-part thesis structure: offline HRG prior and online KGQA retrieval. |
| `14_offline_grammar_statistics.pdf` | Section 21.A | Shows extracted rule counts, unique patterns, and relation coverage. |
| `15_grammar_perturbation_trends.pdf` | Section 21.B | Shows grammar robustness under 10/20/30% node deletion. |
| `16_evidence_coverage_hard_cases.pdf` | Section 21.6 | Shows stress-test executable evidence recovery over strict-spine ablations on MLPQ/KQAPro. |
| `17_online_token_proxy.pdf` | Section 21.6.1 | Shows end-to-end online token proxy ratio, not just final context. |
| `18_bootstrap_hard_case_effects.pdf` | Section 21.6.2 | Shows paired bootstrap confidence intervals for HRG vs Spine-Correction. |
| `19_prior_pilot.pdf` | Section 21.6.3 | Shows candidate-level HRG score vs relation unigram prior pilot. |

Final-use checklist:

- Regenerate numeric figures from the same rerun summary used by the main tables.
- Caption every result figure with dataset, model set, sample size, and aggregation unit.
- Label perturbation figures as `single-model diagnostic` unless they are rerun and summarized across all active models.
- Use stress-test wording for KQAPro; do not describe low-F1 KQAPro results as a solved hard-case success.
- Keep all output as vector PDF/SVG with consistent color and legend conventions.
