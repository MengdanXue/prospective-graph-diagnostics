# Route A Claim--Evidence Map

This map is the contract for claims in the active manuscript, whose entry point is `main_neurocomputing.tex`. A claim may appear in the Abstract as a result only when its status is **supported** and its evidence is available as an auditable proposition or artifact.

| Claim | Evidence | Status | Permitted wording |
|---|---|---|---|
| Fixed-degree neighbor averaging has ratio $\rho^2d/[1+(1-\rho^2)s]$ under the stated binary Gaussian conditional-neighbor model. | Proposition and derivation in `sections/06_fixed_degree_analysis.tex`. | Supported within assumptions | “Exact moment-based discriminability ratio under the stated fixed-degree model.” |
| The denominator is label-mixture covariance and the $\rho^2d$ expression is a noise-dominated approximation. | Conditional covariance derivation in `sections/06_fixed_degree_analysis.tex`. | Supported within assumptions | “The approximation can be inaccurate for strong features.” |
| The aggregate distribution generally remains a Gaussian mixture. | Conditional construction in the fixed-degree proposition. | Supported within assumptions | “The ratio is not a Bayes-error formula.” |
| A prescribed self-feature-plus-neighbor operator has the stated $\kappa_\alpha$ expression. | Proposition~`prop:self_neighbor_mixing` and `tests/test_self_feature_mixing.py`, including boundary reductions and a fixed-seed moment simulation. | Supported within assumptions | “Exact for the prescribed linear surrogate,” not “the exact behavior of GCN.” |
| The exact expression matches empirical moments across a frozen parameter grid. | 2,000 immutable records under `results/discriminability/route_a_grid_v1/` and the audited summary/figure; median absolute relative error 0.94% over nonzero exact ratios. | Supported for the prescribed surrogate grid | “Direct moment validation of the scoped surrogate,” not “validation of trained-GNN accuracy.” |
| The historical combined diagnostic improves model-selection regret over simple baselines under the prospective target. | Completed frozen benchmark in `results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json`: 11 datasets, 110 units, 770 model records, and 110 diagnostic records whose label-dependent graph statistics are training-only. Combined full-set regret is 7.46 pp versus 0.26 pp for always-graph; no comparison supports an advantage for the combined rule after Holm correction. | Rejected by prospective benchmark | “No stable incremental value under the frozen protocol,” not “all possible diagnostics fail.” |
| Two-hop recovery does not uniquely determine the best model. | Audited 10-seed paired results for six heterophilic datasets; failures on Chameleon and Squirrel. | Supported for the audited protocol | “Insufficient as a stand-alone selector,” not “never useful.” |
| The historical combined rule does not improve over trivial baselines under matched scoring. | `paper_release/results/selector_baseline_audit.json` in the audited release branch: historical rule and always-graph both select correctly on 5/6 datasets. | Supported for the audited six-dataset comparison | “No demonstrated incremental value in the audited comparison.” |
| $1-\mathrm{Acc}_{\mathrm{MLP}}$ bounds positive accuracy gain. | Arithmetic range of accuracy. | Supported but tautological | “Arithmetic positive-gain headroom,” never “information-theoretic predictor.” |
| Positive headroom bounds negative aggregation damage. | None; counterexamples exist. | Rejected | Must not appear. |
| Classification-error improvement is bounded by $I(Y;G\mid X)/\log C$. | None; the stated inequality is false. | Rejected | Delete the Structure Information Bound. |
| Efficiency defined as gain divided by headroom lies in $[-1,1]$. | None; negative values can have magnitude greater than one. | Rejected | Delete the range claim and decomposition. |
| The legacy edge shuffle is degree preserving and isolates topology causally. | Legacy source audit confirms discarded self-loops and collapsed duplicate edges. A separate replacement run in `results/diagnostic/route_a_degree_matched_v1/summary/summary.json` contains 30 paired records with verified per-node degree preservation. | Rejected for legacy results; replacement supported as a degree-preserving intervention | Report replacement results separately. They establish sensitivity to edge organization, not causal isolation of homophily. |
| The combined diagnostic attains 55.5% accuracy, 68.2% coverage, and 7.46 pp full-set regret. | `results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json`, method `historical_combined`. | Supported for the frozen prospective protocol | State coverage and fallback policy with the headline. |
| Always-graph attains 80.9% accuracy and 0.26 pp regret; validation selection attains 91.8% and 0.22 pp. | Same prospective audit, methods `always_graph` and `validation_selection`. | Supported for the frozen prospective protocol | Do not interpret raw differences as Holm-corrected significance. |
| Degree-preserving edge randomization reduces GCN accuracy by 45.5, 29.9, and 17.3 pp on Cora, CiteSeer, and PubMed. | `results/diagnostic/route_a_degree_matched_v1/summary/summary.json`; 10 paired seeds per dataset, complete edge lists and invariant checks. | Supported for these datasets and protocol | State that only three dataset clusters are available and overall exact sign-flip $p=0.25$. |
| Selector scores 32/36, 7/9, and 12/12 demonstrate predictive value. | Historical scoring gives ties/abstentions favorable treatment and omits matched trivial baselines. | Rejected as incremental evidence | May appear only in an explicitly retrospective audit table with matched baselines and caveats. |

## Abstract outline

1. **Problem:** cheap graph statistics computed from the training labels are often discussed as performance indicators, but their decision value for graph-vs-MLP selection is unclear.
2. **Protocol:** 11 datasets, 110 dataset--seed units, seven models, four trials, matched baselines, regret/coverage, and test-once accounting frozen before comparative outcomes.
3. **Negative result:** combined diagnostic 55.5%/7.46 pp regret versus always-graph 80.9%/0.26 pp and validation selection 91.8%/0.22 pp; no Holm-corrected advantage for the combined rule.
4. **Mechanism separation:** degree-preserving randomization causes large paired drops on three datasets but does not isolate a single structural property.

## Introduction reverse outline

1. Define the graph-family-versus-MLP-family decision.
2. Position metric-correlation studies and learned graph model selectors.
3. Explain why a diagnostic must be evaluated as an action using regret and coverage.
4. Specify the frozen prospective design and matched baselines.
5. Report the negative selector result without overclaiming corrected significance.
6. Use the degree-preserving intervention to separate diagnostic failure from graph irrelevance.
7. Retain the fixed-degree moment result as scoped mechanistic context rather than the paper's primary novelty.
