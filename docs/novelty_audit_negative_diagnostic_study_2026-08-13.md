# Novelty Audit: Prospective Diagnostics for Graph-vs-MLP Selection

Date: 2026-08-13

Status: first-pass systematic novelty audit; suitable for claim control, not a claim of exhaustive coverage of all future publications. The search snapshot is 2026-08-13.

## Audited research question

Can cheap, train-label-only graph diagnostics decide whether a tuned graph-model family or a tuned feature-only MLP family should be used for node classification, without evaluating candidate models on the test set?

The audited contribution is not a new GNN architecture and not a new homophily metric. It is a frozen decision evaluation of several simple diagnostics, including abstention, against trivial and validation-based baselines, with regret as the main deployment-oriented loss. A degree-preserving edge intervention is used only to test whether the diagnostic failure can be interpreted as evidence that graph structure itself is useless.

## Search scope

Searches covered the following concepts and close variants:

- homophily and heterophily metrics predicting GNN performance;
- GNN superiority relative to MLP or graph-agnostic models;
- graph dataset characterization and graph necessity;
- graph learning model selection, evaluation-free selection, and meta-learning;
- prospective/frozen evaluation, regret, abstention, coverage, and trivial baselines;
- degree-preserving rewiring and graph-structure interventions.

Sources included NeurIPS, ICLR/OpenReview, ICML/PMLR, Springer, arXiv, OpenAlex, and Crossref-indexed metadata. Official proceedings or publisher pages were preferred for the closest papers.

## Closest-work matrix

| Work | What it already establishes | Direct overlap | Remaining distinction, if stated narrowly |
|---|---|---|---|
| Luan et al., *When Do Graph Neural Networks Help with Node Classification?* (NeurIPS 2023) | Proposes feature-based, hypothesis-testing metrics with thresholds intended to reveal the advantage/disadvantage of graph-aware models. | Directly occupies the “metric for when GNNs help” space. | It does not appear to evaluate a pre-frozen family-level decision rule using coverage, abstention, regret, always-graph, always-MLP, random, and validation-selection baselines under the present protocol. |
| Zheng et al., *What Is Missing For Graph Homophily?* (NeurIPS 2024) | Introduces Tri-Hom and compares 17 metrics on 31 real datasets. Its appendix explicitly correlates many metrics with GCN/GraphSAGE/GAT minus MLP performance gaps. | Directly studies whether graph statistics explain graph-aware versus MLP performance differences; also reports weak correlations for many simple metrics. | The present study treats diagnostics as frozen decisions rather than correlations and audits the cost of wrong actions via regret and coverage. It is a negative stress test, not a new superior metric. |
| Platonov et al., *Characterizing Graph Datasets for Node Classification* (NeurIPS 2023) | Shows limitations of common homophily measures and proposes label informativeness, which aligns better with GNN performance. | Establishes that global homophily is insufficient and that alternative dataset characteristics can align with performance. | The present work asks a narrower operational question: do specified cheap diagnostics make reliable graph-family versus MLP-family decisions under frozen scoring? |
| Wang et al., *Understanding Heterophily for Graph Neural Networks* (ICML 2024) | Under HSBM, connects separability gain to neighborhood-distribution distance, average degree, inconsistency, and multi-hop structure. | Strong theoretical overlap with degree/correlation/separability explanations; limits novelty of the scoped discriminability theory. | The current finite fixed-degree calculation may be retained only as a pedagogical scoped analysis, not the paper's primary novelty. |
| Luan et al., *Revisiting Heterophily For Graph Neural Networks* (NeurIPS 2022) | Critiques label-only homophily, develops post-aggregation similarity metrics, and proposes adaptive channel mixing. | Prior evidence that conventional homophily is incomplete and that aggregation/identity/diversification need adaptive treatment. | Supports motivation but prevents a broad “first to show homophily is insufficient” claim. |
| Luan et al., *Revealing the Pitfalls and Re-Evaluating the Advancement of Heterophilic Graph Learning* (arXiv 2024; ICANN 2026 journal reference) | Identifies evaluation pitfalls, tunes baselines on 27 datasets, categorizes heterophilic datasets, and quantitatively evaluates 11 homophily metrics on synthetic graphs. | Very close negative benchmark framing and evaluation-pitfall motivation. | The current study's defensible difference is real-dataset prospective family selection with frozen diagnostics, regret/coverage, trivial baselines, and retained negative outcomes. Do not claim the first negative benchmark of homophily metrics. |
| Park et al., *MetaGL* (ICLR 2023) | Introduces evaluation-free graph learning model selection using prior performance and meta-graph features. | Directly precedes “select graph models without evaluating them on the new graph.” | The current work is not an automatic model-selection method. It is a failure audit of transparent, cheap, train-only heuristics for a binary family decision. |
| Park et al., *GLEMOS* (NeurIPS 2023 Datasets & Benchmarks) | Benchmarks instantaneous graph learning model selection using 366 models on 457 graphs and multiple testbeds/selection algorithms. | Occupies graph model-selection benchmark novelty at a much larger scale. | The present 11-dataset study cannot claim the first or most comprehensive graph-model-selection benchmark. Its narrower value is controlled graph-vs-MLP family selection, explicit trivial/validation baselines, regret, coverage, and an auditable prospective negative result. |
| Katsman et al., *Revisiting Graph Learning Benchmarks: When is the Graph Actually Necessary?* (Neural Processing Letters 2026) | Shows that tuned feature-only MLPs nearly solve several canonical graph benchmarks and analyzes feature leakage of graph information. | Directly precedes the “does the graph add utility beyond features?” question. | The present work does not merely compare tuned MLP/GNN performance; it evaluates whether cheap diagnostics can decide between the families and finds that they cannot. |
| Loveland et al., *On Performance Discrepancies Across Local Homophily Levels* (LoG 2024) | Shows local/global homophily mismatch and node-level GNN performance disparities, including GNN-minus-MLP analyses. | Prior evidence that homophily-performance relationships are heterogeneous and local. | Different estimand: node/subgroup discrepancy versus dataset/seed-level family-selection decisions. |
| Bechler-Speicher et al., *Graph Neural Networks Use Graphs When They Shouldn't* | Compares the same architecture with real versus empty graphs and studies graph-structure overfitting. | Establishes that providing graph structure can hurt even when a model could ignore it. | The present degree-preserving intervention asks whether useful signal depends on edge organization beyond degree; it does not establish a new general fact that graph structure can help or hurt. |

## Findings

### Claims ruled out by prior work

The following claims are not novel and must not appear:

1. “We are the first to ask when GNNs help over MLPs.”
2. “We are the first to propose a diagnostic of GNN usefulness.”
3. “We are the first to show that homophily is insufficient.”
4. “We are the first graph model-selection benchmark.”
5. “We are the first negative evaluation of heterophily metrics.”
6. “We are the first to compare structural metrics with GNN--MLP performance gaps.”
7. “We establish that graph structure matters even when simple homophily fails.”

These themes are already covered, in different forms, by NeurIPS 2022--2024 work, MetaGL/GLEMOS, graph-necessity benchmarks, and graph-structure ablations.

### Potentially defensible novelty

The remaining contribution is methodological and evidentiary rather than conceptual:

1. **Decision rather than correlation.** The study converts proposed diagnostics into explicit graph/MLP/abstain actions and measures the performance cost of those actions, rather than reporting only metric--accuracy correlation.
2. **Frozen prospective failure audit.** Dataset scope, seeds, trial budgets, diagnostic label scope, tie policy, practical margin, selection rules, bootstrap/permutation settings, and test-once access were frozen before comparative outcomes. Operational amendments were documented without changing the scientific estimand.
3. **Matched trivial baselines.** Always-graph, always-MLP, random 50/50, homophily-only, degree-only, homophily-plus-degree, two-hop-only, and validation-set direct selection are scored on identical dataset/seed units.
4. **Deployment-oriented loss.** The primary interpretation uses oracle-family regret and, for abstaining rules, coverage/selective accuracy rather than raw correlation alone.
5. **Mechanism/decision separation.** A degree-preserving intervention shows that the diagnostic failure cannot be simplified to “the graph is useless,” while explicitly acknowledging that the intervention jointly changes multiple degree-independent structural properties.

Even these points should be stated as “to our knowledge, this study provides...” only after the related-work section cites the closest studies and precisely defines the protocol differences. “First” is unnecessary and risky.

## Evidence-aligned contribution statement

A defensible contribution paragraph is:

> We provide a pre-specified, frozen evaluation of whether transparent train-only diagnostics can support a graph-model-versus-MLP family decision in node classification. Unlike prior work that proposes homophily-related metrics, correlates graph characteristics with model performance, or learns cross-dataset model selectors, we evaluate fixed diagnostic actions against matched trivial and validation-based baselines using coverage and oracle-family regret. The diagnostics show no stable incremental value under the frozen protocol. A separately audited degree-preserving intervention provides evidence that this failure should not be interpreted as evidence that edge organization is uninformative.

This wording does not claim that the research question, metrics, negative finding, or graph-model-selection problem is new.

## Required positioning in the rewritten paper

The related-work section should have four explicit subsections:

1. **Graph characteristics and GNN performance:** adjusted homophily, label informativeness, node distinguishability/PBE, Tri-Hom, HSBM, local homophily.
2. **Graph learning model selection:** MetaGL and GLEMOS; distinguish learned cross-dataset selection from a transparent diagnostic stress test.
3. **When graph structure is necessary:** tuned MLP baselines, empty-graph controls, and graph-necessity benchmark work.
4. **Evaluation methodology:** frozen protocols, equal trial budgets, validation selection, trivial policies, regret, and abstention/coverage.

The introduction must lead with the decision-evaluation gap, not with Information Budget, a new threshold, or the general claim that homophily is unreliable.

## Remaining weaknesses after repositioning

- Eleven datasets are far smaller than GLEMOS and Tri-Hom's dataset coverage. Scale cannot be the novelty claim.
- The candidate portfolio contains seven models and only four trials per model. This is controlled but not exhaustive.
- Graph is the oracle family on 89/110 units; selection accuracy is base-rate sensitive, so regret and dataset-level results must remain primary.
- The degree intervention has only three dataset clusters. All seed differences are negative, but the dataset-cluster sign-flip test has minimum attainable two-sided `p=0.25` for the observed direction pattern.
- The edge intervention is not a homophily-only causal intervention: while preserving degree, it jointly changes several higher-order and class-conditional properties. It supports an edge-organization interpretation, not identification of a single mechanism.
- The diagnostics evaluated are deliberately simple. The paper can reject their operational reliability under the protocol, not all possible diagnostics or learned selectors.
- Tri-Hom, PBE/Jeffreys metrics, label informativeness, and MetaGL/GLEMOS are related work, not currently matched baselines. Reviewers may request direct inclusion or a careful explanation of infeasibility and scope.

## Go/No-Go after novelty audit

**Conditional go for a narrow negative diagnostic study.** The study remains publishable in principle only if it is framed as a transparent prospective stress test of simple diagnostics, not as a new graph-utility metric or a new model-selection benchmark.

The strongest reviewer objection will be: “Prior work already showed weak homophily--performance alignment and already benchmarked graph model selection at much larger scale.” The response must be the decision-theoretic protocol difference: fixed actions, train-only inputs, matched trivial/validation policies, regret/coverage, prospective freezing, and a negative result retained without metric redesign.

If the target venue does not consider this protocol-level distinction sufficiently novel, the correct fallback is a technical report or empirical note, not restoration of the rejected theory story.

## Core verified sources

- [Luan et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5ba11de4c74548071899cf41dec078bf-Abstract-Conference.html)
- [Zheng et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html)
- [Platonov et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/01b681025fdbda8e935a66cc5bb6e9de-Abstract-Conference.html)
- [Wang et al., ICML 2024](https://proceedings.mlr.press/v235/wang24u.html)
- [Park et al., MetaGL, ICLR 2023](https://openreview.net/forum?id=C1ns08q9jZ)
- [Park et al., GLEMOS, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html)
- [Katsman et al., Neural Processing Letters 2026](https://link.springer.com/article/10.1007/s11063-026-11833-6)
- [Loveland et al., LoG 2024](https://proceedings.mlr.press/v231/loveland24a.html)
- [Luan et al., heterophily evaluation audit](https://arxiv.org/abs/2409.05755)

## Confirmatory versus exploratory status

- **Confirmatory evidence:** the frozen prospective diagnostic benchmark and the separately frozen degree-matched intervention, including their registered units, seeds, models, decision rules, and inferential outputs.
- **Exploratory synthesis:** this novelty audit, venue-positioning judgment, and the proposed negative-study narrative. These were produced after observing the formal outcomes and must not be described as preregistered conclusions.
