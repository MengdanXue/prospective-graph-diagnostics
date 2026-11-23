# Applied Intelligence rewrite: evidence and editorial contract

Base: publication/tmlr-submission, 6495a64fdbb45670b0ce6f66f33a89a7984d4175. The older local main is not the manuscript baseline. Existing uncommitted files in the TMLR checkout are preserved there.

Target: Applied Intelligence; empirical research article; English; nature-writing generic journal rules. Title retained. This is a rewrite of completed research, not authorization to submit or train.

Core argument: the measured value of a graph-versus-MLP diagnostic depends on the candidate portfolio and the complete decision policy, including fallback. The fixed benchmark quantifies that dependence; its sensitivity analyses define the observed boundaries.

## Terminology ledger

| Canonical term | Meaning / editorial decision |
|---|---|
| graph neural network (GNN) | Expand once; specific architecture names retained. |
| multilayer perceptron (MLP) | Feature-only model; four original trials. |
| graph portfolio | Validation-selected graph architecture and trial; six architectures, 24 trials in the full portfolio. |
| Combined | Historical rule before fallback; distinguish it from the complete rule-and-MLP-fallback policy. |
| oracle-portfolio regret | Test accuracy opportunity loss within the specified selected graph/MLP pair, in percentage points (pp). |
| selection accuracy | Agreement with the one-percentage-point target; distinct from prediction accuracy and positive regret. |
| coverage | Fraction of decisions before fallback that do not abstain. |
| frozen / pre-specified primary analysis | Protocol fixed before primary records; thresholds have an exploratory origin. Not whole-project preregistration. |
| post-hoc sensitivity | Reanalysis or separately recorded diagnostic run; not a new confirmatory result. |
| record reconstruction | Recomputing scores from stored outcomes. |
| training repeatability | Agreement between repeated training executions in a specified configuration. |

## Evidence allocation

| Result | Function | Destination |
|---|---|---|
| Nine policies, 7.46/0.26/0.22 regret | Primary evidence | Main results |
| Fallback decomposition, 75.876% | Conclusion-changing qualification | Main results |
| GCN/GAT reversals, 63 subsets, Cornell/Wisconsin influence | Core finding and boundary | Main sensitivity section |
| Fallback × dataset exclusion | Magnitude qualification | Main sensitivity section |
| LODO calibration and two-dataset preprocessing | Threshold/training qualification | Main sensitivity section |
| Exact Holm resolution bound | Necessary inferential boundary | Main results, compressed synthesis elsewhere |
| Execution amendments and private/public provenance | Traceability | Appendix with main pointer |
| Regret–coverage figure | Supporting description | Appendix with main pointer |
| Three-dataset edge intervention | Auxiliary evidence | Short main section, all statistics retained |
| Architecture, dataset, repeatability and fixed-degree details | Reconstruction / bounded context | Appendices |

No model outcomes, tables' numerical cells, decision formulas or bibliography entries are changed. New 11-dataset input results and MLP24/published-metric results are excluded because the archived acceptance status is incomplete. No full-benchmark robustness claim can be based on software tests.

## Paragraph and claim map

Introduction: decision problem → limitation of correlation-only interpretation → paired decision evaluation → observed portfolio/fallback sensitivity → three existing contributions.

Results: measured policy losses → decomposition → statistical resolution → portfolio changes → threshold and preprocessing boundaries.

Discussion: meaning for model selection → implications of fallback and available headroom → training and dataset limitations. Detailed numerical replay is removed, while limitations that affect the claim stay visible.

Conclusion: one bounded finding and an actionable reporting recommendation. No new method, universal negative claim, or promised acceptance.
