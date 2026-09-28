# Bounded post-hoc diagnostic and MLP-budget extension

Design fixed on 2026-09-28 after inspecting partial input-robustness outcomes. The author approved the additional two metrics and the 24-trial MLP comparison on this date. This is a post-hoc extension, not a retroactive preregistration or an independently confirmed result. Approval does not increase the existing cumulative budget.

## Questions and retained scope

1. How do two published graph-label statistics perform as graph-versus-MLP decisions under a stated calibration rule?
2. How sensitive are those decisions and the existing policies to increasing the MLP candidate count from four to 24?

The eleven datasets, ten split seeds, two input treatments, six graph architectures, data versions, label boundary, practical margin, fallback, and original model implementations remain those of `configs/input_robustness_11_v2.json`. The original frozen benchmark and the four-trial input-robustness records remain separate immutable evidence. The latter run must finish and pass complete validation before the extension begins. No graph model is retrained for this extension. New MLP comparisons reuse graph records from the matching input treatment of that same input-robustness run, not older frozen graph records.

The machine-readable contract is `configs/diagnostics_mlp_budget_extension_v1.json`. No manuscript result or title is changed during preparation. All planned outcomes will be reported, including unfavorable or undefined-metric outcomes.

## Published metrics and adaptation

Source: Oleg Platonov, Denis Kuznedelev, Artem Babenko and Liudmila Prokhorenkova, *Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond*, NeurIPS 2023, [official paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/01b681025fdbda8e935a66cc5bb6e9de-Paper-Conference.pdf), equations (2) and (3), with the edge-sampled LI definition. Reuse bibliography key `platonov2023characterizing`; no unverified identifier is introduced.

Compute on the simple, undirected graph induced by training nodes. Drop loops and deduplicate edges in this diagnostic view only; never alter model inputs. Read labels of training nodes only. Let p_a be the class distribution at a uniformly sampled edge endpoint and p_ab the joint distribution of the two oriented endpoints.

- Adjusted homophily: (edge homophily - sum_a p_a^2) / (1 - sum_a p_a^2).
- Edge label informativeness: I(label_1; label_2) / H(label_1), using the same edge-endpoint distribution and the zero-count entropy convention.

No edges, or a zero denominator/zero entropy, yields an explicitly undefined statistic. It is not converted into an arbitrary finite score. Its decision abstains and uses the declared MLP fallback; the unit remains in the denominator of full-set regret and is excluded from coverage. No undefined result is dropped from the scope. Values and applicability are recorded with split and graph provenance. Because both input treatments retain identical training labels and edges, one metric computation per split is reused for both treatments.

These are published statistics with our stated decision calibration. They are not presented as reproductions of an original end-to-end published selector, or as representatives of feature-based and learned selection methods generally.

## Selection, calibration and losses

Graph architecture selection is exactly the original frozen rule: decreasing validation accuracy, then increasing model-name string. Validation loss breaks ties within a model's hyperparameter trials only; it does not break architecture-selection ties.

For the two new metrics, the policy is graph if score >= threshold, otherwise MLP. Evaluate all 63 nonempty graph portfolios in each input condition and MLP-budget stratum. Calibrate using leave-one-dataset-out: the held-out dataset contributes neither metric values nor outcomes to threshold fitting, threshold direction, portfolio selection, or method selection. Other datasets' selected-model test outcomes are explicitly meta-training targets; this is not a claim that no test outcome is used anywhere in calibration.

Candidate thresholds are midpoints between distinct defined training-fold scores, plus all-graph and all-MLP endpoints. Undefined metrics always abstain under every threshold, including endpoint policies. Optimize equal-weight mean training-dataset raw regret. Ties within the existing 1e-12 numerical comparison tolerance select the smallest threshold; endpoint order is all-graph, finite thresholds ascending, all-MLP. This new threshold family does not replace the original homophily rule's fixed threshold or existing calibration grid.

Report the inherited nine policies unchanged alongside the two calibrated metrics, full-set/covered-set regret, coverage, action counts, model accuracy, and graph-versus-MLP practical targets. Recompute the oracle and every affected policy after changing the MLP choice. Any optional all-test-selected threshold-family optimum is explicitly an optimistic post-hoc loss lower bound, not an estimate of deployable performance. Dataset-level differences remain descriptive; no new confirmatory superiority claim or replacement statistical method is introduced.

## MLP candidate budget

Keep hidden size 64, maximum 500 epochs, patience 100, the same model, initialization seed, splits, transformations, deterministic runtime, device and data bindings. The grid is learning rate {0.001, 0.005, 0.01} x dropout {0, 0.3, 0.5, 0.7} x weight decay {0, 0.0005}.

Preserve the original trial_000 through trial_003, their configurations, full checkpoints and validation histories. Verify their source execution identity, actual file and state hashes, and data/environment bindings before reuse. Enumerate remaining tuples by ascending learning rate, dropout, weight decay as trial_004 through trial_023. Every trial uses the original seed initialization; trial numbering does not change the seed.

There are 220 MLP units, 880 inherited trials, and 4,400 additional trials. Select from all 24 by decreasing validation accuracy, increasing validation loss, then trial id. Save every new checkpoint. Reload the selected real checkpoint and perform one fresh post-selection test evaluation, including when the inherited winner remains selected. Preserve the historical measurement separately as reused evidence. Never copy an old winner's test result onto a different winner, and never select from test scores. Previously observed results remain disclosed as post-hoc context.

The four-trial record schema and validator retain their exact-four requirements. The extension has its own declared record version and exact-24 requirements, not a weakened legacy check. Report two inputs x two MLP budgets. This aligns numbers of candidate training runs, not architecture diversity, wall time, FLOPs or effective search capacity.

## Execution, accounting and acceptance

Do not modify the checkout used by the running base experiment. Development and CPU-only isolated fixtures use a separate checkout and output root. Research dispatch must wait for the base process to exit and for complete scope, checkpoint verification and a trustworthy cumulative budget handoff.

All historical attempts and failures remain charged and preserved. Shared limits remain 384 h total, 380 h formal, 2 h resource and 2 h control/validation/analysis, with 8 h per dataset/condition/model unit and 16 h per dataset/seed/model pair across attempts. The extended MLP inherits the corresponding original unit and pair identities. Reusing historical training does not charge it twice; new preparation, training, saving, validation and failed attempts are charged. Software tests and CI waits retain their existing exclusion. Unused subbudgets do not transfer automatically.

A physical additional ledger segment is allowed only as an append-only continuation with an immutable, hash-bound handoff that carries every cumulative bucket and unit/pair total. It must not start as an independent fresh allowance. An unresolved attempt, unknown usage, missing terminal evidence or untrusted ledger tail blocks new training. Do not reinterpret a conservative unknown-duration bound as a measured overrun.

Reuse the existing outer process supervisor, task-scoped sleep prevention, suspend subscription and five-second monitor-gap constraint. Ordinary pause completes the model unit before stopping; emergency stop terminates its owned worker and retains the unfinished attempt. Preparation, saving, verification and final analysis remain inside supervised and charged phases.

Before research execution: validate actual CPU fixture training, original-four reuse, 20 added trials, selection, real save/reload, strict records, full analysis, ordinary pause, emergency stop and cumulative resume. Freeze and commit source, push, require the three unique CI jobs (lightweight-verification, full-protocol-verification, manuscript-build) to succeed in one run at the execution SHA, and archive evidence separately from execution source. A local test receipt is reported as local and cannot substitute for remote CI. Do not rerun the 48 H2GCN optimization candidates or change H2GCN's four-thread CPU and complete-checkpoint settings.

The approved new scope is recorded here; launch is still conditional on these technical gates and the unchanged budget. No acceptance outcome or future manuscript benefit is guaranteed.
