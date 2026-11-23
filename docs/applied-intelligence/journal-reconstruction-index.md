# Reconstructing the reported graph-versus-MLP decisions

This index accompanies `main_applied_intelligence.tex`. The supplement's
`VERSION.json`, `source-manifest.json`, and `SHA256SUMS.txt` bind its actual
manuscript, code, configurations, summaries and evidence. The source is a
working-tree snapshot; its baseline commit is recorded separately from the
executed commits inside the experimental records.

## Evidence organization

The `evidence/` directory holds three byte-identical original archives and one
explicitly derived repeatability archive:

| Archive | Evidence |
| --- | --- |
| `prospective-graph-formal-artifacts-v0.1.0.zip` | 770 selected-model records, 110 diagnostics and 30 paired edge-intervention records |
| `posthoc_diagnostic_controls_v1.zip` | 200 MLP diagnostic records, 280 preprocessing records, configurations and source snapshots |
| `posthoc_graph_parameterization_v1.zip` | 420 completed graph-diagnostic records; the separate incomplete 198-record attempt remains historical evidence |
| `training_repeatability_audit_journal_v1.zip` | 22 completed repeatability workers, histories, logs, initial/selected states, logits and source snapshots |

The first three ZIPs retain their original checksums. The fourth replaces only
local executable/library paths in 24 JSON files and updates the affected
completion/manifest hashes. `JOURNAL_DERIVATION.json` records the original
archive identity, every original and delivered file digest, permitted JSON
pointers, and canonical hashes with only those path fields removed. Tensors,
logs, histories, source snapshots and every scientific value remain unchanged.
The original compact summary continues to name its original full-summary hash;
the new verifier checks that binding through the derivation mapping rather
than pretending that the delivered summary has the old byte hash.

The original repeatability ZIP is retained by the author. Supplying it to the
verifier's optional `--original-repeatability` argument independently checks
all original byte mappings and exact metadata-only transformations. Without
that original, the package checks the published derivation attestation,
scientific hashes, all delivered bytes and internal bindings. It cannot
independently recover redacted path values.

Historical README files and source snapshots inside evidence archives retain
their original text. Instructions referring to `source.bundle`, the old
three-archive checker, or earlier manuscript entry points describe their
historical distributions. Use the commands below for this supplement.

## Minimal checks and appendix reconstruction

Run from `source/`, using Python 3.12+ and its standard library. These commands
require no raw dataset cache, GPU, model forward pass or training. Create a
new `tmp/rebuild/` directory first; outputs refuse overwriting.

```text
python -m unittest tests.test_reviewer_appendix_summary tests.test_journal_supplement -v
python scripts/verify_journal_supplement.py --evidence-dir ../evidence --report tmp/rebuild/evidence-verification.json
python scripts/summarize_reviewer_appendix.py --audit results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json --metadata-summary docs/applied-intelligence/provenance/reviewer_appendix_summary.pre-2026-09-28.json --output tmp/rebuild/reviewer_appendix_summary.json
python scripts/summarize_winning_architectures.py --output tmp/rebuild/winning_architectures.json
python scripts/summarize_fallback_sensitivity.py --output tmp/rebuild/fallback.json
```

The appendix rebuild recalculates all scientific fields from the retained
110-unit audit and requires exact agreement with the historical summary.
Dataset node/edge/class/feature counts are retained metadata, not independently
reloaded counts. The [provenance note](journal-provenance-note.md) preserves the
historical source-digest discrepancy and its unresolved origin.

Archive verification hashes every recorded entry, all 113 completion-bound
repeatability files, worker model-state/logit artifacts and historical source
bindings. It also reaggregates the 22 repeatability workers. These checks do
not reaggregate all 770 original model records.

## Mapping manuscript tables to inputs

Paths below are relative to `source/`. `A/` abbreviates
`results/diagnostic/route_a_prospective_v2/analysis/`. Scripts are in `scripts/`
unless explicitly prefixed with `experiments/`.

| Table label | Retained input / reconstruction |
| --- | --- |
| `tab:prospective_policy_results` | `A/diagnostic_audit.json`; `assemble_prospective_diagnostics.py` and `experiments/evaluate_diagnostics.py` with all 770 model and 110 diagnostic records |
| `tab:fallback_decomposition`, `tab:action_confusion`, `tab:threshold_headroom` | `A/diagnostic_audit.json`, `A/threshold_headroom.json`; `summarize_threshold_headroom.py` |
| `tab:equal_budget` | `A/equal_budget_sensitivity.json`; `summarize_equal_budget_sensitivity.py` |
| `tab:subset_sensitivity`, `tab:calibrated_threshold` | `A/portfolio_robustness.json`; `summarize_portfolio_robustness.py` (63 subsets and LODO calibration) |
| `tab:fallback_sensitivity`, `tab:preprocessing_fallback` | audit and preprocessing summaries; `summarize_fallback_sensitivity.py`; these result sets remain separate |
| `tab:preprocessing_sensitivity` | `A/preprocessing_sensitivity.json`; `summarize_preprocessing_sensitivity.py`, 280 paired records and manifest/completion marker |
| `tab:degree_intervention` | `results/diagnostic/route_a_degree_matched_v1/summary/summary.json`; `summarize_degree_matched_benchmark.py` and 30 paired records |
| `tab:dataset_properties`, `tab:dataset_decisions` | `A/reviewer_appendix_summary.json`; appendix reconstruction above |
| `tab:winning_architectures` | `A/winning_architectures.json`; `summarize_winning_architectures.py` |
| `tab:mlp_parameterization` | `results/diagnostic/posthoc_mlp_optimization_v1/analysis/mlp_optimization_diagnostic_summary.json`; `summarize_mlp_optimization_diagnostic.py` |
| `tab:graph_parameterization` | `results/diagnostic/posthoc_graph_parameterization_v1/analysis/graph_parameterization_diagnostic_summary.json`; `summarize_graph_parameterization_diagnostic.py` |
| `tab:training_reproducibility` | compact audit summary plus mapped full summary and 22 workers; `verify_journal_supplement.py` |
| `tab:frozen_training`, `tab:diagnostic_policies`, `tab:architecture_details` | `configs/prospective_benchmark_v2.json`, `experiments/prospective_models.py`, `experiments/prospective_data.py`, evaluator and runner |
| `tab:notation`, `tab:threshold` | `sections_applied/11_fixed_degree_analysis.tex`; analytical definitions |

Figures are rebuilt by `plot_prospective_regret_coverage.py`,
`plot_threshold_headroom.py` and `experiments/validate_discriminability_formula.py`
with their respective retained configurations and inputs.

## Reaggregation from original benchmark records

Install the pinned CPU dependencies in `requirements-ci.lock.txt`. After the
archive verifier passes, extract the formal ZIP into `tmp/formal/`, preserving
its `MANIFEST.json` and complete directory tree. Use new output names:

```text
python scripts/assemble_prospective_diagnostics.py --records-root tmp/formal/prospective --config configs/prospective_benchmark_v2.json --output tmp/rebuild/assembled.json
python experiments/evaluate_diagnostics.py --input tmp/rebuild/assembled.json --output tmp/rebuild/diagnostic_audit.json
python scripts/summarize_equal_budget_sensitivity.py --records-root tmp/formal/prospective/records --output tmp/rebuild/equal_budget.json
python scripts/summarize_portfolio_robustness.py --records-root tmp/formal/prospective/records --output tmp/rebuild/portfolio.json
python scripts/summarize_threshold_headroom.py --records-root tmp/formal/prospective/records --output tmp/rebuild/threshold_headroom.json
python scripts/summarize_degree_matched_benchmark.py --config configs/prospective_benchmark_v2.json --input-root tmp/formal/degree_matched --output tmp/rebuild/degree.json
python scripts/check_posthoc_release.py --records-root tmp/formal/prospective/records
```

The evaluator records runtime versions. The release checker compares the
complete scientific audit, keeps actions/counts exact, and permits only
`1e-12` absolute floating-point rounding drift. This is a deeper check than
archive hashing and was not performed when creating this journal package.

## Raw-data, source-history and training requirements

The original protocol, amendments, execution-source mapping and environment
locks are included under `docs/`, `configs/` and the root requirements files.
Raw datasets and all original benchmark checkpoints are not included.
Obtain `roman_empire.npz` and `amazon_ratings.npz` from the exact URLs and
checksums in `configs/preprocessing_sensitivity_v1.json` for feature checks.
Full MLP/graph summary regeneration additionally validates historical source
fingerprints. The retained source snapshots permit inspection; Git-dependent
validators require the matching repository history or author-retained Git
bundle, which is not included here. The old `check_revision_handoff.py` is
retained as historical code and does not accept the journal-derived archive.

Checkpoint-forward replay requires the corresponding state, inputs,
transforms, splits, model code and compatible environment. The repeatability
archive supplies states for its bounded workers, not for all 770 benchmark
models. Retraining is another procedure, with the frozen configurations,
historical runners and `requirements-experiments.lock.txt`. Reconstructed
scores do not imply identical training trajectories across machines.

## Manuscript build

The current entry is `main_applied_intelligence.tex`; its section files,
Springer class/style, bibliography and all referenced figures are included.
Create a new `tmp/pdf/` directory and use Tectonic 0.17.0:

```text
tectonic main_applied_intelligence.tex --outdir tmp/pdf --keep-logs --untrusted
python scripts/check_latex_log.py tmp/pdf/main_applied_intelligence.log
```

Initial TeX dependencies may require network access. No manuscript build or
supplement verification authorizes new training or certifies journal acceptance.
