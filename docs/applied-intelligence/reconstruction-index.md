# Applied Intelligence reconstruction index — 2026-09-28

The versioned local research companion is
`applied-intelligence-research-companion-2026-09-28.zip`. Its root `VERSION.json`
and `SHA256SUMS.txt` identify the delivered source and files. It accompanies
`main_applied_intelligence.tex`, based on source commit
`ef48699cdfdce8137b3fc560ff838977751e6e52` plus the documented review repairs.
It is a local delivery, not a newly published public release. Historical TMLR
packages describe earlier manuscript versions and are not this companion.

The companion has `source/` for the current reproducibility code, results and
manuscript source, and `evidence/` for the four original archives and their
adjacent `.zip.sha256` files. The outer archive has a new identity; the four
evidence ZIPs retain their original bytes. The public entry for the original
benchmark is [release v0.1.0](https://github.com/MengdanXue/prospective-graph-diagnostics/releases/tag/v0.1.0),
whose asset is named `prospective-graph-diagnostics-formal-artifacts-v0.1.0.zip`.
The locally retained copy uses the shorter filename below and has the same
checksum pinned by the repository CI. This revision verified the local copy;
it did not re-query or replace that remote asset.

## Archive identities and scope

All paths in this table are relative to the companion root. On 2026-09-28,
the four ZIP hashes and their recorded entry hashes and lengths were checked.
This verifies retained bytes; it does not independently reconstruct model records.

| Path | Retained content | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `evidence/prospective-graph-formal-artifacts-v0.1.0.zip` | 770 selected-model records, 110 diagnostics, 30 paired edge-intervention records; source/public digest mapping in `MANIFEST.json` | 37421570 | `209d386f078e0c4b2b15eb221b52c52bb002fc136a473494684e01be95b70389` |
| `evidence/posthoc_diagnostic_controls_v1.zip` | 200 MLP diagnostic records, 280 preprocessing records, configurations, manifests and historical source snapshots | 20802216 | `4ab10775a8bbcce924d206d74a101c5a8f7e751a0d50a68ba3d5a0f0a6972c56` |
| `evidence/posthoc_graph_parameterization_v1.zip` | 420 completed graph-diagnostic records and exact source snapshots; separate 198-record incomplete attempt retained only as history | 32463392 | `7ee948ec89856c4223ab7f9043d3858f3f337491c2eaa7167196f14a64f89a06` |
| `evidence/training_reproducibility_audit_v1.zip` | 22 repeatability worker records, histories, logs, initial/selected states and logits, completion hashes and source snapshot | 229022209 | `39d6c1c126be7d3db0b24153ef0a0c43d6a95052d98ff3a6c57bde6e7ea8f1c6` |

The first archive records release commit
`b97ae0ea5e6a5f8a53e853142cc50d1be9152c12` and has 910 manifest-listed JSON
records. The other three have respectively 500, 652 and 124 hash-manifest
entries, excluding each manifest itself. They do not contain the raw datasets
or checkpoints for every original benchmark model. Historical source snapshots
inside the ZIPs remain exact; a source checkout alone is not the complete Git
history needed by every historical-source validator.

## Table-to-input map

Commands below run from `source/`. In this table, `A/` means
`results/diagnostic/route_a_prospective_v2/analysis/`, `D/` means
`results/diagnostic/route_a_degree_matched_v1/summary/`, and script names are
relative to `scripts/` unless an `experiments/` prefix is shown. Stable LaTeX
labels identify tables even if typesetting changes their numbers.

| Table label(s) | Checked-in input | Reconstruction script and required evidence |
| --- | --- | --- |
| `tab:prospective_policy_results` | `A/diagnostic_audit.json` (`units`, policy and comparison summaries) | `assemble_prospective_diagnostics.py` then `experiments/evaluate_diagnostics.py`; all 770 model and 110 diagnostic records plus `configs/prospective_benchmark_v2.json` |
| `tab:fallback_decomposition`, `tab:action_confusion`, `tab:threshold_headroom` | `A/threshold_headroom.json` and `A/diagnostic_audit.json` | `summarize_threshold_headroom.py`; verified formal records, audit, config and release manifest. The primary decomposition/confusion can also be reaggregated directly from the 110 audit units; the architecture threshold rows need model records |
| `tab:equal_budget` | `A/equal_budget_sensitivity.json` | `summarize_equal_budget_sensitivity.py`; formal model/diagnostic records, manifest, audit and config |
| `tab:subset_sensitivity`, `tab:calibrated_threshold` | `A/portfolio_robustness.json` | `summarize_portfolio_robustness.py`; same formal records, manifest, audit and config; enumerates 63 subsets and retained LODO calibrations |
| `tab:fallback_sensitivity`, `tab:preprocessing_fallback` | `A/diagnostic_audit.json`, `A/preprocessing_sensitivity.json` | `summarize_fallback_sensitivity.py`; compact summaries suffice for this reaggregation; preprocessing and frozen results remain separate |
| `tab:preprocessing_sensitivity` | `A/preprocessing_sensitivity.json` | `summarize_preprocessing_sensitivity.py`; all 280 preprocessing records and their manifest/completion marker from the controls ZIP, plus the frozen audit; optional `--data-root` verifies raw NPZ checksums |
| `tab:degree_intervention` | `D/summary.json` | `summarize_degree_matched_benchmark.py`; 30 paired records and explicit v2 config; includes paired seed differences and bootstrap intervals |
| `tab:dataset_properties`, `tab:dataset_decisions` | `A/reviewer_appendix_summary.json` | `summarize_reviewer_appendix.py`; audit plus archived dataset counts for summary reconstruction, or read-only PyG cache to independently reload counts; see the provenance amendment below |
| `tab:winning_architectures` | `A/winning_architectures.json` | `summarize_winning_architectures.py`; audit only |
| `tab:mlp_parameterization` | `results/diagnostic/posthoc_mlp_optimization_v1/analysis/mlp_optimization_diagnostic_summary.json` | `summarize_mlp_optimization_diagnostic.py`; 200 MLP + 280 preprocessing records, the two raw NPZ inputs, and the recorded historical MLP source |
| `tab:graph_parameterization` | `results/diagnostic/posthoc_graph_parameterization_v1/analysis/graph_parameterization_diagnostic_summary.json` | `summarize_graph_parameterization_diagnostic.py`; completed 420 graph + 200 MLP + 280 preprocessing records, two raw NPZ inputs and recorded source bindings |
| `tab:training_reproducibility` | `results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json` | `check_revision_handoff.py` reaggregates the 22 retained workers and checks the full-summary/completion bindings; ZIP supplies the full summary and worker files |
| `tab:frozen_training`, `tab:diagnostic_policies`, `tab:architecture_details` | `configs/prospective_benchmark_v2.json`, `experiments/prospective_models.py`, `experiments/prospective_data.py`, `experiments/evaluate_diagnostics.py` and the runner | Protocol/implementation specification; inspect source and config, not estimated result tables |
| `tab:notation`, `tab:threshold` | `sections_applied/11_fixed_degree_analysis.tex` | Analytical definitions and derivations; no dataset reconstruction |

The regret–coverage and threshold figures use
`plot_prospective_regret_coverage.py` and `plot_threshold_headroom.py` with their
corresponding audit/threshold inputs. Tables are typeset in LaTeX; the scripts
regenerate their supporting JSON rather than rewriting manuscript text.

## Minimal audit-only reconstruction

Python 3.12+ standard library suffices for the appendix repair and its tests.
Create `tmp/rebuild/` first and use new output paths; writers refuse overwrites.

```text
python -m unittest tests.test_reviewer_appendix_summary -v
python scripts/summarize_reviewer_appendix.py --audit results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json --metadata-summary docs/applied-intelligence/provenance/reviewer_appendix_summary.pre-2026-09-28.json --output tmp/rebuild/reviewer_appendix_summary.json
python scripts/summarize_winning_architectures.py --output tmp/rebuild/winning_architectures.json
python scripts/summarize_fallback_sensitivity.py --output tmp/rebuild/fallback.json
```

The appendix command recalculates every scientific field, requires exact
agreement with the preserved summary, and binds the output to the actual audit
bytes. It reuses only node/edge/class/feature counts from the preserved summary;
it does not independently verify those counts against raw data. The old claimed
audit digest remains in the amendment and historical copy. See
[provenance amendment](provenance-amendment-2026-09-28.md).

## Record reconstruction and further requirements

For numerical reaggregation from raw result records, install the pinned CPU
dependencies in `requirements-ci.lock.txt` (see the repository environment
instructions). Verify/extract the formal ZIP as `tmp/formal/`, preserving its
`MANIFEST.json`, `prospective/records`, `prospective/diagnostics` and
`degree_matched/records` tree. Then use fresh output names:

```text
python scripts/assemble_prospective_diagnostics.py --records-root tmp/formal/prospective --config configs/prospective_benchmark_v2.json --output tmp/rebuild/assembled.json
python experiments/evaluate_diagnostics.py --input tmp/rebuild/assembled.json --output tmp/rebuild/diagnostic_audit.json
python scripts/summarize_equal_budget_sensitivity.py --records-root tmp/formal/prospective/records --output tmp/rebuild/equal_budget.json
python scripts/summarize_portfolio_robustness.py --records-root tmp/formal/prospective/records --output tmp/rebuild/portfolio.json
python scripts/summarize_threshold_headroom.py --records-root tmp/formal/prospective/records --output tmp/rebuild/threshold_headroom.json
python scripts/summarize_degree_matched_benchmark.py --config configs/prospective_benchmark_v2.json --input-root tmp/formal/degree_matched --output tmp/rebuild/degree.json
python scripts/check_posthoc_release.py --records-root tmp/formal/prospective/records
python scripts/check_revision_handoff.py --assets-dir ../evidence --output-dir tmp/checked-supplements
```

The first evaluator records runtime versions, so byte identity across different
Python/NumPy versions is not promised. The release check compares complete
scientific audit content, allows only `1e-12` absolute floating rounding drift,
and keeps counts/actions exact. A successful checksum check alone is not that
record reconstruction. The supplementary checker without `--data-root`
verifies archives and worker reaggregation; it does not rebuild feature transforms.

For feature reconstruction and all three preprocessing/MLP/graph summaries,
add `--data-root /path/to/inputs` to the supplementary checker and use another
fresh output directory. Obtain `roman_empire.npz` and `amazon_ratings.npz`
from the source URLs and SHA-256 values in `configs/preprocessing_sensitivity_v1.json`.
This step also requires historical Git objects described in
[`docs/revision_handoff.md`](../revision_handoff.md); the unchanged archive
snapshots permit inspection but a source ZIP is not a Git repository. Use the
retained author `source.bundle` or the matching repository history.

Checkpoint-forward replay additionally needs the appropriate saved states,
architecture, raw inputs, transforms, splits and compatible environment. The
reliability ZIP retains states for its bounded 22-worker audit; it does not
provide checkpoint replay for all 770 original selected models. Repeating
training is a further task: it needs the raw dataset cache, frozen configs,
historical runner and experiment environment (`requirements-experiments.lock.txt`).
Use the explicit v2 runner entry points in the repository README. Summary
agreement does not certify identical training trajectories across backends.

This review repair ran no dataset loader, training, new model/test evaluation,
or complete 770-record reconstruction. It regenerated the appendix from the
110-unit audit, checked exact scientific equality, exercised stale/altered-input
failures, and checked the four retained archive inventories. Broader historical
checks remain attributed to their original reports.
