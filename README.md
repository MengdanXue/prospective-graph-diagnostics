# Prospective Graph Diagnostics

This repository accompanies the manuscript **A Prospective Evaluation of Simple Graph Diagnostics for Graph-vs-MLP Model Selection**. It tests a practical question: can inexpensive diagnostics whose label-dependent graph statistics use training labels only reliably decide whether node classification should use a tuned graph-model portfolio or a tuned feature-only MLP?

The answer under the frozen protocol is negative. Graph structure can be highly predictive, but the evaluated low-dimensional diagnostics do not reliably characterize when it is useful.

## Main results

The prospective benchmark contains 11 datasets, 10 seeds, seven architectures, 770 selected-model records, and 110 diagnostic records whose label-dependent graph statistics use training labels only. Every architecture receives the same four-trial grid, but the total portfolio budgets are asymmetric: the graph action selects among six architectures (24 trials), whereas the feature-only action contains one MLP architecture (four trials). The target requires the selected graph model to beat the selected MLP by more than one percentage point. The estimand therefore concerns these fixed operational portfolios, not an equal-total-compute family comparison.

| Policy | Selection accuracy | Coverage | Full-set regret |
|---|---:|---:|---:|
| Historical combined diagnostic | 55.5% | 68.2% | 7.46 pp |
| Always graph | 80.9% | 100.0% | 0.26 pp |
| Validation selection | 91.8% | 100.0% | 0.22 pp |

After the predeclared Holm correction, the paired comparisons do not establish an advantage for the combined diagnostic. These results should not be interpreted as evidence that graph structure is irrelevant.

A separate degree-preserving intervention changes GCN accuracy by -45.5, -29.9, and -17.3 percentage points on Cora, CiteSeer, and PubMed. Every paired seed difference is negative. Because the intervention changes several forms of edge organization simultaneously and covers only three dataset clusters, it does not isolate homophily or establish a universal causal mechanism.

## Repository map

- `main_neurocomputing.tex`, `sections/`: active manuscript source.
- `configs/`: frozen machine-readable benchmark specifications.
- `experiments/`: prospective runner, models, diagnostics, evaluator, and degree-preserving intervention.
- `scripts/`: immutable-record assembly, statistical summaries, claim audits, and deterministic manuscript-figure generation.
- `results/`: compact deterministic summaries and manuscript figures generated from those audits.
- `docs/`: preregistration, outcome-independent amendments, environment notes, evidence map, and provenance boundaries.
- `tests/`: protocol, provenance, statistics, manuscript-safety, and public-release checks.

The large raw unit-level artifact set is intentionally not stored in ordinary Git history. It is distributed in the versioned [v0.1.0 release](https://github.com/MengdanXue/prospective-graph-diagnostics/releases/tag/v0.1.0); compact summaries needed to check manuscript values remain versioned here. Release archives replace only machine-local `processed_files[].root` values with `data/<dataset>/processed`. Their manifest records both the immutable source SHA-256 and the path-redacted public SHA-256 for every JSON file; no outcome, configuration, data checksum, split, edge list, or selection field is changed.

## Environment

The exact experiment environment was frozen on CPython 3.13.5, Windows x86-64, PyTorch 2.9.1 with CUDA 12.6, and PyTorch Geometric 2.7.0. See `requirements-experiments.lock.txt`. The lightweight analysis environment uses Python 3.12 with NumPy 2.3.5 and Matplotlib 3.11.1.

Windows PowerShell:

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock.txt
.\.venv-analysis\Scripts\Activate.ps1
```

POSIX shells:

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.lock.txt
source .venv-analysis/bin/activate
```

The commands below assume the environment is active. If activation is unavailable,
replace `python` with the explicit environment interpreter shown above.

The experiment lock points to CUDA 12.6 wheels. CPU-only users should install a platform-appropriate PyTorch build before installing the remaining experiment dependencies.

## Verification

Run the lightweight, cross-platform checks:

```bash
python -m unittest \
  tests.test_public_repository \
  tests.test_route_a_claims \
  tests.test_self_feature_mixing \
  tests.test_discriminability_experiment \
  tests.test_diagnostic_scoring \
  tests.test_degree_preserving_shuffle \
  tests.test_submission_safety -v
```

With the full experiment environment installed, run everything:

```bash
python -m unittest discover -s tests -v
```

Audit the active manuscript claims directly:

```bash
python scripts/audit_route_a_claims.py --root . --main main_neurocomputing.tex
```

The prospective and intervention runners expose their complete command-line interfaces:

```bash
python experiments/run_prospective_benchmark.py --help
python experiments/run_degree_matched_benchmark.py --help
```

For current v2 runs, use an entry point that explicitly selects the published
configuration. The original execution files keep their historical v1 defaults
so that their source-provenance mapping remains valid; omitting `--config` from
those older commands does not select the current benchmark. In particular, v1
includes the ineligible Texas dataset. These v2 entry points forward all other
arguments to the frozen implementations:

```bash
python scripts/run_release_v2.py benchmark --data-root /path/to/pyg-cache --output-root tmp/new-prospective
python scripts/run_release_v2.py intervention --data-root /path/to/pyg-cache --output-root tmp/new-intervention
```

The `assemble` and `degree-summary` subcommands provide the same explicit v2
configuration for rebuilding summaries. To run another frozen specification,
use the original scripts with an explicit `--config`. Configuration fields that
are fixed in the execution implementation are not a general hyperparameter API.

Both formal runners require a complete local PyG dataset cache and refuse to download or mutate it during a run. They also refuse stale, mixed-provenance, overwritten, or silently retried artifacts.

## Rebuild the published summaries

Download and unpack the three custom assets from the
[v0.1.0 release](https://github.com/MengdanXue/prospective-graph-diagnostics/releases/tag/v0.1.0).
Verify the ZIP and standalone manifest against `SHA256SUMS.txt`, then extract the
ZIP as `tmp/artifacts/formal-v0.1.0`. The extracted tree must contain
`prospective/records`, `prospective/diagnostics`, and `degree_matched/records`.

With the full experiment environment installed, regenerate both manuscript-facing
summaries without writing into the checked-in `results/` tree:

```bash
python scripts/assemble_prospective_diagnostics.py \
  --records-root tmp/artifacts/formal-v0.1.0/prospective \
  --config configs/prospective_benchmark_v2.json \
  --output tmp/rebuild/assembled_v2.json

python experiments/evaluate_diagnostics.py \
  --input tmp/rebuild/assembled_v2.json \
  --output tmp/rebuild/diagnostic_audit.json

python scripts/summarize_degree_matched_benchmark.py \
  --config configs/prospective_benchmark_v2.json \
  --input-root tmp/artifacts/formal-v0.1.0/degree_matched \
  --output tmp/rebuild/degree_summary.json

python scripts/summarize_reviewer_appendix.py \
  --audit tmp/rebuild/diagnostic_audit.json \
  --data-root /path/to/read-only/pyg-cache \
  --output tmp/rebuild/reviewer_appendix_summary.json
```

The last command reads dataset shapes from the same read-only PyG cache and derives
the appendix's dataset-level action counts, regrets, and post-hoc fallback sensitivity.
It does not train models or alter the frozen evaluator output.

The regenerated files are deterministic Git blobs. These pairs must match:

```bash
git hash-object tmp/rebuild/diagnostic_audit.json
git rev-parse HEAD:results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json

git hash-object tmp/rebuild/degree_summary.json
git rev-parse HEAD:results/diagnostic/route_a_degree_matched_v1/summary/summary.json
```

## Reproducibility boundary

### Post-hoc portfolio analysis

`main_tmlr.tex` and `sections_tmlr/` contain a separate venue-neutral draft:
**Diagnostic Utility Depends on the Model Portfolio: A Frozen Graph-vs-MLP
Decision Benchmark**. It preserves the original submitted manuscript and frozen
execution files. The additional architecture-selection and single-architecture
analyses are post-hoc descriptions of the original records, not an independent
prospective validation. Four trials per action means matched trial count, not
equal training time or FLOPs. A validation-selected graph architecture is not
necessarily a graph-versus-MLP winner.

After verifying the release ZIP and manifest against `SHA256SUMS.txt` and
extracting the archive, rebuild the new summaries to fresh paths:

```bash
python scripts/summarize_winning_architectures.py --output tmp/rebuild/winning_architectures.json
python scripts/summarize_equal_budget_sensitivity.py \
  --records-root tmp/artifacts/formal-v0.1.0/prospective/records \
  --output tmp/rebuild/equal_budget_sensitivity.json
python scripts/audit_route_a_claims.py --root . --main main_tmlr.tex
```

The sensitivity script defaults to the v2 configuration and the extracted
`MANIFEST.json`. It verifies every prospective record checksum, uses the frozen
assembler's scope/split/configuration/trial/provenance checks, and binds the
complete regenerated audit to the supplied audit before computing restrictions.
Audit runtime metadata is excluded from comparison; floating values allow only
`1e-12` absolute rounding differences, while counts and actions remain exact.
The winning-architecture summary uses `math.fsum` for floating sums. Both new
scripts serialize with LF newlines and refuse to overwrite an existing output.
The winning-architecture summary is reconstructed in the lightweight tests;
full reconstruction of the sensitivity summary requires the release records
and is performed by the command above. Unit tests also exercise mismatched
audits, duplicate records, invalid provenance, and altered unselected models.

The original evaluator includes Python and NumPy versions in its output. A
different runtime therefore cannot produce an identical original audit blob
even when all scientific values match; the original byte-comparison commands
assume the frozen execution environment.

The frozen outcomes are empirical results for the named dataset/model portfolio, not a universal model-selection theorem. The fixed-degree Gaussian analysis is a scoped mechanistic calculation and not a new Kesten-Stigum threshold. Protocol amendments and the fresh-history source mapping are documented in `docs/`; `docs/execution_source_mapping.json` gives machine-readable Git blob identifiers for every frozen execution file published here.

## Citation and manuscript status

Citation metadata is provided in `CITATION.cff`. The manuscript is a working research draft targeting Neurocomputing; this repository does not imply acceptance or publication.

The original research code in this repository is released under the MIT License; see `LICENSE`. The unmodified Elsevier class and bibliography-style files under `submission_assets/elsevier/` remain subject to the terms of their upstream distribution and are not relicensed by this repository.


### Post-hoc robustness and continuous manuscript verification

The additional `portfolio_robustness.json` contains all 63 nonempty candidate
subsets, dataset exclusions, and seven leave-one-dataset-out homophily-threshold
controls. All are post-hoc reanalyses, with no model retraining or GPU required.
The held-out dataset never supplies the outcomes used to choose its threshold;
other datasets' selected-model test outcomes are explicitly meta-training data.

```bash
python scripts/summarize_portfolio_robustness.py --records-root tmp/artifacts/formal-v0.1.0/prospective/records --output tmp/rebuild/portfolio_robustness.json
python scripts/check_posthoc_release.py --records-root tmp/artifacts/formal-v0.1.0/prospective/records
tectonic main_tmlr.tex --outdir tmp/manuscript --keep-logs --untrusted
python scripts/check_latex_log.py tmp/manuscript/main_tmlr.log
```

CI downloads the public v0.1.0 archive, verifies its pinned SHA-256, checks the
manifest and full frozen audit, then reconstructs both sensitivity summaries.
The manuscript job uses checksum-pinned Tectonic 0.17.0 and uploads the PDF/log.
Undefined references/citations, duplicate labels and overfull boxes fail the job;
underfull spacing warnings are allowed. This supplements, rather than replaces,
visual review. The draft's publication status and remaining independent-validation
limits are documented in `docs/repositioning_plan.md`.

The post-hoc preprocessing sensitivity is summarized in
`results/diagnostic/route_a_prospective_v2/analysis/preprocessing_sensitivity.json`.
It retrains raw features and PyG 2.7.0 `NormalizeFeatures` on the same ten splits
and seven-model grid for Roman-empire and Amazon-ratings. The committed summary
records its config digest, runtime, complete scope, paired seed differences, and
per-condition outcomes.

The follow-up MLP optimization diagnostic is specified in
[`docs/mlp_optimization_diagnostic_plan.md`](docs/mlp_optimization_diagnostic_plan.md).
It crosses five fixed feature transformations with two weight-decay settings on
the same two datasets and ten seeds, and records selected-checkpoint train and
validation prediction distributions. It performs zero new test evaluations.
The plan includes execution and strict reconstruction commands. The preceding
280-record integrity audit is stored in
`results/diagnostic/route_a_prospective_v2/analysis/preprocessing_record_audit.json`;
the prior scientific summary was reproduced exactly.

The completed 200-record diagnostic and descriptive contrasts are under
`results/diagnostic/posthoc_mlp_optimization_v1/analysis/`.
The [interpretation and scope](docs/mlp_optimization_diagnostic_review.md) explain
the direct prediction-distribution evidence and the joint scale/centering effect.
All 40 original-parameterization controls reproduced their validation metrics and
selected trials exactly. These validation-only results do not replace the frozen
benchmark or establish graph-model or test-generalization claims.

### Revision package and bounded reliability audit (2026-09-08)

The revision now integrates the preprocessing sensitivity in the main text and
the 200-record MLP and 420-record architecture diagnoses in
`sections_tmlr/10_training_diagnostics.tex`. These follow-up diagnoses use only
training and validation labels. They are post-hoc evidence about the training
protocol, not independent confirmation or a new preprocessing algorithm.

A separately frozen 22-process audit reproduces large LINKX training variation
under default CUDA from identical initial states and RNG states. The specified
deterministic configuration gives identical histories and selected checkpoints
within the tested repeat groups. This does not certify every historical run or
all architectures. The [audit report](docs/graph_parameterization_nondeterminism_audit.md)
and compact `results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json`
record the exact scope. Full records, model states and historical source snapshots
are distributed separately from compact Git results.

The [artifact manifest](docs/diagnostic_artifact_manifest.md) describes the three
local supplementary archives and their hashes. The
[revision response pack](docs/revision_response_pack.md) contains ten anticipated
reviewer concerns with evidence-backed English response drafts; these are internal
preparation, not received reviews or correspondence. The
[delivery note](docs/revision_delivery_2026-09-08.md) records the one-paper scope and
remaining claim limits. The submitted Neurocomputing source is preserved.

The manuscript's three new numerical tables are checked against their JSON
summaries by `tests.test_training_diagnostic_manuscript`, included in CI.

The [handoff guide](docs/revision_handoff.md) is the entry point for the
integrated source, manuscript, supplementary archives, and clean-checkout
verification commands. It distinguishes record reconstruction from training
replication and keeps the pending author review explicit.
