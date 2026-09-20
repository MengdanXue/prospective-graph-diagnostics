# Prospective Graph Diagnostics

This repository accompanies **Diagnostic Utility Depends on the Model Portfolio: A Frozen Graph-vs-MLP Decision Benchmark**. The current manuscript is prepared in the anonymous TMLR format in [`main_tmlr.tex`](main_tmlr.tex); it has not been submitted to TMLR. The original Neurocomputing manuscript is preserved in [`main_neurocomputing.tex`](main_neurocomputing.tex).

The study asks whether inexpensive diagnostics whose label-dependent graph statistics use training labels only can decide between a tuned graph-model portfolio and a tuned feature-only MLP. Under the frozen prospective protocol, the evaluated diagnostics do not reliably characterize when graph structure is useful. The TMLR revision examines how this result depends on the model portfolio, using explicitly post-hoc analyses of the retained records and bounded training diagnostics.

The scope is the [September 9 manuscript freeze](docs/submission_freeze_2026-09-09.md). Later eleven-dataset input-robustness preparation is separate work and supplies no results to this manuscript.

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

- `main_tmlr.tex`, `sections_tmlr/`: current TMLR manuscript; shared sections remain in `sections/`.
- `main_neurocomputing.tex`, `sections/`: preserved historical Neurocomputing manuscript.
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

Audit the current manuscript claims and compile the PDF:

```bash
python scripts/audit_route_a_claims.py --root . --main main_tmlr.tex
tectonic main_tmlr.tex --outdir tmp/manuscript --keep-logs --untrusted
python scripts/check_latex_log.py tmp/manuscript/main_tmlr.log
```

CI runs lightweight and full protocol checks, reconstructs the post-hoc summaries
from the checksum-pinned public release, and compiles the manuscript with
checksum-pinned Tectonic 0.17.0. Undefined references/citations, duplicate labels
and overfull boxes fail the manuscript job; underfull spacing warnings are
allowed. These checks supplement visual review.

## Replicate the frozen training protocol

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

In the frozen execution environment, the regenerated files are deterministic Git
blobs. These pairs must match:

```bash
git hash-object tmp/rebuild/diagnostic_audit.json
git rev-parse HEAD:results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json

git hash-object tmp/rebuild/degree_summary.json
git rev-parse HEAD:results/diagnostic/route_a_degree_matched_v1/summary/summary.json
```

The original evaluator includes Python and NumPy versions in its output. A
different runtime therefore cannot produce an identical original audit blob
even when all scientific values match. See the [reproduction guide](docs/revision_handoff.md)
for record reconstruction, historical source snapshots, and the distinction
between reconstruction and training replication.

## Rebuild the post-hoc analyses

Architecture-selection and single-architecture analyses describe the original
records; they are not an independent prospective validation. Four trials per
action means matched trial count, not equal training time or FLOPs. A
validation-selected graph architecture is not necessarily a graph-versus-MLP
winner.

The portfolio robustness analysis contains all 63 nonempty candidate subsets,
dataset exclusions, and seven leave-one-dataset-out homophily-threshold controls.
These analyses require no model retraining or GPU. The held-out dataset never
supplies the outcomes used to choose its threshold; other datasets'
selected-model test outcomes are explicitly meta-training data.

After verifying the release ZIP and manifest against `SHA256SUMS.txt` and
extracting the archive, rebuild the new summaries to fresh paths:

```bash
python scripts/summarize_winning_architectures.py --output tmp/rebuild/winning_architectures.json
python scripts/summarize_equal_budget_sensitivity.py \
  --records-root tmp/artifacts/formal-v0.1.0/prospective/records \
  --output tmp/rebuild/equal_budget_sensitivity.json
python scripts/summarize_portfolio_robustness.py \
  --records-root tmp/artifacts/formal-v0.1.0/prospective/records \
  --output tmp/rebuild/portfolio_robustness.json
python scripts/check_posthoc_release.py --records-root tmp/artifacts/formal-v0.1.0/prospective/records
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

## Additional evidence and limitations

The post-hoc preprocessing sensitivity is summarized in
`results/diagnostic/route_a_prospective_v2/analysis/preprocessing_sensitivity.json`.
It retrains raw features and PyG 2.7.0 `NormalizeFeatures` on the same ten splits
and seven-model grid for Roman-empire and Amazon-ratings. The committed summary
records its config digest, runtime, complete scope, paired seed differences, and
per-condition outcomes.

The follow-up 200-record MLP and 420-record architecture diagnostics use only
training and validation labels and perform zero new test evaluations. They are
post-hoc evidence about the training protocol, not independent confirmation or a
new preprocessing algorithm. The [MLP interpretation](docs/mlp_optimization_diagnostic_review.md)
and [architecture diagnostic plan](docs/graph_parameterization_diagnostic_plan.md)
document their scope and reconstruction commands. These validation-only results
do not replace the frozen benchmark or establish test-generalization claims.

A separately frozen 22-process audit reproduces large LINKX training variation
under default CUDA from identical initial states and RNG states. The specified
deterministic configuration gives identical histories and selected checkpoints
within the tested repeat groups. This does not certify every historical run or
all architectures. The [audit report](docs/graph_parameterization_nondeterminism_audit.md)
and compact `results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json`
record the exact scope. Full records, model states and historical source snapshots
are distributed separately from compact Git results.

The [artifact manifest](docs/diagnostic_artifact_manifest.md) describes the
supplementary archives and their hashes. It distinguishes locally retained
archives from the public v0.1.0 release. The [claim evidence map](docs/claim_evidence_map.md)
links manuscript claims to their supporting records. The manuscript's three
training-diagnostic tables are checked against their JSON summaries in CI.

The frozen outcomes are empirical results for the named dataset/model portfolio,
not a universal model-selection theorem. The fixed-degree Gaussian analysis is a
scoped mechanistic calculation and not a new Kesten-Stigum threshold. Protocol
amendments are retained in `docs/`;
[`docs/execution_source_mapping.json`](docs/execution_source_mapping.json) gives
machine-readable Git blob identifiers for every frozen execution file published
here.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The TMLR manuscript
is a research draft; this repository does not imply acceptance or publication.

The original research code in this repository is released under the MIT License;
see [`LICENSE`](LICENSE). The unmodified Elsevier class and bibliography-style
files under `submission_assets/elsevier/` remain subject to the terms of their
upstream distribution and are not relicensed by this repository. The TMLR
template's license and provenance are retained under `templates/tmlr/`.
