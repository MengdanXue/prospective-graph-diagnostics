# Prospective Graph Diagnostics

This repository accompanies the manuscript **A Prospective Evaluation of Graph Diagnostics for Graph-vs-MLP Model Selection**. It tests a practical question: can inexpensive, train-only graph diagnostics reliably decide whether node classification should use a tuned graph-model family or a tuned feature-only MLP?

The answer under the frozen protocol is negative. Graph structure can be highly predictive, but the evaluated low-dimensional diagnostics do not reliably characterize when it is useful.

## Main results

The prospective benchmark contains 11 datasets, 10 seeds, seven model families, 770 selected-model records, and 110 train-only diagnostic records. The target requires the selected graph family to beat the selected MLP by more than one percentage point.

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
- `scripts/`: immutable-record assembly, statistical summaries, and claim audits.
- `results/`: compact deterministic summaries and the manuscript validation figure.
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
```

The regenerated files are deterministic Git blobs. These pairs must match:

```bash
git hash-object tmp/rebuild/diagnostic_audit.json
git rev-parse HEAD:results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json

git hash-object tmp/rebuild/degree_summary.json
git rev-parse HEAD:results/diagnostic/route_a_degree_matched_v1/summary/summary.json
```

## Reproducibility boundary

The frozen outcomes are empirical results for the named dataset/model portfolio, not a universal model-selection theorem. The fixed-degree Gaussian analysis is a scoped mechanistic calculation and not a new Kesten-Stigum threshold. Protocol amendments and the fresh-history source mapping are documented in `docs/`; `docs/execution_source_mapping.json` gives machine-readable Git blob identifiers for every frozen execution file published here.

## Citation and manuscript status

Citation metadata is provided in `CITATION.cff`. The manuscript is a working research draft targeting Neurocomputing; this repository does not imply acceptance or publication.

The original research code in this repository is released under the MIT License; see `LICENSE`. The unmodified Elsevier class and bibliography-style files under `submission_assets/elsevier/` remain subject to the terms of their upstream distribution and are not relicensed by this repository.
