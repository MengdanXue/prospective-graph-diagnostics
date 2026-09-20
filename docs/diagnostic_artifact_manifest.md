# Post-hoc diagnostic artifact manifest

The graph parameterization result is archived at
`results/diagnostic/posthoc_graph_parameterization_v1/analysis/`. The copied
JSON and Markdown files are byte-identical to the validated output produced
from 420 records.

The large, non-Git record archives are in the sibling working directory
`../revision-final-2026-09-08/assets/`:

| Archive | Contents | SHA-256 |
| --- | --- | --- |
| `posthoc_graph_parameterization_v1.zip` | 420 graph/MLP records from the graph parameterization run, run manifest, completion marker, frozen graph config, summary, graph source snapshot at `c48c34ee160f7f6ab62bfcaf10b5e90f40568466`, the exact `a22606a…` summarizer and historical MLP source snapshot used by control verification, and the earlier 198-record failed attempt | `7ee948ec89856c4223ab7f9043d3858f3f337491c2eaa7167196f14a64f89a06` |
| `posthoc_diagnostic_controls_v1.zip` | 200 MLP optimization records plus the 280-record preprocessing control, their manifests and completion markers, frozen configs, and source snapshots | `4ab10775a8bbcce924d206d74a101c5a8f7e751a0d50a68ba3d5a0f0a6972c56` |
| `training_reproducibility_audit_v1.zip` | 22-worker reliability audit, raw logs, 22 worker records and state/log artifacts, frozen config, complete hash manifest, audit summary, and exact source snapshot | `39d6c1c126be7d3db0b24153ef0a0c43d6a95052d98ff3a6c57bde6e7ea8f1c6` |

The adjacent `.sha256` files record each ZIP byte-stream digest. Each ZIP also
contains `hash_manifest.json`, which lists the SHA-256 and byte length of
every extracted entry except the hash manifest itself. The packaging command
is reproducible. The graph and controls packages exclude NPZ data, virtual
environments, Torch installations, checkpoints, and caches; only the separate
reliability-audit package retains its worker state dictionaries as raw audit
output.

Configuration copies are read from the recorded Git commit and checked against
the manifest's parsed configuration, so checkout line-ending settings cannot
change the rebuilt container. Relative to the earlier local archives, all
experimental record and historical source-snapshot entry hashes are retained.
The standalone preprocessing config copy now uses its Git LF bytes instead of
the previous working-copy CRLF bytes; its JSON content and canonical digest are
unchanged. README content and full manifest coverage account for the other
container changes. The earlier local archives remain preserved separately.

## Frozen graph scope

The graph run is `posthoc_graph_parameterization_v1`, with 2 datasets, 10
seeds, 7 models, and 3 preprocessing conditions: exactly 420 records and
1,680 training trials. Its manifest binds `config_sha256` to
`ed0432fad313d371b3b5fbb6863fa024d05eb6688493c1023cdce653e8834692`, records
`source_commit=c48c34ee160f7f6ab62bfcaf10b5e90f40568466`, and sets
`test_evaluations_after_selection=0`. Every record is checked against those
three bindings before packaging. The generated summary also reports 420
records, 20 split bindings, and verified source content.

The same archive retains the first `graph-parameterization-v1` attempt under
`initial_attempt_198/`. It has 198 partial records, the original manifest, one
`failure_1788846847783089300.json`, a range/scope binding, and the exact source
snapshot at `6c38aab78e8fafda7511ee18c295d0176c42e081`. It has no completion
marker and is audit history only; none of its records enter the completed
420-record summary.

The new summarizer used the current commit `a22606a6a50c118b29b6f4777ce3163e7cce7675`.
Its prior MLP control is checked against the MLP manifest's historical
source commit `3163a5dcb04044c944635fccfb86e7bdd2666c3c` by exact Git-object
hashes. The package retains that historical source tree so the control
provenance can be inspected without changing the current runner to match a
new hash. It also retains the exact summarizer source from this `a22606a…`
commit, including its historical-source verification logic.

## Rebuild and verify

For the portable handoff, start with [the handoff guide](revision_handoff.md).
To regenerate archive containers from the original completed run directories,
run the following from the repository root. Packaging itself needs the retained
records and source history; feature reconstruction additionally needs the two
NPZ inputs and their frozen SHA-256 checks:

```text
python scripts/package_diagnostic_artifacts.py \
  --repo-root . \
  --graph-run-root ../graph-parameterization-v1-fixed \
  --graph-summary-root ../graph-parameterization-v1-analysis-final \
  --initial-graph-run-root ../graph-parameterization-v1 \
  --mlp-run-root ../mlp-optimization-v1 \
  --preprocessing-run-root ../preprocessing-full \
  --audit-run-root ../training-reproducibility-audit-v1 \
  --output-dir ../revision-rebuilt/assets
```

Use a fresh output directory: existing ZIPs and checksums are never replaced.
The script refuses incomplete runs, failed-record artifacts in completed runs, wrong record
counts, provenance drift, mismatched source hashes, or nonzero test-evaluation
bindings for the graph/MLP validation-only runs. It reads historical files via
`git show <recorded-commit>:<path>` and writes those exact bytes into the
archive. The extracted source snapshots can be read directly or placed in a
clean checkout at their recorded commit for an exact runner reconstruction;
the package does not silently substitute the current working-tree runner.

## Why the 280 preprocessing records are included

The graph summary uses the 280-record preprocessing run as a paired control
for `raw` and `normalize_features`, and checks its transformed-feature hashes
against the graph reconstruction. Those records are therefore included in
the separate controls archive with their original `run_manifest.json`,
`complete.json`, and all 280 JSON records. This preserves the evidence needed
to inspect the disclosed control mismatches and feature-hash agreement without
bundling the two large NPZ inputs. The old preprocessing run reports one
post-selection test evaluation per record (`test_evaluations_after_selection=1`);
that is explicitly retained as an attached control fact and is not part of the
graph or MLP validation-only result.

## Reliability audit package

The separate `training_reproducibility_audit_v1.zip` contains the completed
22-worker audit: 22/22 workers succeeded, there are zero failure artifacts and
zero test evaluations, and the complete marker's 113-file hash map was checked
before packaging. It retains the 44 worker state dictionaries and selected
logits as raw audit output. The recorded source commit is
`78a3efebf2f13e271149bceb01ebee1b03552afd`; all committed source files match
their manifest hashes. The audit script was untracked at that commit, so its
exact working-tree bytes are stored separately under
`audit/source_snapshot/working_tree_exact/` and checked against the recorded
hash. A compact review summary is tracked at
`results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json`;
its `full_summary_sha256` field is
`17af44a2535bfbcb6be62eb0b99ae9ae9a2b06c8ede7f4426805da30dad4475c`.
The full summary with its worker records remains unchanged inside the audit ZIP
and is authenticated by that same full-summary hash.
