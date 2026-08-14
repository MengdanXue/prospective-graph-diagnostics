# Source Provenance

The formal prospective and degree-matched experiments were executed from source
commit `6d134b7` in the full research workspace. Unit-level records retain that
commit identifier, the canonical frozen-configuration digest, processed-data
checksums, split identifiers, environment provenance, and test-once accounting.

This public repository intentionally starts from a fresh Git history so that
legacy manuscript drafts, exploratory code, damaged intermediate JSON, and raw
grid outputs are not published as active research artifacts. Before export, a
path-scoped comparison between the execution commit and the verified release
snapshot found no changes under `configs/`, `experiments/`,
`scripts/assemble_prospective_diagnostics.py`, or
`scripts/summarize_degree_matched_benchmark.py`.

The compact prospective audit is stored at
`results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json`; the
degree-matched summary is stored at
`results/diagnostic/route_a_degree_matched_v1/summary/summary.json`. The complete
raw unit-level records are distributed as a checksummed GitHub Release asset.
