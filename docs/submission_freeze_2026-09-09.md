# TMLR-format preparation freeze, 2026-09-09

This freeze prepares an anonymous manuscript and supplement. It does not
submit, withdraw, or contact a journal. According to the author's current
report, the overlapping Neurocomputing manuscript remains With Editor.
The TMLR draft must not be formally submitted in parallel. Author scientific
review remains pending; the [four-question guide](author_review_freeze_2026-09-09.md)
is assistance for that review, not a signed author attestation.

## Research boundary

The existing results are this version's evidence base. No new 11-dataset
experiment, diagnostic comparator, or speculative reviewer-request experiment
is authorized by this freeze. Reopening experiments requires a finding that
a core claim actually lacks evidence. Known limitations already disclosed in
the manuscript do not by themselves reopen the experiment program.

The final changes explain existing observations:

- The complete-portfolio LODO control reduces regret from 7.46 to 2.41
  percentage points but remains above always-graph's 0.26.
- Ten of eleven folds select constant always-graph. Roman-empire accounts for
  89.3698% of the remaining total calibrated regret. This fraction is a
  decomposition of existing records, not a new experiment.
- GCN and GAT retain favorable calibrated descriptive averages; GPR-GNN also
  has a favorable calibrated comparison. Full-portfolio failure is not a
  universal diagnostic-failure result.
- Observed sensitivity is not recast as a new statistical significance or
  causal claim. The improved MLP controls remain scoped to their actual runs.

The original Neurocomputing manuscript, all original results, frozen configs,
and experimental runners are unchanged.

## Review files and author archive

The delivery directory contains:

- `output/pdf/main_tmlr.pdf`: anonymous main manuscript, including appendices.
- `tmlr_anonymous_supplement.zip`: anonymous code, summaries and retained
  records; its actual size and SHA-256 are recorded in VERSION.json.
- `author_review.md`: the Chinese four-question explanation for the author.
- `source.bundle` and `source.zip`: complete author source history and the
  exact frozen source snapshot. These are not anonymous review uploads.
- `assets/`: the original, unmodified archives, including checkpoint tensors.
- `verification/`: scoped test, layout, packaging and synchronization records.
- `VERSION.json` and `SHA256SUMS.txt`: exact version and file identities.

The anonymous supplement contains 770 original model records, 110 diagnostic
records, 30 intervention pairs, 280 preprocessing records, 200 MLP records,
420 completed graph records, and 22 repeatability worker records. The earlier
partial graph attempt remains separately identified. All source archives are
verified before copying. Identifying strings and PDF metadata are removed
only in review copies; all numerical JSON leaves and non-identifying strings
are checked against their originals. The review manifest binds delivered
bytes and identifies modified copies.

The supplement omits 66 large checkpoint tensor files and Git history. This
keeps it below the official 100 MB limit and avoids exposing author identity.
The omission limits checkpoint replay and historical Git-based reconstruction
from the supplement. The complete author archive preserves those materials;
the manuscript and supplement README state this limitation explicitly.

## Verification and rebuilding

The final check consumes existing JSON and performs no model training:

```text
python scripts/check_tmlr_freeze.py
```

When executed from an extracted anonymous supplement, it also verifies every
delivered file against REVIEW_MANIFEST.json. The manuscript's existing
numerical, protocol and scope tests remain in the source repository.

Build a new review supplement from a clean frozen checkout with Python 3.12+
and pypdf 6.10.0 installed:

```text
python scripts/build_tmlr_review_package.py --original-archive ORIGINAL.zip --assets-dir AUTHOR_ASSETS --output-dir NEW_OUTPUT
```

The builder verifies input archive hashes and refuses an existing output
directory. It checks all output hashes, the preserved JSON values, record
counts, anonymity strings, final LODO quantities, and the 100 MB size limit.
It writes an anonymous stage directory, a ZIP and its checksum, and a scoped
verification report. No external service is contacted by the builder.

The unchanged official template is documented in
[templates/tmlr/PROVENANCE.md](../templates/tmlr/PROVENANCE.md). Compile with
Tectonic 0.17.0 and check the log as described in revision_handoff.md. The
standard template displays a review-status header; it does not report an
actual OpenReview submission. References precede appendices. The two existing
AAAI entries supply both volume and issue fields; the official bibliography
style displays the volume and emits two nonfatal field warnings. Their
titles, authors, years, pages and DOIs are retained without changing the
shared bibliography used by the existing submission.

Current official requirements were checked on 2026-09-09:
[author guidelines](https://jmlr.org/tmlr/author-guide.html) require the official
style, anonymous PDF/ZIP supplements and a 100 MB supplement limit;
[editorial policies](https://jmlr.org/tmlr/editorial-policies.html) prohibit
overlapping parallel archival submissions. Only preparation is performed.
