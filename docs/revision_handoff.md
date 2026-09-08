# Revision handoff and reproduction

This is the internal, venue-neutral revision dated 2026-09-08. Its entry point
is `main_tmlr.tex`; the filename does not indicate a TMLR submission. The
submitted Neurocomputing sources remain separate. Author review and any venue
conversion follow the actual editorial status; the package does not attest
that the author has already completed that review.

## Package layout and version

- `manuscript/main_tmlr.pdf`: compiled revision for reading.
- `source.bundle`: Git history containing the integrated revision and every
  public-repository historical commit referenced by the supplementary runs.
- `source.zip`: the same revision's tracked files for convenient inspection.
- `assets/`: three diagnostic ZIPs and adjacent SHA-256 files.
- `verification/`: test, manuscript-build, and clean-checkout reconstruction logs.
- `VERSION.json`: exact source commit and validation scope.
- `SHA256SUMS.txt`: digests for the delivered files, excluding itself.

The old `v0.1.0` public release contains the original benchmark records. The
three supplementary archives accompany this local handoff and are separate
from that release. Raw NPZ inputs are obtained separately. Existing run
artifacts, configs, recorded source fingerprints, and original summaries are
not rewritten when packaging changes.

## Clone the exact source

From the extracted handoff directory:

```text
git clone --config core.autocrlf=false source.bundle source
cd source
git rev-parse HEAD
```

Compare the printed commit with `source_commit` in `../VERSION.json`. Keeping
LF bytes avoids changing historical source fingerprints during checkout on
Windows. The complete public history is included so historical-source checks
can use `git show` offline; a source ZIP alone cannot supply those Git objects.

Install Python 3.12 or 3.13, CPU PyTorch 2.9.1 and the pinned analysis/test
dependencies in `requirements-ci.lock.txt`. A GPU is unnecessary for summary
reconstruction. These commands check retained evidence and perform no model
training or new test-set evaluation:

```text
python -m unittest discover -s tests -v
python scripts/check_revision_handoff.py --assets-dir ../assets --output-dir ../checked-records
```

The second command verifies ZIP digests, exact per-entry manifest coverage,
all entry lengths and digests, the 420/200/280 record counts, the audit's
113-file completion binding, and reaggregates its 22 worker records. The
output directory must be new; existing verification results are preserved.

## Reconstruct features and all supplementary summaries

Obtain `roman_empire.npz` and `amazon_ratings.npz` according to the source URLs
and SHA-256 values in `configs/preprocessing_sensitivity_v1.json`, and put
them in `../inputs/`. Then run:

```text
python scripts/check_revision_handoff.py --assets-dir ../assets --data-root ../inputs --output-dir ../checked-full
```

This additionally reconstructs raw-data checksums, train-fitted transforms,
feature hashes and partition bindings, then compares the preprocessing, MLP
and graph summaries with the tracked results. The checker creates a detached
historical MLP checkout from the supplied history because that run verifies
its own executable bytes. The graph summarizer uses the later graph runner
and independently checks the historical MLP control. No current runner is
edited to imitate a historical fingerprint.

Floating-point feature fingerprints are intentionally exact. A failed check
must be investigated and recorded, not bypassed or treated as a training
replication. Passing these reconstruction checks does not make the original
CUDA training trajectories deterministic across machines.

## Rebuild the manuscript

Using Tectonic 0.17.0, the same version used by the repository's CI:

```text
tectonic main_tmlr.tex --outdir ../manuscript-check --keep-logs --untrusted
python scripts/check_latex_log.py ../manuscript-check/main_tmlr.log
```

Create `../manuscript-check` before compiling. Initial TeX dependency downloads
may require network access. The delivered PDF has also undergone visual
inspection; LaTeX-log checks alone do not verify page appearance.

## Regenerate archive containers

See [the artifact manifest](diagnostic_artifact_manifest.md) for the complete
command against the original run directories. The packager writes a new
output directory and refuses to replace an existing ZIP or checksum. Changing
the container README or its manifest coverage changes the ZIP digest while
leaving its experimental records and recorded source snapshots intact.
