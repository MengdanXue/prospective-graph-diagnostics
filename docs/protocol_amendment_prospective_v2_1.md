# Prospective Benchmark v2.1 Execution-Only Amendment

Date frozen: 2026-08-13

Parent protocol: `route_a_prospective_v2`

Execution code commit: `6d134b74da10c3ce808e660eb858d0bb47a3ab12`

Canonical configuration SHA-256: `3ae6302dbb686ba12ad38ca5a38429074e6015fcfd3ff6d603b63ec8c014b0b9`

## Amendment statement

> **v2.1 execution-only amendment:** batch wall-clock limit increased from 30 to 90 minutes after a blinded feasibility timeout on Squirrel. No comparative outcomes were inspected. Existing immutable artifacts were provenance-verified and resumed without overwrite.

## Trigger and blinding boundary

The single formal Squirrel command reached the external 30-minute hard timeout
after 1,804.1 seconds. At termination, it had written 46 model records and 7
diagnostic records, with no failure artifact. Monitoring and post-timeout checks
were limited to process state, resource use, record counts, JSON validity,
value type/range validation, and provenance. No test accuracy was printed,
ranked, aggregated, or compared, and no diagnostic benchmark outcome was
assembled or evaluated.

## Permitted execution change

The per-dataset hard wall-clock limit is increased from 30 minutes to 90
minutes. The interrupted Squirrel batch may be resumed once with `--resume`.
Before resumption, every existing artifact must pass the runner's immutable
resume checks for run ID, source commit, canonical configuration digest, full
frozen configuration, data provenance, split ID, trial grid, selected
validation result, and test-once count. Existing records must be skipped without
overwrite. The remaining four datasets continue as separate commands under the
same 90-minute limit.

## Unchanged scientific protocol

This amendment changes no dataset eligibility rule, dataset scope, seed, split,
model, hyperparameter trial, early-stopping rule, validation-selection rule,
test-access rule, diagnostic definition, evaluator threshold, bootstrap seed,
permutation seed, or edge-intervention setting. It creates no new run ID and no
new configuration digest. All resumed and subsequent artifacts must continue to
identify execution commit `6d134b74da10c3ce808e660eb858d0bb47a3ab12`.

## Stop and acceptance rules

Execution stops on the first nonzero exit, failure artifact, malformed record,
provenance mismatch, overwrite attempt, tracked-code change, or 90-minute hard
timeout. No automatic retry is permitted after this authorized resume. The
prospective benchmark remains incomplete until all eleven datasets contain
exactly 70 model records and 10 diagnostic records with zero failures.
