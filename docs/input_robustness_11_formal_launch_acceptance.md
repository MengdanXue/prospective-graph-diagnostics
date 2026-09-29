# Input-robustness 11-dataset launch acceptance

This record is an archive of the guarded transition to the formal-run gate. It
does not authorize training and it contains no new research test outcomes.

The accepted runtime and resource execution is bound to execution SHA
`aef8b87a6eb7bf6ea4c3fd1482b0a6882dc176b0`. Its same-SHA CI receipt is run
`34800560156`, with `manuscript-build`, `lightweight-verification`, and
`full-protocol-verification` successful. The shared append-only ledger is the
existing `input-robustness-budget-v2` ledger; its final journal SHA-256 is
`28437c2a9781e48cbb127555b49f7ad490b8b84e4e26872c73edca6a37468a3b`.
The trusted cumulative charge is 1525.045280584 seconds of the 7200-second
resource allocation, with 0 formal seconds and 0 control seconds. The compact
external receipt is `current-runtime-resource-acceptance-final.json`.

The formal record and analysis implementation is execution commit
`3acde08e47ad19e8f0e27599579c21a1fdb38837`. It writes one exclusive record for
each of 1540 dataset-condition-model-seed units and requires 6160 trial rows:
four tuning trials, validation selection, one post-selection test evaluation,
and four complete original checkpoint copies. The validator rejects missing or
duplicate identities, mixed provenance, split or transform drift, incomplete
checkpoint persistence, and test leakage. The analysis adapter consumes only
validated records, evaluates all nine fixed strategies across all 63 non-empty
graph portfolios in both conditions, and fits leave-one-dataset-out thresholds
on the other dataset clusters within the same condition and portfolio.

The committed fixture tests passed (9 tests). The complete local suite passed
377 tests in the approved runtime before this commit; the same-SHA CI receipt
for `3acde08e47ad19e8f0e27599579c21a1fdb38837` is run `34805021727`, with all
three jobs successful. The independent launch report is
`formal-launch-checks-3acde08.json`.

Current state: all technical launch gates pass. `formal_training_enabled` and
`formal_launch_authorized` remain false. The next action is one explicit formal
start authorization; no per-batch confirmations are required after that
authorization, and no formal switch is opened by this archive commit.
