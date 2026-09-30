# MLP-24 Extension Amendment: Reviewed Environmental Execution Failure Retry

Date frozen: 2026-09-30

Parent contract: `configs/diagnostics_mlp_budget_extension_v1.json` and
`docs/diagnostics_mlp_budget_extension_plan.md`

Parent execution commit: `0e6f4b014cea0ae39ed6eeb5ff04abdb1321ff86`

This is an execution-only amendment. It changes no dataset, split, input
treatment, model, seed, trial grid, selection rule, test-access rule, metric,
analysis, budget ceiling, watchdog, pause or emergency-stop rule.

## Trigger

The first formal MLP-24 unit attempt (`Cora`, `normalize_features`, `MLP`,
seed 0) was closed as `failed` after about five seconds. Its worker ran under a
CPU-only PyTorch build while the frozen device policy requires CUDA for MLP. The
worker raised `Torch not compiled with CUDA enabled` while building the runtime
environment snapshot, before the unit training function was entered. No
checkpoint, record, evaluation or test access was produced. The budget journal
has no mechanism to review a `failed` outcome, so the cumulative handoff is
permanently blocked. Before this amendment only an `external_interruption` could
be reviewed and restarted.

## Permitted change

A `failed` model-unit attempt may be retried **once** when a human reviewer
signs a hash-bound terminal failure review classifying it as
`ENVIRONMENTAL_EXECUTION_FAILURE`. Nothing else is permitted:

1. **No automatic retry.** A retry exists only through a signed review. Software
   never classifies or signs a failure.
2. **Classification.** A review has exactly one classification:
   `ENVIRONMENTAL_EXECUTION_FAILURE`, `RESEARCH_EXECUTION_FAILURE` or
   `UNKNOWN`. `UNKNOWN` never permits a retry. `RESEARCH_EXECUTION_FAILURE`
   does not permit a retry under this amendment; the failure is a result and
   stays blocking.
3. **Evidence conditions.** `retry_allowed=true` requires all of: environmental
   classification; `research_computation_started=false`; zero checkpoints, zero
   records and zero test evaluations; no clock, monitor or budget stop reason
   on the attempt; the attempt was not itself a retry; and every bound hash
   (journal, journal head, terminal receipt, close event, charge, failure
   evidence) matches the preserved original exactly.
4. **Immutable history.** The original `failed` outcome, its charged
   nanoseconds, terminal receipt, failure record, worker log, supervision
   receipt, request file and journal hash chain are preserved permanently. The
   original journal is never appended to, rewritten or reconciled. The failed
   attempt is never reopened in place, never relabelled `completed` or
   `external_interruption`, and its charge is never refunded or offset.
5. **Historical versus current blockers.** A cumulative handoff reports every
   historical stop reason unchanged. A valid signed retry review only removes
   that one `unreviewed_failed:<attempt>` reason from the *current* blocking
   set and records the waiver. An invalid review keeps the blocker and adds
   `failure_review_invalid:<attempt>`. A review with `retry_allowed=false`
   keeps the blocker.
6. **New segment only.** A retry runs only in a new inherited budget segment
   created from a handoff that binds the signed review. That segment records a
   `failure_retry_authorized` event after its phase registration and before the
   retry attempt. The event binds the original ledger, attempt, terminal
   receipt, review hashes, unit, paired batch, activity, research identity,
   retry attempt identity, retry ordinal and required environment.
7. **One-time consumption; at most one retry.** A review is consumed when the
   increment is registered in the shared segment index. A consumed review
   cannot authorize a second segment or a second attempt. `max_retries` is 1
   per model unit. If the retry also fails, its failure blocks and no further
   retry can follow from this amendment.
8. **New attempt identity.** A retry has a new deterministic attempt identity
   that includes the segment sequence, unit ordinal and retry ordinal. Attempt
   identities are unique across all segments. No random identifier is used.
   Before any attempt or charge, the runner refuses an identity already present
   in any inherited ledger, worker, failure or preparation artifact.
9. **Same research identity; no new test access.** The retry keeps the
   dataset, input condition, model, seed, reused trials and added trial grid of
   the original unit. It does not gain a second test evaluation: each unit
   record still contains exactly one post-selection test evaluation, and a unit
   with a record cannot be retried.
10. **Environment gate before registration.** Before any formal phase is
    registered and before any attempt or charge, the runner verifies the actual
    interpreter against the frozen device policy and the inherited base
    environment: interpreter path, PyTorch version, CUDA build, CUDA
    availability, device count, device identity and the complete runtime
    environment snapshot compared by the unit trainer. A mismatch refuses the
    launch and leaves only preflight evidence, with no phase, attempt, charge
    or `failed` outcome.
11. **New output root.** A retry under a later source commit writes to a new
    extension output directory, because an extension output is bound to the
    commit that created it. The earlier output, which holds zero records,
    stays unchanged as failure evidence.
12. **Accounting.** The retry is charged to the same cumulative unit and paired
    batch. All existing total, formal, control, resource, unit and batch caps
    apply unchanged.

## Required review artifacts

A machine-readable review plan (`terminal-failure-review-plan/1`) binds the
original identity and evidence listed above, together with the proposed
classification, findings and disposition. The review record
(`terminal-failure-review/1`) binds this amendment, the plan and the human
sign-off. A draft record has status
`DRAFT_PENDING_HUMAN_SIGNOFF_DO_NOT_CONSUME` and never clears a blocker. Only a
record with status `SIGNED` and decision `APPROVE_ENVIRONMENTAL_FAILURE_RETRY`
can authorize a retry.
