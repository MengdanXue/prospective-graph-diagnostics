# Executing the bounded supplement

The author approved this scope on 2026-09-28. Execution remains conditional on the completed base run, a reviewed cumulative budget handoff, actual isolated end-to-end acceptance, and the three required GitHub CI jobs at the execution commit. The preparation command does not launch training. A test count alone is not acceptance.

Use the isolated `diagnostics-mlp24-2026-09-28` checkout. Keep the active base checkout, its records and all old budget segments unchanged. The extension configuration is `configs/diagnostics_mlp_budget_extension_v1.json`; the scientific contract is `docs/diagnostics_mlp_budget_extension_plan.md`.

## Paths and preparation

From this checkout in PowerShell:

```powershell
$supplementWork = Split-Path (Get-Location).Path -Parent
$supplementPython = Join-Path $supplementWork 'venv-gpu/Scripts/python.exe'
$supplementProof = Join-Path $supplementWork 'diagnostics-mlp24-verification-2026-09-28'
$supplementBase = Join-Path $supplementWork 'input-robustness-11-formal-run-r2'
$supplementBinding = Join-Path (Get-Location).Path 'results/diagnostic/posthoc_input_robustness_11_v1/preflight/data_binding.json'
$supplementOutput = Join-Path $supplementWork 'diagnostics-mlp24-formal-run-v1'
$supplementControl = Join-Path $supplementWork 'diagnostics-mlp24-formal-run-v1_control'
$supplementLedger = Join-Path $supplementWork 'diagnostics-mlp24-budget-segment-001'

& $supplementPython -m scripts.mlp_budget_extension_entry `
  --config configs/diagnostics_mlp_budget_extension_v1.json `
  --base-config configs/input_robustness_11_v2.json `
  --base-root $supplementBase --binding $supplementBinding
```

The reviewed handoff and acceptance files below must exist and must bind the committed execution source. Missing evidence is a real blocker; never create a replacement containing only `passed: true`. The handoff includes every original ledger segment and any subsequently registered supplement segments. It cannot erase unresolved attempts or grant fresh limits.

## Execution after all gates pass

Use the archived CI and native-acceptance receipts for the exact checkout. The shared budget remains 384 h total / 380 h formal / 2 h resource / 2 h control, with cumulative unit and pair caps of 8 h and 16 h.

```powershell
$supplementExecution = (git rev-parse HEAD).Trim().Substring(0, 12)
$supplementCi = Join-Path $supplementProof "ci-verified-$supplementExecution.json"
$supplementAcceptance = Join-Path $supplementProof "extension-acceptance-$supplementExecution.json"
$supplementArgs = @(
  '-m', 'scripts.mlp_budget_extension_entry',
  '--config', 'configs/diagnostics_mlp_budget_extension_v1.json',
  '--base-config', 'configs/input_robustness_11_v2.json',
  '--base-root', $supplementBase,
  '--binding', $supplementBinding,
  '--data-root', (Join-Path $supplementWork 'input-robustness-data11'),
  '--output-root', $supplementOutput,
  '--budget-ledger', $supplementLedger,
  '--budget-handoff', (Join-Path $supplementProof 'budget-handoff-reviewed.json'),
  '--acceptance-record', $supplementAcceptance,
  '--ci-receipt', $supplementCi,
  '--execute'
)
& $supplementPython @supplementArgs
```

Dispatch order is dataset configuration order, seed 0–9, then the two conditions. Each unit inherits the four original MLP checkpoints, trains the additional twenty, selects by validation, reloads the selected checkpoint, evaluates test once, and saves a strictly validated record. The six graph architectures are reused from the matching base condition. Preparation, final validation and analysis also run under monitoring and accounting.

## Pause, emergency stop and recovery

The stable control directory is the output directory's sibling with `_control` appended to its name (`$supplementControl` above). It is read during preparation, model units and final validation/analysis, and may be created before the formal output exists. Do not create the formal output directory to request an early pause or stop. Existing markers inside `$supplementOutput` remain supported for compatibility.

For a normal pause, create `PAUSE_REQUESTED` in `$supplementControl`. An active full model unit finishes and is saved; no next unit starts. During preparation, preparation finishes and the immutable output manifest is initialized, then the entry returns paused with no model unit started; resume uses the normal `--resume` route. A pause during final validation/analysis allows that final stage to finish. For an emergency, create `EMERGENCY_STOP_REQUESTED` in the same stable directory. The guardian stops the owned worker and preserves its worker log, supervision receipt, failure and partial files. Preparation artifacts reside in the separately named `<output>_preparation_<attempt-prefix>` directory.

```powershell
New-Item -ItemType Directory -Force -Path $supplementControl
New-Item -ItemType File -Path (Join-Path $supplementControl 'PAUSE_REQUESTED')
# Emergency alternative:
# New-Item -ItemType File -Path (Join-Path $supplementControl 'EMERGENCY_STOP_REQUESTED')
```

Do not delete failed attempts, reset the ledger, change the grid or overwrite output. Preparation and finalization interruptions close as `external_interruption` only when the supervision receipt proves that the owned worker exited and no owned processes remain. Missing or incomplete exit evidence leaves the attempt open. Neither path reviews or retries an interruption automatically.

Before resuming, verify process termination and existing records, review any interruption, bind the new terminal ledger head in a newly named cumulative-handoff receipt, and archive acknowledged markers in the stable directory and any legacy output directory under timestamped names. Preserve every earlier receipt and point `--budget-handoff` to the new one. Then use the same command with `--resume`, the same output and the same incremental ledger. If an emergency interrupted preparation before a manifest was created, retain that preparation directory. After review, use a new output and new append-only ledger segment without `--resume`; the newly reviewed cumulative handoff must include the interrupted segment and all its usage. A stale lock or unresolved stop requires review; the runner does not remove it automatically. Completed valid units are reused. An incomplete unit restarts its twenty additional trials under the same cumulative MLP unit and paired-batch identities.

## Reviewed environmental-failure retry (two stages)

This follows `docs/protocol_amendment_mlp24_environmental_failure_retry_v1.md`. Stage 1 uses a version 2 cumulative handoff that binds the signed review on the failed segment, a new ledger path and `--authorize-only`. The entry runs the environment gate, creates and registers the new segment (this consumes the review once), registers the phase, records `failure_retry_authorized` and stops before any attempt. Stage 2 uses a newly bound handoff that includes the closed Stage 1 segment. It passes the same ledger path with `--resume --new-output-root` and a new output directory that must not exist. The phase must match the registered one: the entry binds a canonical digest of the environment preflight, not the timestamped artifact. The retry uses the attempt identity recorded in the authorization. A consumed authorization is never renewed, and earlier outputs are never reused or overwritten.

## Completion evidence

`complete.json` proves record completeness after strict configuration, data binding and checkpoint verification. `analysis.json` contains both input treatments × both MLP budgets, all 63 portfolios, the inherited nine strategies, the original h1 calibration and the two added metrics. `run_complete.json` binds both outputs. Worker supervision receipts and attempt summaries distinguish normal completion, pause and emergency stop. Ledger closure is additionally required: a completion marker by itself does not settle budget accounting.

Record execution SHA and later evidence-archive commits separately. Neither this runbook nor isolated fixture scores are new research results.
