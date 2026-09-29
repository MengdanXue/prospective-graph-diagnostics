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

For a normal pause, create `PAUSE_REQUESTED` in `$supplementOutput`. The active full model unit finishes and is saved; no next unit starts. For an emergency, create `EMERGENCY_STOP_REQUESTED` there. The guardian stops the owned worker and preserves its attempt, worker log, supervision receipt and all partial files.

```powershell
New-Item -ItemType File -Path (Join-Path $supplementOutput 'PAUSE_REQUESTED')
# Emergency alternative:
# New-Item -ItemType File -Path (Join-Path $supplementOutput 'EMERGENCY_STOP_REQUESTED')
```

Do not delete failed attempts, reset the ledger, change the grid or overwrite output. Before resuming, verify process termination and existing records, review any interruption, bind the new terminal ledger head in a newly named cumulative-handoff receipt, and archive the acknowledged control marker under a timestamped name. Preserve every earlier receipt and point `--budget-handoff` to the new one. Then use the same command with `--resume`, the same output and the same incremental ledger. A stale lock or unresolved stop requires review; the runner does not remove it automatically. Completed valid units are reused. An incomplete unit restarts its twenty additional trials under the same cumulative MLP unit and paired-batch identities.

## Completion evidence

`complete.json` proves record completeness after strict configuration, data binding and checkpoint verification. `analysis.json` contains both input treatments × both MLP budgets, all 63 portfolios, the inherited nine strategies, the original h1 calibration and the two added metrics. `run_complete.json` binds both outputs. Worker supervision receipts and attempt summaries distinguish normal completion, pause and emergency stop. Ledger closure is additionally required: a completion marker by itself does not settle budget accounting.

Record execution SHA and later evidence-archive commits separately. Neither this runbook nor isolated fixture scores are new research results.
