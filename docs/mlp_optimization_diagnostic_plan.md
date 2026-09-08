# MLP optimization diagnostic: design fixed before execution

Design date: 2026-09-08. Status: post hoc development analysis, motivated by the already observed preprocessing results. This is not a preregistered or independent confirmation of those results. Configuration: `configs/mlp_optimization_diagnostic_v1.json`.

## Question and scope

On Roman-empire and Amazon-ratings, does the selected MLP actually concentrate its predictions on the training majority class after PyG NormalizeFeatures? How much do input magnitude, common feature offset, and weight decay contribute under the existing training recipe?

The earlier accuracy coincidence with a majority baseline does not establish constant predictions. We therefore record selected-checkpoint prediction counts and per-class recall on train and validation partitions. We do not evaluate test predictions, compute test metrics, or select settings using new test results. The original test outcomes are already known; this remains a development diagnostic.

## Fixed design

Use the two pinned raw NPZ files, all ten original stratified splits, and the existing two-layer MLP. Keep hidden width 64, Adam, 500 maximum epochs, patience 100, and the four learning-rate/dropout trials from the paired preprocessing study. Select a checkpoint by highest validation accuracy, lowest validation loss, then earliest epoch; select a trial by highest validation accuracy, lowest loss, then trial ID. Preserve full trial histories. Do not change the frozen benchmark code.

Let X be raw float32 features and N the exact existing PyG NormalizeFeatures transform. Using **only training features**, compute normalized column means m, raw centered RMS sX, and normalized centered RMS sN in float64. Set a = sX/sN, rejecting nonfinite values or either RMS <= 1e-12. Cast the transformed output to float32. Apply the same fitted m and a to every node.

| Condition | Features |
|---|---|
| raw | X |
| normalize_features | N |
| normalize_scaled | a N |
| normalize_centered | N - m |
| normalize_centered_scaled | a (N - m) |

Cross all five conditions with Adam weight decay 5e-4 and 0. This produces 2 datasets x 10 seeds x 5 conditions x 2 decay settings = 200 selected records and 800 training trials. Each of the four affine versions of N preserves its information up to floating-point precision. The train-fitted transformations help distinguish sensitivity to the optimization parameterization from the information change caused by row normalization; they do not prove a unique mechanism.

Raw and normalized controls with decay 5e-4 are rerun in the same runtime as all other cells. These controls must be compared with the earlier validation records to identify accidental drift. A mismatch is reported and investigated, not silently replaced by previous records. Numerical nondeterminism is a possible explanation, not an automatic exemption.

## Records and integrity

Before running, commit this plan, the configuration, the runner, and its tests. Bind the manifest and each record to the source commit, canonical config digest, environment, raw file checksums, original split IDs, and transformed-feature digest. Save fitted mean/scale metadata and feature statistics. All writes are exclusive; resume requires identical provenance and valid existing records. The run reports exactly zero test evaluations.

For each selected checkpoint, record train/validation loss, accuracy, label counts, prediction counts, dominant prediction class and fraction, balanced accuracy, per-class recall (null if support is zero), and the accuracy of the training-majority constant classifier on that partition. Store the actual prediction counts so majority collapse can be verified directly rather than inferred from accuracy. Do not store or summarize test prediction statistics.

Acceptance requires exact scope, no duplicates, complete trial grids, consistent checkpoint/trial selection, finite metrics, matched raw data/splits, valid hashes, count totals equal partition sizes, and zero test-evaluation fields. The summarizer must fail before producing a report if these checks fail.

## Fixed descriptive comparisons

Report per-dataset means and sample standard deviations across the ten seeds, individual paired differences, their means/ranges, and positive/zero/negative counts. No population significance claim is planned.

1. At each decay value, compare normalized with raw, scaled with normalized, centered with normalized, and centered-scaled with each of scaled and centered.
2. At each feature condition, compare zero decay with 5e-4.
3. Report the difference between the scale effect at zero and nonzero decay, for centered and uncentered inputs. This describes interaction within the tested optimizer recipe.
4. Report how many selected models predict the training majority class for every validation example, and the full range of dominant prediction fractions. Near-constant predictions remain a continuous measurement; no outcome-dependent cutoff will be introduced.

Scale recovery supports optimization sensitivity of this MLP under this recipe. Recovery after removing decay supports sensitivity to that setting. Additional centered-scaled recovery supports offset/scale interaction. Failure to recover does not establish information loss: learning rate, activation, optimization path, and training duration are still limited. Changes in validation-selected performance do not establish test generalization or explain graph-model results.

## Next decision

After source/data audit and descriptive review, choose whether a separately frozen next-stage graph-model control is justified. Expansion to other models or datasets requires a new dated design and budget; do not grow this experiment after viewing intermediate outcomes. Keep the original benchmark and its manuscript claims unchanged during this diagnostic stage.

## Commands

Use the same experiment environment and device for all 200 cells. Keep outputs outside the tracked source tree so the source remains clean for an exact resume.

```bash
python scripts/validate_preprocessing_records.py --run-root PATH_TO_PREPROCESSING_RUN --data-root PATH_TO_NPZ
python scripts/run_mlp_optimization_diagnostic.py --data-root PATH_TO_NPZ --output-root PATH_TO_NEW_MLP_RUN --device cuda
python scripts/summarize_mlp_optimization_diagnostic.py --run-root PATH_TO_NEW_MLP_RUN --data-root PATH_TO_NPZ --preprocessing-root PATH_TO_PREPROCESSING_RUN --output-root PATH_TO_NEW_SUMMARY
```

The original preprocessing record audit is stored in `results/diagnostic/route_a_prospective_v2/analysis/preprocessing_record_audit.json`. All 280 records passed scope, source/config/environment, selection-history and test-once checks. Independent input reconstruction matched all four raw/normalized feature hashes, all 20 split IDs and the original graph statistics. Recomputing the previous summary reproduced every existing scientific value exactly.
