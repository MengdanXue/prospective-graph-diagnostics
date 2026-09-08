# Bounded graph training reproducibility audit

2026-09-08. This artifact-backed audit supersedes the earlier explanation based only on three immediate CUDA retrainings. The completed graph diagnostic remains a 420-record, 1,680-trial, validation-only result. Structural completeness does not establish deterministic reproducibility, and none of its records are replaced by this audit.

## Frozen scope and preserved evidence

The plan was frozen in [`configs/training_reproducibility_audit_v1.json`](../configs/training_reproducibility_audit_v1.json) before execution. [`scripts/audit_training_reproducibility.py`](../scripts/audit_training_reproducibility.py) launched 22 sequential, independent processes. All 22 succeeded, with zero test evaluations, in 206.5 seconds. No outcome-dependent workers were added.

All workers used the checksum-verified Roman-empire data and original split bindings. GPU trials retained hidden width 64, weight decay 0.0005, a 500-epoch maximum and patience 100, with accuracy/loss/earliest-epoch checkpoint selection. Trial 001 uses learning rate 0.01 and dropout 0.7; trial 003 uses learning rate 0.005 and dropout 0.7. CPU controls were explicitly capped at 12 epochs. Each worker evaluates one fixed trial; this audit does not repeat the original four-trial search.

| Scope | Independent workers |
|---|---:|
| LINKX, normalized input, seed 4: trials 001 and 003, ordinary CUDA, three repeats each | 6 |
| Same two cells, deterministic CUDA, three repeats each | 6 |
| LINKX, seed 4, trial 001: raw and centered-scaled input, deterministic CUDA, two repeats each | 4 |
| MLP, normalized input, seed 4, trial 001, deterministic CUDA | 2 |
| GAT, normalized input, seed 9, trial 001, deterministic CUDA | 2 |
| LINKX, normalized input, seed 4, trial 001, CPU, 12 epochs | 2 |

The deterministic intervention sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch, enables `torch.use_deterministic_algorithms(True)`, sets cuDNN deterministic mode, disables cuDNN benchmarking, and disables matmul/cuDNN TF32. Ordinary workers retain the runtime defaults and unset the CuBLAS workspace override. Both settings are recorded in every worker. The observed environment was Python 3.12.14, Torch 2.9.1+cu128, PyG 2.7.0, CUDA 12.8, and an NVIDIA GeForce RTX 5060 Laptop GPU, with four Torch CPU threads. PyG resolves from the sibling `work/venv` installation; Torch resolves from `work/venv-gpu`.

The checked-in [compact audit summary](../results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json) retains the frozen configuration, source hashes, execution manifest and group summaries. It omits the embedded `records` array to keep full training records outside ordinary Git, and binds the complete source summary by its SHA-256 and byte length. The immutable complete summary at `work/training-reproducibility-audit-v1/audit_summary.json` embeds all 22 worker records, complete epoch histories and early trajectory fingerprints, with SHA-256 `17af44a2535bfbcb6be62eb0b99ae9ae9a2b06c8ede7f4426805da30dad4475c`. That exact complete summary is retained in `training_reproducibility_audit_v1.zip`. The original output directory is beside the repository checkout. Its `complete.json` inventories every produced file by SHA-256. Each worker also preserves initial and selected `state_dict` files, including buffers, and the selected full-node logits. The complete directory is approximately 256 MB. Output creation is exclusive and never overwrites an existing directory.

Early epochs 0, 1, 2, 3, 4, 9 and 11 record train/validation logit hashes, gradient hashes and norms, post-step state hashes, and Python/NumPy/Torch CPU/CUDA RNG hashes. Each selected checkpoint is replayed five times, then loaded into a newly constructed model. Only train and validation partitions are scored. Four focused tests passed, including exact agreement of instrumented and production training histories and selected checkpoints for toy CPU MLP and LINKX runs, plus buffer-sensitive hashing and overwrite refusal.

## Observed results

| Model/input/seed/trial | Backend | Validation accuracy across repeats | Best epochs | Full histories and selected states identical? |
|---|---|---|---|---|
| LINKX normalized / 4 / 001 | Ordinary CUDA | 0.138404, 0.406589, 0.138625 | 3, 190, 4 | No |
| LINKX normalized / 4 / 003 | Ordinary CUDA | 0.399956, 0.426487, 0.283219 | 460, 176, 191 | No |
| LINKX normalized / 4 / 001 | Deterministic CUDA | 0.138404, 0.138404, 0.138404 | 4, 4, 4 | Yes |
| LINKX normalized / 4 / 003 | Deterministic CUDA | 0.428919, 0.428919, 0.428919 | 227, 227, 227 | Yes |
| LINKX raw / 4 / 001 | Deterministic CUDA | 0.600929, 0.600929 | 120, 120 | Yes |
| LINKX centered-scaled / 4 / 001 | Deterministic CUDA | 0.590537, 0.590537 | 127, 127 | Yes |
| MLP normalized / 4 / 001 | Deterministic CUDA | 0.151448, 0.151448 | 23, 23 | Yes |
| GAT normalized / 9 / 001 | Deterministic CUDA | 0.139730, 0.139730 | 492, 492 | Yes |
| LINKX normalized / 4 / 001 | CPU, 12 epochs | 0.138404, 0.138404 | 3, 3 | Yes |

The ordinary-CUDA accuracy ranges were 26.82 percentage points for trial 001 and 14.33 points for trial 003. Within each group, initial parameters, buffers and RNG fingerprints were identical. In both ordinary-CUDA groups, the first recorded epoch already had different train logits, gradients, post-step states and validation logits, while the recorded pre-training RNG states remained matched. All recorded early trajectory fingerprints, full histories and selected states matched within every deterministic-CUDA group and the short CPU group.

Fixed-checkpoint replay is a different phenomenon from retraining. Across this audit's ordinary-CUDA selected checkpoints, the largest observed absolute logit difference was `1.52587890625e-05`, with zero predicted-class disagreements across all nodes. Rebuilding the model and restoring its selected state also produced zero class disagreements. Deterministic-CUDA and CPU replays had zero logit differences. Thus the large accuracy spread observed here arises across different learned checkpoints, not from unstable predictions of a fixed checkpoint.

## What this resolves, and what remains

1. **The large LINKX spread is reproducible in the measured setting.** Independent-process repeats with matching initial weights, buffers, data and RNG state diverged during ordinary-CUDA training. An uninitialized-parameter explanation is not supported by the inspected implementation or these initial-state fingerprints: installed PyG `SparseLinear` initializes its empty allocations, LINKX resets all branches, and selected checkpoints include BatchNorm buffers.
2. **The deterministic backend intervention eliminated the observed repeat variation in these cells.** This is stronger evidence of backend-dependent training repeatability than the earlier informal replay. It does not isolate one CUDA kernel: the intervention bundles deterministic algorithms, CuBLAS configuration and cuDNN/TF32 settings. No claim is made that this identifies a universal mechanism for every graph model or seed.
3. **The limited direction check did not reverse the input-treatment result.** For seed 4 and the same trial 001 under deterministic CUDA, LINKX centered-scaled accuracy was 0.590537 versus normalized accuracy 0.138404, a +45.21-point difference; raw accuracy was 0.600929. This supports the local direction of the diagnostic. It neither reproduces the original four-trial selection nor establishes the stability or magnitude of the ten-seed aggregate contrast. MLP and GAT controls establish repeatability only for their listed cells.

The original graph results therefore remain descriptive for the completed run. LINKX effect magnitudes and individual selected trials require a reproducibility qualification; the present audit does not certify all 420 cells or prove the old control drift harmless. Its deterministic outcomes are supplementary audit evidence and must not replace the original observations. CPU/GPU numerical equality and equality across different random seeds are not required or claimed.

The frozen 770-record benchmark is a separate environment. A read-only check of all 770 archived record environments found CUDA on an RTX 3070 Laptop GPU, Torch 2.9.1+cu126/CUDA 12.6, Python 3.13.5, NumPy 2.2.6 and PyG 2.7.0. These records do not preserve the determinism, cuDNN or TF32 flags. The present audit does not establish that the original 770-record execution has the same repeat variability; it diagnoses the stated cells in the later local environment.

The instrumented runner reads tensors back to the CPU to hash them, which synchronizes CUDA and can alter scheduling. The results establish repeatability of the instrumented fixed recipe under the measured backend settings. A matched CPU test verifies that the instrumentation preserves the production stopping and selection logic, but this does not establish bitwise equivalence between instrumented and uninstrumented CUDA trajectories. The audit is deliberately bounded; no further experiments are implied by completion of this revision package.

中文结论：实际独立进程复核复现了普通 CUDA 下 LINKX 的大幅训练波动；初始参数、缓冲区和随机数状态相同，首轮输出与梯度已分歧。确定性后端设置使本次所测重复的完整训练历史和最终 checkpoint 逐位一致。固定 checkpoint 的预测没有发生类别变化。少量同种子、同 trial 对照仍显示居中缩放优于仅归一化，因此没有推翻所测单元的诊断方向；不能据此确认全部架构、全部种子或总体效果量。原 420 条结果保留，确定性结果只作为附加可靠性证据；原 770 条记录来自另一套 CUDA 硬件和运行环境，不在本次重复性结论的覆盖范围内。
