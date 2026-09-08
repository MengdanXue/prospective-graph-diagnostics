# Graph diagnostic CUDA reproducibility audit

The completed 420-unit run used the pinned CUDA environment and fixed source commit `c48c34ee160f7f6ab62bfcaf10b5e90f40568466`. The run itself is complete and valid, with 420 records, 1,680 trials, zero test evaluations and no failure artifacts.

The old 280-record preprocessing controls share the same data hashes, feature hashes, device and package versions, but their graph metrics drift from the new run. This drift was investigated with a read-only repeated-cell probe using the current runner, the same dataset, split, condition, model, seed and four-trial recipe. Roman-empire, `normalize_features`, LINKX, seed 4 produced the following selected validation results across three immediate CUDA replays:

| Replay | Selected trial | Validation accuracy | Validation loss |
|---:|---|---:|---:|
| 1 | trial_001 | 0.3743091 | 4.0580144 |
| 2 | trial_000 | 0.1384037 | 2.8127158 |
| 3 | trial_001 | 0.3740880 | 6.2213402 |

The corresponding selected prediction counts also changed substantially. This probe wrote no result records and was not used to alter the fixed 420-cell scope. It shows that the large graph-control drift is compatible with nondeterministic CUDA message-passing and optimization trajectories, rather than a transformed-feature mismatch. The graph summary therefore reports control mismatch counts and magnitudes, while the paired contrasts remain descriptive for this one completed run. The MLP control drift is at floating-point scale, and all reconstructed feature hashes and split bindings agree.

This audit does not justify treating the graph contrasts as deterministic estimates or as a test/population claim. A future reproducibility study would need deterministic kernels where available, repeated independent workers, or a separately fixed CPU/controlled backend before using graph results for stronger claims.
