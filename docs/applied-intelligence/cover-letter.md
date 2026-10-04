Dear Editors of Applied Intelligence,

Please consider “Diagnostic Utility Depends on the Model Portfolio: A Frozen Graph-vs-MLP Decision Benchmark” as an original research article.

The manuscript studies a practical model-selection question: when does a graph diagnostic help choose between trained graph and feature-only models? We evaluate diagnostic actions through prediction regret and abstention coverage, making the candidate portfolio, validation selection, fallback, and tuning allocation explicit.

The evidence consists of a pre-specified 11-dataset benchmark and separately identified post-hoc sensitivity analyses. Its central finding is that the measured comparison changes with the models selected by the graph action: an unchanged rule has lower mean regret than always-graph with GCN or GAT alone, but higher regret with the full six-architecture portfolio. The analysis also quantifies the contribution of fallback and the influence of dataset composition. A post-hoc retraining of all 11 datasets under two input parameterizations, with the multilayer-perceptron search expanded from four to 24 trials, changes the feature-only baseline and several regret values but does not reverse the main full-portfolio comparison; it aligns trial counts, while the graph action still chooses among six architectures. Retained records and reconstruction tools accompany the evaluation.

The extension also evaluates two published graph statistics through held-out-dataset calibration. Adjusted homophily attains lower descriptive regret than always-graph, separating the weaknesses of the fixed heuristics from the value of diagnostics more broadly. This empirical contribution is relevant to the journal's interests in neural learning and decision support: it provides an explicit procedure for assessing the cost of model-selection decisions. The manuscript reports the tested settings in which a diagnostic helps, the settings in which it loses accuracy, and the influence of the trained baselines.

The manuscript was previously submitted to TMLR and rejected without external review. It has not been published there. The present version has been rewritten for Applied Intelligence. Mengdan Xue is the sole author; the study received no funding and the author declares no competing interests.

Sincerely,

Mengdan Xue

Faculty of Computational Mathematics and Cybernetics

Lomonosov Moscow State University, Moscow, Russia

17326961775@163.com

---

Local preparation note — not part of the letter: this draft has not been sent (updated 2026-10-04 with the post-hoc 11-dataset and expanded-MLP evidence; the author confirmed on 2026-10-04 that the overlapping Neurocomputing submission is closed and that Applied Intelligence has not been submitted). Before submission, check the then-current status of concurrent submissions and incorporate only completed, validated supplementary experiments. The current rewrite uses the frozen benchmark and already reported bounded sensitivity results.
