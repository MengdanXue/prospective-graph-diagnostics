"""Check manuscript diagnostic numbers against the released result summaries."""

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "sections_tmlr/10_training_diagnostics.tex").read_text(encoding="utf-8")
RESULTS = ROOT / "results/diagnostic"


class TrainingDiagnosticManuscriptTests(unittest.TestCase):
    def test_mlp_table_matches_means_and_sample_standard_deviations(self):
        summary = json.loads((RESULTS / "posthoc_mlp_optimization_v1/analysis/mlp_optimization_diagnostic_summary.json").read_text(encoding="utf-8"))
        table = re.search(r"\\label\{tab:mlp_parameterization\}(.*?)\\end\{table\}", TEXT, re.S).group(1)
        rows = re.findall(r"^.+? & \$([\d.]+)\\pm([\d.]+)\$ & \$([\d.]+)\\pm([\d.]+)\$ \\\\", table, re.M)
        conditions = ["raw", "normalize_features", "normalize_scaled", "normalize_centered", "normalize_centered_scaled"]
        self.assertEqual(len(rows), len(conditions))
        for condition, row in zip(conditions, rows):
            expected = []
            for dataset in ["Roman-empire", "Amazon-ratings"]:
                aggregate = summary["aggregates"][dataset][condition]["wd_5e-4"]
                self.assertEqual(aggregate["n"], 10)
                accuracy = aggregate["validation"]["accuracy"]
                expected.extend(f"{100 * accuracy[key]:.2f}" for key in ["mean", "sample_sd"])
            self.assertEqual(list(row), expected, condition)

    def test_graph_table_matches_all_architectures_and_conditions(self):
        summary = json.loads((RESULTS / "posthoc_graph_parameterization_v1/analysis/graph_parameterization_diagnostic_summary.json").read_text(encoding="utf-8"))
        table = re.search(r"\\label\{tab:graph_parameterization\}(.*?)\\end\{table\}", TEXT, re.S).group(1)
        rows = re.findall(r"^([\w-]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\", table, re.M)
        self.assertEqual({row[0] for row in rows}, {"MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"})
        for model, *values in rows:
            expected = []
            for dataset in ["Roman-empire", "Amazon-ratings"]:
                for condition in ["raw", "normalize_features", "normalize_centered_scaled"]:
                    aggregate = summary["aggregates"][dataset][model][condition]
                    self.assertEqual(aggregate["n"], 10)
                    expected.append(f"{100 * aggregate['validation']['accuracy']['mean']:.2f}")
            self.assertEqual(values, expected, model)

    def test_repeatability_table_matches_the_frozen_independent_workers(self):
        summary = json.loads((RESULTS / "training_reproducibility_audit_v1/analysis/audit_summary.json").read_text(encoding="utf-8"))
        table = re.search(r"\\label\{tab:training_reproducibility\}(.*?)\\end\{table\}", TEXT, re.S).group(1)
        rows = re.findall(r"^(001|003) & (Default CUDA|Deterministic CUDA) & ([\d.]+)--([\d.]+) & (Yes|No) \\\\", table, re.M)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["worker_count"], 22)
        self.assertEqual(summary["success_count"], 22)
        self.assertEqual(summary["test_evaluations"], 0)
        self.assertNotIn("records", summary)
        self.assertEqual(summary["full_summary_sha256"], "17af44a2535bfbcb6be62eb0b99ae9ae9a2b06c8ede7f4426805da30dad4475c")
        backend_ids = {"Default CUDA": "cuda_default", "Deterministic CUDA": "cuda_deterministic"}
        for trial, backend, low, high, agreement in rows:
            group = next(g for g in summary["groups"] if g["model"] == "LINKX" and g["condition"] == "normalize_features" and g["trial_id"] == f"trial_{trial}" and g["backend"] == backend_ids[backend])
            self.assertEqual(group["success_count"], 3)
            self.assertEqual(low, f"{100 * min(group['validation_accuracy']):.2f}")
            self.assertEqual(high, f"{100 * max(group['validation_accuracy']):.2f}")
            self.assertEqual(agreement, "Yes" if group["identical_histories"] and group["identical_selected_states"] else "No")


if __name__ == "__main__":
    unittest.main()
