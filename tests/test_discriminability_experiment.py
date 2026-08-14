import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
EXPERIMENT = ROOT / "experiments" / "validate_discriminability_formula.py"
SUMMARIZER = ROOT / "scripts" / "summarize_discriminability.py"


def tiny_config(path: Path, *, invalid_degree: bool = False) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": "tiny_v1",
        "description": "Tiny deterministic test grid.",
        "model": {
            "dimension": 2,
            "covariance": "identity",
            "samples_per_class": 600,
            "simulation_method": "direct_binomial_label_sum_and_gaussian_noise",
        },
        "grid": {
            "d": [0 if invalid_degree else 3],
            "rho": [0.0],
            "s": [0.5],
            "alpha": [0.0, 0.5],
        },
        "seeds": [17],
        "estimands": {
            "primary": "absolute_relative_error_empirical_kappa_vs_exact_kappa_alpha",
            "primary_zero_policy": "when abs(exact_kappa_alpha) <= 1e-12, relative error is null and absolute error is reported",
            "secondary": "signed_and_absolute_error_of_rho_squared_d_for_alpha_zero_only",
        },
        "summary": {
            "estimator": "median",
            "interval": "95_percentile_bootstrap",
            "bootstrap_samples": 200,
            "bootstrap_seed": 1234,
        },
        "output": {
            "root": "results/discriminability",
            "record_layout": "<run_id>/<config_id>/seed_<seed>.json",
            "overwrite": False,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiscriminabilityExperimentTests(unittest.TestCase):
    def run_experiment(self, config: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(PYTHON),
                str(EXPERIMENT),
                "--config",
                str(config),
                "--output-root",
                str(output),
                "--git-commit",
                "test-commit",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_tiny_grid_writes_complete_immutable_per_seed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "grid.json"
            output = tmp_path / "out"
            tiny_config(config)

            result = self.run_experiment(config, output)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = sorted(output.rglob("seed_*.json"))
            self.assertEqual(len(records), 2)
            self.assertFalse(any(output.rglob("summary.*")))

            required = {
                "schema_version",
                "run_id",
                "config",
                "seed",
                "derived_rng_seed",
                "git_commit",
                "environment",
                "sample_count",
                "empirical_moments",
                "predictions",
                "errors",
                "status",
                "exception",
            }
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in records]
            for payload in payloads:
                self.assertTrue(required.issubset(payload))
                self.assertEqual(payload["status"], "success")
                self.assertIsNone(payload["exception"])
                self.assertEqual(payload["git_commit"], "test-commit")
                self.assertEqual(payload["environment"]["numpy"], np.__version__)
                self.assertEqual(payload["environment"]["matplotlib"], matplotlib.__version__)

            by_alpha = {payload["config"]["alpha"]: payload for payload in payloads}
            self.assertEqual(by_alpha[0.0]["predictions"]["exact_kappa_alpha"], 0.0)
            self.assertIsNone(by_alpha[0.0]["errors"]["exact_relative_error"])
            self.assertIsNotNone(by_alpha[0.0]["errors"]["exact_absolute_error"])
            self.assertIsNotNone(by_alpha[0.0]["predictions"]["rho_squared_d"])
            self.assertIsNone(by_alpha[0.5]["predictions"]["rho_squared_d"])
            self.assertIsNone(by_alpha[0.5]["errors"]["approximation_signed_error"])

            before = {path.relative_to(output): sha256(path) for path in records}
            repeated = self.run_experiment(config, output)
            after = {path.relative_to(output): sha256(path) for path in records}
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("refusing to overwrite", (repeated.stdout + repeated.stderr).lower())
            self.assertEqual(before, after)

    def test_repeated_runs_in_separate_directories_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "grid.json"
            tiny_config(config)
            out_a = tmp_path / "a"
            out_b = tmp_path / "b"

            first = self.run_experiment(config, out_a)
            second = self.run_experiment(config, out_b)

            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            files_a = sorted(path.relative_to(out_a) for path in out_a.rglob("seed_*.json"))
            files_b = sorted(path.relative_to(out_b) for path in out_b.rglob("seed_*.json"))
            self.assertEqual(files_a, files_b)
            for relative in files_a:
                self.assertEqual((out_a / relative).read_bytes(), (out_b / relative).read_bytes())

    def test_invalid_configuration_is_preserved_as_an_error_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "invalid.json"
            output = tmp_path / "out"
            tiny_config(config, invalid_degree=True)

            result = self.run_experiment(config, output)

            self.assertNotEqual(result.returncode, 0)
            records = sorted(output.rglob("seed_*.json"))
            self.assertEqual(len(records), 2)
            for record in records:
                payload = json.loads(record.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "error")
                self.assertEqual(payload["exception"]["type"], "ValueError")
                self.assertIn("d must be", payload["exception"]["message"])

    def test_summarizer_writes_auditable_statistics_data_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "grid.json"
            output = tmp_path / "out"
            summary_dir = tmp_path / "summary"
            tiny_config(config)
            specification = json.loads(config.read_text(encoding="utf-8"))
            specification["grid"]["alpha"] = [0.0, 0.25, 0.5, 0.75, 1.0]
            config.write_text(json.dumps(specification, indent=2), encoding="utf-8")
            experiment = self.run_experiment(config, output)
            self.assertEqual(experiment.returncode, 0, experiment.stdout + experiment.stderr)

            result = subprocess.run(
                [
                    str(PYTHON),
                    str(SUMMARIZER),
                    "--records-root",
                    str(output),
                    "--config",
                    str(config),
                    "--output-dir",
                    str(summary_dir),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("UserWarning", result.stderr)
            expected = {
                "summary.json",
                "records.csv",
                "validation_figure.pdf",
                "validation_figure.png",
                "figure_manifest.json",
            }
            self.assertEqual(expected, {path.name for path in summary_dir.iterdir()})
            summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["record_counts"]["success"], 5)
            self.assertEqual(summary["record_counts"]["primary_relative_error_defined"], 4)
            self.assertEqual(summary["record_counts"]["approximation_defined"], 1)
            self.assertEqual(summary["uncertainty"]["method"], "95_percentile_bootstrap")
            manifest = json.loads(
                (summary_dir / "figure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["missing_data"], "Exact-zero relative errors are omitted and counted.")
            self.assertIn("95% percentile bootstrap", manifest["uncertainty"])


if __name__ == "__main__":
    unittest.main()
