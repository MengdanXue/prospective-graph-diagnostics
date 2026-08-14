import json
import math
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_self_feature_mixing.py"


class SelfFeatureMixingTests(unittest.TestCase):
    def run_json(self, *args: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def formula(self, alpha: float, rho: float, degree: int, strength: float) -> dict:
        return self.run_json(
            "formula",
            "--alpha",
            str(alpha),
            "--rho",
            str(rho),
            "--degree",
            str(degree),
            "--strength",
            str(strength),
        )

    def test_alpha_zero_reduces_to_neighbor_only_ratio(self):
        rho, degree, strength = 0.4, 7, 1.8
        result = self.formula(0.0, rho, degree, strength)
        expected = rho**2 * degree / (1.0 + (1.0 - rho**2) * strength)
        self.assertAlmostEqual(result["ratio"], expected, places=12)

    def test_alpha_one_preserves_self_discriminability(self):
        result = self.formula(1.0, -0.7, 5, 3.0)
        self.assertAlmostEqual(result["ratio"], 1.0, places=12)
        self.assertAlmostEqual(result["label_mixture_coefficient"], 0.0, places=12)

    def test_perfect_edge_correlation_removes_label_mixture_variance(self):
        alpha, degree, strength = 0.3, 9, 2.0
        result = self.formula(alpha, 1.0, degree, strength)
        coefficient = alpha**2 + (1.0 - alpha) ** 2 / degree
        self.assertAlmostEqual(result["label_mixture_coefficient"], 0.0, places=12)
        self.assertAlmostEqual(result["denominator"], coefficient, places=12)
        self.assertAlmostEqual(result["ratio"], 1.0 / coefficient, places=12)

    def test_denominator_is_positive_on_declared_domain(self):
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
                result = self.formula(alpha, rho, 4, 1.0)
                self.assertGreater(result["denominator"], 0.0)
                self.assertTrue(math.isfinite(result["ratio"]))

    def test_simulated_conditional_moments_match_theory(self):
        result = self.run_json(
            "simulate",
            "--alpha",
            "0.35",
            "--rho",
            "0.4",
            "--degree",
            "8",
            "--strength",
            "1.5",
            "--samples",
            "120000",
            "--seed",
            "20260808",
        )
        self.assertLess(abs(result["empirical_mean_coefficient"] - result["theory_mean_coefficient"]), 0.01)
        self.assertLess(abs(result["empirical_variance"] - result["theory_variance"]), 0.02)


if __name__ == "__main__":
    unittest.main()
