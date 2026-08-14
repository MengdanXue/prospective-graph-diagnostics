import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_route_a_claims.py"


class RouteAClaimAuditTests(unittest.TestCase):
    def run_audit(self, root: Path, main: str = "main.tex") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root), "--main", main],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_safe_active_sources_pass_and_commented_claims_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sections").mkdir()
            (root / "main.tex").write_text(
                "\\input{sections/body}\n% 32/36 is historical and inactive\n",
                encoding="utf-8",
            )
            (root / "sections" / "body.tex").write_text(
                "A scoped discriminability calculation.\n",
                encoding="utf-8",
            )
            (root / "unused.tex").write_text("7/9\n", encoding="utf-8")

            result = self.run_audit(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Route A claim audit passed", result.stdout)

    def test_dangerous_active_claims_fail_with_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sections").mkdir()
            (root / "main.tex").write_text("\\input{sections/body}\n", encoding="utf-8")
            (root / "sections" / "body.tex").write_text(
                "A frozen rule obtains 32/36 accuracy.\n"
                "This holds regardless of architecture.\n"
                "\\begin{proposition}[Structure Information Bound]\n",
                encoding="utf-8",
            )

            result = self.run_audit(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("sections/body.tex:1:historical_selector_score_32_36", result.stdout)
            self.assertIn("sections/body.tex:2:architecture_independent_claim", result.stdout)
            self.assertIn("sections/body.tex:3:structure_information_bound", result.stdout)

    def test_degree_preserving_language_requires_checksum_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(
                "A degree-preserving edge shuffle was used.\n",
                encoding="utf-8",
            )

            unsafe = self.run_audit(root)
            self.assertEqual(unsafe.returncode, 1, unsafe.stdout + unsafe.stderr)
            self.assertIn("unverified_degree_preserving_claim", unsafe.stdout)

            (root / "main.tex").write_text(
                "A degree-preserving edge shuffle was used; the degree-sequence checksum "
                "was identical before and after randomization.\n",
                encoding="utf-8",
            )
            safe = self.run_audit(root)
            self.assertEqual(safe.returncode, 0, safe.stdout + safe.stderr)

    def test_degree_preserving_language_accepts_a_verified_formal_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "results" / "diagnostic" / "route_a_degree_matched_v1" / "summary"
            summary.mkdir(parents=True)
            (summary / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "run_id": "route_a_degree_matched_v1",
                        "record_count": 30,
                        "datasets": {"Cora": {}, "CiteSeer": {}, "PubMed": {}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "main.tex").write_text(
                "A degree-preserving edge randomization was completed.\n",
                encoding="utf-8",
            )

            result = self.run_audit(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_false_information_and_absolute_gap_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(
                "Bayes error improvement is at most "
                "\\frac{I(Y; G \\mid X)}{\\log C}.\n"
                "\\mathcal{B}<5\\% \\Rightarrow |\\Delta|<5\\%.\n",
                encoding="utf-8",
            )

            result = self.run_audit(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("main.tex:1:false_conditional_mi_error_bound", result.stdout)
            self.assertIn("main.tex:2:headroom_absolute_gap_claim", result.stdout)

    def test_current_manuscript_reports_only_remaining_route_a_work(self):
        result = self.run_audit(ROOT, "main_neurocomputing.tex")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Route A claim audit passed", result.stdout)
        self.assertNotIn("historical_selector_score_32_36", result.stdout)
        self.assertNotIn("structure_information_bound", result.stdout)
        self.assertNotIn("false_conditional_mi_error_bound", result.stdout)
        self.assertNotIn("unverified_degree_preserving_claim", result.stdout)
        self.assertNotIn("architecture_independent_claim", result.stdout)


if __name__ == "__main__":
    unittest.main()
