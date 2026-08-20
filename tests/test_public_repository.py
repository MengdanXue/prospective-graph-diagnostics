import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RESULTS = {
    "results/diagnostic/route_a_degree_matched_v1/summary/summary.json",
    "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json",
    "results/diagnostic/route_a_prospective_v2/analysis/prospective_regret_coverage.pdf",
    "results/diagnostic/route_a_prospective_v2/analysis/prospective_regret_coverage.png",
    "results/discriminability/route_a_grid_v1/summary/figure_manifest.json",
    "results/discriminability/route_a_grid_v1/summary/records.csv",
    "results/discriminability/route_a_grid_v1/summary/summary.json",
    "results/discriminability/route_a_grid_v1/summary/validation_figure.pdf",
    "results/discriminability/route_a_grid_v1/summary/validation_figure.png",
}
ACTIVE_SECTIONS = [
    "01_introduction",
    "02_related_work",
    "03_decision_protocol",
    "04_prospective_results",
    "05_edge_intervention",
    "06_fixed_degree_analysis",
    "07_discussion",
    "08_conclusion",
]


class PublicRepositoryTests(unittest.TestCase):
    def test_public_entry_documents_exist(self):
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / "CITATION.cff").is_file())
        self.assertTrue((ROOT / "LICENSE").is_file())

    def test_public_license_is_explicit_and_preserves_third_party_terms(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 Mengdan Xue", license_text)
        self.assertIn("released under the MIT License", readme)
        self.assertIn("not relicensed by this repository", readme)
        self.assertIn("license: MIT", citation)

    def test_internal_and_legacy_surfaces_are_absent(self):
        forbidden = (
            "code",
            "tables",
            "docs/plans",
            "docs/neurocomputing_cover_letter_draft.md",
            "docs/neurocomputing_submission_metadata.md",
            "docs/neurocomputing_rewrite_self_review_2026-08-13.md",
            "docs/route_a_adversarial_review.md",
            "docs/route_a_go_no_go_2026-08-13.md",
            "scripts/build_neurocomputing_submission.py",
            "tests/test_neurocomputing_submission_bundle.py",
        )
        present = []
        for path in forbidden:
            candidate = ROOT / path
            if candidate.is_dir():
                if any(item.is_file() for item in candidate.rglob("*")):
                    present.append(path)
            elif candidate.exists():
                present.append(path)
        self.assertEqual(present, [])

    def test_only_compact_results_are_published(self):
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "results").rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, ALLOWED_RESULTS)

    def test_all_published_json_is_valid(self):
        failures = []
        for path in ROOT.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual(failures, [])

    def test_latex_inputs_resolve(self):
        missing = []
        manuscript = (ROOT / "main_neurocomputing.tex").read_text(encoding="utf-8")
        for target in re.findall(r"\\input\{([^}]+)\}", manuscript):
            if not (ROOT / f"{target}.tex").is_file():
                missing.append(target)
        self.assertEqual(missing, [])

    def test_bibliography_contains_only_cited_entries(self):
        manuscript_paths = [
            ROOT / "main_neurocomputing.tex",
            *sorted((ROOT / "sections").glob("*.tex")),
        ]
        manuscript = "\n".join(
            path.read_text(encoding="utf-8") for path in manuscript_paths
        )
        cited = {
            key.strip()
            for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript)
            for key in group.split(",")
        }
        bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
        published = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
        self.assertEqual(published, cited)

    def test_bibliography_has_no_duplicate_titles(self):
        bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
        titles = [
            re.sub(r"[{}\s-]+", "", title).lower()
            for title in re.findall(
                r"^\s*title\s*=\s*\{([^}]*)\}", bibliography, re.MULTILINE
            )
        ]
        duplicates = sorted({title for title in titles if titles.count(title) > 1})
        self.assertEqual(duplicates, [])

    def test_manuscript_uses_neutral_ordered_section_names(self):
        manuscript = (ROOT / "main_neurocomputing.tex").read_text(encoding="utf-8")
        actual = re.findall(r"\\input\{sections/([^}]+)\}", manuscript)
        self.assertEqual(actual, ACTIVE_SECTIONS)
        section_files = {
            path.stem for path in (ROOT / "sections").glob("*.tex")
        }
        self.assertEqual(section_files, set(ACTIVE_SECTIONS))

    def test_bounded_ai_declaration_remains_visible(self):
        manuscript = (ROOT / "main_neurocomputing.tex").read_text(encoding="utf-8")
        protocol = (ROOT / "sections" / "03_decision_protocol.tex").read_text(encoding="utf-8")
        self.assertIn("Declaration of generative AI and AI-assisted technologies", manuscript)
        self.assertIn("OpenAI Codex", manuscript)
        self.assertIn("did not generate experimental data or make reporting decisions", protocol)

    def test_no_private_paths_or_informal_process_traces(self):
        forbidden = (
            "d:\\users\\",
            "documents\\毕业论文",
            "based on codex",
            "generated by gemini",
            "for codex:",
            "codex suggested",
            "3-ai review",
            "fsd-gnn",
            "fsd framework",
            "tkde submission",
        )
        hits = []
        this_file = Path(__file__).resolve()
        for suffix in ("*.py", "*.md", "*.tex", "*.json", "*.yml", "*.cff"):
            for path in ROOT.rglob(suffix):
                if path.resolve() == this_file:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                hits.extend(
                    f"{path.relative_to(ROOT)}: {phrase}"
                    for phrase in forbidden
                    if phrase in text
                )
        self.assertEqual(hits, [])

    def test_readme_states_the_bounded_result_and_verification_path(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("55.5%", readme)
        self.assertIn("80.9%", readme)
        self.assertIn("91.8%", readme)
        self.assertIn("python -m unittest discover -s tests -v", readme)
        self.assertIn("raw unit-level", readme.lower())
        self.assertIn(
            "diagnostic records whose label-dependent graph statistics use training labels only",
            readme,
        )
        self.assertNotIn("train-only diagnostic records", readme)

    def test_supporting_docs_use_the_precise_diagnostic_label_scope(self):
        for relative in (
            "docs/novelty_audit_negative_diagnostic_study_2026-08-13.md",
            "docs/protocol_amendment_prospective_v2.md",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("train-only diagnostics", text)
            self.assertNotIn("train-only heuristics", text)
            self.assertNotIn("train-only inputs", text)

    def test_readme_installs_into_the_created_virtual_environment(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            r".\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock.txt",
            readme,
        )
        self.assertIn(
            ".venv-analysis/bin/python -m pip install -r requirements-analysis.lock.txt",
            readme,
        )

    def test_readme_documents_release_to_summary_reconstruction(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        required = (
            "scripts/assemble_prospective_diagnostics.py",
            "experiments/evaluate_diagnostics.py",
            "scripts/summarize_degree_matched_benchmark.py",
            "configs/prospective_benchmark_v2.json",
            "prospective/records",
            "degree_matched/records",
        )
        for text in required:
            self.assertIn(text, readme)

    def test_tests_do_not_hardcode_a_windows_virtualenv(self):
        hits = []
        this_file = Path(__file__).resolve()
        for path in (ROOT / "tests").glob("test_*.py"):
            if path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8")
            if '".venv-analysis" / "Scripts" / "python.exe"' in text:
                hits.append(path.name)
        self.assertEqual(hits, [])

    def test_ci_cache_tracks_the_published_lockfile(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "cache-dependency-path: requirements-analysis.lock.txt",
            workflow,
        )

    def test_ci_runs_the_public_release_builder_tests(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("tests.test_public_release_builder", workflow)

    def test_ci_runs_the_full_protocol_suite_on_cpu(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("full-protocol-verification:", workflow)
        self.assertIn("requirements-ci.lock.txt", workflow)
        self.assertIn("python -m unittest discover -s tests -v", workflow)

    def test_execution_source_mapping_matches_public_files(self):
        mapping_path = ROOT / "docs" / "execution_source_mapping.json"
        self.assertTrue(mapping_path.is_file())
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        self.assertEqual(
            mapping["execution_source_commit"],
            "6d134b74da10c3ce808e660eb858d0bb47a3ab12",
        )
        self.assertEqual(
            mapping["public_release_commit"],
            "b97ae0ea5e6a5f8a53e853142cc50d1be9152c12",
        )
        self.assertEqual(len(mapping["files"]), 12)
        for relative, expected in mapping["files"].items():
            data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
            blob = hashlib.sha1(
                f"blob {len(data)}\0".encode("ascii") + data,
                usedforsecurity=False,
            ).hexdigest()
            self.assertEqual(blob, expected["git_blob_sha1"], relative)


if __name__ == "__main__":
    unittest.main()
