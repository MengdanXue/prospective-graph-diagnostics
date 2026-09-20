#!/usr/bin/env python3
"""Check supplied revision archives and rebuild summaries without model training.

Run from a clean clone of the delivered source.bundle. With --data-root, also
reconstruct preprocessing, MLP and graph summaries using the raw NPZ inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_training_reproducibility import summarize_records
from scripts.verify_diagnostic_artifacts import ARCHIVES, file_sha256, verify_archive


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(command, cwd, log):
    print(f"Running {log.name}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    require(process.returncode == 0, f"command failed; inspect {log}")


def compare_preprocessing_summary(actual, expected):
    # Later validators add audit metadata to this older published summary.
    # Compare every original field exactly and retain the new validation in
    # the verification report instead of rewriting the historical result.
    comparable = json.loads(json.dumps(actual))
    validation = comparable["source_run"].get("validation")
    if "validation" not in expected["source_run"]:
        require(isinstance(validation, dict) and validation.get("status") == "validated"
                and validation.get("record_count") == 280 and validation.get("raw_npz_hashes_checked") is True,
                "missing preprocessing reconstruction validation")
        comparable["source_run"].pop("validation")
    require(comparable == expected, "preprocessing summary differs from tracked result")
    return validation


def compare_mlp_summary(actual, expected, data_root):
    comparable = json.loads(json.dumps(actual))
    require(comparable["transform_reconstruction"]["data_root"] == str(data_root),
            "MLP reconstruction used an unexpected data directory")
    relocation = {"original": expected["transform_reconstruction"]["data_root"], "verified": str(data_root)}
    comparable["transform_reconstruction"]["data_root"] = relocation["original"]
    require(comparable == expected, "MLP summary differs from tracked result")
    return relocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = [verify_archive(args.assets_dir / name) for name in ARCHIVES]
    roots = []
    for name, folder in zip(ARCHIVES, ("graph_archive", "controls_archive", "audit_archive")):
        target = output / folder
        with zipfile.ZipFile(args.assets_dir / name) as archive:
            # Historical commit snapshots can exceed Windows' legacy 260-char
            # path limit even when the handoff directory itself is ordinary.
            extraction_path = str(target)
            if os.name == "nt" and not extraction_path.startswith("\\\\?\\"):
                extraction_path = "\\\\?\\" + extraction_path
            archive.extractall(extraction_path)  # Entry names and digests verified above.
        roots.append(target)
    graph, controls, audit_root = roots[0] / "graph", roots[1], roots[2] / "audit"
    for prefix, count in ((graph, 420), (controls / "mlp", 200), (controls / "preprocessing", 280)):
        require(len(list((prefix / "records").rglob("*.json"))) == count, f"record count mismatch: {prefix}")
    manifest = load(audit_root / "run_manifest.json")
    summary = load(audit_root / "audit_summary.json")
    records = [load(audit_root / "workers" / w["worker_id"] / "record.json") for w in manifest["workers"]]
    rebuilt = summarize_records(manifest["config"], records)
    for key, value in rebuilt.items():
        require(value == summary[key], f"audit summary mismatch: {key}")
    require(summary["records"] == records, "audit embedded records differ from worker files")
    require(rebuilt["worker_count"] == rebuilt["success_count"] == 22, "audit worker scope mismatch")
    complete = load(audit_root / "complete.json")
    for name, expected in complete["files_sha256"].items():
        require(file_sha256(audit_root / name.replace("\\", "/")) == expected, f"audit completion hash mismatch: {name}")
    compact = load(ROOT / "results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json")
    require(file_sha256(audit_root / "audit_summary.json") == compact["full_summary_sha256"], "audit full-summary binding mismatch")
    report = {"status": "verified", "archives": reports, "records": {"graph": 420, "mlp": 200, "preprocessing": 280},
              "audit_workers_reaggregated": len(records), "audit_completion_files_verified": len(complete["files_sha256"]),
              "feature_reconstruction": "not requested", "training_performed": False}
    if args.data_root:
        data = args.data_root.resolve()
        python = sys.executable
        pre_out = output / "preprocessing_summary.json"
        run([python, "scripts/summarize_preprocessing_sensitivity.py", "--root", str(controls / "preprocessing"),
             "--data-root", str(data), "--output", str(pre_out)], ROOT, output / "preprocessing.log")
        expected_pre = ROOT / "results/diagnostic/route_a_prospective_v2/analysis/preprocessing_sensitivity.json"
        report["preprocessing_validation"] = compare_preprocessing_summary(load(pre_out), load(expected_pre))
        # The MLP summarizer verifies the executable bytes from its own run.
        # A historical checkout avoids substituting the later graph helper.
        mlp_source = output / "mlp_source"
        mlp_commit = load(controls / "mlp/run_manifest.json")["source_commit"]
        run(["git", "clone", "--no-hardlinks", "--no-checkout", "--config", "core.autocrlf=false", "--config", "core.longpaths=true",
             str(ROOT), str(mlp_source)], ROOT, output / "mlp_clone.log")
        run(["git", "checkout", "--detach", mlp_commit], mlp_source, output / "mlp_checkout.log")
        # The analysis was added after the execution commit. Use the delivered
        # analysis with the untouched historical runner it authenticates.
        mlp_analysis = ROOT / "scripts/summarize_mlp_optimization_diagnostic.py"
        shutil.copyfile(mlp_analysis, mlp_source / "scripts" / mlp_analysis.name)
        report["mlp_analysis_sha256"] = file_sha256(mlp_analysis)
        report["mlp_runner_source_commit"] = mlp_commit
        mlp_out = output / "mlp_summary"
        run([python, "scripts/summarize_mlp_optimization_diagnostic.py", "--run-root", str(controls / "mlp"),
             "--data-root", str(data), "--preprocessing-root", str(controls / "preprocessing"),
             "--output-root", str(mlp_out)], mlp_source, output / "mlp.log")
        expected_mlp = ROOT / "results/diagnostic/posthoc_mlp_optimization_v1/analysis/mlp_optimization_diagnostic_summary.json"
        report["mlp_data_directory_relocation"] = compare_mlp_summary(load(mlp_out / expected_mlp.name), load(expected_mlp), data)
        graph_out = output / "graph_summary"
        run([python, "scripts/summarize_graph_parameterization_diagnostic.py", "--run-root", str(graph),
             "--data-root", str(data), "--preprocessing-root", str(controls / "preprocessing"),
             "--mlp-root", str(controls / "mlp"), "--output-root", str(graph_out)], ROOT, output / "graph.log")
        expected_graph = ROOT / "results/diagnostic/posthoc_graph_parameterization_v1/analysis/graph_parameterization_diagnostic_summary.json"
        require(load(graph_out / expected_graph.name) == load(expected_graph), "graph summary differs from tracked result")
        require(load(graph_out / expected_graph.name) == load(graph / "summary" / expected_graph.name), "graph archive summary differs")
        report["feature_reconstruction"] = "raw checksums, transforms, partitions, and all three summaries verified"
    with (output / "verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
