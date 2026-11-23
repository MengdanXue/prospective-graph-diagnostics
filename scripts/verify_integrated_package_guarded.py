"""Guard one journal-reader reconstruction with the already accepted controller.

Default is read-only preparation. This has no training or arbitrary-command
interface. The externally retained package-manifest digest binds the reader.
The accepted execution checkout is read only; a fresh incremental control
segment inherits every prior charge through the existing handoff implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

CONTROLLER_SHA = "51f168217e5b55fc190dd3fd885878408df43fe5"
OPERATION = "journal_reader_reconstruction"
READER = "reconstruct_integrated_analysis.py"
VERIFIER = "verify_integrated_supplement.py"
ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fsha(path, poll=None):
    with Path(path).open("rb") as stream:
        result, previous = hashlib.sha256(), time.monotonic()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
            if poll is not None and time.monotonic() - previous >= 0.25:
                poll()
                previous = time.monotonic()
        return result.hexdigest()


def source_identity(root, expected=None):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(expected is None or head == expected, "wrong controller execution source")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
            "execution requires a clean committed checkout")
    return head


def package_binding(root, expected_manifest_sha256, poll=None):
    """Verify bytes before importing or launching any package-supplied code."""
    root = Path(root).resolve()
    manifest = root / "PACKAGE_MANIFEST.json"
    require(len(expected_manifest_sha256) == 64 and fsha(manifest, poll) == expected_manifest_sha256,
            "package manifest differs from external binding")
    files = json.loads(manifest.read_bytes())["files"]
    require(all(name in files for name in (READER, VERIFIER)), "package reader/verifier missing")
    for name, entry in files.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), "unsafe or missing package member")
        require(path.stat().st_size == entry["bytes"] and fsha(path, poll) == entry["sha256"],
                "package file hash differs: " + name)
        if poll is not None:
            poll()
    # Extra Python modules could shadow verified imports even with a valid manifest.
    require(all(path.relative_to(root).as_posix() in files for path in root.rglob("*.py")),
            "unlisted executable Python member")
    require(not any(path.suffix.lower() in {".pyc", ".pyo", ".pyd", ".so"} for path in root.rglob("*")),
            "compiled Python import can bypass source bindings")
    return {"manifest_sha256": expected_manifest_sha256,
            "reader_sha256": files[READER]["sha256"], "verifier_sha256": files[VERIFIER]["sha256"],
            "package_files": len(files)}


def reader_command(package_root, output):
    return [sys.executable, "-B", str(Path(package_root).resolve() / READER), "--output", str(Path(output).resolve())]


def make_controller(api, package_root, output, binding):
    """Only the fixed child changes; accounting, poll, power and stop stay inherited."""
    class ReaderController(api.ExtensionExecutionController):
        def _child(self, request, *, attempt_id, cap_seconds=7200):
            require(request == {"operation": OPERATION}, "unsupported reader operation or arguments")
            require(fsha(package_root / "PACKAGE_MANIFEST.json") == binding["manifest_sha256"]
                    and fsha(package_root / READER) == binding["reader_sha256"]
                    and fsha(package_root / VERIFIER) == binding["verifier_sha256"],
                    "reader or manifest replaced after preparation")
            workers = self.writer.root / "workers"
            workers.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            supervised = api.supervise_process(reader_command(package_root, output),
                cwd=package_root, environment=environment, log_path=workers / f"{attempt_id}.log",
                worker_cap_seconds=min(float(cap_seconds), 7200), poll_seconds=0.25,
                on_poll=self._supervisor_poll)
            api._exclusive_json(workers / f"{attempt_id}.supervision.json", supervised)
            if supervised.get("stop_reasons"):
                raise api.FormalEmergencyStop("journal reader guardian stopped worker: " + "; ".join(supervised["stop_reasons"]))
            require(supervised.get("returncode") == 0 and not supervised.get("owned_processes_remaining")
                    and output.is_file(), "reader did not complete with owned workers closed")
            require(package_binding(package_root, binding["manifest_sha256"], self._monitor) == binding,
                    "package binding changed during reconstruction")
            return {"status": "reconstructed_exactly", "operation": OPERATION, **binding,
                    "output": str(output), "output_sha256": fsha(output, self._monitor), "training_started": False}
    return ReaderController


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("controller-root", "package-root", "handoff", "ledger-root", "output-root",
                 "controller-ci-receipt", "wrapper-ci-receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--package-manifest-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    controller = args.controller_root.resolve()
    source_identity(controller, CONTROLLER_SHA)
    binding = package_binding(args.package_root, args.package_manifest_sha256)
    require(not args.output_root.exists() and not args.ledger_root.exists(),
            "preserve previous output and ledger; choose fresh paths")
    require(not args.output_root.resolve().is_relative_to(args.package_root.resolve()),
            "outputs must remain outside the immutable reader package")
    plan = {"operation": OPERATION, "budget_group": "control", "controller_source": CONTROLLER_SHA,
            **binding, "training_started": False}
    if not args.execute:
        print(json.dumps({**plan, "status": "preparation_only", "launch_enabled": False}))
        return 0
    wrapper_sha = source_identity(ROOT)
    # This process has imported only standard-library modules; resolve all accepted
    # controller dependencies from its fixed clean checkout, never this manuscript.
    sys.path.insert(0, str(controller))
    if "scripts" in sys.modules:
        sys.modules["scripts"].__path__ = [str(controller / "scripts")]
    api = importlib.import_module("scripts.mlp_budget_extension_entry")
    h = importlib.import_module("scripts.supplement_budget_handoff")
    require(Path(api.__file__).resolve() == controller / "scripts/mlp_budget_extension_entry.py",
            "controller import shadowed")
    ci = api.strict_ci_gate(args.controller_ci_receipt, CONTROLLER_SHA)
    api.strict_ci_gate(args.wrapper_ci_receipt, wrapper_sha)
    config = api.read_json(controller / "configs/diagnostics_mlp_budget_extension_v1.json")
    with h.open_inherited_ledger(args.handoff, args.ledger_root, base_root=controller.parent,
            expected_config_sha=api.digest(config), expected_source_commit=CONTROLLER_SHA,
            required_inventory_roots=config["budget"]["inventory_roots"]) as ledger:
        ledger.register_phase(OPERATION, provenance={"source_commit": CONTROLLER_SHA,
            "config_sha256": api.digest(config), "data_binding_sha256": fsha(controller /
                "results/diagnostic/posthoc_input_robustness_11_v1/preflight/data_binding.json"),
            "source_files": {"journal_wrapper": fsha(Path(__file__)),
                             "reader": binding["reader_sha256"], "verifier": binding["verifier_sha256"]},
            "environment": {"python": sys.executable, "wrapper_commit": wrapper_sha, **binding}},
            gate_receipt={"ci_receipt_file_sha256": fsha(args.controller_ci_receipt),
                          "review_record_sha256": fsha(args.handoff), "ci_receipt": ci,
                          "wrapper_ci_receipt_sha256": fsha(args.wrapper_ci_receipt)})
        output = args.output_root.resolve() / "reconstructed-analysis.json"
        cls = make_controller(api, args.package_root.resolve(), output, binding)
        runner = cls(writer=SimpleNamespace(root=args.output_root.resolve(), config=config, synthetic=False),
                     ledger=ledger, phase_id=OPERATION)
        attempt = api.attempt_prefix(ledger) + "_journal_reader"
        result = runner.run_preparation({"operation": OPERATION}, attempt_id=attempt, activity_id=attempt)
        api._exclusive_json(args.output_root / "execution.json", {"result": result, **plan,
            "wrapper_source": wrapper_sha, "monitor_gap_seconds": 5, "monitor_samples": runner.monitor_samples,
            "normal_pause_observed": runner.pause_requested, "power_request": runner.power_request.snapshot(),
            "power_watcher": runner.power_watcher.snapshot(), "ledger": ledger.snapshot()})
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
