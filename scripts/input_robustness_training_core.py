"""Formal execution adapter around the frozen prospective training primitives.

The public benchmark runner remains byte-for-byte frozen.  This module calls
its private trial trainer/model builder and adds the formal persistence steps
required by the guarded record contract.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from experiments.run_prospective_benchmark import (
    _accuracy, _train_trial, seed_everything, select_trial,
)
from experiments.prospective_models import build_model
from scripts.input_robustness_checkpoint_store import load_checkpoint, save_checkpoint


def run_model_unit_with_checkpoints(*, checkpoint_dir: Path, run_id: str,
                                    dataset: str, model_id: str, seed: int,
                                    split_id: str, x: torch.Tensor, y: torch.Tensor,
                                    edge_index: torch.Tensor, train_indices: torch.Tensor,
                                    validation_indices: torch.Tensor, test_indices: torch.Tensor,
                                    training: dict[str, Any], source_commit: str,
                                    environment: dict[str, Any], data_provenance: dict[str, Any],
                                    device: torch.device, h2_adjacencies: tuple[torch.Tensor, torch.Tensor] | None = None,
                                    config_sha256: str = "test-only", frozen_config: dict[str, Any] | None = None,
                                    **_: Any) -> dict[str, Any]:
    """Run four frozen trials, persist all states, reload the selected state, test once."""
    started = time.perf_counter()
    checkpoint_dir = Path(checkpoint_dir)
    trial_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for index, trial in enumerate(training["trials"]):
        trial_id = f"trial_{index:03d}"
        row, state = _train_trial(
            model_id=model_id, seed=seed, trial_id=trial_id, trial=trial, x=x, y=y,
            edge_index=edge_index, train_indices=train_indices,
            validation_indices=validation_indices,
            hidden_channels=int(training["hidden_channels"]), max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]), weight_decay=float(training["weight_decay"]),
            device=device, h2_adjacencies=h2_adjacencies,
        )
        trial_rows.append(row)
        manifests.append(save_checkpoint(checkpoint_dir, trial_id, state))
    selected = select_trial(trial_rows)
    selected_configuration = selected["configuration"]
    selected_state = load_checkpoint(checkpoint_dir / f"{selected['trial_id']}.pt")
    seed_everything(seed)
    model = build_model(
        model_id, num_nodes=x.size(0), in_channels=x.size(1),
        hidden_channels=int(training["hidden_channels"]),
        out_channels=int(y.max().item()) + 1,
        dropout=float(selected_configuration["dropout"]), edge_index=edge_index,
        h2_adjacencies=h2_adjacencies,
    ).to(device)
    model.load_state_dict(selected_state)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device))
        test_accuracy = _accuracy(logits, y.to(device), test_indices.to(device))
    return {
        "schema_version": "1.0", "run_id": run_id, "status": "success",
        "dataset": dataset, "model": model_id,
        "family": "mlp" if model_id == "MLP" else "graph", "seed": int(seed), "split_id": split_id,
        "validation_accuracy": selected["validation_accuracy"], "validation_loss": selected["validation_loss"],
        "test_accuracy": test_accuracy, "test_evaluations_after_selection": 1,
        "test_evaluation": {"selected_trial_id": selected["trial_id"], "count": 1},
        "selected_trial_id": selected["trial_id"], "trials": trial_rows,
        "training_configuration": dict(training), "source_commit": source_commit,
        "environment": environment, "data_provenance": data_provenance,
        "config_sha256": config_sha256, "frozen_config": frozen_config,
        "checkpoint_mode": "original_full_state_copy", "checkpoint_manifest": manifests,
        "checkpoint_complete": len(manifests) == 4, "record_complete": len(manifests) == 4,
        "validation_evaluations": len(trial_rows),
        "duration_seconds": time.perf_counter() - started,
    }

