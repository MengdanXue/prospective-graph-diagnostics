"""Exclusive full-state checkpoint persistence for formal model units."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointStoreError(ValueError):
    """A checkpoint could not be written, loaded, or verified."""


def state_sha256(state: Mapping[str, Any]) -> str:
    """Hash state keys and tensor values independently of torch's pickle bytes.

    Sparse buffers are fingerprinted from their canonical sparse components;
    they are never expanded to a dense ``N x N`` tensor.
    """
    if not isinstance(state, Mapping) or not state:
        raise CheckpointStoreError("checkpoint state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise CheckpointStoreError("checkpoint state must contain tensor values")
        tensor = value.detach().cpu()
        layout = str(tensor.layout)
        components: list[tuple[str, torch.Tensor]]
        if tensor.layout == torch.strided:
            components = [("dense", tensor.contiguous())]
        elif tensor.layout == torch.sparse_coo:
            canonical = tensor.coalesce()
            components = [("indices", canonical.indices().contiguous()),
                          ("values", canonical.values().contiguous())]
        elif tensor.layout in (torch.sparse_csr, torch.sparse_csc,
                               torch.sparse_bsr, torch.sparse_bsc):
            # Conversion between sparse layouts preserves sparsity and lets
            # COO coalesce provide a canonical index/value ordering without
            # allocating a dense matrix.
            canonical = tensor.to_sparse_coo().coalesce()
            components = [("indices", canonical.indices().contiguous()),
                          ("values", canonical.values().contiguous())]
        else:
            raise CheckpointStoreError(f"unsupported checkpoint tensor layout: {layout}")
        metadata = json.dumps(
            {"name": name, "layout": layout, "shape": list(tensor.shape),
             "component_names": [label for label, _ in components]},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        for label, component in components:
            component_meta = json.dumps(
                {"label": label, "dtype": str(component.dtype),
                 "shape": list(component.shape)},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            raw = component.numpy().tobytes()
            digest.update(len(component_meta).to_bytes(8, "big"))
            digest.update(component_meta)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def save_checkpoint(root: Path, trial_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Write one full state exactly once and return its immutable manifest row."""
    if not isinstance(trial_id, str) or not trial_id:
        raise CheckpointStoreError("trial_id is required")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{trial_id}.pt"
    state_copy = {name: value.detach().cpu().clone() for name, value in state.items()}
    state_hash = state_sha256(state_copy)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CheckpointStoreError(f"checkpoint already exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            torch.save(state_copy, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    raw = path.read_bytes()
    return {
        "trial_id": trial_id,
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "state_sha256": state_hash,
        "bytes": len(raw),
        "saved": True,
    }


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointStoreError(f"checkpoint file is missing: {path}")
    try:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before weights_only
            loaded = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CheckpointStoreError(f"checkpoint cannot be loaded: {path}: {exc}") from exc
    if not isinstance(loaded, Mapping) or not loaded:
        raise CheckpointStoreError("checkpoint payload must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise CheckpointStoreError("checkpoint payload must contain tensor values")
        result[name] = value.detach().cpu()
    return result


def verify_checkpoint(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify path confinement, byte/file hashes, and reloaded state hash."""
    relative = manifest.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CheckpointStoreError("checkpoint path must be a relative path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CheckpointStoreError("checkpoint path escapes the run root") from exc
    if not path.is_file():
        raise CheckpointStoreError(f"checkpoint file is missing: {path}")
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    if manifest.get("sha256") != file_hash:
        raise CheckpointStoreError(f"checkpoint file hash mismatch: {path}")
    if manifest.get("bytes") != len(raw):
        raise CheckpointStoreError(f"checkpoint byte count mismatch: {path}")
    state = load_checkpoint(path)
    state_hash = state_sha256(state)
    if manifest.get("state_sha256") != state_hash:
        raise CheckpointStoreError(f"checkpoint state hash mismatch: {path}")
    return {"path": str(path), "sha256": file_hash, "state_sha256": state_hash, "bytes": len(raw)}
