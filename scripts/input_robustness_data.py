"""Exclusive data preparation and read-only binding for the post-hoc 11-dataset study.

Acquisition is an explicit operation, never a side effect of training-time load.
The unchanged PyG loaders reconstruct the original untransformed processed data;
its byte digest must equal the original released diagnostic's processed digest.
No predictive validation or test metrics are calculated in this module.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.prospective_data import (
    canonical_undirected_edges, dataset_specifications, make_stratified_split,
    split_identifier, train_only_diagnostics, validate_dataset_eligibility,
)
from experiments.run_prospective_benchmark import _bidirectional_edge_index, _load_dataset
from scripts.run_preprocessing_sensitivity import transform_features

DATASETS = tuple(json.loads((ROOT / "configs/prospective_benchmark_v2.json").read_text())["datasets"])
PARTITIONS = ("train", "validation", "test")
CONDITIONS = ("normalize_features", "normalize_centered_scaled")
# Published v0.1.0 SHA256SUMS and GitHub release asset digests; verified against
# the retained 37,421,570-byte ZIP before this preparation was frozen.
RELEASE_MANIFEST_SHA256 = "32768777518701dae9c1184200413daf24fa3ddade2af347a0e04dc0647d6eb0"
RELEASE_ARCHIVE_SHA256 = "209d386f078e0c4b2b15eb221b52c52bb002fc136a473494684e01be95b70389"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash dtype, shape, then C-order bytes, streaming contiguous input."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.ascontiguousarray(array)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\n")
    view = memoryview(array).cast("B")
    for begin in range(0, view.nbytes, 1024 * 1024):
        digest.update(view[begin:begin + 1024 * 1024])
    return digest.hexdigest()


def estimate_transform_peak_bytes(shape, train_count: int, *, block_bytes=64 * 1024**2) -> int:
    """Conservative live feature arrays, excluding interpreter/model/graph memory.

Includes raw+normalized+output float32 and three train float64 arrays during
the unchanged mean-square reduction, and output-block temporaries. Sparse inputs
are rejected: this estimate never authorizes an implicit sparse-to-dense cast.
"""
    n, d = map(int, shape)
    if n < 1 or d < 1 or not 0 < train_count <= n or block_bytes < 1:
        raise ValueError("invalid transform shape, training count or block size")
    return 12 * n * d + 24 * train_count * d + 3 * block_bytes + 32 * d


def fit_transform_features_bounded(x, train_indices, condition, *,
                                   max_feature_bytes=4 * 1024**3,
                                   block_bytes=64 * 1024**2):
    """Existing train-fitted affine formula with bounded allocation, exact reductions.

The train arrays and NumPy reductions have the same C-order values and order as
the historical implementation. Only array lifetimes and output-row blocking
change. NormalizeFeatures retains its historical global-min semantics.
"""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown paired condition: {condition}")
    if x.layout != torch.strided or x.ndim != 2 or x.dtype != torch.float32 or x.device.type != "cpu":
        raise ValueError("expected finite dense CPU float32 features; no implicit densification")
    indices = torch.as_tensor(train_indices).detach().cpu().numpy()
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("train indices must be a nonempty integer vector")
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(x):
        raise ValueError("invalid or repeated training indices")
    estimate = estimate_transform_peak_bytes(x.shape, len(indices), block_bytes=block_bytes)
    if estimate > max_feature_bytes:
        raise MemoryError(f"feature allocation estimate {estimate} exceeds cap {max_feature_bytes}")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite input features")
    normalized = transform_features(x, "normalize_features")
    statistics = []
    for matrix in (x, normalized):
        train = matrix.numpy()[indices].astype(np.float64)
        mean = train.mean(axis=0, dtype=np.float64)
        rms = float(np.sqrt(np.mean((train - mean) ** 2, dtype=np.float64)))
        statistics.append((mean, rms))
        del train
    (_, raw_rms), (normalized_mean, normalized_rms) = statistics
    if not all(np.isfinite(value) and value > 1e-12 for value in (raw_rms, normalized_rms)):
        raise ValueError("nonfinite or near-zero train feature RMS")
    scale = raw_rms / normalized_rms
    if not np.isfinite(scale):
        raise ValueError("nonfinite fitted feature scale")
    if condition == "normalize_features":
        output = normalized.clone()
    else:
        array = np.empty(tuple(x.shape), dtype=np.float32)
        rows = max(1, block_bytes // (8 * x.shape[1]))
        for begin in range(0, len(x), rows):
            end = min(len(x), begin + rows)
            array[begin:end] = (normalized.numpy()[begin:end].astype(np.float64) - normalized_mean) * scale
        output = torch.from_numpy(array)
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite transformed features")
    metadata = {"condition": condition, "fit_partition": "train_features_only",
                "statistics_dtype": "float64", "output_dtype": "float32", "epsilon": 1e-12,
                "fitted_normalized_mean": normalized_mean.tolist(), "fitted_scale": float(scale),
                "feature_statistics": {"raw_train_centered_rms": raw_rms,
                                       "normalized_train_centered_rms": normalized_rms}}
    return output, metadata


def check_original_binding(name, x, y, edges, original_units):
    """Verify all ten original splits plus train-only homophily and graph degree."""
    if name not in DATASETS:
        raise ValueError(f"dataset outside fixed original 11: {name}")
    if x.ndim != 2 or len(x) != len(y) or x.dtype != torch.float32 or x.layout != torch.strided:
        raise ValueError("invalid raw feature shape/type")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite raw features")
    labels = y.numpy()
    counts = validate_dataset_eligibility(labels, minimum_class_count=3)
    relevant = [row for row in original_units if row["dataset"] == name]
    if len(relevant) != 10 or {row["seed"] for row in relevant} != set(range(10)):
        raise ValueError("original diagnostic scope must contain exactly ten seeds")
    splits, checks = {}, []
    for unit in sorted(relevant, key=lambda row: row["seed"]):
        seed = int(unit["seed"])
        split = make_stratified_split(labels, seed=seed)
        identifier = split_identifier(split)
        if identifier != unit["split_id"]:
            raise ValueError(f"original split mismatch: {name}/{seed}")
        mask = np.zeros(len(y), dtype=bool)
        mask[split["train"]] = True
        eligible = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
        h = float(np.mean(labels[eligible[:, 0]] == labels[eligible[:, 1]]))
        degree = 2 * len(edges) / len(y)
        if abs(h - unit["homophily"]) > 1e-12 or abs(degree - unit["mean_degree"]) > 1e-12:
            raise ValueError(f"original graph diagnostic mismatch: {name}/{seed}")
        splits[seed] = (split, identifier)
        checks.append({"seed": seed, "split_id": identifier, "homophily": h,
                       "mean_degree": degree, "train_edge_count": len(eligible),
                       "partition_sizes": {part: len(split[part]) for part in PARTITIONS},
                       "partition_sha256": {part: array_sha256(split[part]) for part in PARTITIONS}})
    return splits, checks, counts


def _exclusive_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def acquire_dataset(name, data_root: Path, *, copy_heterophilous_from=None):
    """Explicit acquisition to a dedicated new tree, preserving original inputs."""
    if name not in DATASETS:
        raise ValueError("dataset outside frozen scope")
    from torch_geometric import datasets
    specification = dataset_specifications()[name]
    cls = getattr(datasets, specification["class"])
    kwargs = {key: val for key, val in specification.items() if key != "class"}
    cache_root = Path(data_root) / "pyg"
    if copy_heterophilous_from and specification["class"] == "HeterophilousGraphDataset":
        filename = name.lower().replace("-", "_") + ".npz"
        destination = cache_root / name.lower().replace("-", "_") / "raw" / filename
        source = Path(copy_heterophilous_from) / filename
        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    # Download canonical raw GitHub content directly. This is equivalent to
    # PyG's configured URLs, without fsspec's directory-listing HTTP requests.
    # It also prevents partial downloads from becoming accepted raw files.
    import requests
    for relative, url in raw_source_urls(name).items():
        destination = Path(data_root) / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        failure = None
        for attempt in range(3):
            partial = destination.with_name(destination.name + f".partial-{time.time_ns()}")
            try:
                with requests.get(url, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    size = 0
                    with partial.open("xb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > 256 * 1024**2:
                                raise ValueError("raw public file exceeds 256 MiB acquisition cap")
                            handle.write(chunk)
                    if size == 0:
                        raise ValueError("empty raw download")
                partial.rename(destination)
                failure = None
                break
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise RuntimeError(f"raw acquisition failed after three attempts: {url}") from failure
    # The constructor has precisely the original dataset kwargs (including the
    # original WikipediaNetwork geom_gcn_preprocess=True default).
    return cls(root=str(cache_root), transform=None, **kwargs)


def raw_source_urls(name):
    spec = dataset_specifications()[name]
    kind, token = spec["class"], name.lower()
    base = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
    if kind == "Planetoid":
        return {f"pyg/{name}/raw/ind.{token}.{suffix}":
                f"https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{token}.{suffix}"
                for suffix in ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")}
    if kind == "Coauthor":
        return {"pyg/CS/raw/ms_academic_cs.npz":
                "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/ms_academic_cs.npz"}
    if kind == "HeterophilousGraphDataset":
        token = token.replace("-", "_")
        return {f"pyg/{token}/raw/{token}.npz":
                f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{token}.npz"}
    if kind == "WikipediaNetwork":
        root = f"pyg/{token}/geom_gcn/raw"
        base += "f1fc0d14b3b019c562737240d06ec83b07d16a8f"
    elif kind == "Actor":
        root, token, base = "pyg/raw", "film", base + "master"
    else:
        root, base = f"pyg/{token}/raw", base + "master"
    result = {f"{root}/{filename}": f"{base}/new_data/{token}/{filename}"
              for filename in ("out1_node_feature_label.txt", "out1_graph_edges.txt")}
    result.update({f"{root}/{token}_split_0.6_0.2_{seed}.npz":
                   f"{base}/splits/{token}_split_0.6_0.2_{seed}.npz" for seed in range(10)})
    return result


def materialize_dataset(name, data_root, original_audit, release_root):
    """Read verified PyG cache, bind the release, write an exclusive canonical NPZ."""
    data_root, release_root = Path(data_root), Path(release_root)
    dataset = _load_dataset(name, data_root / "pyg")
    dataset.transform = None
    data = dataset[0]
    x, y = data.x.detach().cpu().float(), data.y.detach().cpu().long().view(-1)
    edges = canonical_undirected_edges(data.edge_index.numpy(), node_count=len(y))
    edge_index = _bidirectional_edge_index(edges)
    splits, checks, counts = check_original_binding(name, x, y, edges, original_audit["units"])
    original_path = release_root / "prospective" / "diagnostics" / name / "seed_000.json"
    release_manifest = verify_release_manifest(release_root)
    release_entry = next(row for row in release_manifest["files"] if row["path"] == original_path.relative_to(release_root).as_posix())
    if file_sha256(original_path) != release_entry["public_sha256"]:
        raise ValueError("original released diagnostic checksum mismatch")
    original = json.loads(original_path.read_text())
    expected = original["provenance"]["data"]["processed_files"]
    actual = [{"path": Path(path).name, "size": Path(path).stat().st_size,
               "sha256": file_sha256(Path(path))} for path in dataset.processed_paths]
    projected = [{key: row[key] for key in ("path", "size", "sha256")} for row in expected]
    exact_processed = actual == projected
    if not exact_processed:
        raise ValueError(f"original processed file checksum mismatch: {name}; actual={actual}; expected={projected}")
    raw_files = [{"path": Path(path).relative_to(data_root).as_posix(),
                  "source_url": raw_source_urls(name)[Path(path).relative_to(data_root).as_posix()],
                  "size": Path(path).stat().st_size, "sha256": file_sha256(Path(path))}
                 for path in dataset.raw_paths]
    tensors = {"x": x.numpy(), "y": y.numpy(), "edge_index": edge_index.numpy()}
    for seed, (split, _) in splits.items():
        tensors.update({f"seed_{seed}_{part}": split[part] for part in PARTITIONS})
    output = data_root / "materialized" / f"{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez(handle, **tensors)
    condition_checks = []
    for seed, (split, _) in splits.items():
        transformed, metadata = fit_transform_features_bounded(x, torch.from_numpy(split["train"]), "normalize_centered_scaled")
        condition_checks.append({"seed": seed, "raw_train_centered_rms": metadata["feature_statistics"]["raw_train_centered_rms"],
                                 "normalized_train_centered_rms": metadata["feature_statistics"]["normalized_train_centered_rms"],
                                 "fitted_scale": metadata["fitted_scale"],
                                 "transformed_feature_sha256": array_sha256(transformed)})
        del transformed
        gc.collect()
    normalized = transform_features(x, "normalize_features")
    row = {"dataset": name, "status": "bound", "loader": dataset_specifications()[name],
           "original_processed_byte_match": exact_processed, "processed_files": actual,
           "original_diagnostic": {"path": original_path.relative_to(release_root).as_posix(),
                                   "sha256": file_sha256(original_path)},
           "raw_files": raw_files, "materialized_path": output.relative_to(data_root).as_posix(),
           "materialized_sha256": file_sha256(output), "materialized_bytes": output.stat().st_size,
           "tensor_sha256": {key: array_sha256(value) for key, value in tensors.items()},
           "feature_shape": list(x.shape), "feature_dtype": str(x.dtype), "feature_layout": "dense_strided",
           "feature_min": float(x.min()), "feature_max": float(x.max()),
           "zero_feature_rows": int((x.count_nonzero(dim=1) == 0).sum()),
           "nonzero_features": int(x.count_nonzero()), "class_counts": counts,
           "canonical_undirected_edge_count": len(edges), "split_checks": checks,
           "normalize_features_sha256": array_sha256(normalized), "candidate_checks": condition_checks,
           "max_feature_allocation_estimate_bytes": max(estimate_transform_peak_bytes(x.shape, len(value[0]["train"])) for value in splits.values()),
           "candidate_applicability": "all_ten_train_fits_finite_nonzero_and_outputs_finite",
           "candidate_optimality_claim": False, "validation_or_test_score_evaluations": 0}
    _exclusive_json(data_root / "bindings" / f"{name}.json", row)
    return row


def verify_release_manifest(release_root):
    path = Path(release_root) / "MANIFEST.json"
    if file_sha256(path) != RELEASE_MANIFEST_SHA256:
        raise ValueError("release manifest does not match published v0.1.0 digest")
    return json.loads(path.read_text())


def enrich_dataset_binding(row, data_root, release_root):
    """Recompute 50k-walk diagnostics and retain all ten complete affine fits.

Stage-one materializations and their individual bindings remain immutable.
The complete verified row is a distinct new artifact; it reads no predictions.
"""
    name = row["dataset"]
    x, y, edge_index, splits, _ = load_bound_dataset(name, data_root, {"datasets": {name: row}})
    edges = canonical_undirected_edges(edge_index.numpy(), node_count=len(y))
    manifest = verify_release_manifest(release_root)
    entries = {item["path"]: item for item in manifest["files"]}
    new_row = dict(row)
    candidate_checks, full_diagnostics = [], []
    for seed, (split, identifier) in splits.items():
        relative = f"prospective/diagnostics/{name}/seed_{seed:03d}.json"
        path = Path(release_root) / relative
        digest = file_sha256(path)
        if digest != entries[relative]["public_sha256"]:
            raise ValueError(f"released diagnostic digest mismatch: {name}/{seed}")
        original = json.loads(path.read_text())
        if original["split_id"] != identifier or original["status"] != "success":
            raise ValueError("released diagnostic split/status mismatch")
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[split["train"]] = True
        actual = train_only_diagnostics(edges, y.numpy(), train_mask, sample_count=50000, seed=200000 + seed)
        if actual != original["details"]:
            raise ValueError(f"recomputed original 50k-walk diagnostic mismatch: {name}/{seed}")
        full_diagnostics.append({"seed": seed, "split_id": identifier, "diagnostics": actual,
                                 "original_record_path": relative, "original_record_sha256": digest,
                                 "all_original_diagnostic_fields_exact": True})
        transformed, metadata = fit_transform_features_bounded(x, torch.from_numpy(split["train"]), CONDITIONS[1])
        check = next(item for item in row["candidate_checks"] if item["seed"] == seed)
        if array_sha256(transformed) != check["transformed_feature_sha256"]:
            raise ValueError("candidate feature digest drift during verification")
        candidate_checks.append({**check, "fit_metadata": metadata})
        del transformed
    normalized = transform_features(x, CONDITIONS[0])
    if array_sha256(normalized) != row["normalize_features_sha256"]:
        raise ValueError("NormalizeFeatures digest drift during verification")
    new_row.update({"candidate_checks": candidate_checks, "full_diagnostic_checks": full_diagnostics,
                    "normalize_features_fit_metadata": {
                        "binding": "candidate_checks[seed].fit_metadata with condition replaced by normalize_features",
                        "explanation": "Identical raw/normalized train statistics are computed for both conditions; the control uses no fitted affine correction."},
                    "all_original_diagnostics_exact": True,
                    "stage_one_binding_sha256": file_sha256(Path(data_root) / "bindings" / f"{name}.json")})
    _exclusive_json(Path(data_root) / "bindings_verified" / f"{name}.json", new_row)
    return new_row


def load_bound_dataset(name, data_root, binding):
    """Read-only load; verify file, tensors and every split against the frozen map."""
    if isinstance(binding, (str, Path)):
        binding = json.loads(Path(binding).read_text())
    rows = binding["datasets"]
    entry = rows[name] if isinstance(rows, dict) else next(row for row in rows if row["dataset"] == name)
    if name not in DATASETS or entry["status"] != "bound" or not entry["original_processed_byte_match"]:
        raise ValueError("data binding is incomplete or outside frozen scope")
    root = Path(data_root).resolve()
    path = (root / entry["materialized_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("materialized path escapes data root")
    if file_sha256(path) != entry["materialized_sha256"]:
        raise ValueError("materialized checksum mismatch")
    with np.load(path, allow_pickle=False) as contents:
        if set(contents.files) != set(entry["tensor_sha256"]):
            raise ValueError("materialized tensor set mismatch")
        arrays = {key: contents[key] for key in contents.files}
    for key, expected in entry["tensor_sha256"].items():
        if array_sha256(arrays[key]) != expected:
            raise ValueError(f"materialized tensor checksum mismatch: {key}")
    splits = {}
    for row in entry["split_checks"]:
        seed = row["seed"]
        split = {part: arrays[f"seed_{seed}_{part}"] for part in PARTITIONS}
        if split_identifier(split) != row["split_id"]:
            raise ValueError("materialized split mismatch")
        splits[seed] = (split, row["split_id"])
    if set(splits) != set(range(10)):
        raise ValueError("materialized seed scope mismatch")
    return (torch.from_numpy(arrays["x"]), torch.from_numpy(arrays["y"]),
            torch.from_numpy(arrays["edge_index"]), splits, entry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--binding-output", required=True, type=Path)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--copy-heterophilous-from", type=Path)
    parser.add_argument("--resume", action="store_true", help="Recheck existing immutable dataset bindings")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    audit_path = ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json"
    audit = json.loads(audit_path.read_text())
    verify_release_manifest(args.release_root)
    results = {}
    for name in DATASETS:
        binding_path = args.data_root / "bindings" / f"{name}.json"
        if binding_path.exists():
            if not args.resume:
                raise FileExistsError(binding_path)
            row = json.loads(binding_path.read_text())
            loaded = load_bound_dataset(name, args.data_root, {"datasets": {name: row}})
            del loaded
        else:
            print(json.dumps({"dataset": name, "phase": "acquisition"}), flush=True)
            if args.acquire:
                dataset = acquire_dataset(name, args.data_root, copy_heterophilous_from=args.copy_heterophilous_from)
                del dataset
                gc.collect()
            row = materialize_dataset(name, args.data_root, audit, args.release_root)
        verified_path = args.data_root / "bindings_verified" / f"{name}.json"
        if verified_path.exists():
            if not args.resume:
                raise FileExistsError(verified_path)
            verified = json.loads(verified_path.read_text())
            if verified["stage_one_binding_sha256"] != file_sha256(binding_path):
                raise ValueError("stage-one binding drift")
            load_bound_dataset(name, args.data_root, {"datasets": {name: verified}})
            row = verified
        else:
            row = enrich_dataset_binding(row, args.data_root, args.release_root)
        results[name] = row
        print(json.dumps({"dataset": name, "status": row["status"], "shape": row["feature_shape"],
                          "original_processed_byte_match": row["original_processed_byte_match"]}), flush=True)
        gc.collect()
    result = {"schema_version": "1.0", "analysis_status": "post_hoc_training_recipe_robustness",
              "run_id": "posthoc_input_robustness_11_v1",
              "phase": "preflight_data_binding_only", "status": "passed",
              "original_audit_sha256": file_sha256(audit_path),
              "release_manifest_sha256": file_sha256(args.release_root / "MANIFEST.json"),
              "published_release_archive_sha256": RELEASE_ARCHIVE_SHA256,
              "published_release_url": "https://github.com/MengdanXue/prospective-graph-diagnostics/releases/tag/v0.1.0",
              "datasets": results, "dataset_count": 11, "bound_split_count": 110,
              "validation_or_test_score_evaluations": 0}
    _exclusive_json(args.binding_output, result)


if __name__ == "__main__":
    main()
