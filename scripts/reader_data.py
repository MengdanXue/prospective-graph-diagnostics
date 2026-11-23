"""Public-data reconstruction with tensor-level bindings and train-fitted inputs.

The hash and feature-transform functions retain the scientific implementation in
controller revision 51f168217e5b55fc190dd3fd885878408df43fe5. The reader requires
only the compact published binding, never the author controller or ledger.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from experiments.prospective_data import canonical_undirected_edges, make_stratified_split, split_identifier

CONDITIONS = ("normalize_features", "normalize_centered_scaled")

def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def acquire_raw(entry, root):
    """Explicitly fetch only checksum-pinned public files into the reader cache."""
    from urllib.request import Request, urlopen
    import time
    root = Path(root).resolve()
    receipts = []
    for item in entry['raw_files']:
        destination = (root / item['path']).resolve()
        destination.relative_to(root)
        if not destination.exists():
            if not item['source_url'].startswith('https://raw.githubusercontent.com/'):
                raise ValueError('binding must name the public upstream HTTPS source')
            destination.parent.mkdir(parents=True, exist_ok=True)
            failure = None
            for attempt in range(3):
                token = time.time_ns()
                partial = destination.with_name(destination.name + f'.download-{token}')
                # Fresh full-file requests avoid intermediary-cached empty/range responses.
                request = Request(item['source_url'] + f'?reader_download={token}',
                    headers={'User-Agent': 'graph-diagnostics-reader', 'Accept-Encoding': 'identity'})
                try:
                    with urlopen(request, timeout=90) as response, partial.open('xb') as output:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > item['size']:
                                raise ValueError('download exceeds published byte count')
                            output.write(chunk)
                    if partial.stat().st_size != item['size'] or file_sha256(partial) != item['sha256']:
                        raise ValueError('incomplete download or changed public content')
                    partial.rename(destination)
                    failure = None
                    break
                except (OSError, ValueError) as exc:
                    failure = exc
            if failure is not None:
                raise RuntimeError(f'public download failed integrity checks: {item["source_url"]}') from failure
        if destination.stat().st_size != item['size'] or file_sha256(destination) != item['sha256']:
            raise ValueError(f'raw checksum mismatch: {item["path"]}')
        receipts.append(dict(item))
    return receipts


def load_public_dataset(name, seed, root, binding, *, download=False):
    """Regenerate tensors and splits; compare array hashes, not PyG pickle bytes."""
    from torch_geometric import datasets
    entry = binding['datasets'][name]
    root = Path(root).resolve()
    if download:
        acquire_raw(entry, root)
    for item in entry['raw_files']:
        path = (root / item['path']).resolve()
        path.relative_to(root)
        if not path.is_file() or file_sha256(path) != item['sha256']:
            raise ValueError(f'missing or changed raw input; use acquire first: {item["path"]}')
    spec = dict(entry['loader'])
    cls = getattr(datasets, spec.pop('class'))
    graph = cls(root=str(root / 'pyg'), transform=None, **spec)[0]
    x, y = graph.x.detach().cpu().float(), graph.y.detach().cpu().long().view(-1)
    undirected = canonical_undirected_edges(graph.edge_index.numpy(), node_count=len(y))
    both = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    edge_index = torch.from_numpy(both[order].T.copy()).long()
    split = make_stratified_split(y.numpy(), seed=seed)
    tensors = {'x': x, 'y': y, 'edge_index': edge_index,
               **{f'seed_{seed}_{k}': v for k, v in split.items()}}
    hashes = {key: array_sha256(value) for key, value in tensors.items()}
    for key, digest in hashes.items():
        if digest != entry['tensor_sha256'][key]:
            raise ValueError(f'tensor mismatch: {name}/{key}')
    split_id = split_identifier(split)
    expected = next(s for s in entry['split_checks'] if s['seed'] == seed)
    if split_id != expected['split_id']:
        raise ValueError('split identifier mismatch')
    return x, y, edge_index, split, split_id, hashes

def transform_features(x, condition):
    if condition != "normalize_features":
        raise ValueError("only NormalizeFeatures is used before the fitted transform")
    return NormalizeFeatures()(Data(x=x.clone())).x

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
