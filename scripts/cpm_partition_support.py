"""Replay retained CPM partitions and count class support without fitting models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reader_data import array_sha256, load_public_dataset
from scripts.reader_reproduce import assert_same


def replay(y, seed, repeats=100, cap=500):
    classes = np.unique(y)
    groups = [np.flatnonzero(y == c) for c in classes]
    rng = np.random.default_rng(seed)
    digest, sizes, supports = hashlib.sha256(), [], []
    for _ in range(repeats):
        if len(y) <= cap:
            sample = np.arange(len(y))
        else:
            quota = int(round(cap / len(classes)))
            sample = np.sort(np.concatenate([rng.permutation(ids)[:quota] for ids in groups]))
        ys = y[sample]
        quota = int(round(0.6 * len(sample) / len(classes)))
        fit_parts, hold_parts = [], []
        for c in classes:
            indices = rng.permutation(np.flatnonzero(ys == c))
            fit_parts.append(indices[:quota])
            hold_parts.append(indices[quota:])
        fit, hold = np.concatenate(fit_parts), np.concatenate(hold_parts)
        sizes.append([len(sample), len(fit), len(hold)])
        digest.update(np.asarray(sizes[-1], '<i8').tobytes())
        for indices in (sample, fit, hold):
            digest.update(np.asarray(indices, '<i8').tobytes())
        supports.append({'sample': [int(np.sum(ys == c)) for c in classes],
                         'fit': [len(ids) for ids in fit_parts],
                         'holdout': [len(ids) for ids in hold_parts]})
    return {'partition_sha256': digest.hexdigest(), 'partition_sizes': sizes,
            'classes': classes.tolist(), 'supports_by_repeat': supports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--materialized-root', type=Path)
    source.add_argument('--public-data-root', type=Path)
    p.add_argument('--download', action='store_true')
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--cpm', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+')
    a = p.parse_args()
    if a.output.exists():
        p.error('preserve existing output')
    binding = json.loads(a.binding.read_text())
    metrics = json.loads(a.cpm.read_text())['metric_records']
    selected = set(a.datasets or binding['datasets'])
    if not selected <= set(binding['datasets']):
        p.error('unknown dataset')
    rows = []
    for name in sorted(selected):
        entry = binding['datasets'][name]
        for seed in range(10):
            if a.materialized_root:
                with np.load(a.materialized_root / (name + '.npz'), allow_pickle=False) as data:
                    y, train = data['y'], data[f'seed_{seed}_train']
            else:
                _, labels, _, splits, _, _ = load_public_dataset(name, seed, a.public_data_root,
                    binding, download=a.download)
                y, train = labels.numpy(), splits['train']
            for key, value in [('y', y), (f'seed_{seed}_train', train)]:
                if array_sha256(value) != entry['tensor_sha256'][key]:
                    raise ValueError('label/split binding mismatch')
            saved = [r for r in metrics if r['dataset'] == name and r['seed'] == seed]
            if len(saved) != 2 or len({r['condition'] for r in saved}) != 2:
                raise ValueError('expected both conditions')
            result = replay(y[train], saved[0]['resampling_seed'])
            for record in saved:
                assert_same({k: result[k] for k in ('partition_sha256', 'partition_sizes')},
                            {k: record[k] for k in ('partition_sha256', 'partition_sizes')})
                if record['outer_train_count'] != len(train) or record['resampling_seed'] != saved[0]['resampling_seed']:
                    raise ValueError('outer scope mismatch')
            rows.append({'dataset': name, 'seed': seed, 'outer_train_count': len(train),
                         'resampling_seed': saved[0]['resampling_seed'], **result})
    payload = {'schema': 'cpm-partition-support/1', 'status': 'verified',
               'new_classifier_or_neural_fits': 0, 'condition_records_verified': 2 * len(rows),
               'binding_sha256': hashlib.sha256(a.binding.read_bytes()).hexdigest(),
               'cpm_sha256': hashlib.sha256(a.cpm.read_bytes()).hexdigest(), 'rows': rows}
    with a.output.open('x', encoding='utf-8') as out:
        json.dump(payload, out, indent=2, allow_nan=False)
        out.write('\n')
    for name in sorted(selected):
        r = next(r for r in rows if r['dataset'] == name)
        print(name, r['partition_sizes'][0], r['supports_by_repeat'][0])


if __name__ == '__main__':
    main()
