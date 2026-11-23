"""Bounded CPM diagnostic extension. Author execution uses cpm_extension_guarded.

Compute metrics before opening candidate outcome records. Reuse bound graph data,
the accepted feature transformer and all declared saved-score evaluation cells.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpm_probe import aggregation_operator, probe_training_rows, seed_for
from major_revision_audit import audit_actions, calibrate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind_probe_record(identity, metric):
    """Prevent numerical outputs from overwriting dataset/split identities."""
    if set(identity) & set(metric):
        raise ValueError('probe output collides with outer record identity')
    return {**identity, **metric}


def evaluate(package, metrics, protocol):
    evidence = package / 'new-evidence'
    base = [json.loads(p.read_bytes()) for p in sorted((evidence / 'input-retraining/records').rglob('*.json'))]
    extended = [json.loads(p.read_bytes()) for p in sorted((evidence / 'mlp24/records').rglob('*.json'))]
    if len(base) != 1540 or len(extended) != 220:
        raise ValueError('incomplete saved records')
    key = lambda r: (r['condition'], r['dataset'], r['seed'], r['model'])
    replacements = {key(r): r for r in extended}
    lookup = {(r['condition'], r['dataset'], r['seed']): r for r in metrics}
    output = []
    for budget in protocol['mlp_budgets']:
        records = base if budget == 4 else [replacements.get(key(r), r) for r in base]
        for condition in protocol['conditions']:
            units = defaultdict(dict)
            for r in records:
                if r['condition'] == condition:
                    units[r['dataset'], r['seed']][r['model']] = r
            if len(units) != 110:
                raise ValueError('unexpected units')
            for portfolio in protocol['portfolios']:
                rows = []
                for (dataset, seed), models in sorted(units.items()):
                    g = sorted((models[m] for m in portfolio), key=lambda r: (-r['validation_accuracy'], r['model']))[0]
                    m = models['MLP']
                    metric = lookup[condition, dataset, seed]
                    if g['split_id'] != m['split_id'] or m['split_id'] != metric['split_id']:
                        raise ValueError('split identity mismatch')
                    rows.append({'dataset': dataset, 'seed': seed, 'graph_test': g['test_accuracy'],
                        'mlp_test': m['test_accuracy'], 'graph_model': g['model'],
                        'cpm_gnb': metric['cpm_gnb'], 'probe_gap': metric['probe_gap'],
                        'validation_action': 'graph' if g['validation_accuracy'] - m['validation_accuracy'] - 0.01 > 0 else 'mlp'})
                actions = {
                    'cpm_nt05': ['graph' if r['cpm_gnb'] >= .5 else 'mlp' for r in rows],
                    'cpm_sst005': ['graph' if r['cpm_gnb'] > .95 else 'mlp' if r['cpm_gnb'] < .05 else 'abstain' for r in rows],
                    'gap_zero': ['graph' if r['probe_gap'] >= 0 else 'mlp' for r in rows],
                    'always_graph': ['graph'] * len(rows),
                    'validation_selection': [r['validation_action'] for r in rows]}
                output.append({'budget': budget, 'condition': condition, 'portfolio': portfolio, 'rows': rows,
                    'fixed': {name: {**audit_actions(rows, values), 'actions': values} for name, values in actions.items()},
                    'calibrated': {metric: calibrate(rows, metric) for metric in ('cpm_gnb', 'probe_gap')}})
    return output


def retained_records(path, expected_sha, protocol, binding):
    """Reuse only a closed, externally hash-bound complete JSONL prefix."""
    if path is None:
        if expected_sha is not None:
            raise ValueError('resume hash without input')
        return []
    if expected_sha is None or sha(path) != expected_sha:
        raise ValueError('resume records differ from external binding')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    order = [(d,s,c) for d in protocol['datasets'] for s in protocol['seeds'] for c in protocol['conditions']]
    if [(r['dataset'],r['seed'],r['condition']) for r in rows] != order[:len(rows)] or len(rows) > 220:
        raise ValueError('resume records are not a unique declared prefix')
    from cpm_probe import directional_score
    for r in rows:
        entry = binding['datasets'][r['dataset']]
        split = next(s for s in entry['split_checks'] if s['seed'] == r['seed'])
        feature = (entry['normalize_features_sha256'] if r['condition']=='normalize_features' else
            next(s['transformed_feature_sha256'] for s in entry['candidate_checks'] if s['seed']==r['seed']))
        if r['split_id'] != split['split_id'] or r['materialized_sha256'] != entry['materialized_sha256'] or r['transformed_feature_sha256'] != feature:
            raise ValueError('resume data/split/feature binding mismatch')
        if r['resampling_seed'] != seed_for(r['dataset'],r['split_id']):
            raise ValueError('resume probe seed mismatch')
        if any(len(r[k]) != protocol['repeats'] for k in ('raw_accuracy','graph_accuracy','partition_sizes')):
            raise ValueError('incomplete probe repeats')
        if directional_score(r['raw_accuracy'],r['graph_accuracy']) != r['cpm_gnb']:
            raise ValueError('retained CPM score differs from its repeats')
        if not math.isclose(math.fsum(g-x for g,x in zip(r['graph_accuracy'],r['raw_accuracy']))/protocol['repeats'],r['probe_gap'],abs_tol=1e-14):
            raise ValueError('retained gap differs from repeats')
    return rows


def run(controller, data_root, package, output, resume_records=None, resume_sha=None):
    import numpy as np
    import scipy
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=1)
    protocol_path = ROOT / 'configs/cpm_train_only_v1.json'
    protocol = json.loads(protocol_path.read_bytes())
    sys.path.insert(0, str(controller))
    data = importlib.import_module('scripts.input_robustness_data')
    if Path(data.__file__).resolve() != controller / 'scripts/input_robustness_data.py':
        raise ValueError('accepted data loader shadowed')
    binding_path = controller / 'results/diagnostic/posthoc_input_robustness_11_v1/preflight/data_binding.json'
    binding = json.loads(binding_path.read_bytes())
    started = time.monotonic()
    records = retained_records(resume_records, resume_sha, protocol, binding)
    resumed_count = len(records)
    completed = {(r['dataset'],r['seed'],r['condition']) for r in records}
    partial = output.with_suffix('.records.jsonl')
    with partial.open('x', encoding='utf-8') as progress:
        for row in records:
            progress.write(json.dumps(row,allow_nan=False)+'\n')
        progress.flush()
        for dataset in protocol['datasets']:
            if all((dataset,s,c) in completed for s in protocol['seeds'] for c in protocol['conditions']):
                continue
            load_started = time.monotonic()
            x, y, edges, splits, entry = data.load_bound_dataset(dataset, data_root, binding)
            operator = aggregation_operator(edges.numpy(), len(x))
            load_seconds = time.monotonic() - load_started
            for seed in protocol['seeds']:
                split, split_id = splits[seed]
                train = split['train']
                train_labels = y.numpy()[train].copy()
                for condition in protocol['conditions']:
                    if (dataset,seed,condition) in completed:
                        continue
                    cell_start = time.monotonic()
                    transformed, meta = data.fit_transform_features_bounded(x, train, condition)
                    transformed_sha = data.array_sha256(transformed)
                    expected_sha = (entry['normalize_features_sha256'] if condition == 'normalize_features' else
                        next(c['transformed_feature_sha256'] for c in entry['candidate_checks'] if c['seed'] == seed))
                    if transformed_sha != expected_sha:
                        raise ValueError('probe feature map differs from bound training input')
                    # Retain sparse graph propagation; never materialize n-by-n adjacency.
                    aggregated_train = operator[train] @ transformed.numpy()
                    prepared = time.monotonic()
                    metric = probe_training_rows(transformed.numpy()[train], aggregated_train, train_labels,
                        seed=seed_for(dataset, split_id), repeats=protocol['repeats'], cap=protocol['nominal_sample_cap'])
                    row = {**bind_probe_record({'dataset': dataset, 'seed': seed, 'split_id': split_id, 'condition': condition}, metric),
                        'feature_metadata': meta, 'transformed_feature_sha256': transformed_sha,
                        'materialized_sha256': entry['materialized_sha256'],
                        'load_seconds_shared_across_dataset': load_seconds,
                        'feature_and_aggregation_seconds': prepared - cell_start,
                        'probe_seconds': time.monotonic() - prepared,
                        'metric_seconds': time.monotonic() - cell_start}
                    records.append(row)
                    progress.write(json.dumps(row, allow_nan=False) + '\n')
                    progress.flush()
                    print(f'CPM {dataset} {seed} {condition} complete', flush=True)
                    del transformed, aggregated_train
            del x, y, edges, operator
    if len(records) != 220:
        raise ValueError('metric scope incomplete')
    metrics_seconds = time.monotonic() - started
    paired = []
    for dataset in protocol['datasets']:
        for seed in protocol['seeds']:
            a, b = [next(r for r in records if (r['dataset'], r['seed'], r['condition']) == (dataset, seed, c))
                    for c in protocol['conditions']]
            if a['partition_sha256'] != b['partition_sha256']:
                raise ValueError('N/CS probe partitions differ')
            paired.append({'dataset': dataset, 'seed': seed,
                'identical_replicate_accuracies': a['raw_accuracy'] == b['raw_accuracy'] and a['graph_accuracy'] == b['graph_accuracy'],
                'absolute_score_difference': abs(a['cpm_gnb'] - b['cpm_gnb']),
                'absolute_gap_difference': abs(a['probe_gap'] - b['probe_gap'])})
    policies = evaluate(package, records, protocol)
    result = {'schema': 'cpm-extension/1', 'status': 'complete_posthoc', 'protocol': protocol,
        'protocol_sha256': sha(protocol_path), 'data_binding_sha256': sha(binding_path),
        'metric_records': records, 'policies': policies, 'input_pair_audit': paired,
        'identical_input_pair_count': sum(r['identical_replicate_accuracies'] for r in paired),
        'max_input_score_difference': max(r['absolute_score_difference'] for r in paired),
        'metrics_wall_seconds': metrics_seconds,
        'metrics_wall_seconds_scope': 'current attempt only; includes retained-record validation on resume',
        'included_cell_seconds': math.fsum(r['metric_seconds'] for r in records),
        'resumed_record_count': resumed_count, 'resumed_records_sha256': resume_sha,
        'worker_wall_seconds': time.monotonic() - started,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__},
        'neural_training_started': False, 'probe_fits': 44000,
        'label_scope': 'outer_training_only; model outcomes opened after all probe metrics completed'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('controller-root', 'data-root', 'package-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--resume-records',type=Path)
    p.add_argument('--resume-records-sha256')
    args = p.parse_args()
    run(args.controller_root.resolve(), args.data_root.resolve(), args.package_root.resolve(), args.output.resolve(),
        args.resume_records,args.resume_records_sha256)


if __name__ == '__main__':
    main()
