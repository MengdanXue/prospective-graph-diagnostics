"""Reader commands for public inputs, CPM reconstruction, scoring, and new training.

Paths are explicit. No author controller, Git checkout, ledger or CI receipt is
required. New reader runs use exclusive output files and retain their own identity.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

def versions():
    import numpy, scipy, sklearn, torch, torch_geometric
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                torch=torch.__version__, torch_geometric=torch_geometric.__version__)

def assert_same(left, right, path='root', tolerance=1e-12):
    """Reject structural changes and compare finite numeric fields within tolerance."""
    import math
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f'field mismatch at {path}')
        for key in left:
            assert_same(left[key], right[key], f'{path}/{key}', tolerance)
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError(f'length mismatch at {path}')
        for i, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f'{path}/{i}', tolerance)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        if not math.isclose(left, right, abs_tol=tolerance, rel_tol=0):
            raise ValueError(f'numeric mismatch at {path}: {left} != {right}')
    elif left != right:
        raise ValueError(f'value mismatch at {path}')

def score(evidence, output):
    from scripts.cpm_extension_worker import evaluate
    from scripts.major_revision_audit import audit_actions
    started = time.monotonic()
    retained = read(evidence / 'cpm-extension.json')
    policies = evaluate(evidence, retained['metric_records'], retained['protocol'])
    assert_same(policies, retained['policies'])
    for cell in policies:
        actions = ['mlp'] * len(cell['rows'])
        cell['fixed']['always_mlp'] = {**audit_actions(cell['rows'], actions), 'actions': actions}
    resolution = {}
    for condition in retained['protocol']['conditions']:
        grouped = defaultdict(list)
        for row in retained['metric_records']:
            if row['condition'] == condition:
                grouped[row['dataset']].append(row['cpm_gnb'])
        resolution[condition] = {name: {'minimum': min(v), 'maximum': max(v),
            'exact_zero': v.count(0.0), 'exact_one': v.count(1.0), 'distinct_scores': len(set(v))}
            for name, v in sorted(grouped.items())}
    result = dict(schema='reader-score/1', status='passed', tolerance=1e-12,
        original_policy_cells_verified=112, additional_constant_reference_cells=16,
        policies=policies, score_resolution=resolution, wall_seconds=time.monotonic()-started,
        scope='Saved-score reconstruction and new constant-reference arithmetic; no model fitting')
    write(output, result)
    return result

def train(name, seed, condition, model_id, trials, data_root, output, binding, device_name):
    """Fit the declared grid afresh; retain checkpoints and evaluate its winner once."""
    import torch
    from scripts.reader_data import load_public_dataset, fit_transform_features_bounded
    from experiments.run_prospective_benchmark import _train_trial, select_trial, seed_everything, _accuracy
    from experiments.prospective_models import build_model, prepare_h2_adjacencies
    started = time.monotonic()
    config = read(ROOT / 'configs/prospective_benchmark_v2.json')['training']
    if trials == 24:
        if model_id != 'MLP':
            raise ValueError('the expanded grid applies only to MLP')
        config = read(ROOT / 'configs/reader_mlp24_grid.json')
    x, y, edge, split, split_id, hashes = load_public_dataset(name, seed, data_root, binding)
    x, metadata = fit_transform_features_bounded(x, split['train'], condition)
    device = torch.device(device_name)
    h2 = prepare_h2_adjacencies(edge, num_nodes=len(x)) if model_id == 'H2GCN' else None
    indices = {k: torch.from_numpy(v) for k, v in split.items()}
    checkpoints = output.with_suffix('.checkpoints')
    checkpoints.mkdir(parents=True, exist_ok=False)
    rows, states = [], {}
    for i, spec in enumerate(config['trials']):
        trial_id = f'trial_{i:03d}'
        row, state = _train_trial(model_id=model_id, seed=seed, trial_id=trial_id,
            trial=spec, x=x, y=y, edge_index=edge, train_indices=indices['train'],
            validation_indices=indices['validation'], hidden_channels=config['hidden_channels'],
            max_epochs=config['max_epochs'], patience=config['patience'],
            weight_decay=spec.get('weight_decay', config.get('weight_decay', .0005)),
            device=device, h2_adjacencies=h2)
        rows.append(row); states[trial_id] = state
        torch.save(state, checkpoints / f'{trial_id}.pt')
    winner = select_trial(rows)
    seed_everything(seed)
    model = build_model(model_id, num_nodes=len(x), in_channels=x.shape[1],
        hidden_channels=config['hidden_channels'], out_channels=int(y.max())+1,
        dropout=winner['configuration']['dropout'], edge_index=edge, h2_adjacencies=h2).to(device)
    model.load_state_dict(states[winner['trial_id']]); model.eval()
    with torch.no_grad():
        accuracy = _accuracy(model(x.to(device), edge.to(device)), y.to(device), indices['test'].to(device))
    result = dict(schema='reader-training/1', status='complete_new_reproduction', dataset=name,
        seed=seed, split_id=split_id, condition=condition, model=model_id, trials=rows,
        selected_trial_id=winner['trial_id'], test_accuracy=accuracy,
        validation_accuracy=winner['validation_accuracy'], test_evaluations_after_selection=1,
        grid=config, versions=versions(), feature_metadata=metadata, input_tensor_sha256=hashes,
        wall_seconds=time.monotonic()-started, checkpoints=checkpoints.name,
        scope='Fresh fits of every declared candidate; no inherited author checkpoint; no claim of bitwise neural training repeatability')
    write(output, result)
    return result

def probe(name, seed, data_root, evidence, output, binding):
    from scripts.reader_data import load_public_dataset, fit_transform_features_bounded, array_sha256
    from scripts.cpm_probe import aggregation_operator, probe_training_rows, seed_for
    from scripts.major_revision_audit import audit_actions
    started = time.monotonic()
    protocol = read(ROOT / 'configs/cpm_train_only_v1.json')
    x, y, edge, split, split_id, tensors = load_public_dataset(name, seed, data_root, binding)
    operator = aggregation_operator(edge.numpy(), len(x))
    train = split['train']
    records = []
    for condition in protocol['conditions']:
        transformed, _ = fit_transform_features_bounded(x, train, condition)
        digest = array_sha256(transformed)
        entry = binding['datasets'][name]
        expected = entry['normalize_features_sha256'] if condition == 'normalize_features' else next(
            v['transformed_feature_sha256'] for v in entry['candidate_checks'] if v['seed'] == seed)
        if digest != expected:
            raise ValueError(f'transformed input mismatch: {condition}')
        metric = probe_training_rows(transformed.numpy()[train], operator[train] @ transformed.numpy(),
            y.numpy()[train].copy(), seed=seed_for(name, split_id), repeats=100, cap=500)
        records.append(dict(dataset=name, seed=seed, split_id=split_id, condition=condition,
                            transformed_feature_sha256=digest, **metric))
    # Candidate outcomes are opened only after both independent probe computations.
    retained = read(evidence / 'cpm-extension.json')
    scored = []
    for row in records:
        original = next(r for r in retained['metric_records'] if
            (r['dataset'],r['seed'],r['condition']) == (name,seed,row['condition']))
        assert_same(row, {key: original[key] for key in row})
        for cell in retained['policies']:
            if cell['condition'] != row['condition']:
                continue
            unit = next(r for r in cell['rows'] if (r['dataset'],r['seed']) == (name,seed))
            action = 'graph' if row['cpm_gnb'] >= .5 else 'mlp'
            scored.append(dict(budget=cell['budget'], portfolio=cell['portfolio'],
                condition=cell['condition'], action=action, **audit_actions([unit], [action])))
    result = dict(schema='reader-probe/1', status='passed', dataset=name, seed=seed,
        tensor_sha256=tensors, records=records, policy_scores=scored, versions=versions(),
        tolerance=1e-12, paired_inner_repeats=200, classifier_fits=400,
        wall_seconds=time.monotonic()-started, neural_training_started=False,
        scope='Public-raw-data CPM reproduction for this specified dataset and seed only')
    write(output, result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('acquire', 'probe', 'train'):
        p = sub.add_parser(command)
        p.add_argument('--dataset', required=True)
        p.add_argument('--seed', type=int, choices=range(10), default=0)
        p.add_argument('--data-root', type=Path, required=True)
        p.add_argument('--binding', type=Path, default=ROOT/'configs/reader_data_bindings_v1.json')
        p.add_argument('--output', type=Path, required=True)
        if command == 'probe':
            p.add_argument('--evidence-root', type=Path, required=True)
        if command == 'train':
            p.add_argument('--model', required=True, choices=('MLP','GCN','GAT','GraphSAGE','H2GCN','LINKX','GPR-GNN'))
            p.add_argument('--condition', required=True, choices=('normalize_features','normalize_centered_scaled'))
            p.add_argument('--trials', type=int, choices=(4,24), default=4)
            p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p = sub.add_parser('score')
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; use a fresh path to preserve prior results')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if args.command == 'score':
            result = score(args.evidence_root, args.output)
        else:
            binding = read(args.binding)
            if args.dataset not in binding['datasets']:
                parser.error('dataset outside the published binding')
            if args.command == 'acquire':
                from scripts.reader_data import acquire_raw, load_public_dataset
                started = time.monotonic()
                receipts = acquire_raw(binding['datasets'][args.dataset], args.data_root)
                *_, split_id, hashes = load_public_dataset(args.dataset, args.seed, args.data_root, binding)
                result = dict(status='passed', dataset=args.dataset, seed=args.seed, raw_files=receipts,
                    split_id=split_id, tensor_sha256=hashes, versions=versions(), wall_seconds=time.monotonic()-started)
                write(args.output, result)
            elif args.command == 'probe':
                result = probe(args.dataset, args.seed, args.data_root, args.evidence_root, args.output, binding)
            else:
                result = train(args.dataset, args.seed, args.condition, args.model, args.trials,
                    args.data_root, args.output, binding, args.device)
    print(json.dumps({k: result[k] for k in ('status','wall_seconds') if k in result}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
