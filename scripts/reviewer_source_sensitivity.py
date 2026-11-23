"""Descriptive influence and source-group calibration from retained records.

The scope is supplied as a separately saved protocol. This module uses the
standard library and the existing pure calibration/accounting functions only;
it never loads graph data, fits a classifier or trains a neural predictor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.major_revision_audit import action, audit_actions, calibrate, candidates, dataset_mean
from scripts.reader_reproduce import assert_same


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_rows(rows):
    keys = [(r['dataset'], r['seed']) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate dataset-seed rows')
    for r in rows:
        if not all(math.isfinite(r[k]) and 0 <= r[k] <= 1 for k in ('graph_test', 'mlp_test')):
            raise ValueError('invalid candidate accuracy')


def fit_with_trace(rows, metric, tolerance=1e-12):
    """Keep every candidate/loss; fit receives training rows only."""
    if not rows:
        raise ValueError('empty calibration training set')
    if any(r[metric] is not None and not math.isfinite(r[metric]) for r in rows):
        raise ValueError('nonfinite metric')
    threshold_list = candidates(rows, metric)
    losses = []
    for threshold in threshold_list:
        decisions = [action(r[metric], threshold) for r in rows]
        loss = [max(r['graph_test'], r['mlp_test']) - r[('mlp' if a == 'abstain' else a) + '_test']
                for r, a in zip(rows, decisions)]
        losses.append(dataset_mean(rows, loss)[0])
    index = next(i for i, loss in enumerate(losses) if loss <= min(losses) + tolerance)
    return {'threshold': threshold_list[index], 'selected_candidate_index': index,
            'training_datasets': sorted({r['dataset'] for r in rows}),
            'candidate_count': len(threshold_list), 'training_mean_regret_pp': 100 * losses[index],
            'candidates': [{'threshold': t, 'training_mean_regret_pp': 100 * loss}
                           for t, loss in zip(threshold_list, losses)]}


def calibrate_groups(rows, metric, groups, tolerance=1e-12):
    validate_rows(rows)
    grouped_names = [d for names in groups.values() for d in names]
    if any(not names for names in groups.values()) or len(grouped_names) != len(set(grouped_names)):
        raise ValueError('empty or overlapping source groups')
    if set(grouped_names) != {r['dataset'] for r in rows} or len(groups) < 2:
        raise ValueError('groups must partition every dataset with at least two groups')
    folds, actions_by_unit = {}, {}
    for group, names in groups.items():
        training = [r for r in rows if r['dataset'] not in names]
        held = [r for r in rows if r['dataset'] in names]
        fitted = fit_with_trace(training, metric, tolerance)
        decisions = [action(r[metric], fitted['threshold']) for r in held]
        if set(fitted['training_datasets']) & set(names):
            raise ValueError('held-out group leaked into fitting')
        for row, decision in zip(held, decisions):
            actions_by_unit[row['dataset'], row['seed']] = decision
        folds[group] = {**fitted, 'heldout_datasets': sorted(names),
            'training_inputs': [{k: r[k] for k in ('dataset', 'seed', metric, 'graph_test', 'mlp_test')}
                                for r in training],
            'heldout_outcomes': [{**r, 'action': a} for r, a in zip(held, decisions)],
            'heldout': audit_actions(held, decisions)}
    decisions = [actions_by_unit[r['dataset'], r['seed']] for r in rows]
    return {**audit_actions(rows, decisions), 'folds': folds,
            'outcomes': [{'dataset': r['dataset'], 'seed': r['seed'], 'score': r[metric], 'action': a}
                         for r, a in zip(rows, decisions)]}


def influence(rows, fixed_actions):
    validate_rows(rows)
    names = sorted({r['dataset'] for r in rows})
    if len(names) < 2 or any(len(a) != len(rows) for a in fixed_actions.values()):
        raise ValueError('invalid influence scope')
    output = {}
    for omitted in [None, *names]:
        indices = [i for i, r in enumerate(rows) if r['dataset'] != omitted]
        output['all' if omitted is None else omitted] = {
            policy: audit_actions([rows[i] for i in indices], [actions[i] for i in indices])
            for policy, actions in fixed_actions.items()}
    return output


def run(evidence_root, published_metrics, reference, protocol):
    retained = read(evidence_root / 'cpm-extension.json')
    metrics = {(r['dataset'], r['seed']): r for r in published_metrics['metric_records']}
    datasets = {d for names in protocol['source_groups'].values() for d in names}
    if len(metrics) != 110 or {k[0] for k in metrics} != datasets:
        raise ValueError('incomplete published statistics')
    base_paths = sorted((evidence_root / 'new-evidence/input-retraining/records').glob('*.json'))
    extended_paths = sorted((evidence_root / 'new-evidence/mlp24/records').glob('*.json'))
    if (len(base_paths), len(extended_paths)) != (1540, 220):
        raise ValueError('incomplete candidate records')
    key = lambda r: (r['condition'], r['dataset'], r['seed'], r['model'])
    base = {key(r): r for r in map(read, base_paths)}
    extended = {key(r): r for r in map(read, extended_paths)}
    if len(base) != 1540 or len(extended) != 220 or set(extended) != {k for k in base if k[-1] == 'MLP'}:
        raise ValueError('duplicate or unpaired candidates')
    observed = set()
    cells = []
    for cell in retained['policies']:
        condition, budget, portfolio = cell['condition'], cell['budget'], cell['portfolio']
        identity = (condition, budget, tuple(sorted(portfolio)))
        if identity in observed:
            raise ValueError('duplicate policy cell')
        observed.add(identity)
        rows = []
        for saved in cell['rows']:
            dataset, seed = saved['dataset'], saved['seed']
            graph = min((base[condition, dataset, seed, model] for model in portfolio),
                        key=lambda r: (-r['validation_accuracy'], r['model']))
            mlp = (base if budget == 4 else extended)[condition, dataset, seed, 'MLP']
            metric = metrics[dataset, seed]
            if graph['split_id'] != mlp['split_id'] or graph['split_id'] != metric['split_id']:
                raise ValueError('candidate/statistic split mismatch')
            assert_same({k: saved[k] for k in ('graph_test', 'mlp_test', 'graph_model')},
                        {'graph_test': graph['test_accuracy'], 'mlp_test': mlp['test_accuracy'], 'graph_model': graph['model']})
            rows.append({**saved, 'homophily': graph['diagnostics']['homophily'],
                         **{name: value['value'] for name, value in metric['metrics'].items()}})
        if len(rows) != 110 or {r['dataset'] for r in rows} != datasets:
            raise ValueError('incomplete cell')
        fixed = {name: (['mlp'] * len(rows) if name == 'always_mlp' else cell['fixed'][name]['actions'])
                 for name in protocol['influence']['policies']}
        for name, actions in fixed.items():
            if name != 'always_mlp':
                assert_same(audit_actions(rows, actions), {k: v for k, v in cell['fixed'][name].items() if k != 'actions'})
        saved_reference = next(r for r in reference['by_budget'][str(budget)][condition].values()
                               if set(r['portfolio']) == set(portfolio))
        group_results, lodo_results = {}, {}
        for metric in protocol['calibration']['metrics']:
            lodo_results[metric] = calibrate(rows, metric)
            assert_same(lodo_results[metric], saved_reference['matched_calibration'][metric])
            group_results[metric] = calibrate_groups(rows, metric, protocol['source_groups'],
                protocol['calibration']['tie_tolerance_accuracy_units'])
        cells.append({'budget': budget, 'condition': condition, 'portfolio': portfolio,
            'rows': rows, 'fixed_actions': fixed, 'influence': influence(rows, fixed),
            'matched_lodo_verified': lodo_results, 'source_group_calibration': group_results})
    expected = {(c, b, tuple(sorted(p))) for c in protocol['conditions']
                for b in protocol['mlp_budgets'] for p in protocol['portfolios']}
    if observed != expected:
        raise ValueError('observed policy cells differ from fixed protocol')
    decision_vectors = defaultdict(set)
    for cell in cells:
        for metric, result in cell['source_group_calibration'].items():
            decision_vectors[('+'.join(sorted(cell['portfolio'])), metric)].add(tuple(r['action'] for r in result['outcomes']))
    return {'schema': 'reviewer-influence-source-results/1', 'status': 'complete_descriptive_analysis',
        'scope': {'cells': 16, 'influence_scenarios': 192, 'fixed_policy_evaluations': 768,
                  'matched_lodo_folds_verified': 528, 'source_group_folds': 288,
                  'new_classifier_or_neural_fits': 0},
        'protocol': protocol, 'cells': cells,
        'distinct_group_action_vectors_across_four_input_budget_settings':
            [{'portfolio': p, 'metric': m, 'count': len(v)} for (p, m), v in sorted(decision_vectors.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--published-metrics', type=Path, required=True)
    parser.add_argument('--matched-reference', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--scope-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    receipt = read(args.scope_receipt)
    for key, path in [('protocol_sha256', args.protocol), ('published_metrics_sha256', args.published_metrics),
                      ('matched_reference_sha256', args.matched_reference), ('cpm_sha256', args.evidence_root/'cpm-extension.json')]:
        if receipt[key] != sha(path):
            raise ValueError('input differs from scope receipt: ' + key)
    started = time.monotonic()
    result = run(args.evidence_root, read(args.published_metrics), read(args.matched_reference), read(args.protocol))
    result['provenance'] = {**receipt, 'script_sha256': sha(Path(__file__)),
                            'scope_receipt_sha256': sha(args.scope_receipt)}
    result['wall_seconds'] = time.monotonic() - started
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'scope': result['scope'], 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
