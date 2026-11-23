"""Finite post-hoc saved-score audit; launch via major_revision_guarded.py.

No training, checkpoint evaluation, graph-metric recomputation, bootstrap or
test-selected reporting scope. All four input/budget strata and four declared
portfolios are retained. The source package must be externally hash verified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

METRICS = ('homophily', 'adjusted_homophily', 'edge_label_informativeness')
FALLBACKS = ('mlp', 'graph', 'validation_selection')


def mean(values):
    values = list(values)
    if not values:
        raise ValueError('empty mean')
    return math.fsum(values) / len(values)


def dataset_mean(rows, values):
    groups = defaultdict(list)
    if len(rows) != len(values):
        raise ValueError('unpaired outcomes')
    for row, value in zip(rows, values):
        groups[row['dataset']].append(value)
    per_dataset = {key: mean(value) for key, value in sorted(groups.items())}
    return mean(per_dataset.values()), per_dataset


def candidates(rows, metric):
    values = sorted({r[metric] for r in rows if r[metric] is not None})
    result = [{'kind': 'all_graph', 'value': None}]
    for low, high in zip(values, values[1:]):
        mid = low + (high - low) / 2
        result.append({'kind': 'cutoff', 'value': mid if mid > low else high})
    return result + [{'kind': 'all_mlp', 'value': None}]


def action(value, threshold):
    if value is None:
        return 'abstain'
    if threshold['kind'] == 'all_graph':
        return 'graph'
    if threshold['kind'] == 'all_mlp':
        return 'mlp'
    return 'graph' if value >= threshold['value'] else 'mlp'


def audit_actions(rows, actions):
    """Dataset-weighted benefit/harm identity relative to always choosing graph."""
    if len(rows) != len(actions) or not set(actions) <= {'graph', 'mlp', 'abstain'}:
        raise ValueError('invalid actions')
    effective = ['mlp' if a == 'abstain' else a for a in actions]
    regret, benefit, harm, headroom = [], [], [], []
    for row, a in zip(rows, effective):
        gap = row['mlp_test'] - row['graph_test']
        headroom.append(max(gap, 0) * 100)
        benefit.append(max(gap, 0) * 100 if a == 'mlp' else 0)
        harm.append(max(-gap, 0) * 100 if a == 'mlp' else 0)
        regret.append((max(row['graph_test'], row['mlp_test']) - row[a + '_test']) * 100)
    loss, per_dataset = dataset_mean(rows, regret)
    b, bd = dataset_mean(rows, benefit)
    h, hd = dataset_mean(rows, harm)
    ceiling, cd = dataset_mean(rows, headroom)
    if not math.isclose(ceiling - loss, b - h, abs_tol=1e-10, rel_tol=0):
        raise ValueError('benefit/harm identity failed')
    return {'mean_regret_pp': loss, 'always_graph_headroom_pp': ceiling,
            'benefit_pp': b, 'harm_pp': h, 'net_gain_vs_graph_pp': b - h,
            'datasets': {d: {'regret_pp': per_dataset[d], 'benefit_pp': bd[d],
                              'harm_pp': hd[d], 'headroom_pp': cd[d]} for d in per_dataset},
            'coverage': 1 - actions.count('abstain') / len(actions),
            'effective_mlp_count': effective.count('mlp'), 'units': len(rows)}


def fit(rows, metric):
    thresholds = candidates(rows, metric)
    losses = []
    for threshold in thresholds:
        values = []
        for row in rows:
            a = action(row[metric], threshold)
            a = 'mlp' if a == 'abstain' else a
            values.append(max(row['graph_test'], row['mlp_test']) - row[a + '_test'])
        losses.append(dataset_mean(rows, values)[0])
    minimum = min(losses)
    index = next(i for i, loss in enumerate(losses) if loss <= minimum + 1e-12)
    return {'threshold': thresholds[index], 'candidate_count': len(thresholds),
            'training_datasets': sorted({r['dataset'] for r in rows}),
            'training_mean_regret_pp': 100 * losses[index]}


def calibrate(rows, metric):
    datasets = sorted({r['dataset'] for r in rows})
    if len(datasets) < 2:
        raise ValueError('LODO requires at least two datasets')
    folds = {d: fit([r for r in rows if r['dataset'] != d], metric) for d in datasets}
    actions = [action(r[metric], folds[r['dataset']]['threshold']) for r in rows]
    return {**audit_actions(rows, actions), 'folds': folds,
            'outcomes': [{'dataset': r['dataset'], 'seed': r['seed'], 'score': r[metric],
                          'action': a} for r, a in zip(rows, actions)]}


def sensitivity(rows, fixed):
    datasets = sorted({r['dataset'] for r in rows})
    excluded = {'all': [], 'exclude_chameleon_squirrel': ['Chameleon', 'Squirrel'],
                'exclude_cornell_wisconsin': ['Cornell', 'Wisconsin'],
                **{'leave_out:' + d: [d] for d in datasets}}
    result = {}
    for scenario, omitted in excluded.items():
        indices = [i for i, r in enumerate(rows) if r['dataset'] not in omitted]
        subset = [rows[i] for i in indices]
        result[scenario] = {}
        for fallback in FALLBACKS:
            actions = []
            for i in indices:
                a = fixed['historical_combined'][i]
                if a == 'abstain':
                    a = fixed['validation_selection'][i] if fallback == 'validation_selection' else fallback
                actions.append(a)
            scored = audit_actions(subset, actions)
            scored['original_diagnostic_coverage'] = 1 - sum(fixed['historical_combined'][i] == 'abstain' for i in indices) / len(indices)
            result[scenario][fallback] = scored
    return result


def run(package):
    evidence = package / 'new-evidence'
    base = [json.loads(p.read_bytes()) for p in sorted((evidence / 'input-retraining/records').rglob('*.json'))]
    extended = [json.loads(p.read_bytes()) for p in sorted((evidence / 'mlp24/records').rglob('*.json'))]
    expected = json.loads((evidence / 'mlp24/analysis.json').read_bytes())
    if len(base) != 1540 or len(extended) != 220:
        raise ValueError('incomplete saved record scope')
    key = lambda r: (r['condition'], r['dataset'], r['seed'], r['model'])
    replacements = {key(r): r for r in extended}
    if len(replacements) != 220 or set(replacements) != {key(r) for r in base if r['model'] == 'MLP'}:
        raise ValueError('unpaired extension')
    metrics = {(r['dataset'], r['seed'], r['split_id']): r for r in expected['metric_records']}
    output = {}
    for budget in (4, 24):
        records = base if budget == 4 else [replacements.get(key(r), r) for r in base]
        output[str(budget)] = {}
        for condition in expected['conditions']:
            units = defaultdict(dict)
            for r in records:
                if r['condition'] == condition:
                    units[r['dataset'], r['seed']][r['model']] = r
            prior = expected['by_mlp_budget'][str(budget)]
            policies = prior['original_strategies_and_h1_lodo']['portfolios_by_condition'][condition]['portfolios']
            published = {p['label']: p for p in prior['published_metrics']['portfolios_by_condition'][condition]}
            selected = [p for p in policies if len(p['portfolio']) == 6 or set(p['portfolio']) in ({'GCN'}, {'GAT'}, {'GCN', 'GAT'})]
            if len(units) != 110 or len(selected) != 4:
                raise ValueError('unexpected audit scope')
            output[str(budget)][condition] = {}
            for p in selected:
                rows = []
                for unit, models in sorted(units.items()):
                    g = sorted((models[m] for m in p['portfolio']), key=lambda r: (-r['validation_accuracy'], r['model']))[0]
                    m = models['MLP']
                    if g['split_id'] != m['split_id']:
                        raise ValueError('unpaired splits')
                    metric = metrics[g['dataset'], g['seed'], g['split_id']]
                    rows.append({'dataset': g['dataset'], 'seed': g['seed'], 'graph_test': g['test_accuracy'],
                                 'mlp_test': m['test_accuracy'], 'homophily': g['diagnostics']['homophily'],
                                 **{name: metric['metrics'][name]['value'] for name in METRICS[1:]}})
                fixed = {}
                audits = {}
                for name, strategy in p['strategies'].items():
                    if name == 'random_50_50':
                        continue
                    by_unit = {(r['dataset'], r['seed']): r['action'] for r in strategy['outcomes']}
                    actions = [by_unit[r['dataset'], r['seed']] for r in rows]
                    if 'expected_random' in actions:
                        continue
                    fixed[name] = actions
                    audits[name] = audit_actions(rows, actions)
                    if not math.isclose(audits[name]['mean_regret_pp'], strategy['mean_regret_pp'], abs_tol=1e-10, rel_tol=0):
                        raise ValueError('saved fixed-policy regret differs')
                calibrated = {metric: calibrate(rows, metric) for metric in METRICS}
                for metric in METRICS[1:]:
                    saved = published[p['label']]['metrics'][metric]
                    actual = calibrated[metric]
                    if actual['folds'] != saved['folds'] or [r['action'] for r in actual['outcomes']] != [r['action'] for r in saved['outcomes']]:
                        raise ValueError('matched published calibration reconstruction differs')
                output[str(budget)][condition][p['label']] = {
                    'portfolio': p['portfolio'], 'fixed_policy_audit': audits,
                    'matched_calibration': calibrated, 'fallback_exclusion': sensitivity(rows, fixed)}
                print(f'completed {budget} {condition} {p["label"]}', flush=True)
    return {'schema_version': '1.0', 'status': 'post_hoc_saved_score_audit', 'by_budget': output,
            'scope': {'mlp_budgets': [4, 24], 'conditions': expected['conditions'], 'portfolios_per_stratum': 4,
                      'fallback_exclusion_cases': 672, 'matched_calibration_folds': 528},
            'contract': 'Identical midpoint/constant candidate algorithm, loss, dataset weighting, missing-score MLP fallback and 1e-12 smallest-threshold ties for three statistics. Held-out scores and outcomes do not enter fitting. Existing graph construction retained.',
            'inference': 'Descriptive only; no new significance tests or confidence intervals.',
            'source_analysis_sha256': hashlib.sha256((evidence / 'mlp24/analysis.json').read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    result = run(args.package_root)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
