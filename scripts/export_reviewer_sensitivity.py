"""Export complete descriptive review analyses and manuscript tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def table(caption, label, spec, header, rows):
    return '\n'.join([r'\begin{table}[tbp]', r'\centering', r'\caption{' + caption + '}',
        r'\label{' + label + '}', r'\small', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}' + spec + '@{}}', r'\toprule', header + r' \\',
        r'\midrule', *[r' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def csv_file(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def name(c):
    return ('N' if c['condition'] == 'normalize_features' else 'CS') + '/' + str(c['budget'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('analysis', 'support', 'primary', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    data, support, primary = [json.loads(path.read_text()) for path in (a.analysis, a.support, a.primary)]
    cells = sorted(data['cells'], key=lambda c: (c['condition'] != 'normalize_features', c['budget'], len(c['portfolio']), c['portfolio']))
    full = [c for c in cells if len(c['portfolio']) == 6]
    scalar = lambda obj: {k: v for k, v in obj.items() if isinstance(v, (int, float))}
    influence, calibration, folds, per_dataset = [], [], [], []
    for c in cells:
        context = {'input_budget': name(c), 'portfolio': '+'.join(c['portfolio'])}
        for omit, policies in c['influence'].items():
            for policy, result in policies.items():
                influence.append({**context, 'omitted_dataset': omit, 'policy': policy, **scalar(result)})
        for metric, result in c['source_group_calibration'].items():
            calibration.append({**context, 'metric': metric, **scalar(result)})
            for group, fold in result['folds'].items():
                folds.append({**context, 'metric': metric, 'heldout_group': group,
                    'heldout_datasets': '+'.join(fold['heldout_datasets']),
                    'training_datasets': '+'.join(fold['training_datasets']),
                    'threshold_kind': fold['threshold']['kind'], 'threshold_value': fold['threshold']['value'],
                    'candidate_count': fold['candidate_count'], **scalar(fold['heldout'])})
        for dataset, result in c['influence']['all']['cpm_nt05']['datasets'].items():
            per_dataset.append({**context, 'dataset': dataset, **result})
    for stem, records in [('influence', influence), ('group-calibration', calibration),
                          ('group-folds', folds), ('cpm-dataset-accounting', per_dataset)]:
        csv_file(a.output / (stem + '.csv'), records)

    rows = []
    for c in full:
        all_ = c['influence']['all']; omit = c['influence']['Roman-empire']
        share = 100 * all_['cpm_nt05']['datasets']['Roman-empire']['regret_pp'] / (11 * all_['cpm_nt05']['mean_regret_pp'])
        rows.append([name(c), f"{all_['cpm_nt05']['mean_regret_pp']:.3f}", f"{all_['always_graph']['mean_regret_pp']:.3f}",
                     f'{share:.1f}\\%', f"{omit['cpm_nt05']['mean_regret_pp']:.3f}", f"{omit['always_graph']['mean_regret_pp']:.3f}"])
    (a.output/'cpm-influence-main.tex').write_text(table(
        'CPM-.5 full-portfolio regret and influence of Roman-empire (RE). All-11 results remain primary; omission is a descriptive sensitivity. The RE share is its contribution to total CPM regret, not to net harm. Values are percentage points except the share.',
        'tab:cpm_influence', 'lrrrrr', r'Input/trials & CPM & Graph & RE share & CPM w/o RE & Graph w/o RE', rows))

    rows = []
    for c in full:
        group = c['source_group_calibration']; lodo = c['matched_lodo_verified']
        rows.append([name(c), f"{c['influence']['all']['always_graph']['mean_regret_pp']:.3f}",
            f"{lodo['adjusted_homophily']['mean_regret_pp']:.3f}",
            *[f"{group[m]['mean_regret_pp']:.3f}" for m in ('adjusted_homophily','homophily','edge_label_informativeness')]])
    (a.output/'source-group-main.tex').write_text(table(
        'Calibration transfer in the full portfolio. Adjusted-LODO leaves one dataset out; Adjusted-G, $h_1$-G and LI-G leave the entire source group out. Regret is averaged equally over the eleven held-out datasets, not over six groups. All values are descriptive percentage points.',
        'tab:source_group_main', 'lrrrrr', r'Input/trials & Graph & Adjusted-LODO & Adjusted-G & $h_1$-G & LI-G', rows))

    appendix = []
    rows = []
    for omit in ['all', *sorted(full[0]['influence'].keys() - {'all'})]:
        rows.append(['All 11' if omit == 'all' else omit, *[f"{c['influence'][omit]['cpm_nt05']['mean_regret_pp'] - c['influence'][omit]['always_graph']['mean_regret_pp']:+.3f}" for c in full]])
    appendix.append(table('All single-dataset influence checks for CPM-.5 in the full portfolio. Entries are CPM-minus-always-graph regret in percentage points; negative favors CPM. The omitted dataset is named in the first column. Actions are held fixed and the remaining datasets are reweighted equally.',
        'tab:cpm_all_omissions', 'lrrrr', 'Omission & N/4 & N/24 & CS/4 & CS/24', rows))
    for c in full:
        rows = [[d, *[f'{r[k]:.3f}' for k in ('headroom_pp','regret_pp','benefit_pp','harm_pp')], f"{r['benefit_pp']-r['harm_pp']:+.3f}"]
                for d, r in sorted(c['influence']['all']['cpm_nt05']['datasets'].items())]
        appendix.append(table('CPM-.5 full-portfolio dataset accounting, '+name(c)+'. Dataset means over ten seeds, in percentage points. Net gain equals benefit minus harm relative to always-graph; headroom equals always-graph regret.',
            'tab:cpm_account_'+name(c).replace('/','_'), 'lrrrrr', 'Dataset & Headroom & Regret & Benefit & Harm & Net gain', rows))
    for input_name in ('N','CS'):
        rows = []
        for c in cells:
            if name(c).split('/')[0] != input_name:
                continue
            for metric, label in [('homophily', '$h_1$'), ('adjusted_homophily', 'Adjusted'), ('edge_label_informativeness', 'LI')]:
                r = c['source_group_calibration'][metric]; w = r['folds']['webkb']
                rows.append(['Full' if len(c['portfolio'])==6 else '+'.join(c['portfolio']), str(c['budget']), label,
                    f"{r['mean_regret_pp']:.3f}", f"{r['net_gain_vs_graph_pp']:+.3f}",
                    f"{w['heldout']['net_gain_vs_graph_pp']:+.3f}"])
        appendix.append(table('Complete source-group calibration, '+input_name+' inputs. All-dataset regret and net gain use equal dataset weights; WebKB gain averages Cornell and Wisconsin only, both held out from fitting. Values are descriptive percentage points.',
            'tab:group_all_'+input_name, 'lllr rr'.replace(' ', ''), 'Portfolio & Trials & Metric & Regret & Net gain & WebKB gain', rows))
    (a.output/'sensitivity-appendix-tables.tex').write_text('\n'.join(appendix))

    rows = []
    for dataset in sorted({r['dataset'] for r in support['rows']}):
        records = [r for r in support['rows'] if r['dataset']==dataset]
        sizes = {tuple(s) for r in records for s in r['partition_sizes']}
        counts = [s for r in records for s in r['supports_by_repeat']]
        if len(sizes) != 1:
            raise ValueError('table needs explicit size ranges')
        n,f,h = next(iter(sizes))
        fit = [v for c in counts for v in c['fit']]; hold = [v for c in counts for v in c['holdout']]
        rows.append([dataset,str(n),str(f),str(h),f'{100*f/n:.1f}',f'{min(fit)}--{max(fit)}',f'{min(hold)}--{max(hold)}',str(sum(v==0 for v in counts[0]['holdout']))])
    (a.output/'partition-support-table.tex').write_text(table(
        'Realized inner-probe partitions. Counts and support ranges cover all ten seeds and 100 repeats, shared by N and CS. The last column counts classes absent from each holdout; this count is constant across repeats and seeds. Full per-class supports and matching partition hashes are retained. No classifier was refitted for this audit.',
        'tab:cpm_support', 'lrrrrrrr', r'Dataset & Sample & Fit & Hold & Fit (\%) & Fit/class & Hold/class & Absent', rows))

    labels = dict(always_mlp='Always-MLP',always_graph='Always-graph',random_50_50='Random 50/50',homophily_only='Homophily-only',degree_only='Degree-only',homophily_plus_degree='Homophily+degree',validation_selection='Validation selection',two_hop_only='Two-hop-only')
    rows, inference = [], []
    def regret(u, method):
        g,m = u['selected_graph_test'],u['selected_mlp_test']
        action = u['decisions'][method]['action']
        chosen = (g+m)/2 if action=='expected_random' else g if action=='graph' else m
        return max(g,m)-chosen
    for r in primary['paired_comparisons']:
        diffs = defaultdict(list)
        for u in primary['units']:
            diffs[u['dataset']].append(regret(u,r['method'])-regret(u,r['reference']))
        values = [mean(v) for v in diffs.values()]
        assert abs(mean(values)-r['mean_difference']) < 1e-12
        count = sum(abs(v)>1e-12 for v in values)
        lo,hi = [100*v for v in r['bootstrap_95_ci']]
        rows.append([labels[r['method']], f"{100*r['mean_difference']:+.2f}", f'[{lo:.2f}, {hi:.2f}]',str(count), f"{r['raw_p']:.6f}".rstrip('0').rstrip('.'),f"{r['holm_adjusted_p']:.5f}".rstrip('0').rstrip('.')])
        inference.append({**r, 'nonzero_dataset_count':count})
    (a.output/'primary-inference-table.tex').write_text(table(
        'Complete primary eight-comparison family. Effect is comparator-minus-Combined mean regret (percentage points); negative favors the comparator. Bootstrap intervals are unadjusted 95\\% intervals over eleven dataset means. $n_{\\ne0}$ counts nonzero dataset differences; raw sign-flip and Holm-adjusted $p$ values retain the frozen audit.',
        'tab:primary_inference', 'lrlrrr', r'Comparator & Effect & 95\% interval & $n_{\ne0}$ & Raw $p$ & Holm $p$',rows))
    (a.output/'primary-inference.json').write_text(json.dumps(inference,indent=2)+'\n')
    print(json.dumps({'influence_rows':len(influence),'calibration_rows':len(calibration),'fold_rows':len(folds),'dataset_accounting_rows':len(per_dataset),'primary_comparisons':len(inference)}))


if __name__ == '__main__':
    main()
