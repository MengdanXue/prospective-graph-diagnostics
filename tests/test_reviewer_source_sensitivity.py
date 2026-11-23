import copy
import unittest

from scripts.reviewer_source_sensitivity import calibrate_groups, fit_with_trace, influence


class SourceSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'dataset':'A','seed':0,'score':0.1,'graph_test':0.2,'mlp_test':0.8},
            {'dataset':'A','seed':1,'score':0.2,'graph_test':0.3,'mlp_test':0.9},
            {'dataset':'B','seed':0,'score':0.8,'graph_test':0.9,'mlp_test':0.4},
            {'dataset':'C','seed':0,'score':0.7,'graph_test':0.7,'mlp_test':0.5}]
        self.groups = {'one':['A'], 'two':['B','C']}

    def test_whole_heldout_group_excluded_from_candidates_and_loss(self):
        first = calibrate_groups(self.rows, 'score', self.groups)
        poisoned = copy.deepcopy(self.rows)
        for row in poisoned:
            if row['dataset'] in ['B','C']:
                row.update(score=1000, graph_test=0.0, mlp_test=1.0)
        second = calibrate_groups(poisoned, 'score', self.groups)
        for key in ('threshold','candidates','training_inputs','training_datasets'):
            self.assertEqual(first['folds']['two'][key], second['folds']['two'][key])
        self.assertEqual(first['folds']['two']['training_datasets'], ['A'])

    def test_training_loss_weights_datasets_not_seed_rows(self):
        original = fit_with_trace(self.rows, 'score')
        replicated = self.rows + [{**r,'seed':r['seed']+10} for r in self.rows if r['dataset']=='A']
        self.assertEqual(original, fit_with_trace(replicated, 'score'))

    def test_omission_keeps_actions_and_dataset_mean_arithmetic(self):
        acts = {'probe':['mlp','graph','mlp','graph'], 'graph':['graph']*4}
        result = influence(self.rows, acts)
        self.assertAlmostEqual(result['all']['probe']['mean_regret_pp'], (30+50+0)/3)
        self.assertAlmostEqual(result['A']['probe']['mean_regret_pp'], 25)
        self.assertAlmostEqual(result['B']['probe']['mean_regret_pp'], 15)
        self.assertEqual(result['A']['probe']['effective_mlp_count'], 1)

    def test_ties_keep_existing_first_constant_rule(self):
        rows = [{**r,'mlp_test':r['graph_test']} for r in self.rows]
        self.assertEqual(fit_with_trace(rows, 'score')['threshold'], {'kind':'all_graph','value':None})

    def test_groups_must_cover_each_dataset_exactly_once(self):
        for groups in ({'x':['A'],'y':['B']}, {'x':['A','B'],'y':['B','C']}, {'x':['A','B','C']}):
            with self.assertRaises(ValueError):
                calibrate_groups(self.rows, 'score', groups)

    def test_duplicate_units_and_nonfinite_values_fail(self):
        with self.assertRaises(ValueError):
            influence(self.rows+[self.rows[0]], {'graph':['graph']*5})
        with self.assertRaises(ValueError):
            fit_with_trace([{**self.rows[0],'score':float('nan')}], 'score')


if __name__ == '__main__':
    unittest.main()
