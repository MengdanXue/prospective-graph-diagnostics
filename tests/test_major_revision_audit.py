import copy
import unittest
from scripts import major_revision_audit as audit
from scripts import major_revision_guarded as guard
from tests.test_integrated_package_guarded import GuardedReaderTests
from unittest.mock import patch


class MatchedCalibrationTests(unittest.TestCase):
    def rows(self):
        return [dict(dataset=d, seed=0, homophily=s, graph_test=g, mlp_test=m)
                for d, s, g, m in [('A', .1, .2, .8), ('B', .9, .9, .3), ('C', .4, .7, .6)]]

    def test_heldout_outcomes_and_scores_never_choose_its_threshold(self):
        rows = self.rows()
        first = audit.calibrate(rows, 'homophily')['folds']['C']
        rows[-1].update(homophily=-100, graph_test=0, mlp_test=1)
        self.assertEqual(first, audit.calibrate(rows, 'homophily')['folds']['C'])

    def test_duplicate_seed_does_not_reweight_dataset(self):
        rows = self.rows()
        result = audit.fit(rows, 'homophily')
        self.assertEqual(result, audit.fit(rows + [copy.deepcopy(rows[0])] * 20, 'homophily'))

    def test_missing_score_has_declared_fallback_even_for_constant_graph(self):
        self.assertEqual(audit.action(None, {'kind': 'all_graph'}), 'abstain')
        self.assertEqual(audit.action(-100, {'kind': 'all_graph'}), 'graph')
        self.assertEqual(audit.action(100, {'kind': 'all_mlp'}), 'mlp')

    def test_benefit_harm_identity_with_opposing_decision_errors(self):
        rows = self.rows()
        result = audit.audit_actions(rows, ['mlp', 'abstain', 'graph'])
        self.assertAlmostEqual(result['benefit_pp'], 20)
        self.assertAlmostEqual(result['harm_pp'], 20)
        self.assertAlmostEqual(result['net_gain_vs_graph_pp'], 0)
        self.assertAlmostEqual(result['coverage'], 2 / 3)

    def test_fallback_changes_only_abstentions(self):
        rows = self.rows()
        fixed = {'historical_combined': ['mlp', 'abstain', 'graph'],
                 'validation_selection': ['graph', 'graph', 'mlp']}
        result = audit.sensitivity(rows, fixed)['all']
        self.assertEqual(result['graph']['effective_mlp_count'], 1)
        self.assertEqual(result['mlp']['effective_mlp_count'], 2)
        self.assertEqual(result['graph']['mean_regret_pp'], result['validation_selection']['mean_regret_pp'])


class RevisionGuardTests(GuardedReaderTests):
    def setUp(self):
        # Reuse controller boundary tests against the new fixed-child wrapper.
        self.patch = patch('tests.test_integrated_package_guarded.reader', guard)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        super().setUp()

    def test_default_preparation_is_read_only_and_never_imports_runtime(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with patch.object(guard, 'source_identity', return_value=guard.CONTROLLER_SHA), \
             patch.object(guard.importlib, 'import_module', side_effect=AssertionError('execution import')), \
             redirect_stdout(StringIO()):
            self.assertEqual(guard.main(self.arguments()), 0)
        self.assertFalse(self.output_root.exists())

    def test_cli_rejects_arbitrary_command_and_reader_path(self):
        from contextlib import redirect_stderr
        from io import StringIO
        for option in ('--command', '--reader', '--worker'):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                guard.parser().parse_args(self.arguments() + [option, 'anything'])
        command = guard.reader_command(self.package, self.root / 'out.json')
        self.assertEqual(command[2], str(guard.WORKER))
        self.assertEqual(command[3], '--package-root')
        self.assertEqual(len(command), 7)


if __name__ == '__main__':
    unittest.main()
