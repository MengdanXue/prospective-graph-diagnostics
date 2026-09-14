"""Check that the proposed budget remains consistent and cannot imply activation."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/diagnostic/h2gcn_cpu_performance_v1'


class H2BudgetProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = json.loads((RESULTS / 'budget_proposal.json').read_text())
        cls.summary = json.loads((RESULTS / 'summary.json').read_text())
        cls.parent_path = ROOT / cls.proposal['parent_config']
        cls.parent = json.loads(cls.parent_path.read_text())

    def test_pending_proposal_preserves_actual_source_and_effective_caps(self):
        p = self.proposal
        self.assertEqual(hashlib.sha256((RESULTS/'summary.json').read_bytes()).hexdigest(), p['diagnostic_summary_file_sha256'])
        receipt = self.summary['source_ci_receipt']
        self.assertEqual(hashlib.sha256((ROOT/receipt['path']).read_bytes()).hexdigest(), receipt['sha256'])
        self.assertEqual(p['status'], 'AWAITING_USER_CONFIRMATION')
        self.assertTrue(p['proposal_only'])
        for key in ('budget_change_authorized', 'formal_training_enabled', 'formal_launch_authorized'):
            self.assertIs(p[key], False)
        self.assertEqual(hashlib.sha256(self.parent_path.read_bytes()).hexdigest(), p['parent_config_file_sha256'])
        self.assertIs(self.parent['formal_training_enabled'], False)
        actual = self.parent['resource_budget']
        self.assertEqual(actual['formal_total_worker_wall_seconds'], p['unchanged_effective_caps_seconds']['formal_total'])
        self.assertEqual(actual['formal_model_unit_wall_seconds'], p['unchanged_effective_caps_seconds']['four_trial_unit'])
        self.assertEqual(p['retained_h2_execution']['threads'], self.parent['execution']['torch_num_threads'])

    def test_total_contains_reserves_and_attempt_caps_do_not_reset(self):
        p = self.proposal
        caps = p['proposed_caps_seconds']
        self.assertEqual(caps['total_cumulative'], sum(caps[k] for k in
            ('resource_acceptance_subbudget', 'coordination_validation_analysis_subbudget', 'formal_all_attempts_subbudget')))
        self.assertEqual(caps['paired_batch_all_attempts'], 2*caps['four_trial_unit_all_attempts'])
        self.assertGreater(caps['formal_all_attempts_subbudget'], p['basis']['v2_conservative_seconds'])
        self.assertFalse(p['basis']['use_two_dataset_speedup_extrapolation'])
        self.assertTrue(p['accounting']['all_attempts_cumulative_per_unit_batch_subbudget_and_total'])
        for key in ('unit_clock_reset_on_retry', 'batch_clock_reset_on_retry', 'unused_reserves_auto_transfer'):
            self.assertFalse(p['accounting'][key])
        self.assertTrue(p['accounting']['live_process_wall_including_suspend_counts'])
        self.assertEqual(p['stop_resume']['automatic_retries'], 0)
        self.assertTrue(p['stop_resume']['insufficient_remaining_budget_requires_user_amendment'])

    def test_pairing_scope_and_original_training_rules_are_unchanged(self):
        p = self.proposal
        for key, value in p['scope'].items():
            self.assertEqual(value, self.parent[key])
        b = p['batching']
        self.assertEqual(b['paired_batches'], len(p['scope']['datasets'])*len(p['scope']['seeds'])*len(p['scope']['models']))
        self.assertEqual(b['paired_batches']*b['units_per_batch'], p['scope']['expected_records'])
        self.assertEqual(p['scope']['expected_records']*b['trials_per_unit'], p['scope']['expected_trials'])
        for key, value in p['training_rules_unchanged'].items():
            self.assertEqual(value, self.parent['training'][key])
        self.assertTrue(b['completion_requires_both_conditions_validated'])
        self.assertTrue(b['resume_missing_partner_before_next_batch'])

    def test_partial_evidence_is_neither_full_scope_nor_qualified_optimization(self):
        s = self.summary
        n = s['scope']
        self.assertEqual(n['planned_workers'], n['completed_workers']+n['interrupted_workers']+n['not_launched_workers'])
        self.assertEqual(n['launched_workers'], len(s['completed_workers'])+len(s['failed_jobs']))
        self.assertEqual(n['known_durable_steps'], n['complete_worker_optimization_steps']+n['durable_partial_step_records'])
        self.assertFalse(s['fixed_scope_complete'])
        self.assertFalse(s['resource']['time_caps_passed'])
        self.assertEqual(s['qualification']['qualifying_settings'], [])
        self.assertEqual(s['recommendation'], self.proposal['recommendation'])
        for kind, counts in s['comparison_counts'].items():
            selected = [c for c in s['comparisons'] if c['kind']==kind]
            self.assertEqual(sum(counts.values()), len(selected))
            for status, number in counts.items():
                self.assertEqual(number, sum(c['evidence_status']==status for c in selected))
            for c in selected:
                self.assertEqual(c['evidence_status']=='unavailable', 'comparison' not in c)
        for field in ('formal_records', 'validation_evaluations', 'test_evaluations'):
            self.assertEqual(s[field], 0)
        self.assertFalse(s['formal_training_enabled'])


if __name__ == '__main__':
    unittest.main()
