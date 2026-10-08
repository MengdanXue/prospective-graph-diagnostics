"""Synthetic correctness, label boundary, invariance and launch-contract checks."""
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile

import numpy as np
from scipy.stats import ttest_ind
from scripts.cpm_probe import aggregation_operator, directional_score, probe_training_rows, seed_for
from scripts import cpm_extension_guarded as guard
from scripts.cpm_extension_worker import bind_probe_record, retained_records


class CPMTests(unittest.TestCase):
    def test_sparse_operator_matches_dense_with_duplicates_isolates_and_selfloops(self):
        edge = np.array([[0,0,1,2,2],[1,1,0,2,1]])
        actual = aggregation_operator(edge, 4).toarray()
        expected = np.array([[1,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]], float)
        expected /= expected.sum(axis=1)[:,None]
        np.testing.assert_allclose(actual, expected)

    def test_invalid_edges_rejected(self):
        with self.assertRaises(ValueError):
            aggregation_operator(np.array([[0],[3]]), 3)

    def test_score_matches_published_welch_and_win_fraction_direction(self):
        x=np.array([.2,.3,.4,.35]); g=np.array([.5,.6,.4,.7])
        expected=1-float(ttest_ind(x,g,equal_var=False).pvalue)/2
        self.assertAlmostEqual(directional_score(x,g),expected)
        self.assertAlmostEqual(directional_score(g,x),1-expected)

    def test_degenerate_equal_and_unequal_scores(self):
        self.assertEqual(directional_score([.5]*4,[.5]*4),.5)
        self.assertEqual(directional_score([.5]*4,[.8]*4),1)
        self.assertEqual(directional_score([.8]*4,[.5]*4),0)

    def test_affine_invariance_and_identical_resampling(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(120,5)); y=np.repeat([0,1,2],40)
        g=x + y[:,None]*.9
        kwargs={'seed':72,'repeats':10,'cap':70}
        a=probe_training_rows(x,g,y,**kwargs)
        shift=np.array([3,-1,2,.2,8])
        b=probe_training_rows(7*(x-shift),7*(g-shift),y,**kwargs)
        for key in ('raw_accuracy','graph_accuracy','partition_sha256','cpm_gnb','probe_gap'):
            self.assertEqual(a[key],b[key])

    def test_identity_graph_no_probe_advantage(self):
        rng=np.random.default_rng(71)
        x=rng.normal(size=(90,6)); y=np.arange(90)%3
        a=probe_training_rows(x,x,y,seed=11,repeats=8)
        self.assertEqual(a['cpm_gnb'],.5)
        self.assertEqual(a['probe_gap'],0)

    def test_zero_features_have_prior_only_predictions(self):
        a=probe_training_rows(np.zeros((30,3)),np.zeros((30,3)),np.arange(30)%3,seed=1,repeats=3)
        self.assertEqual(a['cpm_gnb'],.5)
        self.assertEqual(a['raw_accuracy'],[1/3]*3)

    def test_api_excludes_full_labels_and_model_outcomes(self):
        params=list(inspect.signature(probe_training_rows).parameters)
        self.assertEqual(params,['raw_train','aggregated_train','train_labels','seed','repeats','cap'])
        with self.assertRaises(ValueError):
            probe_training_rows(np.ones((9,2)),np.ones((9,2)),np.arange(20)%2,seed=1)

    def test_seed_binding(self):
        self.assertEqual(seed_for('a','b'),seed_for('a','b'))
        self.assertNotEqual(seed_for('a','b'),seed_for('a','c'))

    def test_probe_to_record_preserves_outer_seed_end_to_end(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(60,5)); labels=np.arange(60)%3
        metric=probe_training_rows(x,x,labels,seed=seed_for('fixture','split'),repeats=3)
        identity={'dataset':'fixture','seed':0,'split_id':'split','condition':'normalize_features'}
        record=json.loads(json.dumps(bind_probe_record(identity,metric)))
        self.assertEqual(record['seed'],0)
        self.assertEqual(record['resampling_seed'],seed_for('fixture','split'))
        self.assertEqual(record['split_id'],'split')
        with self.assertRaisesRegex(ValueError,'collides'):
            bind_probe_record(identity,{'seed':100})

    def test_guard_launch_is_fixed_and_default_is_read_only(self):
        cmd=guard.reader_command(Path('package'),Path('out'),Path('controller'),Path('data'))
        self.assertEqual(Path(cmd[2]).name,'cpm_extension_worker.py')
        self.assertIn('--data-root',cmd)
        self.assertNotIn('--execute',cmd)
        protocol=json.loads((guard.ROOT/'configs/cpm_train_only_v1.json').read_bytes())
        self.assertEqual(protocol['repeats'],100)
        self.assertEqual(protocol['worker_cap_seconds'],2700)
        self.assertEqual(len(protocol['datasets'])*len(protocol['seeds'])*len(protocol['conditions']),220)

    def test_control_admission_and_supervision_remain_inherited(self):
        class Accepted:
            def run_preparation(self, request, *, attempt_id, activity_id):
                self.ledger.begin_attempt(attempt_id, budget_group='control')
                return self._child(request, attempt_id=attempt_id)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); package=root/'package'; package.mkdir()
            for name in (guard.READER,guard.VERIFIER):
                (package/name).write_text('# fixture')
            manifest=package/'PACKAGE_MANIFEST.json'
            manifest.write_text(json.dumps({'files':{p.name:{'sha256':guard.fsha(p),'bytes':p.stat().st_size} for p in package.iterdir()}}))
            binding=guard.package_binding(package,guard.fsha(manifest))
            output=root/'result.json'; calls=[]; charges=[]
            def supervised(command,**kw):
                calls.append((command,kw)); Path(command[command.index('--output')+1]).write_text('{}')
                return {'returncode':0,'stop_reasons':[],'owned_processes_remaining':[]}
            api=SimpleNamespace(ExtensionExecutionController=Accepted,supervise_process=supervised,
                _exclusive_json=lambda *a:None,FormalEmergencyStop=RuntimeError)
            cls=guard.make_controller(api,package,output,binding,root/'controller',root/'data')
            self.assertIs(cls.run_preparation,Accepted.run_preparation)
            instance=cls(); instance.writer=SimpleNamespace(root=root/'output')
            instance._supervisor_poll=lambda *_:[]; instance._monitor=lambda:None
            instance.ledger=SimpleNamespace(begin_attempt=lambda *a,**kw:charges.append(kw['budget_group']))
            result=instance.run_preparation({'operation':guard.OPERATION},attempt_id='fixture',activity_id='fixture')
            self.assertEqual(charges,['control']); self.assertTrue(result['probe_fitting'])
            self.assertEqual(calls[0][1]['poll_seconds'],.25)
            self.assertEqual(calls[0][1]['worker_cap_seconds'],2700)
            with self.assertRaisesRegex(ValueError,'unsupported'):
                instance._child({'operation':'unit'},attempt_id='wrong')

    def test_default_never_imports_execution_controller(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=[s for name in ('controller-root','data-root','package-root','handoff','ledger-root',
                'output-root','controller-ci-receipt','wrapper-ci-receipt') for s in ('--'+name,str(root/name))]
            args += ['--package-manifest-sha256','a'*64]
            with patch.object(guard,'source_identity'),patch.object(guard,'package_binding',return_value={}), \
                 patch.object(guard.importlib,'import_module',side_effect=AssertionError('runtime import')),redirect_stdout(StringIO()):
                self.assertEqual(guard.main(args),0)
            self.assertFalse((root/'ledger-root').exists())

    def test_resume_prefix_requires_hash_split_features_and_replicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'records.jsonl'
            x=np.arange(60,dtype=float).reshape(20,3); labels=np.arange(20)%2
            metric=probe_training_rows(x,x,labels,seed=seed_for('fixture','split'),repeats=3)
            row=bind_probe_record({'dataset':'fixture','seed':0,'condition':'N','split_id':'split'},metric)
            row.update(materialized_sha256='raw',transformed_feature_sha256='features')
            protocol={'datasets':['fixture'],'seeds':[0],'conditions':['N','CS'],'repeats':3}
            binding={'datasets':{'fixture':{'materialized_sha256':'raw','split_checks':[{'seed':0,'split_id':'split'}],
                'candidate_checks':[{'seed':0,'transformed_feature_sha256':'features'}]}}}
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertEqual(retained_records(path,guard.fsha(path),protocol,binding),[row])
            with self.assertRaisesRegex(ValueError,'external binding'):
                retained_records(path,'0'*64,protocol,binding)
            path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'unique declared prefix'):
                retained_records(path,guard.fsha(path),protocol,binding)
            row['cpm_gnb']=.9; path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differs from its repeats'):
                retained_records(path,guard.fsha(path),protocol,binding)

    def test_package_verification_polls_by_time_without_skipping_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in (guard.READER,guard.VERIFIER,'data.txt'):
                (root/name).write_text('fixture')
            manifest=root/'PACKAGE_MANIFEST.json'
            manifest.write_text(json.dumps({'files':{p.name:{'sha256':guard.fsha(p),'bytes':p.stat().st_size} for p in root.iterdir()}}))
            calls=[]
            with patch.object(guard.time,'monotonic',return_value=1):
                guard.package_binding(root,guard.fsha(manifest),lambda:calls.append(1))
            self.assertEqual(calls,[1])
            (root/'data.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'file hash'):
                guard.package_binding(root,guard.fsha(manifest),lambda:None)


if __name__ == '__main__':
    unittest.main()
