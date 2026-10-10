"""Integrity failures and the reader training boundary, without scientific runs."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scripts.reader_data import acquire_raw, array_sha256, load_public_dataset
from scripts.reader_reproduce import assert_same, train, write


class ReaderTests(unittest.TestCase):
    def test_truncated_download_is_never_accepted_and_retry_is_verified(self):
        content = b'complete public fixture'
        item = {'path':'pyg/X/raw/data', 'source_url':'https://raw.githubusercontent.com/example/data',
                'size':len(content), 'sha256':hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen',
                side_effect=[io.BytesIO(b'partial'),io.BytesIO(content)]) as fetch:
            acquire_raw({'raw_files':[item]}, Path(temp))
            self.assertEqual((Path(temp)/item['path']).read_bytes(), content)
            self.assertEqual(fetch.call_count, 2)

    def test_cached_corruption_stops_before_dataset_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'pyg/raw/data';p.parent.mkdir(parents=True);p.write_bytes(b'corrupt')
            binding={'datasets':{'fixture':{'raw_files':[{'path':'pyg/raw/data','sha256':'0'*64}]}}}
            with self.assertRaisesRegex(ValueError,'missing or changed raw'):
                load_public_dataset('fixture',0,Path(temp),binding)

    def test_binding_cannot_escape_reader_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                acquire_raw({'raw_files':[{'path':'../outside'}]},Path(temp))

    def test_tensor_binding_includes_shape_and_dtype(self):
        a=np.arange(4,dtype=np.int32)
        self.assertNotEqual(array_sha256(a),array_sha256(a.reshape(2,2)))
        self.assertNotEqual(array_sha256(a),array_sha256(a.astype(np.int64)))
        self.assertEqual(array_sha256(a),array_sha256(torch.from_numpy(a)))

    def test_reconstruction_rejects_changed_scope_and_nonfinite_values(self):
        assert_same({'x':[.2]},{'x':[.2+1e-14]})
        for bad in ({'x':[.201]}, {'x':[]}, {'x':[float('nan')]}, {'z':[.2]}):
            with self.assertRaises(ValueError):
                assert_same({'x':[.2]},bad)

    def test_exclusive_outputs_preserve_previous_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'result.json';write(p,{'original':True})
            with self.assertRaises(FileExistsError):
                write(p,{'original':False})
            self.assertTrue(json.loads(p.read_text())['original'])

    def test_expanded_training_uses_per_trial_decay_and_tests_only_winner(self):
        x=torch.ones(9,3);y=torch.tensor([0,1,2]*3);edge=torch.tensor([[0,1],[1,0]])
        split={'train':np.array([0,1,2]),'validation':np.array([3,4,5]),'test':np.array([6,7,8])}
        calls=[]
        def fake_trial(**kw):
            calls.append(kw)
            i=int(kw['trial_id'][-3:])
            return {'trial_id':kw['trial_id'],'validation_accuracy':i/100,
                    'validation_loss':1.0,'configuration':kw['trial']}, {'fixture':torch.tensor(i)}
        class Model:
            def to(self,*args): return self
            def load_state_dict(self,state): self.selected=int(state['fixture'])
            def eval(self): return self
            def __call__(self,*args): return torch.zeros(9,3)
        model=Model()
        with tempfile.TemporaryDirectory() as temp, \
             patch('scripts.reader_data.load_public_dataset',return_value=(x,y,edge,split,'split',{})), \
             patch('scripts.reader_data.fit_transform_features_bounded',return_value=(x,{})), \
             patch('experiments.run_prospective_benchmark._train_trial',side_effect=fake_trial), \
             patch('experiments.prospective_models.build_model',return_value=model), \
             patch('experiments.run_prospective_benchmark._accuracy',return_value=.5) as accuracy:
            result=train('fixture',0,'normalize_features','MLP',24,Path(temp),Path(temp)/'result.json',{},'cpu')
            self.assertEqual(len(calls),24)
            self.assertEqual({c['weight_decay'] for c in calls},{0,.0005})
            self.assertTrue(all('test_indices' not in c for c in calls))
            self.assertEqual(model.selected,23)
            self.assertEqual(accuracy.call_count,1)
            self.assertEqual(result['test_evaluations_after_selection'],1)
            self.assertEqual(len(list((Path(temp)/'result.checkpoints').glob('*.pt'))),24)

if __name__ == '__main__':
    unittest.main()
