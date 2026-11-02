"""Historical diagnostics must retain immutable source and split provenance."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from consistent_cdm import runtime



class HistoricalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.data=self.root/'source'; self.data.mkdir()
        y,x=np.mgrid[:16,:16].astype(np.float32)
        for case in ['Bone_0001','Bone_0002','Bone_0003']:
            folder=self.data/case;folder.mkdir()
            for variant in ['std','aug']:
                f=.5+x/8+y/16; d=(x-7)/80
                for name,value in [('full',f),('left',f/2+d),('right',f/2-d)]:
                    np.save(folder/f'{variant}_{name}.npy',value.astype(np.float32))
        common=dict(type='case_folders',case_root=str(self.data),split=None,variants=['std','aug'],
                    condition_template='{variant}_full.npy',target_templates=['{variant}_left.npy','{variant}_right.npy'],
                    target_mode='dual',normalize='fixed_range_m11',value_range=[0,6],clip_range=[0,6],image_size=[8,8],
                    d_only_normalization={'scale':.25})
        self.cfg={'dataset':{'train':dict(common,case_ids=['Bone_0001','Bone_0002']),
                             'val':dict(common,case_ids=['Bone_0003'])}}
        self.checkpoint=self.root/'checkpoint.pt';torch.save({'config':self.cfg},self.checkpoint)
        self.calibration=self.root/'calibration.json'
        self.calibration.write_text(json.dumps(dict(scope='Bone_train_only',raw_full_quantiles=[.01,2,5])))
        self.prepared=runtime.prepare_historical_diagnostics(self.checkpoint,self.calibration,self.root/'prepared')

    def test_preserves_source_order_split_scale_and_all_val_hashes(self):
        p=self.prepared
        self.assertEqual(p['side_contract'],'historical_stored_order_unverified')
        self.assertEqual(p['split']['val'],['Bone_0003']);self.assertEqual(p['d_normalization']['scale'],.25)
        self.assertEqual(len(p['sources']),6)
        self.assertTrue(all(source['path'].startswith('Bone_0003/') for source in p['sources']))
        runtime.verify_prepared(p)
        runtime.verify_diagnostic_checkpoint(p,self.cfg,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'cannot be used for training'):
            runtime.training_config('resunet',p)

    def test_changed_source_checkpoint_scale_and_split_are_rejected(self):
        changed=copy.deepcopy(self.cfg);changed['dataset']['train']['d_only_normalization']['scale']=.3
        with self.assertRaisesRegex(ValueError,'binding differs'):
            runtime.verify_diagnostic_checkpoint(self.prepared,changed,self.checkpoint)
        path=self.root/'other.pt';torch.save({'config':changed},path)
        with self.assertRaisesRegex(ValueError,'binding differs'):
            runtime.verify_diagnostic_checkpoint(self.prepared,self.cfg,path)
        forged=copy.deepcopy(self.prepared);forged['split']['val']=['SY1'];forged['prepared_id']=runtime._historical_id(forged)
        with self.assertRaisesRegex(ValueError,'frozen/non-Bone'):
            runtime.verify_prepared(forged)
        np.save(self.data/'Bone_0003/std_full.npy',np.ones((16,16),np.float32))
        with self.assertRaisesRegex(ValueError,'source changed'):
            runtime.verify_prepared(self.prepared)

    def test_historical_condition_preserves_normalize_before_resize(self):
        source=np.arange(256,dtype=np.float32).reshape(16,16)/25
        expected=runtime.resize_raw(np.clip(source,0,6)/3-1,8)
        with patch.object(runtime,'predict',return_value=(None,None,None)) as mocked:
            runtime.diagnostic_predict(None,'diffusion',{},.25,source,self.prepared)
        np.testing.assert_array_equal(mocked.call_args.kwargs['normalized_condition'],expected)

    def test_stress_default_keeps_statistics_without_array_exports(self):
        args=Namespace(prepared=self.root/'prepared',checkpoint=self.checkpoint,device='cpu',seed=1234,
                       output=self.root/'compact_stress',operators=['blur'],mapping='raw',max_cases=1)
        with patch.object(runtime,'load_predictor',return_value=(None,self.cfg,'resunet',.25,'ema')), \
             patch.object(runtime,'diagnostic_predict',return_value=(np.zeros((8,8),np.float32),None,None)):
            runtime.stress_test(args)
        self.assertFalse((args.output/'arrays').exists())
        self.assertTrue((args.output/'metrics.csv').is_file())
        self.assertFalse(json.loads((args.output/'summary.json').read_text())['save_arrays'])

    def test_quantile_information_erasure_is_skipped_without_predicting(self):
        args=Namespace(prepared=self.root/'prepared',checkpoint=self.checkpoint,device='cpu',seed=1234,
                       output=self.root/'stress',operators=['invert'],mapping='quantile',max_cases=1,save_arrays=True)
        with patch.object(runtime,'load_predictor',return_value=(None,self.cfg,'resunet',.25,'ema')), \
             patch.object(runtime,'diagnostic_predict',return_value=(np.zeros((8,8),np.float32),None,None)) as pred:
            runtime.stress_test(args)
        skipped=json.loads((args.output/'skipped.json').read_text())
        self.assertEqual(len(skipped),2)
        self.assertTrue(all(row['operator']=='invert' and row['level']==.5 for row in skipped))
        self.assertEqual(pred.call_count,8) # (clean + .25/.75/1) x 2 views, not .5.
        self.assertTrue((args.output/'arrays/Bone_0003/std/invert_1/d.npy').is_file())
        with self.assertRaisesRegex(ValueError,'positive'):
            args.max_cases=-1
            with patch.object(runtime,'load_predictor',return_value=(None,self.cfg,'resunet',.25,'ema')):
                runtime.stress_test(args)

if __name__=='__main__':unittest.main()
