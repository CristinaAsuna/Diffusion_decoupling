import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from cbct_preprocess.config import load_config
from cbct_preprocess.nnunet import preflight, predict_segmentation


class NnunetTests(unittest.TestCase):
    def test_missing_resources_explained(self):
        cfg=load_config()['nnunet']
        with patch('cbct_preprocess.nnunet.shutil.which', return_value=None):
            with self.assertRaisesRegex(FileNotFoundError,'segmentation extra'):
                preflight(cfg)
        with patch('cbct_preprocess.nnunet.shutil.which', return_value='/bin/mock'), patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(FileNotFoundError,'nnUNet_results'):
                preflight(cfg)

    def test_uncompressed_input_is_converted_and_no_plans_rewrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'Dataset015_Teeth/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
            (model/'fold_all').mkdir(parents=True)
            plans=json.dumps({'normalization_schemes':['ZScoreNormalization']})
            (model/'plans.json').write_text(plans);(model/'dataset.json').write_text('{}');(model/'fold_all/checkpoint_final.pth').write_bytes(b'mock')
            cfg=load_config()['nnunet'];cfg['results_dir']=str(root)
            source=root/'case.nii';sitk.WriteImage(sitk.GetImageFromArray(np.zeros((3,3,3),np.int16)),str(source))
            def run(command, **kwargs):
                staged=root/'work/input/case_0000.nii.gz'
                self.assertFalse(staged.is_symlink())
                self.assertEqual(staged.read_bytes()[:2],b'\x1f\x8b')
                sitk.WriteImage(sitk.ReadImage(str(staged)),str(root/'work/output/case.nii.gz'))
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch('cbct_preprocess.nnunet.shutil.which',return_value='/bin/mock'),patch('cbct_preprocess.nnunet.subprocess.run',side_effect=run):
                result=predict_segmentation(source,root/'work',cfg)
            self.assertTrue(result.exists());self.assertEqual((model/'plans.json').read_text(),plans)
            self.assertEqual(len(json.loads((root/'work/resources.json').read_text())),3)
