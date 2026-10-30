import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from cbct_preprocess.config import load_config
from cbct_preprocess.pipeline import process_case, resolve_variants


class EndToEndTests(unittest.TestCase):
    def test_packaged_config_and_deterministic_variants(self):
        config = load_config()
        self.assertEqual(config['projection']['output_size'], [512, 512])
        self.assertTrue(config['nnunet']['enabled'])
        self.assertEqual(resolve_variants('patient', config['projection']), resolve_variants('patient', config['projection']))

    def test_asymmetric_phantom(self):
        config = load_config()
        config['projection'].update(device='cpu', sad_mm=100., sid_mm=150., detector_pixel_spacing_mm=[1.,1.], detector_padding_pixels=[2,2], step_size_mm=1., chunk_size=256, output_size=[32,32], variants=[{'name':'std','primary_angle_deg':0.}, {'name':'aug','primary_angle_deg':5.}])
        config['projection']['crop']={'mode':'none'}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            array=np.full((9,9,9),-1000,dtype=np.int16)
            array[1:8,1:8,1:4]=100;array[1:8,1:8,4:8]=900
            seg=np.zeros_like(array,dtype=np.uint8);seg[4,4,6]=8;seg[4,4,2]=9
            image_path=root/'phantom.nii';seg_path=root/'seg.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(array),str(image_path));sitk.WriteImage(sitk.GetImageFromArray(seg),str(seg_path))
            manifest=process_case(image_path,seg_path,root/'out',config)
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['side_contract']['branches'][0]['label'],8)
            case=root/'out/phantom'
            self.assertFalse((case/'landmarks').exists())
            for variant in ('std','aug'):
                full,left,right=[np.load(case/f'projections/{variant}_{branch}.npy') for branch in ('full','left','right')]
                self.assertEqual(full.dtype,np.float32)
                self.assertGreater(left.sum(),right.sum())
                np.testing.assert_allclose(full,left+right,atol=1e-6)
            (case/'projections/stale.npy').touch();(case/'user.txt').touch()
            config['runtime']['overwrite']=True
            process_case(image_path,seg_path,root/'out',config)
            self.assertFalse((case/'projections/stale.npy').exists());self.assertTrue((case/'user.txt').exists())

    def test_reject_swapped_contract(self):
        config=load_config();config['labels'].update(left=9,right=8)
        with self.assertRaisesRegex(ValueError,'left=8, right=9'):
            process_case('unused',None,'unused',config)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.yaml';path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'left=8, right=9'):
                load_config(path)
