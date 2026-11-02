"""Domain statistics must describe the tensor actually given to the model."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

from PIL import Image

import numpy as np

from consistent_cdm import runtime

spec = importlib.util.spec_from_file_location(
    "domain_input_statistics", Path(__file__).resolve().parents[1] / "scripts/domain_stats.py")
domain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(domain)


class DomainInputTests(unittest.TestCase):
    def test_historical_input_clips_before_resize(self):
        source = np.array([[-1., 1., 7., 4.], [2., 5., 8., -2.],
                           [3., 1., 4., 2.], [0., 3., 2., 1.]], np.float32)
        mapped, normalized = domain.model_condition_from_source(
            source, 2, "historical_normalize_before_resize")
        expected = runtime.resize_raw(np.clip(source / 6, 0, 1) * 2 - 1, 2)
        np.testing.assert_array_equal(normalized, expected)
        np.testing.assert_array_equal(mapped, (expected + 1) * 3)
        self.assertGreater(np.max(np.abs(mapped - np.clip(runtime.resize_raw(source, 2), 0, 6))), .1)

    def test_domain_cli_array_exports_are_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data'
            y,x=np.mgrid[:16,:16].astype(np.float32)
            full=1+x/8+y/16;difference=(x-7)/80
            for case in ['case_a','case_b']:
                folder=data/case; (folder/'projections').mkdir(parents=True)
                (folder/'manifest.json').write_text(json.dumps({'side_contract':runtime.SIDE_CONTRACT}))
                for name,array in [('full',full),('left',full/2+difference),('right',full/2-difference)]:
                    np.save(folder/'projections'/f'std_{name}.npy',array)
            prepared=root/'prepared'
            runtime.prepare_data(data,prepared,image_size=8,variants=('std',))
            real=root/'real.png';Image.fromarray((full/6*255).astype(np.uint8)).save(real)
            for enabled in (False,True):
                output=root/('arrays_on' if enabled else 'compact')
                argv=['--prepared',str(prepared),'--real',str(real),'--output',str(output),'--max-cases','1']
                domain.main(argv+(['--save-arrays'] if enabled else []))
                self.assertEqual((output/'arrays').exists(),enabled)
                self.assertTrue((output/'domain_stats.csv').is_file())
                self.assertEqual(json.loads((output/'summary.json').read_text())['save_arrays'],enabled)

    def test_modern_input_resizes_before_clipping(self):
        source = np.array([[-4., 8.], [4., 2.]], np.float32)
        mapped, normalized = domain.model_condition_from_source(
            source, 1, "raw_resize_then_normalize")
        expected = np.clip(runtime.resize_raw(source, 1), 0, 6) / 3 - 1
        np.testing.assert_array_equal(normalized, expected)
        np.testing.assert_array_equal(mapped, (expected + 1) * 3)


if __name__ == "__main__":
    unittest.main()
