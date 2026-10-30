"""Portable pipeline contracts using synthetic patients and random tiny models."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
import copy

import numpy as np
from PIL import Image
import torch

from consistent_cdm import runtime
from consistent_cdm.factories import build_deterministic_generator, build_generative_model


torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / 'data'
        y, x = np.mgrid[:16, :16].astype(np.float32)
        for index, case in enumerate(('patient_a', 'patient_b', 'patient_c')):
            projection = self.data / case / 'projections'
            projection.mkdir(parents=True)
            (projection.parent / 'manifest.json').write_text(json.dumps({
                'status': 'complete', 'side_contract': {'version': runtime.SIDE_CONTRACT}}))
            full = .6 + (x + y) / 16 + index * .1
            d = (x - 7.5) / 80 + index * .01
            for branch, value in (('full', full), ('left', full / 2 + d), ('right', full / 2 - d)):
                np.save(projection / f'std_{branch}.npy', value)
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({'train': ['patient_a', 'patient_b'], 'val': ['patient_c']}))

    def prepare(self, name='prepared'):
        return runtime.prepare_data(self.data, self.root / name, split_file=self.split,
                                    image_size=16, variants=['std'])

    def checkpoint(self, family, prepared):
        cfg = runtime.training_config(family, prepared, tiny=True)
        torch.manual_seed(17)
        if family in ('diffusion', 'flow_matching'):
            model, _ = build_generative_model(cfg, condition_channels=1, dataset_target_channels=1, image_size=16)
        else:
            model_cfg = cfg['model'].get('generator', cfg['model'])
            model = build_deterministic_generator(model_cfg, condition_channels=1, target_channels=1)
        path = self.root / f'{family}.pt'
        torch.save({'model': model.state_dict(), 'ema_model': model.state_dict(), 'config': cfg,
                    'd_normalization': prepared['d_normalization']}, path)
        return path

    def test_statistics_use_train_patients_only(self):
        before = self.prepare()
        p = self.data / 'patient_c' / 'projections'
        full = np.load(p / 'std_full.npy') * 20
        delta = np.arange(256, dtype=np.float32).reshape(16, 16) / 10
        for name, value in (('full', full), ('left', full / 2 + delta), ('right', full / 2 - delta)):
            np.save(p / f'std_{name}.npy', value)
        after = self.prepare('prepared_after')
        self.assertEqual(before['d_normalization'], after['d_normalization'])
        self.assertEqual(before['calibration'], after['calibration'])
        self.assertNotEqual(before['prepared_id'], after['prepared_id'])
        self.assertEqual(len(after['sources']), 12)
        for source in after['sources']:
            self.assertEqual(source['sha256'], runtime.sha256(self.data / source['path']))

    def test_changed_sources_are_rejected_after_preparation(self):
        prepared = self.prepare()
        runtime.verify_prepared(prepared)
        altered = copy.deepcopy(prepared)
        altered['d_normalization']['scale'] *= 2
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            runtime.verify_prepared(altered)
        source = self.data / next(item['path'] for item in prepared['sources'] if item['path'].endswith('.npy'))
        np.save(source, np.load(source) + .02)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runtime.verify_prepared(prepared)

    def test_missing_or_wrong_anatomical_provenance_is_rejected(self):
        manifest = self.data / 'patient_a' / 'manifest.json'
        manifest.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'provenance'):
            self.prepare('missing_manifest')
        manifest.write_text(json.dumps({'side_contract': {'version': 'legacy'}}))
        with self.assertRaisesRegex(ValueError, 'side contract'):
            self.prepare('legacy_manifest')

    def test_validation_export_contains_full_half_reference(self):
        prepared = self.prepare()
        checkpoint = self.checkpoint('resunet', prepared)
        output = self.root / 'validation'
        scores = runtime.evaluate(Namespace(prepared=self.root / 'prepared', checkpoint=checkpoint,
                                            output=output, device='cpu', seed=1234))
        self.assertEqual(set(scores), {'resunet', 'full_half'})
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['cohort'], 'validation_only')
        self.assertEqual(summary['prepared_id'], prepared['prepared_id'])
        import csv
        with (output / 'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row['case'] for row in rows}, {'patient_c'})
        self.assertEqual({row['method'] for row in rows}, {'resunet', 'full_half'})
        self.assertTrue((output / 'patient_c__std' / 'd.npy').is_file())

    def test_d_metrics_distinguish_perfect_collapse_and_swap(self):
        from consistent_cdm.evaluation.data_metrics import asymmetry_metrics
        d = np.array([[-.5,.25],[.2,-.1]], dtype=np.float32)
        for prediction,nmae,identity in ((d,0,1),(np.zeros_like(d),1,0),(-d,2,-1)):
            scores = asymmetry_metrics(prediction,d)
            self.assertAlmostEqual(scores['nmae'],nmae,places=6)
            self.assertAlmostEqual(scores['identity'],identity,places=6)

    def test_bad_split_and_nonadditive_triplets_are_rejected(self):
        for index, split in enumerate((
                {'train': ['patient_a', 'patient_b'], 'val': ['patient_b', 'patient_c']},
                {'train': ['patient_a'], 'val': ['patient_b']},
                {'train': [], 'val': ['patient_a', 'patient_b', 'patient_c']})):
            self.split.write_text(json.dumps(split))
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.prepare(f'bad_split_{index}')
        self.split.write_text(json.dumps({'train': ['patient_a', 'patient_b'], 'val': ['patient_c']}))
        p = self.data / 'patient_a' / 'projections' / 'std_full.npy'
        np.save(p, np.load(p) + 1)
        with self.assertRaisesRegex(ValueError, 'L\\+R|additi'):
            self.prepare('bad_additivity')

    def test_npy_defaults_raw_and_16bit_images_keep_intensity(self):
        full = np.arange(256, dtype=np.float32).reshape(16, 16) / 64
        np.save(self.root / 'condition.npy', full)
        value, info = runtime.load_input(self.root / 'condition.npy', size=16)
        np.testing.assert_array_equal(value, full)
        self.assertEqual(info['mapping'], 'raw')
        pixels = (np.arange(256).reshape(16, 16) * 257).astype(np.uint16)
        path = self.root / 'sixteen.png'
        Image.fromarray(pixels).save(path)
        raw, _ = runtime.load_input(path, size=16, mapping='raw')
        np.testing.assert_array_equal(raw, pixels.astype(np.float32))
        calibration = {'raw_full_quantiles': [.1, 2., 5.5]}
        quantile, transform = runtime.load_input(path, size=16, calibration=calibration)
        expected = np.interp(pixels, np.percentile(pixels, [1, 50, 99]), [.1, 2., 5.5])
        np.testing.assert_allclose(quantile, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(transform['mapping'], 'quantile')
        with self.assertRaisesRegex(ValueError, 'calibration'):
            runtime.load_input(path, size=16)

    def test_geometry_transform_and_invalid_inputs(self):
        array = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = self.root / 'condition.npy'
        np.save(path, array)
        result, _ = runtime.load_input(path, size=16, crop=[2, 3, 10, 11],
                                       nose_direction='left', invert=True)
        cropped = array[3:11, 2:10][:, ::-1]
        expected = runtime.resize_raw(cropped.max() + cropped.min() - cropped, 16)
        np.testing.assert_array_equal(result, expected)
        with self.assertRaises(ValueError):
            runtime.load_input(path, size=16, crop=[0, 0, 17, 16])
        array[0, 0] = np.nan
        np.save(path, array)
        with self.assertRaises(ValueError):
            runtime.load_input(path, size=16)

    def test_four_family_condition_only_inference(self):
        prepared = self.prepare()
        # The only inference input is F; no side files are present in this directory.
        input_dir = self.root / 'unpaired'
        input_dir.mkdir()
        full = np.load(self.data / 'patient_c' / 'projections' / 'std_full.npy')
        np.save(input_dir / 'condition.npy', full)
        pixels = (np.arange(256).reshape(16, 16) * 257).astype(np.uint16)
        Image.fromarray(pixels).save(input_dir / 'image16.tiff')
        Image.fromarray(pixels).save(input_dir / 'image16.png')
        Image.fromarray((pixels / 257).astype(np.uint8)).save(input_dir / 'image.jpg')
        for family in runtime.FAMILIES:
            with self.subTest(family=family):
                checkpoint = self.checkpoint(family, prepared)
                output = self.root / f'inference_{family}'
                manifest = runtime.infer(Namespace(checkpoint=checkpoint, input=input_dir, output=output,
                    device='cpu', family=None, config=None, calibration=None, mapping=None,
                    nose_direction='right', invert=False, crop=None, seed=1234))
                self.assertEqual(manifest['family'], family)
                self.assertEqual(manifest['scope'], 'condition_only_no_paired_accuracy_metrics')
                self.assertEqual(manifest['checkpoint_sha256'], runtime.sha256(checkpoint))
                self.assertEqual(len(manifest['images']), 4)
                for row in manifest['images']:
                    item = output / row['output']
                    f, l, r = [np.load(item / f'{key}.npy') for key in ('full', 'left', 'right')]
                    np.testing.assert_allclose(l + r, f, rtol=1e-6, atol=1e-6)
                    self.assertTrue(np.isfinite(l).all() and np.isfinite(r).all())
                folder = output / next(row['output'] for row in manifest['images'] if row['input'].endswith('condition.npy'))
                left, right, saved_full = [np.load(folder / f'{key}.npy') for key in ('left', 'right', 'full')]
                self.assertTrue(np.isfinite(left).all() and np.isfinite(right).all())
                np.testing.assert_allclose(left + right, saved_full, rtol=1e-6, atol=1e-6)
                self.assertTrue((folder / 'preview.png').is_file())
                self.assertFalse((output / 'metrics.csv').exists())
                # Repeating the seed yields the same raw signed difference.
                model, cfg, selected_family, scale, _ = runtime.load_predictor(checkpoint)
                d1, _, _ = runtime.predict(model, selected_family, cfg, scale, full, seed=1234)
                d2, _, _ = runtime.predict(model, selected_family, cfg, scale, full, seed=1234)
                np.testing.assert_array_equal(d1, d2)
                np.testing.assert_array_equal(d1, np.load(folder / 'd.npy'))

    def test_original_generative_checkpoints_infer_family_without_rewriting_config(self):
        prepared = self.prepare()
        from consistent_cdm.utils.checkpointing import load_checkpoint_file
        for family in ('diffusion', 'flow_matching'):
            checkpoint = self.checkpoint(family, prepared)
            state = load_checkpoint_file(checkpoint, torch.device('cpu'))
            state['config'].pop('family')
            state['config'].pop('side_contract')
            if family == 'diffusion':
                state['config'].pop('generative')
            torch.save(state, checkpoint)
            model, cfg, detected, scale, weights = runtime.load_predictor(checkpoint)
            self.assertEqual(detected, family)
            self.assertEqual(weights, 'ema_model')
            self.assertNotIn('family', cfg)
            self.assertEqual(cfg, state['config'])
            output = self.root / f'legacy_inference_{family}'
            manifest = runtime.infer(Namespace(checkpoint=checkpoint,
                input=self.data / 'patient_a' / 'projections' / 'std_full.npy', output=output,
                device='cpu', family=None, config=None, calibration=None, mapping=None,
                nose_direction='right', invert=False, crop=None, seed=1234))
            self.assertEqual(manifest['side_contract'], 'historical_stored_order_unverified')
            wrong = 'diffusion' if family == 'flow_matching' else 'flow_matching'
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                runtime.load_predictor(checkpoint, family=wrong)


if __name__ == '__main__':
    unittest.main()
