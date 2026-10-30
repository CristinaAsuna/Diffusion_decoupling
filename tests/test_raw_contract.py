"""Regression checks for the raw physical target and explicit patient split."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from consistent_cdm.data.dataset import CaseFolderNpyDataset
from consistent_cdm.baseline.d_only import DOnlyDataset, DNormalization, decode_standardized_d, resolve_d_normalization


class RawContractTests(unittest.TestCase):
    def test_clipped_input_retains_raw_d_and_full_for_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in ['train', 'val']:
                folder = root / case
                folder.mkdir()
                for name, value in [('full', 10.), ('left', 8.), ('right', 2.)]:
                    np.save(folder / (name + '.npy'), np.full((4, 4), value, np.float32))
            cfg = {'normalize': 'fixed_range_m11', 'value_range': [0., 6.], 'clip_range': [0., 6.]}
            base = CaseFolderNpyDataset(str(root), condition_file='full.npy', target_files=['left.npy', 'right.npy'], target_mode='dual', case_ids=['train'], split=None, image_size=[4, 4], **cfg)
            self.assertEqual([item.name for item in base.samples], ['train__full'])
            normalization = DNormalization(0., 2., 2., 'fixed', None, 16, 16, -3., 3.)
            sample = DOnlyDataset(base, cfg, normalization)[0]
            self.assertTrue(torch.all(sample['condition'] == 1.))
            self.assertTrue(torch.all(sample['target'] == 1.5))
            _, branches, _ = decode_standardized_d(sample['target'][None], sample['condition'][None], cfg, normalization, raw_condition=sample['raw_condition'][None])
            self.assertTrue(torch.all(branches[:, 0] == 8.))
            self.assertTrue(torch.all(branches[:, 1] == 2.))
            self.assertEqual(resolve_d_normalization(base, dict(cfg, d_only_normalization=normalization.to_dict())), normalization)

    def test_checkpoint_random_states_restore(self):
        import random
        from consistent_cdm.baseline.common import save_checkpoint
        from consistent_cdm.utils.checkpointing import load_checkpoint_file, restore_rng_state
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            save_checkpoint(checkpoint, {"model": {}})
            expected = (random.random(), float(np.random.random()), torch.rand(2))
            restore_rng_state(load_checkpoint_file(checkpoint, torch.device("cpu"))["rng_state"])
            actual = (random.random(), float(np.random.random()), torch.rand(2))
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(torch.equal(expected[2], actual[2]))
