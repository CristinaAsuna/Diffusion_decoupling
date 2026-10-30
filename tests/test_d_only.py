from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from consistent_cdm.baseline.d_only import (
    DOnlyDataset,
    assert_allowed_case_prefixes,
    assert_no_forbidden_cases,
    compute_d_normalization,
    decode_standardized_d,
    raw_difference,
)
from consistent_cdm.baseline.train_pix2pix_d_only import validate_d_only


CFG = {"normalize": "fixed_range_m11", "value_range": [0.0, 6.0]}


class _Sample:
    def __init__(self, name: str) -> None:
        self.name = name


class TinyBranches(Dataset):
    def __init__(self) -> None:
        self.samples = [_Sample("DEV_A__std"), _Sample("DEV_B__aug")]
        self.condition_channels = 1
        self._full = torch.tensor([[[0.0, 6.0], [3.0, 3.0]]])
        self._left = torch.tensor([[[0.0, 4.0], [2.0, 2.0]]])
        self._right = self._full - self._left

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        full = self._full / 3.0 - 1.0
        target = torch.cat((self._left / 3.0 - 1.0, self._right / 3.0 - 1.0), dim=0)
        return {"name": self.samples[index].name, "condition": full, "target": target}


class DOnlyTests(unittest.TestCase):
    def test_d_only_pix2pix_validation_uses_raw_d_nmae(self) -> None:
        """The GAN selector must score decoded raw D, not normalized image MAE."""
        base = TinyBranches()
        normalization = compute_d_normalization(base, CFG, {"method": "raw"})
        wrapped = DOnlyDataset(base, CFG, normalization)
        loader = torch.utils.data.DataLoader(wrapped, batch_size=2)

        class PerfectD(torch.nn.Module):
            def forward(self, condition):
                # For this fixture D is (L-R)/2=(0, 1, .5, .5).
                return torch.tensor([[[[0.0, 1.0], [0.5, 0.5]]]], device=condition.device).expand(condition.shape[0], -1, -1, -1)

        scores = validate_d_only(PerfectD(), loader, torch.device("cpu"), CFG, normalization)
        self.assertAlmostEqual(scores["val_d_nmae"], 0.0, places=6)
        self.assertAlmostEqual(scores["val_d_l1"], 0.0, places=6)
        self.assertAlmostEqual(scores["val_left_mae"], 0.0, places=6)
        self.assertAlmostEqual(scores["val_right_mae"], 0.0, places=6)
        self.assertAlmostEqual(scores["val_side_ssim"], 1.0, places=6)
        self.assertGreater(scores["val_side_psnr"], 100.0)

    def test_raw_d_fixed_scaling_and_exact_branch_reconstruction(self) -> None:
        base = TinyBranches()
        stats = compute_d_normalization(base, CFG)
        sample = base[0]
        expected_d = torch.tensor([[[0.0, 1.0], [0.5, 0.5]]])
        self.assertTrue(torch.allclose(raw_difference(sample["target"], CFG), expected_d))
        wrapped = DOnlyDataset(base, CFG, stats)
        # The formal probe uses D / s_D only: no subtraction of the non-zero
        # dataset mean and no per-image target scaling.
        self.assertTrue(torch.allclose(wrapped[0]["target"], expected_d / stats.scale))
        restored_d, branches, _ = decode_standardized_d(wrapped[0]["target"].unsqueeze(0), sample["condition"].unsqueeze(0), CFG, stats)
        self.assertTrue(torch.allclose(restored_d[0], expected_d))
        self.assertTrue(torch.allclose(branches.sum(dim=1, keepdim=True), self._batch_raw_full(sample)))
        self.assertTrue(torch.allclose(branches[0, 0:1], base._left))
        self.assertTrue(torch.allclose(branches[0, 1:2], base._right))

    def test_raw_ablation_uses_unit_global_scale(self) -> None:
        stats = compute_d_normalization(TinyBranches(), CFG, {"method": "raw"})
        self.assertEqual(stats.scale, 1.0)
        self.assertEqual(stats.scale_method, "raw")

    @staticmethod
    def _batch_raw_full(sample):
        return (sample["condition"].unsqueeze(0) + 1.0) * 3.0

    def test_frozen_patient_guard(self) -> None:
        base = TinyBranches()
        assert_no_forbidden_cases(base, ["HELD_OUT"])
        with self.assertRaisesRegex(ValueError, "DEV_A"):
            assert_no_forbidden_cases(base, ["DEV_A"])
        assert_allowed_case_prefixes(base, ["DEV_"])
        with self.assertRaisesRegex(ValueError, "DEV_A"):
            assert_allowed_case_prefixes(base, ["OTHER_"])





if __name__ == "__main__":
    unittest.main()
