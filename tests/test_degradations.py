"""Stress-test degradations keep identities and the F=L+R contract."""
import unittest

import numpy as np

from consistent_cdm.evaluation.degradations import DEGRADATIONS, degrade


class DegradationTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:32, :32].astype(np.float32)
        self.left = .5 + np.exp(-((x-12)**2 + (y-16)**2) / 40)
        self.right = .4 + np.exp(-((x-20)**2 + (y-16)**2) / 60)
        self.full = self.left + self.right

    def test_identity_level_returns_inputs(self):
        for name, spec in DEGRADATIONS.items():
            outputs = degrade(name, spec.identity, self.full, self.left, self.right, model_size=16)
            for output, source in zip(outputs, (self.full, self.left, self.right)):
                np.testing.assert_array_equal(output, source, err_msg=name)

    def test_levels_are_finite_and_respect_operator_kind(self):
        for name, spec in DEGRADATIONS.items():
            for level in spec.levels:
                full, left, right = degrade(name, level, self.full, self.left, self.right, model_size=16, seed=3)
                self.assertEqual(full.shape, self.full.shape)
                self.assertEqual(full.dtype, np.float32)
                self.assertTrue(np.isfinite(full).all(), f"{name}={level}")
                self.assertFalse(np.allclose(full, self.full), f"{name}={level} had no effect")
                if spec.kind == "intensity":
                    np.testing.assert_array_equal(left, self.left)
                    np.testing.assert_array_equal(right, self.right)
                else:
                    np.testing.assert_allclose(full, left + right, atol=1e-5, err_msg=f"{name}={level}")

    def test_equalization_preserves_tied_background_pixels(self):
        full = np.zeros((8, 8), dtype=np.float32)
        full[2:6, 2:6] = 2.
        actual = degrade("equalize", 1., full, full/2, full/2, model_size=8)[0]
        self.assertEqual(np.unique(actual[full == 0]).size, 1)
        self.assertEqual(np.unique(actual[full == 2]).size, 1)
        np.testing.assert_array_equal(actual, full)

    def test_noise_is_seeded(self):
        first = degrade("noise", .1, self.full, self.left, self.right, model_size=16, seed=5)[0]
        second = degrade("noise", .1, self.full, self.left, self.right, model_size=16, seed=5)[0]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
