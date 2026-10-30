import unittest

import numpy as np
import SimpleITK as sitk

from cbct_preprocess.alignment import build_alignment
from cbct_preprocess.geometry import ImageGeometry, rodrigues_from_a_to_b
from cbct_preprocess.midplane import estimate_midplane
from cbct_preprocess.split import complementary_masks, hu_to_mu, split_hu_image
from cbct_preprocess.io import image_array


class GeometryTests(unittest.TestCase):
    def test_index_physical_roundtrip_nonidentity_anisotropic(self):
        geometry = ImageGeometry(
            (7, 8, 9), (0.4, 0.7, 1.3), (10.0, -20.0, 5.0),
            (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        index = np.array([[0.0, 0.0, 0.0], [2.3, 4.1, 6.2]])
        np.testing.assert_allclose(geometry.physical_to_continuous_index(geometry.index_to_physical(index)), index)
        np.testing.assert_allclose(
            geometry.renderer_local_to_physical(geometry.physical_to_renderer_local(geometry.index_to_physical(index))),
            geometry.index_to_physical(index),
        )

    def test_rotation_maps_requested_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.2, 0.9, -0.3]); b /= np.linalg.norm(b)
        np.testing.assert_allclose(rodrigues_from_a_to_b(a, b) @ a, b, atol=1e-12)

    def test_label8_is_positive_patient_left(self):
        label_array = np.zeros((5, 5, 7), dtype=np.uint8)
        label_array[2, 2, 5] = 8
        label_array[2, 2, 1] = 9
        label = sitk.GetImageFromArray(label_array)
        plane, left, right, angle = estimate_midplane(label)
        self.assertGreater(plane.signed_distance(left), 0.1)
        self.assertLess(plane.signed_distance(right), -0.1)
        self.assertGreater(left[0], right[0])  # DICOM LPS +X is patient left.
        self.assertAlmostEqual(angle, 0.0)

    def test_legacy_high_x_split_is_patient_left_for_identity_direction(self):
        label_array = np.zeros((5, 5, 7), dtype=np.uint8)
        label_array[2, 2, 5] = 8
        label_array[2, 2, 1] = 9
        label = sitk.GetImageFromArray(label_array)
        plane, *_ = estimate_midplane(label)
        corrected_left, corrected_right, _ = complementary_masks(label, plane)
        # The historical Split/Left retained x >= floor((max(label8)+min(label9))/2).
        x_mid = (5 + 1) // 2
        legacy_left = np.zeros_like(corrected_left)
        legacy_left[:, :, x_mid:] = True
        np.testing.assert_array_equal(corrected_left, legacy_left)
        np.testing.assert_array_equal(corrected_right, ~legacy_left)

    def test_swapping_side_names_does_not_change_alignment_rotation(self):
        label_array = np.zeros((7, 7, 7), dtype=np.uint8)
        label_array[4, 2, 5] = 8
        label_array[2, 4, 1] = 9
        label = sitk.GetImageFromArray(label_array)
        corrected_mapping, corrected_plane, *_ = build_alignment(label, 8, 9)
        legacy_mapping, legacy_plane, *_ = build_alignment(label, 9, 8)
        # Reversing both the source normal and its signed target leaves the
        # rigid alignment unchanged; only branch sign/identity changes.
        np.testing.assert_allclose(corrected_mapping.rotation, legacy_mapping.rotation, atol=1e-12)
        np.testing.assert_allclose(corrected_mapping.center_lps, legacy_mapping.center_lps, atol=1e-12)
        np.testing.assert_allclose(corrected_plane.point, legacy_plane.point, atol=1e-12)
        np.testing.assert_allclose(corrected_plane.normal, -legacy_plane.normal, atol=1e-12)

    def test_reversed_direction_uses_lps_not_array_index_for_patient_left(self):
        label_array = np.zeros((5, 5, 7), dtype=np.uint8)
        # With index-x pointing toward LPS -X, label 8/patient-left is at the
        # lower array index even though it remains on the greater physical X.
        label_array[2, 2, 1] = 8
        label_array[2, 2, 5] = 9
        label = sitk.GetImageFromArray(label_array)
        label.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        plane, left, right, angle = estimate_midplane(label)
        self.assertGreater(plane.signed_distance(left), 0.1)
        self.assertLess(plane.signed_distance(right), -0.1)
        self.assertAlmostEqual(angle, 0.0)
        mapping, _, aligned_plane, *_ = build_alignment(label)
        np.testing.assert_allclose(mapping.rotation, np.eye(3), atol=1e-12)
        self.assertGreater(aligned_plane.signed_distance(mapping.input_to_output(left)), 0.0)
        mask_left, mask_right, _ = complementary_masks(label, aligned_plane)
        self.assertTrue(mask_left[2, 2, 1])
        self.assertTrue(mask_right[2, 2, 5])
        self.assertFalse(mask_left[2, 2, 5])
        self.assertTrue(np.all(np.logical_xor(mask_left, mask_right)))

    def test_split_mu_is_additive(self):
        array = np.arange(5 * 5 * 7, dtype=np.float32).reshape(5, 5, 7) - 1000.0
        image = sitk.GetImageFromArray(array)
        label_array = np.zeros_like(array, dtype=np.uint8)
        label_array[2, 2, 5] = 8
        label_array[2, 2, 1] = 9
        label = sitk.GetImageFromArray(label_array)
        plane, *_ = estimate_midplane(label)
        left_mask, right_mask, _ = complementary_masks(image, plane)
        left, right = split_hu_image(image, left_mask, right_mask)
        np.testing.assert_allclose(
            hu_to_mu(array), hu_to_mu(image_array(left)) + hu_to_mu(image_array(right)), atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
