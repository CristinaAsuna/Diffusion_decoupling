import tempfile
import unittest
from pathlib import Path

import numpy as np

from cbct_preprocess.drr_geometry import (
    PixelTransform,
    detector_shape_for_volume,
    generate_rays,
    make_camera,
    project_points,
    resolve_horizontal_flip,
)
from cbct_preprocess.geometry import ImageGeometry
from cbct_preprocess.drr_renderer import (
    additivity_metrics,
    apply_pixel_transform,
    content_bbox,
    render_line_integral,
)


class ProjectionTests(unittest.TestCase):
    def test_anatomical_anterior_right_respects_image_direction(self):
        camera = make_camera(0.0, 0.0, 1000.0, 1500.0, (0.5, 0.5), (101, 101))
        identity = ImageGeometry((9, 9, 9), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
        reversed_y = ImageGeometry(
            (9, 9, 9), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
        )
        identity_flip, identity_meta = resolve_horizontal_flip(
            "anatomical_anterior_right", identity, camera
        )
        reversed_flip, reversed_meta = resolve_horizontal_flip(
            "anatomical_anterior_right", reversed_y, camera
        )
        self.assertTrue(identity_flip)
        self.assertFalse(reversed_flip)
        self.assertGreater(identity_meta["displayed_col_dot_lps_anterior"], 0.0)
        self.assertGreater(reversed_meta["displayed_col_dot_lps_anterior"], 0.0)

    def test_content_bbox_supports_outward_padding_and_inward_trim(self):
        image = np.zeros((100, 120), dtype=np.float32)
        image[10:90, 20:100] = 1.0
        bbox = content_bbox(
            image, threshold_fraction=0.02,
            padding=(2, 3, 4, 5), trim=(7, 8, 9, 10),
        )
        self.assertEqual(bbox, (15, 85, 25, 95))

    def test_central_point_projects_to_detector_center_at_arbitrary_angle(self):
        camera = make_camera(37.0, -4.0, 1000.0, 1500.0, (0.6, 0.7), (101, 121))
        coordinates, valid = project_points(np.array([[0.0, 0.0, 0.0]]), camera)
        np.testing.assert_allclose(coordinates[0], [50.0, 60.0], atol=1e-10)
        self.assertTrue(valid[0])
        ray_to_iso = -camera.source / np.linalg.norm(camera.source)
        ray_to_detector = (camera.detector_center - camera.source)
        ray_to_detector /= np.linalg.norm(ray_to_detector)
        np.testing.assert_allclose(ray_to_iso, ray_to_detector, atol=1e-12)

    def test_flip_crop_resize_half_pixel_mapping(self):
        transform = PixelTransform(100, 120, (10, 90, 20, 100), 160, 200, True)
        final, valid = transform.apply(np.array([[10.0, 99.0], [89.0, 20.0]]))
        # preflip col 99 becomes 20, exactly the left crop boundary.
        np.testing.assert_allclose(final[0], [0.5, 0.75])
        self.assertTrue(np.all(valid))

    def test_resize_preserves_nonnegative_line_integrals(self):
        image = np.zeros((9, 11), dtype=np.float32)
        image[2:7, 3:9] = 2.0
        transform = PixelTransform(9, 11, (0, 9, 0, 11), 32, 37, False)
        resized = apply_pixel_transform(image, transform)
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 2.0)

    def test_deterministic_drr_additivity(self):
        camera = make_camera(7.0, 0.0, 100.0, 150.0, (1.0, 1.0), (15, 17))
        rays = generate_rays(camera)
        full = np.zeros((9, 11, 13), dtype=np.float32)
        full[2:7, 2:9, 2:11] = 0.02
        left, right = np.zeros_like(full), np.zeros_like(full)
        left[:, :, 6:] = full[:, :, 6:]
        right[:, :, :6] = full[:, :, :6]
        kwargs = dict(spacing_xyz=(1.0, 1.2, 1.5), raw_shape_hw=(15, 17), step_size_mm=0.7, chunk_size=128, device="cpu")
        projections = [render_line_integral(volume, rays, **kwargs) for volume in (full, left, right)]
        metrics = additivity_metrics(*projections)
        self.assertLess(metrics["nmae"], 1e-6)

    def test_angle_specific_detector_contains_volume_corners(self):
        geometry = ImageGeometry((21, 31, 41), (1.5, 1.0, 0.8), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
        seed_camera = make_camera(35.0, 0.0, 1000.0, 1500.0, (0.6, 0.7), (1, 1))
        shape = detector_shape_for_volume(geometry, seed_camera, (3, 4))
        camera = make_camera(35.0, 0.0, 1000.0, 1500.0, (0.6, 0.7), shape)
        half = geometry.boundary_extent_xyz_mm / 2.0
        corners = np.array(
            [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
        )
        coordinates, valid = project_points(corners, camera)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(coordinates[:, 0] >= 0) and np.all(coordinates[:, 0] < shape[0]))
        self.assertTrue(np.all(coordinates[:, 1] >= 0) and np.all(coordinates[:, 1] < shape[1]))



if __name__ == "__main__":
    unittest.main()
