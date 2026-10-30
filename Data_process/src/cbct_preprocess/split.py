from __future__ import annotations

import numpy as np

from .geometry import ImageGeometry
from .io import image_array, image_from_array


def signed_distance_volume(reference_image, plane, slab_depth: int = 16) -> np.ndarray:
    geometry = ImageGeometry.from_sitk(reference_image)
    nx, ny, nz = geometry.size_xyz
    distances = np.empty((nz, ny, nx), dtype=np.float32)
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    for start in range(0, nz, slab_depth):
        stop = min(start + slab_depth, nz)
        z = np.arange(start, stop, dtype=np.float64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        indices = np.stack((xx, yy, zz), axis=-1)
        points = geometry.index_to_physical(indices)
        distances[start:stop] = plane.signed_distance(points).astype(np.float32)
    return distances


def complementary_masks(reference_image, plane, boundary_goes_to: str = "left"):
    distances = signed_distance_volume(reference_image, plane)
    if boundary_goes_to == "left":
        left = distances >= 0.0
    elif boundary_goes_to == "right":
        left = distances > 0.0
    else:
        raise ValueError("boundary_goes_to must be 'left' or 'right'")
    right = ~left
    return left, right, distances


def split_hu_image(image, left_mask: np.ndarray, right_mask: np.ndarray, air_hu: float = -1000.0):
    if left_mask.dtype != np.bool_ or right_mask.dtype != np.bool_:
        raise TypeError("Split masks must be boolean")
    if not np.all(np.logical_xor(left_mask, right_mask)):
        raise ValueError("left/right masks are not a strict complementary partition")
    full = image_array(image, np.float32)
    if full.shape != left_mask.shape:
        raise ValueError(f"Mask/image shape mismatch: {left_mask.shape} != {full.shape}")
    left_hu = np.where(left_mask, full, np.float32(air_hu)).astype(np.float32)
    right_hu = np.where(right_mask, full, np.float32(air_hu)).astype(np.float32)
    return image_from_array(left_hu, image), image_from_array(right_hu, image)


def hu_to_mu(array_hu: np.ndarray, water_mu_per_mm: float = 0.022) -> np.ndarray:
    return np.clip((np.asarray(array_hu, np.float32) / 1000.0 + 1.0) * water_mu_per_mm, 0.0, None).astype(np.float32)
