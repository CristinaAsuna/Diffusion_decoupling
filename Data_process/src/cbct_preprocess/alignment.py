from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .geometry import ImageGeometry, RigidMapping, normalize, rodrigues_from_a_to_b, validate_same_geometry
from .midplane import SignedPlane, estimate_midplane


def build_alignment(label_image, label_left: int = 8, label_right: int = 9):
    plane, left, right, angle = estimate_midplane(label_image, label_left, label_right)
    geometry = ImageGeometry.from_sitk(label_image)
    image_x = normalize(geometry.direction_matrix[:, 0], name="image x direction")
    # Preserve the anatomical normal sign when index-x is reversed.
    target = image_x if np.dot(plane.normal, image_x) >= 0.0 else -image_x
    # Resample accepts output->input.  Therefore R maps desired output normal to
    # the original normal and R.T maps original points/normals into output space.
    rotation_output_to_input = rodrigues_from_a_to_b(target, plane.normal)
    mapping = RigidMapping(tuple(rotation_output_to_input.ravel()), tuple(plane.point))
    aligned_plane = plane.transformed_input_to_output(mapping)
    return mapping, plane, aligned_plane, left, right, angle


def resample_pair(image, label, mapping: RigidMapping, image_fill_hu: float = -1000.0):
    validate_same_geometry(image, label)
    transform = mapping.to_sitk()
    # Never quantize linear interpolation back into an integer CBCT type.
    image = sitk.Cast(image, sitk.sitkFloat32)
    aligned_image = sitk.Resample(
        image, image, transform, sitk.sitkLinear, float(image_fill_hu), sitk.sitkFloat32
    )
    aligned_label = sitk.Resample(
        label, label, transform, sitk.sitkNearestNeighbor, 0, label.GetPixelID()
    )
    return aligned_image, aligned_label


def expanded_reference(image, mapping: RigidMapping):
    """Create an input-oriented grid containing all rotated voxel centres."""
    geometry = ImageGeometry.from_sitk(image)
    size = np.asarray(geometry.size_xyz)
    corners_index = np.array(
        [[x, y, z] for x in (0, size[0] - 1) for y in (0, size[1] - 1) for z in (0, size[2] - 1)],
        dtype=np.float64,
    )
    corners_output = mapping.input_to_output(geometry.index_to_physical(corners_index))
    direction = geometry.direction_matrix
    coordinates_in_basis = corners_output @ direction
    lower, upper = coordinates_in_basis.min(axis=0), coordinates_in_basis.max(axis=0)
    spacing = np.asarray(geometry.spacing_xyz)
    output_size = np.ceil((upper - lower) / spacing).astype(int) + 1
    output_origin = lower @ direction.T
    reference = sitk.Image([int(value) for value in output_size], sitk.sitkFloat32)
    reference.SetSpacing(geometry.spacing_xyz)
    reference.SetOrigin(tuple(output_origin))
    reference.SetDirection(geometry.direction)
    return reference


def resample_pair_with_grid(
    image, label, mapping: RigidMapping, image_fill_hu: float = -1000.0, preserve_input_grid: bool = True
):
    validate_same_geometry(image, label)
    reference = sitk.Cast(image, sitk.sitkFloat32) if preserve_input_grid else expanded_reference(image, mapping)
    transform = mapping.to_sitk()
    aligned_image = sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32), reference, transform, sitk.sitkLinear,
        float(image_fill_hu), sitk.sitkFloat32,
    )
    aligned_label = sitk.Resample(
        label, reference, transform, sitk.sitkNearestNeighbor, 0, label.GetPixelID()
    )
    return aligned_image, aligned_label


def residual_angle_deg(aligned_label, label_left: int = 8, label_right: int = 9) -> float:
    return estimate_midplane(aligned_label, label_left, label_right)[-1]
