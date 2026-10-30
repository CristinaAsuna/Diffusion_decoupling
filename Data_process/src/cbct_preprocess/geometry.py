from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def _v3(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {result.shape}")
    return result


def normalize(vector: Iterable[float], *, name: str = "vector") -> np.ndarray:
    vector = _v3(vector)
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError(f"Cannot normalize zero-length {name}")
    return vector / length


@dataclass(frozen=True)
class ImageGeometry:
    """SimpleITK geometry with xyz index order and LPS physical coordinates."""

    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction: tuple[float, ...]

    @classmethod
    def from_sitk(cls, image) -> "ImageGeometry":
        return cls(
            tuple(int(x) for x in image.GetSize()),
            tuple(float(x) for x in image.GetSpacing()),
            tuple(float(x) for x in image.GetOrigin()),
            tuple(float(x) for x in image.GetDirection()),
        )

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=np.float64).reshape(3, 3)

    @property
    def center_lps(self) -> np.ndarray:
        center_index = (np.asarray(self.size_xyz, dtype=np.float64) - 1.0) / 2.0
        return self.index_to_physical(center_index)

    @property
    def center_extent_xyz_mm(self) -> np.ndarray:
        return np.asarray(self.spacing_xyz) * (np.asarray(self.size_xyz) - 1.0)

    @property
    def boundary_extent_xyz_mm(self) -> np.ndarray:
        return np.asarray(self.spacing_xyz) * np.asarray(self.size_xyz)

    def index_to_physical(self, index_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(index_xyz, dtype=np.float64)
        scaled = points * np.asarray(self.spacing_xyz, dtype=np.float64)
        return np.asarray(self.origin_lps) + scaled @ self.direction_matrix.T

    def physical_to_continuous_index(self, points_lps: np.ndarray) -> np.ndarray:
        points = np.asarray(points_lps, dtype=np.float64)
        local = (points - np.asarray(self.origin_lps)) @ self.direction_matrix
        return local / np.asarray(self.spacing_xyz)

    def physical_to_renderer_local(self, points_lps: np.ndarray) -> np.ndarray:
        """Map LPS to an image-axis local frame centred on the volume."""
        points = np.asarray(points_lps, dtype=np.float64)
        return (points - self.center_lps) @ self.direction_matrix

    def renderer_local_to_physical(self, points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=np.float64)
        return self.center_lps + points @ self.direction_matrix.T

    def contains_physical(self, points_lps: np.ndarray, tolerance_voxels: float = 0.5) -> np.ndarray:
        indices = self.physical_to_continuous_index(points_lps)
        lower = -float(tolerance_voxels)
        upper = np.asarray(self.size_xyz, dtype=np.float64) - 1.0 + float(tolerance_voxels)
        return np.all((indices >= lower) & (indices <= upper), axis=-1)

    def as_dict(self) -> dict:
        return asdict(self)


def validate_same_geometry(image, label, atol: float = 1e-5) -> None:
    a, b = ImageGeometry.from_sitk(image), ImageGeometry.from_sitk(label)
    if a.size_xyz != b.size_xyz:
        raise ValueError(f"Image/label size mismatch: {a.size_xyz} != {b.size_xyz}")
    for name, x, y in (
        ("spacing", a.spacing_xyz, b.spacing_xyz),
        ("origin", a.origin_lps, b.origin_lps),
        ("direction", a.direction, b.direction),
    ):
        if not np.allclose(x, y, atol=atol, rtol=0.0):
            raise ValueError(f"Image/label {name} mismatch: {x} != {y}")


def rodrigues_from_a_to_b(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    """Return a proper rotation matrix R such that R @ a == b."""
    a, b = normalize(a, name="a"), normalize(b, name="b")
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-12:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, candidate), name="180-degree axis")
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(a, b)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / np.dot(cross, cross))


@dataclass(frozen=True)
class RigidMapping:
    """Output-to-input mapping used by SimpleITK Resample."""

    matrix: tuple[float, ...]
    center_lps: tuple[float, float, float]

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)

    def output_to_input(self, points_output_lps: np.ndarray) -> np.ndarray:
        points = np.asarray(points_output_lps, dtype=np.float64)
        center = np.asarray(self.center_lps)
        return center + (points - center) @ self.rotation.T

    def input_to_output(self, points_input_lps: np.ndarray) -> np.ndarray:
        """Transform physical points into the aligned output space."""
        points = np.asarray(points_input_lps, dtype=np.float64)
        center = np.asarray(self.center_lps)
        return center + (points - center) @ self.rotation

    def vector_input_to_output(self, vector_input: Iterable[float]) -> np.ndarray:
        return self.rotation.T @ _v3(vector_input)

    def to_sitk(self):
        import SimpleITK as sitk

        transform = sitk.AffineTransform(3)
        transform.SetCenter(self.center_lps)
        transform.SetMatrix(self.matrix)
        return transform

    def as_dict(self) -> dict:
        rotation = self.rotation
        center = np.asarray(self.center_lps)
        inverse_matrix = rotation.T
        inverse_translation = center - inverse_matrix @ center
        forward_translation = center - rotation @ center
        return {
            "semantics": "SimpleITK Resample output_to_input",
            "matrix_3x3": rotation.tolist(),
            "center_lps_mm": center.tolist(),
            "homogeneous_output_to_input": np.block(
                [[rotation, forward_translation[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
            ).tolist(),
            "homogeneous_input_to_output": np.block(
                [[inverse_matrix, inverse_translation[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
            ).tolist(),
        }
