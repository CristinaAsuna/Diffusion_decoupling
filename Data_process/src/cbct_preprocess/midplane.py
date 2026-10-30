from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ImageGeometry, normalize
from .io import centroid_for_label


@dataclass(frozen=True)
class SignedPlane:
    point_lps_mm: tuple[float, float, float]
    normal_lps: tuple[float, float, float]
    positive_side: str = "left"

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.point_lps_mm, dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self.normal_lps, dtype=np.float64)

    def signed_distance(self, points_lps: np.ndarray) -> np.ndarray:
        return (np.asarray(points_lps, dtype=np.float64) - self.point) @ self.normal

    def transformed_input_to_output(self, rigid_mapping) -> "SignedPlane":
        point = rigid_mapping.input_to_output(self.point)
        normal = normalize(rigid_mapping.vector_input_to_output(self.normal))
        return SignedPlane(tuple(point), tuple(normal), self.positive_side)

    def as_dict(self) -> dict:
        return {
            "point_lps_mm": list(self.point_lps_mm),
            "normal_lps": list(self.normal_lps),
            "positive_side": self.positive_side,
            "equation": "dot(normal_lps, point_lps - point_lps_mm) = signed_distance_mm",
        }


def estimate_midplane(label_image, label_left: int = 8, label_right: int = 9):
    """Estimate a physical midsagittal proxy from patient-left/right labels."""
    left = centroid_for_label(label_image, label_left)
    right = centroid_for_label(label_image, label_right)
    # With the v2 side contract, positive points from label 9 (patient right)
    # to label 8 (patient left). Keep this anatomical sign in physical LPS.
    normal = normalize(left - right, name="left-right centroid vector")
    geometry = ImageGeometry.from_sitk(label_image)
    image_x = normalize(geometry.direction_matrix[:, 0], name="image x direction")
    point = (left + right) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(normal, image_x)), -1.0, 1.0))))
    return SignedPlane(tuple(point), tuple(normal)), left, right, angle
