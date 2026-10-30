from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import normalize


@dataclass(frozen=True)
class CameraGeometry:
    """Cone-beam camera in renderer-local xyz coordinates (millimetres)."""

    source_xyz_mm: tuple[float, float, float]
    detector_center_xyz_mm: tuple[float, float, float]
    u_vector_mm: tuple[float, float, float]
    v_vector_mm: tuple[float, float, float]
    raw_height: int
    raw_width: int
    sad_mm: float
    sid_mm: float
    primary_angle_deg: float
    secondary_angle_deg: float

    @property
    def source(self) -> np.ndarray:
        return np.asarray(self.source_xyz_mm, dtype=np.float64)

    @property
    def detector_center(self) -> np.ndarray:
        return np.asarray(self.detector_center_xyz_mm, dtype=np.float64)

    @property
    def u_vector(self) -> np.ndarray:
        return np.asarray(self.u_vector_mm, dtype=np.float64)

    @property
    def v_vector(self) -> np.ndarray:
        return np.asarray(self.v_vector_mm, dtype=np.float64)

    def as_dict(self) -> dict:
        return {
            "coordinate_frame": "renderer-local xyz: image direction columns, volume center is origin",
            "source_xyz_mm": list(self.source_xyz_mm),
            "detector_center_xyz_mm": list(self.detector_center_xyz_mm),
            "u_vector_per_col_mm": list(self.u_vector_mm),
            "v_vector_per_row_mm": list(self.v_vector_mm),
            "raw_shape_hw": [self.raw_height, self.raw_width],
            "sad_mm": self.sad_mm,
            "sid_mm": self.sid_mm,
            "primary_angle_deg": self.primary_angle_deg,
            "secondary_angle_deg": self.secondary_angle_deg,
        }


def make_camera(
    primary_angle_deg: float,
    secondary_angle_deg: float,
    sad_mm: float,
    sid_mm: float,
    detector_spacing_uv_mm: tuple[float, float],
    raw_shape_hw: tuple[int, int],
    isocenter_xyz_mm=(0.0, 0.0, 0.0),
) -> CameraGeometry:
    primary, secondary = np.radians([primary_angle_deg, secondary_angle_deg])
    isocenter = np.asarray(isocenter_xyz_mm, dtype=np.float64)
    radial = np.array(
        [np.cos(secondary) * np.cos(primary), np.cos(secondary) * np.sin(primary), np.sin(secondary)]
    )
    source = isocenter + sad_mm * radial
    detector = isocenter - (sid_mm - sad_mm) * radial
    spacing_u, spacing_v = detector_spacing_uv_mm
    u = spacing_u * np.array([-np.sin(primary), np.cos(primary), 0.0])
    v = spacing_v * np.array(
        [np.sin(secondary) * np.cos(primary), np.sin(secondary) * np.sin(primary), -np.cos(secondary)]
    )
    return CameraGeometry(
        tuple(source), tuple(detector), tuple(u), tuple(v), int(raw_shape_hw[0]), int(raw_shape_hw[1]),
        float(sad_mm), float(sid_mm), float(primary_angle_deg), float(secondary_angle_deg),
    )


def resolve_horizontal_flip(mode, image_geometry, camera: CameraGeometry) -> tuple[bool, dict]:
    """Resolve display orientation from the image physical geometry.

    DICOM LPS anterior is physical -Y.  In anatomical_anterior_right mode,
    detector columns after the optional flip must increase toward anterior.
    """
    if isinstance(mode, bool):
        return mode, {"mode": "explicit", "flip_lr": mode}
    if mode != "anatomical_anterior_right":
        raise ValueError(
            "projection.flip_lr must be true, false, or 'anatomical_anterior_right'"
        )
    u_local = normalize(camera.u_vector, name="detector u direction")
    u_lps = normalize(image_geometry.direction_matrix @ u_local, name="detector u in LPS")
    anterior_lps = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    raw_col_toward_anterior = float(np.dot(u_lps, anterior_lps))
    if abs(raw_col_toward_anterior) < 1e-3:
        raise ValueError(
            "Detector horizontal axis is nearly orthogonal to LPS anterior; "
            "cannot enforce anatomical_anterior_right"
        )
    flip_lr = raw_col_toward_anterior < 0.0
    displayed_u_lps = -u_lps if flip_lr else u_lps
    return flip_lr, {
        "mode": mode,
        "flip_lr": flip_lr,
        "raw_positive_col_direction_lps": u_lps.tolist(),
        "displayed_positive_col_direction_lps": displayed_u_lps.tolist(),
        "displayed_col_dot_lps_anterior": float(np.dot(displayed_u_lps, anterior_lps)),
        "target": "LPS anterior [0,-1,0] points toward increasing displayed columns",
    }


def auto_detector_shape(image_geometry, sid_mm: float, sad_mm: float, spacing_uv_mm, padding_uv=(32, 32)):
    """Size a lateral detector from image y/z boundary extents, never x/y."""
    extent = image_geometry.boundary_extent_xyz_mm
    magnification = float(sid_mm) / float(sad_mm)
    spacing_u, spacing_v = (float(x) for x in spacing_uv_mm)
    pad_u, pad_v = (int(x) for x in padding_uv)
    width = int(np.ceil(extent[1] * magnification / spacing_u)) + 2 * pad_u
    height = int(np.ceil(extent[2] * magnification / spacing_v)) + 2 * pad_v
    return height, width


def detector_points(camera: CameraGeometry) -> np.ndarray:
    rows, cols = np.meshgrid(
        np.arange(camera.raw_height, dtype=np.float64),
        np.arange(camera.raw_width, dtype=np.float64),
        indexing="ij",
    )
    return (
        camera.detector_center
        + (cols[..., None] - (camera.raw_width - 1.0) / 2.0) * camera.u_vector
        + (rows[..., None] - (camera.raw_height - 1.0) / 2.0) * camera.v_vector
    )


def generate_rays(camera: CameraGeometry) -> np.ndarray:
    pixels = detector_points(camera)
    origins = np.broadcast_to(camera.source, pixels.shape)
    directions = pixels - origins
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.concatenate((origins, directions), axis=-1).reshape(-1, 6).astype(np.float32)


def project_points(points_xyz_mm: np.ndarray, camera: CameraGeometry):
    """Analytically project renderer-local points to pre-flip raw (row, col)."""
    points = np.atleast_2d(np.asarray(points_xyz_mm, dtype=np.float64))
    source = camera.source
    center = camera.detector_center
    detector_normal = normalize(center - source, name="detector normal")
    direction = points - source
    denominator = direction @ detector_normal
    numerator = float(np.dot(center - source, detector_normal))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = numerator / denominator
        intersection = source + scale[:, None] * direction
    relative = intersection - center
    col = (camera.raw_width - 1.0) / 2.0 + relative @ camera.u_vector / np.dot(camera.u_vector, camera.u_vector)
    row = (camera.raw_height - 1.0) / 2.0 + relative @ camera.v_vector / np.dot(camera.v_vector, camera.v_vector)
    valid_ray = np.isfinite(row) & np.isfinite(col) & (denominator > 1e-9) & (scale > 0.0)
    return np.stack((row, col), axis=-1), valid_ray


def detector_shape_for_volume(image_geometry, camera: CameraGeometry, padding_uv=(32, 32)):
    """Fit all eight volume-boundary corners for this exact camera angle."""
    half = image_geometry.boundary_extent_xyz_mm / 2.0
    corners = np.array(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
    )
    row_col, valid = project_points(corners, camera)
    if not np.all(valid):
        raise ValueError("A volume corner is not projectable by the configured camera")
    pad_u, pad_v = [int(value) for value in padding_uv]
    half_height = int(np.ceil(np.max(np.abs(row_col[:, 0]))))
    half_width = int(np.ceil(np.max(np.abs(row_col[:, 1]))))
    height = 2 * half_height + 1 + 2 * pad_v
    width = 2 * half_width + 1 + 2 * pad_u
    return height, width


@dataclass(frozen=True)
class PixelTransform:
    raw_height: int
    raw_width: int
    bbox_yxyx: tuple[int, int, int, int]
    output_height: int
    output_width: int
    flip_lr: bool = True

    def apply(self, raw_row_col: np.ndarray):
        coordinates = np.atleast_2d(np.asarray(raw_row_col, dtype=np.float64)).copy()
        row, col = coordinates[:, 0], coordinates[:, 1]
        if self.flip_lr:
            col = (self.raw_width - 1.0) - col
        ymin, ymax, xmin, xmax = self.bbox_yxyx
        in_raw = (row >= 0.0) & (row <= self.raw_height - 1.0) & (col >= 0.0) & (col <= self.raw_width - 1.0)
        in_crop = (row >= ymin) & (row < ymax) & (col >= xmin) & (col < xmax)
        # Pillow resize uses a half-pixel centre convention.
        out_row = (row - ymin + 0.5) * (self.output_height / (ymax - ymin)) - 0.5
        out_col = (col - xmin + 0.5) * (self.output_width / (xmax - xmin)) - 0.5
        return np.stack((out_row, out_col), axis=-1), in_raw & in_crop

    def as_dict(self) -> dict:
        return {
            "raw_shape_hw": [self.raw_height, self.raw_width],
            "flip_lr": self.flip_lr,
            "bbox_ymin_ymax_xmin_xmax_half_open": list(self.bbox_yxyx),
            "output_shape_hw": [self.output_height, self.output_width],
            "resize_kernel": "Pillow bilinear",
            "resize_coordinate_formula": "out=(cropped_in+0.5)*out_size/crop_size-0.5",
        }
