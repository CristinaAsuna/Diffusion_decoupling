from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .drr_geometry import PixelTransform


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA projection requested but torch.cuda.is_available() is false")
    return requested


def _ray_box(rays, half_boundary_extent):
    minimum, maximum = -half_boundary_extent, half_boundary_extent
    direction = rays[:, 3:6]
    safe = torch.where(torch.abs(direction) < 1e-9, torch.full_like(direction, 1e-9), direction)
    t1 = (minimum - rays[:, :3]) / safe
    t2 = (maximum - rays[:, :3]) / safe
    near = torch.max(torch.minimum(t1, t2), dim=-1).values
    far = torch.min(torch.maximum(t1, t2), dim=-1).values
    return near, far


def _integrate_chunk(rays, volume_zyx, spacing_xyz, step_size_mm):
    size_xyz = torch.tensor(
        [volume_zyx.shape[2], volume_zyx.shape[1], volume_zyx.shape[0]], device=rays.device, dtype=torch.float32
    )
    center_extent = spacing_xyz * torch.clamp(size_xyz - 1.0, min=1.0)
    half_boundary_extent = spacing_xyz * size_xyz / 2.0
    near, far = _ray_box(rays, half_boundary_extent)
    near = torch.clamp(near, min=0.0)
    valid = far > near
    result = torch.zeros(rays.shape[0], device=rays.device, dtype=torch.float32)
    if not torch.any(valid):
        return result
    selected = rays[valid]
    near_selected, far_selected = near[valid], far[valid]
    sample_count = max(int(torch.ceil(torch.max(far_selected - near_selected) / step_size_mm).item()), 1)
    fractions = (torch.arange(sample_count, device=rays.device, dtype=torch.float32) + 0.5) / sample_count
    sample_t = near_selected[:, None] + (far_selected - near_selected)[:, None] * fractions[None, :]
    points = selected[:, None, :3] + selected[:, None, 3:6] * sample_t[..., None]
    normalized = points / (center_extent / 2.0)  # center coordinates map exactly to [-1,1]
    grid = normalized[None, :, None, :, :]
    samples = functional.grid_sample(
        volume_zyx[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0, :, 0, :]
    delta = (far_selected - near_selected) / sample_count
    result[valid] = torch.sum(samples, dim=1) * delta
    return result


def render_line_integral(
    mu_volume_zyx: np.ndarray,
    rays: np.ndarray,
    spacing_xyz,
    raw_shape_hw,
    *,
    step_size_mm: float = 0.5,
    chunk_size: int = 8192,
    device: str = "auto",
) -> np.ndarray:
    """Deterministic midpoint DRR. Reusing rays makes split projections linear."""
    device = resolve_device(device)
    volume = torch.as_tensor(mu_volume_zyx, dtype=torch.float32, device=device)
    spacing = torch.as_tensor(spacing_xyz, dtype=torch.float32, device=device)
    rays_tensor = torch.as_tensor(rays, dtype=torch.float32, device=device)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(rays_tensor), int(chunk_size)):
            pieces.append(_integrate_chunk(rays_tensor[start:start + chunk_size], volume, spacing, step_size_mm))
    return torch.cat(pieces).reshape(*raw_shape_hw).cpu().numpy().astype(np.float32)


def content_bbox(
    image: np.ndarray,
    threshold_fraction=0.02,
    padding=(20, 20, 20, 20),
    trim=(0, 0, 0, 0),
):
    image = np.asarray(image)
    dynamic = float(image.max() - image.min())
    if dynamic <= 0.0:
        return (0, image.shape[0], 0, image.shape[1])
    coordinates = np.argwhere(image > image.min() + threshold_fraction * dynamic)
    if coordinates.size == 0:
        return (0, image.shape[0], 0, image.shape[1])
    top, bottom, left, right = [int(v) for v in padding]
    trim_top, trim_bottom, trim_left, trim_right = [int(v) for v in trim]
    ymin = max(0, int(coordinates[:, 0].min()) - top + trim_top)
    ymax = min(image.shape[0], int(coordinates[:, 0].max()) + 1 + bottom - trim_bottom)
    xmin = max(0, int(coordinates[:, 1].min()) - left + trim_left)
    xmax = min(image.shape[1], int(coordinates[:, 1].max()) + 1 + right - trim_right)
    if ymin >= ymax or xmin >= xmax:
        raise ValueError(f"Degenerate projection crop {(ymin, ymax, xmin, xmax)}")
    return ymin, ymax, xmin, xmax


def apply_pixel_transform(image: np.ndarray, transform: PixelTransform) -> np.ndarray:
    data = np.fliplr(image) if transform.flip_lr else image
    ymin, ymax, xmin, xmax = transform.bbox_yxyx
    crop = data[ymin:ymax, xmin:xmax]
    pil = Image.fromarray(crop.astype(np.float32), mode="F")
    # Bilinear interpolation is a convex, linear resampling operation: a
    # non-negative line integral remains non-negative, and full=left+right is
    # preserved. Lanczos can introduce negative ringing at sharp air edges.
    resized = pil.resize((transform.output_width, transform.output_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def additivity_metrics(full: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    residual = np.asarray(full) - (np.asarray(left) + np.asarray(right))
    signal = np.abs(full) > np.percentile(np.abs(full), 20)
    if not np.any(signal):
        signal = np.ones_like(full, dtype=bool)
    mae = float(np.mean(np.abs(residual[signal])))
    nmae = mae / (float(np.mean(np.abs(full[signal]))) + 1e-12)
    a, b = full[signal].ravel(), (left + right)[signal].ravel()
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else 1.0
    return {"mae": mae, "nmae": float(nmae), "correlation": correlation, "max_abs": float(np.max(np.abs(residual)))}
