"""Controlled condition degradations for synthetic-to-real stress tests.

Every operator takes a raw line-integral image and one strength parameter; the
operator's ``identity`` strength returns the input unchanged.  Intensity
operators model detector/vendor effects and alter only the condition F, so the
paired L/R stay valid targets.  Geometric operators are linear resamplings and
are applied identically to F, L and R, preserving F=L+R.  Pixel parameters are
expressed in model-resolution pixels so a level means the same at any raw size.
These are diagnostic approximations, not a calibrated scanner model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter


@dataclass(frozen=True)
class Degradation:
    name: str
    kind: str  # "intensity" (F only) or "geometric" (F, L and R)
    identity: float
    levels: tuple[float, ...]
    apply: Callable[..., np.ndarray]
    description: str


def _pixels(value, shape, model_size):
    return float(value) * max(shape) / float(model_size)


def exponential(full, k, **_):
    """Detector brightness without log: (1-exp(-kF))/k, tending to F as k->0."""
    if k == 0: return full
    return ((1 - np.exp(-k * full)) / k).astype(np.float32)


def gamma(full, value, **_):
    """Global display gamma relative to the image maximum."""
    peak = max(float(full.max()), 1e-8)
    return (peak * np.power(np.clip(full, 0, None) / peak, value)).astype(np.float32)


def blur(full, sigma, *, model_size, **_):
    """Gaussian focal-spot/detector blur; sigma in model pixels."""
    if sigma == 0: return full
    return gaussian_filter(full, _pixels(sigma, full.shape, model_size), mode="nearest").astype(np.float32)


def noise(full, sigma, *, rng, **_):
    """Signal-dependent quantum-like noise with std sigma*sqrt(F)."""
    if sigma == 0: return full
    std = sigma * np.sqrt(np.clip(full, 0, None) + 1e-3)
    return (full + std * rng.standard_normal(full.shape)).astype(np.float32)


def scatter(full, fraction, *, model_size, **_):
    """Add a broad low-pass scatter fraction in transmission, then re-log."""
    if fraction == 0: return full
    intensity = np.exp(-full.astype(np.float64))
    halo = gaussian_filter(intensity, _pixels(32, full.shape, model_size), mode="nearest")
    return (-np.log((intensity + fraction * halo) / (1 + fraction))).astype(np.float32)


def unsharp(full, amount, *, model_size, **_):
    """Multi-scale band boosting resembling vendor contrast enhancement."""
    if amount == 0: return full
    bands = [full - gaussian_filter(full, _pixels(s, full.shape, model_size), mode="nearest")
             for s in (1, 4, 16)]
    return (full + amount * sum(bands) / len(bands)).astype(np.float32)


def equalize(full, blend, **_):
    """Blend toward global histogram equalization within the image's range."""
    if blend == 0: return full
    ranks = np.argsort(np.argsort(full, axis=None)).reshape(full.shape) / max(full.size - 1, 1)
    low, high = float(full.min()), float(full.max())
    return ((1 - blend) * full + blend * (low + ranks * (high - low))).astype(np.float32)


def invert(full, blend, **_):
    """Blend toward max+min-F, the opposite brightness convention."""
    if blend == 0: return full
    flipped = float(full.max()) + float(full.min()) - full
    return ((1 - blend) * full + blend * flipped).astype(np.float32)


def _affine(image, matrix, offset):
    return affine_transform(image, matrix, offset=offset, order=1, mode="constant", cval=0.).astype(np.float32)


def scale(image, factor, **_):
    """Zoom about the centre; factor<1 adds empty field of view around the head."""
    if factor == 1: return image
    centre = (np.asarray(image.shape, dtype=np.float64) - 1) / 2
    matrix = np.eye(2) / factor
    return _affine(image, matrix, centre - matrix @ centre)


def rotate(image, degrees, **_):
    """In-plane rotation about the centre (head tilt)."""
    if degrees == 0: return image
    angle = np.deg2rad(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = (np.asarray(image.shape, dtype=np.float64) - 1) / 2
    return _affine(image, matrix, centre - matrix @ centre)


def translate(image, fraction, **_):
    """Shift down and right by a fraction of the image size."""
    if fraction == 0: return image
    offset = -fraction * np.asarray(image.shape, dtype=np.float64)
    return _affine(image, np.eye(2), offset)


DEGRADATIONS = {d.name: d for d in (
    Degradation("exponential", "intensity", 0., (.25, .5, 1., 2.), exponential, "(1-exp(-kF))/k"),
    Degradation("gamma", "intensity", 1., (.5, .75, 1.5, 2.), gamma, "global gamma"),
    Degradation("blur", "intensity", 0., (.5, 1., 2., 4.), blur, "Gaussian sigma, model px"),
    Degradation("noise", "intensity", 0., (.01, .03, .1, .3), noise, "std sigma*sqrt(F)"),
    Degradation("scatter", "intensity", 0., (.1, .3, 1., 3.), scatter, "low-pass scatter fraction"),
    Degradation("unsharp", "intensity", 0., (.5, 1., 2., 4.), unsharp, "multi-scale band gain"),
    Degradation("equalize", "intensity", 0., (.25, .5, .75, 1.), equalize, "histogram equalization blend"),
    Degradation("invert", "intensity", 0., (.25, .5, .75, 1.), invert, "inversion blend"),
    Degradation("scale", "geometric", 1., (.7, .85, 1.15, 1.3), scale, "central zoom factor"),
    Degradation("rotate", "geometric", 0., (-10., -5., 5., 10.), rotate, "in-plane degrees"),
    Degradation("translate", "geometric", 0., (-.1, -.05, .05, .1), translate, "shift, image fraction"),
)}


def degrade(name, level, full, left, right, *, model_size, seed=0):
    """Return degraded (F, L, R); L/R change only for geometric operators."""
    spec = DEGRADATIONS[name]
    arrays = [np.asarray(a, dtype=np.float32) for a in (full, left, right)]
    if level == spec.identity:
        return tuple(arrays)
    options = dict(model_size=model_size, rng=np.random.default_rng(seed))
    if spec.kind == "geometric":
        return tuple(spec.apply(a, level, **options) for a in arrays)
    degraded = spec.apply(arrays[0], level, **options)
    if not np.isfinite(degraded).all():
        raise ValueError(f"Degradation {name}={level} produced nonfinite values")
    return degraded, arrays[1], arrays[2]
