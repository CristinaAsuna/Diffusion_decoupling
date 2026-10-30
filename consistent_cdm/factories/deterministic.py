"""Deterministic model construction shared by ResUNet and Pix2Pix entrypoints."""

from __future__ import annotations

from consistent_cdm.baseline.models import (
    MultiScalePatchDiscriminator,
    ResUNetGenerator,
    build_discriminator,
    build_generator,
)


def build_deterministic_generator(model_cfg: dict, condition_channels: int, target_channels: int) -> ResUNetGenerator:
    """Build a generator without changing its historical state-dict structure."""
    return build_generator(model_cfg, condition_channels, target_channels)


def build_deterministic_discriminator(model_cfg: dict, condition_channels: int, target_channels: int) -> MultiScalePatchDiscriminator:
    """Build the Pix2Pix discriminator with the same legacy constructor."""
    return build_discriminator(model_cfg, condition_channels, target_channels)
