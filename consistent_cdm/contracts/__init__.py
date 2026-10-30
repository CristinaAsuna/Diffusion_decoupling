"""Stable task contracts shared by deterministic and generative model families."""

from .targets import TargetContract, canonical_target_parameterization

__all__ = ["TargetContract", "canonical_target_parameterization"]
