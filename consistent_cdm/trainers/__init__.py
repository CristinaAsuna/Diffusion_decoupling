"""Small shared setup units for training entrypoints.

Training loops intentionally remain in their family-specific applications; this module
contains only invariant preflight and construction work.
"""

from .generative import GenerativeTrainingSetup, prepare_generative_training

__all__ = ["GenerativeTrainingSetup", "prepare_generative_training"]
