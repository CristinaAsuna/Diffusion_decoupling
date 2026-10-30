"""Invariant setup for diffusion and Flow Matching trainers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from consistent_cdm.contracts import TargetContract
from consistent_cdm.factories import build_generative_model


@dataclass(frozen=True)
class GenerativeTrainingSetup:
    model: Any
    contract: TargetContract
    image_size: int


def prepare_generative_training(cfg: dict[str, Any], dataset: Any) -> GenerativeTrainingSetup:
    """Build a model only after validating the data/model D-only scale contract."""
    target_cfg = cfg["diffusion"].get("target_parameterization") or cfg.get("target_parameterization")
    value_range = cfg["dataset"]["train"].get("value_range", [0.0, 1.0])
    contract = TargetContract.from_config(target_cfg, raw_value_scale=float(value_range[1] - value_range[0]))
    if contract.is_d_only:
        d_norm = cfg["dataset"]["train"].get("d_only_normalization")
        if d_norm is None:
            raise ValueError("D-only diffusion requires dataset.train.d_only_normalization")
        data_scale = float(d_norm["scale"])
        if not math.isclose(data_scale, contract.difference_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"D-only dataset/model scale mismatch: {data_scale} != {contract.difference_scale}")
    image_size = cfg["model"].get("image_size") or dataset[0]["condition"].shape[-1]
    model, built_contract = build_generative_model(
        cfg,
        condition_channels=dataset.condition_channels,
        dataset_target_channels=dataset.target_channels,
        image_size=image_size,
        num_side_classes=getattr(dataset, "num_side_classes", None),
    )
    if built_contract != contract:
        raise RuntimeError("Factory target contract differs from validated training contract")
    return GenerativeTrainingSetup(model=model, contract=contract, image_size=image_size)
