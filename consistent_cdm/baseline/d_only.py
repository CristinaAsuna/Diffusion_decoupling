"""Shared data, normalization, and decoding utilities for the D-only probe.

The probe deliberately learns only ``D=(left-right)/2`` in raw line-integral
space.  Full is passed to the network using the normal fixed-range image
normalization; D uses one train-patient-only *scale* (never per-image
normalization and never centering) so the target retains its physical signed
amplitude while receiving a numerically useful L1 objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import denormalize_tensor, normalize_tensor
from consistent_cdm.evaluation.data_metrics import branch_metrics


@dataclass(frozen=True)
class DNormalization:
    """Train-only raw-D statistics plus the fixed target scale ``s_D``."""

    mean: float
    std: float
    scale: float
    scale_method: str
    scale_quantile: float | None
    scale_sample_count: int
    count: int
    raw_min: float
    raw_max: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DNormalization":
        return cls(
            mean=float(value["mean"]),
            std=float(value["std"]),
            # ``scale`` was introduced after the engineering-only z-score
            # smoke checkpoint.  Retain read compatibility but never use that
            # checkpoint for the formal no-centering experiment.
            scale=float(value.get("scale", value["std"])),
            scale_method=str(value.get("scale_method", "legacy_std")),
            scale_quantile=(None if value.get("scale_quantile") is None else float(value["scale_quantile"])),
            scale_sample_count=int(value.get("scale_sample_count", value["count"])),
            count=int(value["count"]),
            raw_min=float(value["raw_min"]),
            raw_max=float(value["raw_max"]),
        )


def raw_difference(target: torch.Tensor, data_norm_cfg: dict[str, Any]) -> torch.Tensor:
    """Return raw ``D=(left-right)/2`` from a normalized two-branch target."""

    if target.shape[-3] < 2:
        raise ValueError(f"D-only probe expects at least two target channels, got {target.shape}")
    raw = denormalize_tensor(target[..., 0:2, :, :] if target.ndim == 4 else target[0:2], data_norm_cfg)
    return (raw[..., 0:1, :, :] - raw[..., 1:2, :, :]) / 2.0 if raw.ndim == 4 else (raw[0:1] - raw[1:2]) / 2.0


def compute_d_normalization(
    dataset: Dataset,
    data_norm_cfg: dict[str, Any],
    scale_cfg: dict[str, Any] | None = None,
) -> DNormalization:
    """Compute train-only raw-D statistics and a fixed global ``s_D``.

    ``abs_percentile`` is calculated from a deterministic spatial subsample of
    every training view.  It is robust to the small number of extreme pixels,
    has no patient-level adaptation, and is recorded in the checkpoint.
    """

    scale_cfg = dict(scale_cfg or {})
    method = str(scale_cfg.get("method", "std"))
    if method not in {"raw", "std", "abs_percentile"}:
        raise ValueError(f"Unknown D scale method {method!r}; expected raw, std, or abs_percentile")
    quantile = float(scale_cfg.get("quantile", 0.95))
    if method == "abs_percentile" and not 0.5 < quantile < 1.0:
        raise ValueError("abs_percentile D scale quantile must be in (0.5, 1.0)")
    max_samples = int(scale_cfg.get("max_samples", 4_000_000))
    if max_samples <= 0:
        raise ValueError("D scale max_samples must be positive")
    per_item = max(1, int(np.ceil(max_samples / max(len(dataset), 1))))
    count = 0
    total = 0.0
    total_square = 0.0
    raw_min = float("inf")
    raw_max = float("-inf")
    scale_samples: list[np.ndarray] = []
    for sample in dataset:
        target = sample["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("Dataset target must be a tensor")
        raw_target = sample.get("raw_target")
        value = ((raw_target[0:1] - raw_target[1:2]) / 2.0 if raw_target is not None else raw_difference(target, data_norm_cfg)).detach().cpu().double()
        count += int(value.numel())
        total += float(value.sum())
        total_square += float((value * value).sum())
        raw_min = min(raw_min, float(value.min()))
        raw_max = max(raw_max, float(value.max()))
        if method == "abs_percentile":
            flattened = value.abs().reshape(-1).numpy()
            stride = max(1, int(np.ceil(flattened.size / per_item)))
            scale_samples.append(flattened[::stride][:per_item])
    if count == 0:
        raise ValueError("Cannot normalize D from an empty dataset")
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std <= 1e-8:
        raise ValueError("Raw D has near-zero training variance; D-only training is undefined")
    if method == "raw":
        scale, sample_count, stored_quantile = 1.0, 0, None
    elif method == "std":
        scale, sample_count, stored_quantile = std, count, None
    else:
        sampled = np.concatenate(scale_samples)
        scale, sample_count, stored_quantile = float(np.quantile(sampled, quantile)), int(sampled.size), quantile
    if scale <= 1e-8:
        raise ValueError("Raw D scale is near zero; D-only training is undefined")
    return DNormalization(
        mean=mean, std=std, scale=scale, scale_method=method,
        scale_quantile=stored_quantile, scale_sample_count=sample_count,
        count=count, raw_min=raw_min, raw_max=raw_max,
    )


def case_ids(dataset: Dataset) -> set[str]:
    """Extract patient IDs from the repository's stable ``CASE__variant`` names."""

    samples = getattr(dataset, "samples", None)
    if samples is None and hasattr(dataset, "base"):
        samples = getattr(dataset.base, "samples", None)
    if samples is None:
        raise TypeError("Dataset does not expose case-folder sample metadata")
    return {str(sample.name).split("__", 1)[0] for sample in samples}


def assert_no_forbidden_cases(dataset: Dataset, forbidden: Iterable[str]) -> None:
    forbidden_ids = {str(value) for value in forbidden}
    overlap = sorted(case_ids(dataset) & forbidden_ids)
    if overlap:
        raise ValueError(
            "Frozen test patients were selected for development work: " + ", ".join(overlap)
        )


def assert_allowed_case_prefixes(dataset: Dataset, prefixes: Iterable[str]) -> None:
    """Fail closed when a development root is not the declared patient cohort."""

    allowed = tuple(str(value) for value in prefixes)
    if not allowed:
        return
    unexpected = sorted(case_id for case_id in case_ids(dataset) if not case_id.startswith(allowed))
    if unexpected:
        raise ValueError(
            "Development dataset contains IDs outside its declared cohort: "
            + ", ".join(unexpected[:10])
        )


class DOnlyDataset(Dataset):
    """Turn paired branches into ``Full -> D/s_D`` samples."""

    def __init__(
        self,
        base: Dataset,
        data_norm_cfg: dict[str, Any],
        normalization: DNormalization,
    ) -> None:
        self.base = base
        self.data_norm_cfg = data_norm_cfg
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        target = sample["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("Dataset target must be a tensor")
        result = dict(sample)
        # Retain the paired normalized target for validation-only hard decoding
        # metrics.  The model and its losses still see only ``target`` below.
        result["branch_target"] = target
        raw_target = sample.get("raw_target")
        raw_d = (raw_target[0:1] - raw_target[1:2]) / 2.0 if raw_target is not None else raw_difference(target, self.data_norm_cfg)
        result["target"] = raw_d / self.normalization.scale
        return result

    @property
    def condition_channels(self) -> int:
        return int(getattr(self.base, "condition_channels"))

    @property
    def target_channels(self) -> int:
        return 1

    @property
    def samples(self):
        return getattr(self.base, "samples")


def decode_standardized_d(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    data_norm_cfg: dict[str, Any],
    normalization: DNormalization,
    raw_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode globally scaled D and reconstruct raw/normalized branches exactly."""

    if prediction.shape[-3] != 1 or condition.shape[-3] < 1:
        raise ValueError("D-only decoding expects [B,1,H,W] prediction and full condition")
    raw_d = prediction * normalization.scale
    raw_full = raw_condition[..., 0:1, :, :] if raw_condition is not None else denormalize_tensor(condition[..., 0:1, :, :], data_norm_cfg)
    raw_branches = torch.cat((raw_full / 2.0 + raw_d, raw_full / 2.0 - raw_d), dim=-3)
    normalized_branches = normalize_tensor(raw_branches, data_norm_cfg)
    return raw_d, raw_branches, normalized_branches


@torch.no_grad()
def hard_decoded_branch_metrics(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    branch_target: torch.Tensor,
    data_norm_cfg: dict[str, Any],
    normalization: DNormalization,
    raw_condition: torch.Tensor | None = None,
    raw_target: torch.Tensor | None = None,
) -> dict[str, float]:
    """Average raw L/R metrics after mandatory ``Full/2 ± D`` decoding.

    This is validation/reporting only.  It deliberately does not clamp decoded
    branches: reported image quality must describe the same hard-additive
    prediction that is exported for formal D-space evaluation.
    """

    _, raw_prediction, _ = decode_standardized_d(prediction, condition, data_norm_cfg, normalization, raw_condition=raw_condition)
    raw_target = raw_target if raw_target is not None else denormalize_tensor(branch_target, data_norm_cfg)
    rows: list[dict[str, Any]] = []
    for pred_item, target_item, full_item in zip(
        raw_prediction.detach().cpu().numpy(),
        raw_target.detach().cpu().numpy(),
        (raw_condition[:, 0:1] if raw_condition is not None else denormalize_tensor(condition[:, 0:1], data_norm_cfg)).detach().cpu().numpy(),
    ):
        rows.append(branch_metrics(
            pred_item[0], pred_item[1], target_item[0], target_item[1], full=full_item[0], data_range=6.0,
        ))
    if not rows:
        raise ValueError("Cannot summarize an empty D-only validation batch")
    return {
        "left_mae": float(np.mean([row["left"]["mae"] for row in rows])),
        "right_mae": float(np.mean([row["right"]["mae"] for row in rows])),
        "side_mae": float(np.mean([row["side_average"]["mae"] for row in rows])),
        "left_psnr": float(np.mean([row["left"]["psnr"] for row in rows])),
        "right_psnr": float(np.mean([row["right"]["psnr"] for row in rows])),
        "side_psnr": float(np.mean([row["side_average"]["psnr"] for row in rows])),
        "left_ssim": float(np.mean([row["left"]["ssim"] for row in rows])),
        "right_ssim": float(np.mean([row["right"]["ssim"] for row in rows])),
        "side_ssim": float(np.mean([row["side_average"]["ssim"] for row in rows])),
        "sum_nmae": float(np.mean([row["sum"]["nmae"] for row in rows])),
    }



def resolve_d_normalization(dataset, data_norm_cfg, scale_cfg=None):
    """Use the prepared training record unchanged, or fit on training data."""
    record = data_norm_cfg.get("d_only_normalization")
    normalization = DNormalization.from_dict(record) if record else compute_d_normalization(dataset, data_norm_cfg, scale_cfg)
    if not np.isfinite(normalization.scale) or normalization.scale <= 0:
        raise ValueError("D normalization scale must be positive and finite")
    return normalization
