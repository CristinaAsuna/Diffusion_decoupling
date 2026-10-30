"""Target-space contracts for branch, sum/difference, and D-only decoding.

The contract has no trainable state.  Keeping it separate from a model preserves
historical checkpoint state dicts while making the task representation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def canonical_target_parameterization(value: str | None) -> str:
    raw = str(value or "left_right").lower()
    aliases = {
        "identity": "left_right", "left-right": "left_right", "left_right": "left_right", "lr": "left_right",
        "sum-difference": "sum_difference", "sum_difference": "sum_difference", "sumdiff": "sum_difference", "rd": "sum_difference",
        "d": "d_only", "d-only": "d_only", "d_only": "d_only", "difference_only": "d_only",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError("target_parameterization.type must be one of: left_right, sum_difference, d_only") from exc


@dataclass(frozen=True)
class TargetContract:
    """Non-trainable encode/decode contract for a model target space."""

    kind: str
    residual_scale: float = 1.0
    difference_scale: float = 1.0
    d_only_already_standardized: bool = False
    clamp_decoded_target: bool = False

    @classmethod
    def from_config(cls, cfg: dict | None, *, raw_value_scale: float) -> "TargetContract":
        cfg = dict(cfg or {})
        kind = canonical_target_parameterization(cfg.get("type"))
        residual_scale = float(cfg.get("r_scale", cfg.get("residual_scale", cfg.get("intensity_scale", raw_value_scale))))
        difference_scale = float(cfg.get("d_scale", cfg.get("difference_scale", cfg.get("intensity_scale", raw_value_scale))))
        if residual_scale <= 0.0 or difference_scale <= 0.0:
            raise ValueError("target scales must be positive")
        return cls(kind, residual_scale, difference_scale, bool(cfg.get("already_standardized", False)), bool(cfg.get("clamp_decoded", False)))

    @property
    def is_d_only(self) -> bool:
        return self.kind == "d_only"

    @property
    def expected_model_channels(self) -> int:
        return 1 if self.is_d_only else 2

    @property
    def channel_names(self) -> tuple[str, str]:
        if self.kind == "sum_difference":
            return "r", "d"
        if self.kind == "d_only":
            return "d", "d"
        return "left", "right"

    def validate(self, *, target_channels: int, branch_decoder: bool, difference_loss: bool, difference_edge_loss: bool) -> None:
        if target_channels != self.expected_model_channels:
            raise ValueError(f"{self.kind} target_parameterization expects {self.expected_model_channels} model target channel(s)")
        if branch_decoder and self.is_d_only:
            raise ValueError("d_only target_parameterization forbids branch_decoder")
        if self.is_d_only and (difference_loss or difference_edge_loss):
            raise ValueError("d_only target_parameterization forbids difference and difference-edge losses")

    @staticmethod
    def _split_left_right(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[1] != 2:
            raise ValueError(f"Expected 2 channels for left/right split, got {tensor.shape[1]}")
        return tensor[:, 0:1], tensor[:, 1:2]

    def encode(
        self, target: torch.Tensor, condition: torch.Tensor, *, to_raw: Callable[[torch.Tensor], torch.Tensor], full_raw: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.kind == "left_right":
            return target
        if self.is_d_only and self.d_only_already_standardized:
            if target.shape[1] != 1:
                raise ValueError("standardized d_only target must have one channel")
            return target
        left, right = self._split_left_right(target)
        left_raw, right_raw = to_raw(left), to_raw(right)
        if self.is_d_only:
            return (left_raw - right_raw) / (2.0 * self.difference_scale)
        residual = (left_raw + right_raw - full_raw(condition)) / self.residual_scale
        difference = (left_raw - right_raw) / self.difference_scale
        return torch.cat([residual, difference], dim=1)

    def decode(
        self, target_model_space: torch.Tensor, condition: torch.Tensor, *, from_raw: Callable[[torch.Tensor], torch.Tensor], full_raw: Callable[[torch.Tensor], torch.Tensor], clamp: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.kind == "left_right":
            return clamp(target_model_space) if self.clamp_decoded_target else target_model_space
        if self.is_d_only:
            if target_model_space.shape[1] != 1:
                raise ValueError("d_only decoding expects [B,1,H,W]")
            return target_model_space
        residual, difference = self._split_left_right(target_model_space)
        total_raw = full_raw(condition) + residual * self.residual_scale
        diff_raw = difference * self.difference_scale
        decoded = torch.cat([from_raw(0.5 * (total_raw + diff_raw)), from_raw(0.5 * (total_raw - diff_raw))], dim=1)
        return clamp(decoded) if self.clamp_decoded_target else decoded

    def raw_d_from_target(self, target: torch.Tensor, *, to_raw: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.is_d_only and self.d_only_already_standardized:
            if target.shape[1] != 1:
                raise ValueError("standardized d_only target must have one channel")
            return target * self.difference_scale
        left, right = self._split_left_right(target)
        return (to_raw(left) - to_raw(right)) / 2.0

    def raw_d_from_model_space(self, value: torch.Tensor) -> torch.Tensor:
        if not self.is_d_only or value.shape[1] != 1:
            raise ValueError("raw_d_from_model_space is only valid for one-channel d_only output")
        return value * self.difference_scale
