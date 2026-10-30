"""Factory for diffusion and Flow Matching without changing model checkpoints."""

from __future__ import annotations

from typing import Any

from consistent_cdm.contracts import TargetContract
from consistent_cdm.model.diffusion import GaussianConditionalDiffusion
from consistent_cdm.model.flow_matching import ConditionalFlowMatching


FLOW_TYPES = {"flow", "flow_matching", "rectified_flow"}


def generative_run_group(cfg: dict[str, Any]) -> str:
    return "flow_matching" if str(cfg.get("generative", {}).get("type", "diffusion")).lower() in FLOW_TYPES else "diffusion"


def build_generative_model(cfg: dict[str, Any], *, condition_channels: int, dataset_target_channels: int, image_size: int, num_side_classes: int | None = None):
    target_cfg = cfg.get("diffusion", {}).get("target_parameterization") or cfg.get("target_parameterization")
    contract = TargetContract.from_config(target_cfg, raw_value_scale=float(cfg["dataset"]["train"].get("value_range", [0.0, 1.0])[1] - cfg["dataset"]["train"].get("value_range", [0.0, 1.0])[0]))
    if not contract.is_d_only:
        raise ValueError("Only d_only target parameterization is supported")
    target_channels = 1
    model_cfg, diffusion_cfg = cfg["model"], cfg["diffusion"]
    model_cls = ConditionalFlowMatching if generative_run_group(cfg) == "flow_matching" else GaussianConditionalDiffusion
    model = model_cls(
        condition_channels=condition_channels, target_channels=target_channels, image_size=image_size,
        inner_channel=model_cfg["inner_channel"], channel_mults=model_cfg["channel_mults"], attn_res=model_cfg["attn_res"],
        res_blocks=model_cfg["res_blocks"], dropout=model_cfg.get("dropout", 0.0), beta_schedule=diffusion_cfg["beta_schedule"],
        prediction_type=diffusion_cfg.get("prediction_type", "epsilon"), target_parameterization_cfg=target_cfg,
        num_side_classes=num_side_classes, branch_decoder_cfg=model_cfg.get("branch_decoder"),
        branch_loss_weights=diffusion_cfg.get("branch_loss_weights"), consistency_cfg=diffusion_cfg.get("consistency_loss"),
        x0_reconstruction_cfg=diffusion_cfg.get("x0_reconstruction_loss"), min_snr_cfg=diffusion_cfg.get("min_snr"),
        difference_cfg=diffusion_cfg.get("difference_loss") or diffusion_cfg.get("difference"),
        difference_edge_cfg=diffusion_cfg.get("difference_edge_loss") or diffusion_cfg.get("difference_edge"),
        data_norm_cfg=cfg["dataset"]["train"], num_head_channels=model_cfg.get("num_head_channels", 32),
        resblock_updown=model_cfg.get("resblock_updown", True), use_scale_shift_norm=model_cfg.get("use_scale_shift_norm", True),
        **({"flow_cfg": cfg.get("flow_matching", {})} if model_cls is ConditionalFlowMatching else {}),
    )
    return model, contract
