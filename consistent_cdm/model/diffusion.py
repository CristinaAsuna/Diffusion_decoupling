
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from consistent_cdm.contracts import TargetContract

try:
    from .models import UNet
except ImportError:
    from models import UNet


def extract(a: torch.Tensor, t: torch.Tensor, x_shape=(1, 1, 1, 1)) -> torch.Tensor:
    batch, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(batch, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-6,
    linear_end: float = 1e-2,
    cosine_s: float = 8e-3,
    zero_terminal_snr: bool = False,
):
    if schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        betas = (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999).cpu().numpy()
    else:
        raise NotImplementedError(f"Unsupported beta schedule: {schedule}")
    return _rescale_zero_terminal_snr(betas) if zero_terminal_snr else betas


def _rescale_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    first = sqrt_alphas_cumprod[0].copy()
    last = sqrt_alphas_cumprod[-1].copy()
    if abs(first - last) < 1e-12:
        return betas
    sqrt_alphas_cumprod = (sqrt_alphas_cumprod - last) * first / (first - last)
    alphas_cumprod = sqrt_alphas_cumprod**2
    alphas = np.empty_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / np.clip(alphas_cumprod[:-1], 1e-12, None)
    return np.clip(1.0 - alphas, 1e-8, 0.999)


class GaussianConditionalDiffusion(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        target_channels: int,
        image_size: int,
        inner_channel: int,
        channel_mults: list[int],
        attn_res: list[int],
        res_blocks: int,
        dropout: float,
        beta_schedule: dict,
        prediction_type: str = "epsilon",
        target_parameterization_cfg: dict | None = None,
        num_side_classes: int | None = None,
        branch_decoder_cfg: dict | None = None,
        branch_loss_weights: dict[str, float] | None = None,
        consistency_cfg: dict | None = None,
        x0_reconstruction_cfg: dict | None = None,
        min_snr_cfg: dict | None = None,
        difference_cfg: dict | None = None,
        difference_edge_cfg: dict | None = None,
        data_norm_cfg: dict | None = None,
        num_head_channels: int = 32,
        resblock_updown: bool = True,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.target_channels = target_channels
        branch_decoder_cfg = dict(branch_decoder_cfg or {})
        target_parameterization_cfg = dict(target_parameterization_cfg or {})
        consistency_cfg = dict(consistency_cfg or {})
        x0_reconstruction_cfg = dict(x0_reconstruction_cfg or {})
        min_snr_cfg = dict(min_snr_cfg or {})
        difference_cfg = dict(difference_cfg or {})
        difference_edge_cfg = dict(difference_edge_cfg or {})
        data_norm_cfg = dict(data_norm_cfg or {})
        self.normalize_mode = str(data_norm_cfg.get("normalize", "none"))
        self.value_range = data_norm_cfg.get("value_range")
        self.raw_value_scale = self._default_raw_scale()
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("prediction_type must be one of: epsilon, v_prediction")
        self.target_contract = TargetContract.from_config(target_parameterization_cfg, raw_value_scale=self.raw_value_scale)
        # Kept as public compatibility attributes for historical adapters/checkpoints.
        self.target_parameterization = self.target_contract.kind
        self.d_only_target_is_standardized = self.target_contract.d_only_already_standardized
        self.residual_scale = self.target_contract.residual_scale
        self.difference_scale = self.target_contract.difference_scale
        self.clamp_decoded_target = self.target_contract.clamp_decoded_target
        self.denoise_fn = UNet(
            image_size=image_size,
            in_channel=condition_channels + target_channels,
            out_channel=target_channels,
            inner_channel=inner_channel,
            channel_mults=tuple(channel_mults),
            attn_res=attn_res,
            res_blocks=res_blocks,
            dropout=dropout,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            num_side_classes=num_side_classes,
            branch_decoder=bool(branch_decoder_cfg.get("enabled", False)),
            branch_split_blocks=branch_decoder_cfg.get("shared_blocks"),
        )
        channel_name_0, channel_name_1 = self._target_channel_names()
        branch_loss_weights = branch_loss_weights or {}
        self.branch_loss_weights = {
            "left": float(branch_loss_weights.get("left", 1.0)),
            "right": float(branch_loss_weights.get("right", 1.0)),
            "r": float(branch_loss_weights.get("r", branch_loss_weights.get(channel_name_0, 1.0))),
            "d": float(branch_loss_weights.get("d", branch_loss_weights.get(channel_name_1, 1.0))),
        }
        if self.denoise_fn.branch_decoder and self.target_channels != 2:
            raise ValueError("branch_decoder currently expects target_channels=2 for left/right prediction")
        self.consistency_enabled = bool(consistency_cfg.get("enabled", False))
        if self.consistency_enabled and self.target_channels != 2:
            raise ValueError("consistency_loss expects a 2-channel left/right target")
        self.consistency_lambda = float(consistency_cfg.get("lambda", 1e-4))
        if self.consistency_lambda < 0.0:
            raise ValueError("consistency_loss.lambda must be non-negative")
        consistency_max_ratio = consistency_cfg.get("max_ratio", 0.1)
        self.consistency_max_ratio = None if consistency_max_ratio is None else float(consistency_max_ratio)
        if self.consistency_max_ratio is not None and self.consistency_max_ratio < 0.0:
            raise ValueError("consistency_loss.max_ratio must be non-negative or null")
        self.consistency_gamma_power = float(consistency_cfg.get("gamma_power", 1.0))
        self.consistency_beta = float(consistency_cfg.get("smooth_l1_beta", 1.0))
        self.consistency_eps = float(consistency_cfg.get("eps", 1e-6))
        self.consistency_gamma_threshold = float(consistency_cfg.get("gamma_threshold", 0.8))
        self.consistency_start_step = int(consistency_cfg.get("start_step", 0))
        self.consistency_warmup_steps = int(consistency_cfg.get("warmup_steps", 0))
        self.consistency_mask_percentile = float(consistency_cfg.get("mask_percentile", 20.0))
        self.consistency_intensity_scale = float(consistency_cfg.get("intensity_scale", self.raw_value_scale))
        if self.consistency_intensity_scale <= 0.0:
            raise ValueError("consistency_loss.intensity_scale must be positive")
        self.x0_reconstruction_enabled = bool(x0_reconstruction_cfg.get("enabled", False))
        self.x0_reconstruction_lambda = float(x0_reconstruction_cfg.get("lambda", 0.0))
        if self.x0_reconstruction_lambda < 0.0:
            raise ValueError("x0_reconstruction_loss.lambda must be non-negative")
        x0_max_ratio = x0_reconstruction_cfg.get("max_ratio", None)
        self.x0_reconstruction_max_ratio = None if x0_max_ratio is None else float(x0_max_ratio)
        if self.x0_reconstruction_max_ratio is not None and self.x0_reconstruction_max_ratio < 0.0:
            raise ValueError("x0_reconstruction_loss.max_ratio must be non-negative or null")
        self.x0_reconstruction_start_step = int(x0_reconstruction_cfg.get("start_step", 0))
        self.x0_reconstruction_loss_type = str(x0_reconstruction_cfg.get("loss", "l1")).lower()
        if self.x0_reconstruction_loss_type not in {"l1", "smooth_l1", "mse"}:
            raise ValueError("x0_reconstruction_loss.loss must be one of: l1, smooth_l1, mse")
        self.x0_reconstruction_beta = float(x0_reconstruction_cfg.get("smooth_l1_beta", 1.0))
        self.x0_reconstruction_clamp = bool(x0_reconstruction_cfg.get("clamp", False))
        self.min_snr_enabled = bool(min_snr_cfg.get("enabled", False))
        self.min_snr_gamma = float(min_snr_cfg.get("gamma", 5.0))
        if self.min_snr_gamma <= 0.0:
            raise ValueError("min_snr.gamma must be positive")
        self.difference_enabled = bool(difference_cfg.get("enabled", False))
        self.difference_lambda = float(difference_cfg.get("lambda", 0.0))
        self.difference_intensity_scale = float(difference_cfg.get("intensity_scale", self.raw_value_scale))
        if self.difference_lambda < 0.0 or self.difference_intensity_scale <= 0.0:
            raise ValueError("difference loss lambda must be non-negative and intensity_scale must be positive")
        self.difference_edge_enabled = bool(difference_edge_cfg.get("enabled", False))
        self.difference_edge_lambda = float(difference_edge_cfg.get("lambda", 0.0))
        self.difference_edge_intensity_scale = float(difference_edge_cfg.get("intensity_scale", self.raw_value_scale))
        if self.difference_edge_lambda < 0.0 or self.difference_edge_intensity_scale <= 0.0:
            raise ValueError("difference_edge loss lambda must be non-negative and intensity_scale must be positive")
        self.target_contract.validate(
            target_channels=self.target_channels,
            branch_decoder=self.denoise_fn.branch_decoder,
            difference_loss=self.difference_enabled,
            difference_edge_loss=self.difference_edge_enabled,
        )
        self.last_loss_breakdown: dict[str, float] = {}
        self.beta_schedule = beta_schedule
        self._set_noise_schedule()

    def _default_raw_scale(self) -> float:
        if self.value_range is None:
            return 1.0
        return max(float(self.value_range[1]) - float(self.value_range[0]), 1e-8)

    def _split_left_right(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[1] != 2:
            raise ValueError(f"Expected 2 channels for left/right split, got {tensor.shape[1]}")
        return tensor[:, 0:1], tensor[:, 1:2]

    def _target_channel_names(self) -> tuple[str, str]:
        return self.target_contract.channel_names

    @property
    def is_d_only(self) -> bool:
        return self.target_contract.is_d_only

    def _denormalize_to_raw(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "none":
            return tensor
        if self.normalize_mode == "fixed_range_m11":
            if self.value_range is None:
                raise ValueError("value_range is required when fixed_range_m11 normalization is used")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return ((tensor + 1.0) * 0.5) * (max_val - min_val) + min_val
        if self.normalize_mode == "fixed_range_01":
            if self.value_range is None:
                raise ValueError("value_range is required when fixed_range_01 normalization is used")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return tensor * (max_val - min_val) + min_val
        raise ValueError(
            "An invertible normalization mode is required for raw-space physics losses. "
            f"Got normalize={self.normalize_mode!r}; supported modes are none, fixed_range_01, fixed_range_m11."
        )

    def _normalize_from_raw(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "none":
            return tensor
        if self.normalize_mode == "fixed_range_m11":
            if self.value_range is None:
                raise ValueError("value_range is required when fixed_range_m11 normalization is used")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return ((tensor - min_val) / max(max_val - min_val, 1e-8)) * 2.0 - 1.0
        if self.normalize_mode == "fixed_range_01":
            if self.value_range is None:
                raise ValueError("value_range is required when fixed_range_01 normalization is used")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return (tensor - min_val) / max(max_val - min_val, 1e-8)
        raise ValueError(f"Unsupported normalize mode: {self.normalize_mode!r}")

    def _maybe_clamp_data_range(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.clamp_decoded_target:
            return tensor
        if self.normalize_mode == "fixed_range_m11":
            return tensor.clamp(-1.0, 1.0)
        if self.normalize_mode == "fixed_range_01":
            return tensor.clamp(0.0, 1.0)
        return tensor

    def _full_condition_raw(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[1] < 1:
            raise ValueError("condition must contain the full projection in channel 0")
        return self._denormalize_to_raw(condition[:, 0:1])

    def encode_target(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.target_contract.encode(target, condition, to_raw=self._denormalize_to_raw, full_raw=self._full_condition_raw)

    def decode_target(self, target_model_space: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.target_contract.decode(target_model_space, condition, from_raw=self._normalize_from_raw, full_raw=self._full_condition_raw, clamp=self._maybe_clamp_data_range)

    def raw_d_from_target(self, target: torch.Tensor) -> torch.Tensor:
        """Return raw D from a paired normalized L/R training target."""
        return self.target_contract.raw_d_from_target(target, to_raw=self._denormalize_to_raw)

    def raw_d_from_model_space(self, d_model_space: torch.Tensor) -> torch.Tensor:
        return self.target_contract.raw_d_from_model_space(d_model_space)

    def _per_sample_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).pow(2).flatten(1).mean(dim=1)

    def _weighted_mean(self, per_sample: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        return (per_sample * sample_weights.view(-1)).mean()

    def _compute_training_loss(
        self,
        model_output: torch.Tensor,
        objective_target: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.target_channels != 2:
            return self._weighted_mean(self._per_sample_mse(model_output, objective_target), sample_weights), {}
        pred_first, pred_second = self._split_left_right(model_output)
        target_first, target_second = self._split_left_right(objective_target)
        first_loss_per_sample = self._per_sample_mse(pred_first, target_first)
        second_loss_per_sample = self._per_sample_mse(pred_second, target_second)
        first_name, second_name = self._target_channel_names()
        first_weight = self.branch_loss_weights[first_name]
        second_weight = self.branch_loss_weights[second_name]
        total_per_sample = (first_weight * first_loss_per_sample + second_weight * second_loss_per_sample) / max(first_weight + second_weight, 1e-8)
        first_loss = self._weighted_mean(first_loss_per_sample, sample_weights)
        second_loss = self._weighted_mean(second_loss_per_sample, sample_weights)
        return self._weighted_mean(total_per_sample, sample_weights), {
            f"{first_name}_diff_loss": first_loss,
            f"{second_name}_diff_loss": second_loss,
        }

    def _reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.x0_reconstruction_loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.x0_reconstruction_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=self.x0_reconstruction_beta)
        if self.x0_reconstruction_loss_type == "mse":
            return F.mse_loss(pred, target)
        raise RuntimeError(f"Unsupported x0 reconstruction loss: {self.x0_reconstruction_loss_type}")

    def _compute_x0_reconstruction_loss(
        self,
        y_0_hat: torch.Tensor,
        target_model_space: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.target_channels != 2:
            loss = self._reconstruction_loss(y_0_hat, target_model_space)
            return loss, {}
        pred_first, pred_second = self._split_left_right(y_0_hat)
        target_first, target_second = self._split_left_right(target_model_space)
        first_loss = self._reconstruction_loss(pred_first, target_first)
        second_loss = self._reconstruction_loss(pred_second, target_second)
        first_name, second_name = self._target_channel_names()
        first_weight = self.branch_loss_weights[first_name]
        second_weight = self.branch_loss_weights[second_name]
        total_loss = (first_weight * first_loss + second_weight * second_loss) / max(first_weight + second_weight, 1e-8)
        return total_loss, {
            f"{first_name}_x0_loss": first_loss,
            f"{second_name}_x0_loss": second_loss,
        }

    def _compute_consistency_loss(
        self,
        condition: torch.Tensor,
        sample_gammas: torch.Tensor,
        decoded_y_0_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_left, pred_right = self._split_left_right(decoded_y_0_hat)
        full_raw = self._full_condition_raw(condition)
        left_raw = self._denormalize_to_raw(pred_left)
        right_raw = self._denormalize_to_raw(pred_right)
        residual = (left_raw + right_raw - full_raw) / self.consistency_intensity_scale
        flat_full = full_raw.detach().flatten(1)
        q = min(max(self.consistency_mask_percentile / 100.0, 0.0), 1.0)
        threshold = torch.quantile(flat_full, q, dim=1).view(-1, 1, 1, 1)
        spatial_mask = (full_raw > threshold).to(dtype=full_raw.dtype)
        gamma_mask = (sample_gammas > self.consistency_gamma_threshold).to(dtype=full_raw.dtype).view(-1, 1, 1, 1)
        consistency_map = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            beta=self.consistency_beta,
        )
        gamma_weight = sample_gammas.view(-1, 1, 1, 1).pow(self.consistency_gamma_power) * gamma_mask
        weight = gamma_weight * spatial_mask
        denom = weight.sum()
        if float(denom.detach().item()) <= 0.0:
            zero = torch.zeros((), device=condition.device, dtype=condition.dtype)
            return zero, {
                "consistency_active_ratio": gamma_mask.mean(),
                "consistency_mask_ratio": spatial_mask.mean(),
            }
        return (weight * consistency_map).sum() / denom, {
            "consistency_active_ratio": gamma_mask.mean(),
            "consistency_mask_ratio": spatial_mask.mean(),
        }

    def _raw_difference(self, tensor: torch.Tensor) -> torch.Tensor:
        left, right = self._split_left_right(tensor)
        return self._denormalize_to_raw(left) - self._denormalize_to_raw(right)

    def _compute_difference_loss(self, decoded_y_0_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = self._raw_difference(decoded_y_0_hat) / self.difference_intensity_scale
        target_diff = self._raw_difference(target) / self.difference_intensity_scale
        return F.l1_loss(pred_diff, target_diff)

    def _sobel_edges(self, tensor: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3, 3) / 4.0
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3, 3) / 4.0
        padded = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
        return torch.cat([F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)], dim=1)

    def _compute_difference_edge_loss(self, decoded_y_0_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = self._raw_difference(decoded_y_0_hat) / self.difference_edge_intensity_scale
        target_diff = self._raw_difference(target) / self.difference_edge_intensity_scale
        return F.l1_loss(self._sobel_edges(pred_diff), self._sobel_edges(target_diff))

    def _scheduled_lambda(self, base_lambda: float, start_step: int, warmup_steps: int, global_step: int | None) -> tuple[float, float]:
        if base_lambda <= 0.0:
            return 0.0, 0.0
        if global_step is None:
            return base_lambda, 1.0
        if global_step < start_step:
            return 0.0, 0.0
        if warmup_steps <= 0:
            return base_lambda, 1.0
        progress = min(max((global_step - start_step + 1) / max(warmup_steps, 1), 0.0), 1.0)
        return base_lambda * progress, progress

    def _set_noise_schedule(self) -> None:
        device = next(self.denoise_fn.parameters()).device
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1)))
        self.register_buffer("sqrt_gammas", to_torch(np.sqrt(gammas)))
        self.register_buffer("sqrt_one_minus_gammas", to_torch(np.sqrt(1.0 - gammas)))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)))

    def q_sample(self, y_0: torch.Tensor, sample_gammas: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def _training_objective(self, y_0: torch.Tensor, sample_gammas: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        gamma = sample_gammas.view(-1, 1, 1, 1)
        return gamma.sqrt() * noise - (1.0 - gamma).sqrt() * y_0

    def _min_snr_weights(self, sample_gammas: torch.Tensor) -> torch.Tensor:
        if not self.min_snr_enabled:
            return torch.ones((sample_gammas.shape[0],), device=sample_gammas.device, dtype=sample_gammas.dtype)
        snr = sample_gammas.view(-1) / (1.0 - sample_gammas.view(-1)).clamp_min(1e-8)
        clipped = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
        if self.prediction_type == "v_prediction":
            return clipped / (snr + 1.0).clamp_min(1e-8)
        return clipped / snr.clamp_min(1e-8)

    def predict_start_from_output_gamma(self, y_t: torch.Tensor, sample_gammas: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        gamma = sample_gammas.view(-1, 1, 1, 1)
        if self.prediction_type == "epsilon":
            return (y_t - (1.0 - gamma).sqrt() * model_output) / gamma.sqrt().clamp_min(1e-8)
        return gamma.sqrt() * y_t - (1.0 - gamma).sqrt() * model_output

    def predict_noise_from_output_gamma(self, y_t: torch.Tensor, sample_gammas: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        gamma = sample_gammas.view(-1, 1, 1, 1)
        if self.prediction_type == "epsilon":
            return model_output
        return (1.0 - gamma).sqrt() * y_t + gamma.sqrt() * model_output

    def predict_start_from_noise(self, y_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise

    def predict_start_from_model_output(self, y_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device).view(-1)
        return self.predict_start_from_output_gamma(y_t, gammas, model_output)

    def predict_noise_from_model_output(self, y_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device).view(-1)
        return self.predict_noise_from_output_gamma(y_t, gammas, model_output)

    def predict_noise_from_start(self, y_t: torch.Tensor, t: torch.Tensor, y_0: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - y_0) / extract(self.sqrt_recipm1_gammas, t, y_t.shape)

    def q_posterior(self, y_0_hat: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance

    def p_mean_variance(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor, side_labels: torch.Tensor | None = None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        model_output = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, side=side_labels)
        y_0_hat = self.predict_start_from_model_output(y_t, t=t, model_output=model_output)
        return self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)

    @torch.no_grad()
    def p_sample(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor, side_labels: torch.Tensor | None = None) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(y_t=y_t, t=t, y_cond=y_cond, side_labels=side_labels)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        sample_steps: int | None = None,
        side_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = sample_steps or self.num_timesteps
        y_t = torch.randn(condition.shape[0], self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        for i in reversed(range(steps)):
            t = torch.full((condition.shape[0],), i, device=condition.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=condition, side_labels=side_labels)
        return self.decode_target(y_t, condition)

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: torch.Tensor,
        sample_steps: int = 50,
        eta: float = 0.0,
        side_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sample_ddim_trace(condition, sample_steps=sample_steps, eta=eta, side_labels=side_labels)["final"]

    def _alpha_sigma_lambda(self, t: torch.Tensor, x_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = extract(self.sqrt_gammas, t, x_shape).clamp_min(1e-8)
        sigma = extract(self.sqrt_one_minus_gammas, t, x_shape).clamp_min(1e-8)
        return alpha, sigma, torch.log(alpha) - torch.log(sigma)

    def _dpm_solver_pp_first_update(
        self,
        y_t: torch.Tensor,
        y_0_hat: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, sigma_t, lambda_t = self._alpha_sigma_lambda(t, y_t.shape)
        alpha_prev, sigma_prev, lambda_prev = self._alpha_sigma_lambda(t_prev, y_t.shape)
        h = lambda_prev - lambda_t
        y_prev = (sigma_prev / sigma_t) * y_t - alpha_prev * (torch.exp(-h) - 1.0) * y_0_hat
        return y_prev, lambda_t, h

    @torch.no_grad()
    def sample_dpm_solver_pp(
        self,
        condition: torch.Tensor,
        sample_steps: int = 30,
        solver_order: int = 2,
        side_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        if solver_order not in {1, 2}:
            raise ValueError("solver_order must be 1 or 2")
        batch = condition.shape[0]
        y_t = torch.randn(batch, self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        time_pairs = torch.linspace(self.num_timesteps - 1, 0, sample_steps, device=condition.device).long()
        prev_pairs = torch.cat([time_pairs[1:], torch.tensor([-1], device=condition.device, dtype=torch.long)])
        last_y_0_hat: torch.Tensor | None = None
        last_lambda: torch.Tensor | None = None
        for t_now, t_prev in zip(time_pairs, prev_pairs):
            t = torch.full((batch,), int(t_now.item()), device=condition.device, dtype=torch.long)
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            model_output = self.denoise_fn(torch.cat([condition, y_t], dim=1), noise_level, side=side_labels)
            y_0_hat = self.predict_start_from_model_output(y_t, t=t, model_output=model_output)
            if int(t_prev.item()) < 0:
                y_t = y_0_hat
                break
            t_prev_tensor = torch.full((batch,), int(t_prev.item()), device=condition.device, dtype=torch.long)
            y_prev, lambda_now, h = self._dpm_solver_pp_first_update(y_t, y_0_hat, t, t_prev_tensor)
            if solver_order == 2 and last_y_0_hat is not None and last_lambda is not None:
                alpha_prev, _, _ = self._alpha_sigma_lambda(t_prev_tensor, y_t.shape)
                r0 = ((lambda_now - last_lambda) / h.clamp_min(1e-8)).clamp_min(1e-8)
                d1 = (y_0_hat - last_y_0_hat) / r0
                y_prev = y_prev - 0.5 * alpha_prev * (torch.exp(-h) - 1.0) * d1
            last_y_0_hat = y_0_hat
            last_lambda = lambda_now
            y_t = y_prev
        return self.decode_target(y_t, condition)

    @torch.no_grad()
    def sample_ddim_trace(
        self,
        condition: torch.Tensor,
        sample_steps: int = 50,
        eta: float = 0.0,
        side_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | list[int]]:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        batch = condition.shape[0]
        y_t = torch.randn(batch, self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        time_pairs = torch.linspace(self.num_timesteps - 1, 0, sample_steps, device=condition.device).long()
        prev_pairs = torch.cat([time_pairs[1:], torch.tensor([-1], device=condition.device, dtype=torch.long)])
        samples: list[torch.Tensor] = [self.decode_target(y_t, condition).detach().cpu().clone()]
        predictions: list[torch.Tensor] = []
        timesteps: list[int] = [int(time_pairs[0].item())]
        for t_now, t_prev in zip(time_pairs, prev_pairs):
            t = torch.full((batch,), int(t_now.item()), device=condition.device, dtype=torch.long)
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            model_output = self.denoise_fn(torch.cat([condition, y_t], dim=1), noise_level, side=side_labels)
            y_0_hat = self.predict_start_from_model_output(y_t, t=t, model_output=model_output)
            predictions.append(self.decode_target(y_0_hat, condition).detach().cpu().clone())
            if int(t_prev.item()) < 0:
                y_t = y_0_hat
                samples.append(self.decode_target(y_t, condition).detach().cpu().clone())
                timesteps.append(-1)
                continue
            t_prev_tensor = torch.full((batch,), int(t_prev.item()), device=condition.device, dtype=torch.long)
            alpha_t = extract(self.gammas, t, y_t.shape)
            alpha_prev = extract(self.gammas, t_prev_tensor, y_t.shape)
            pred_noise = self.predict_noise_from_model_output(y_t, t=t, model_output=model_output)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(y_t) if eta > 0 else torch.zeros_like(y_t)
            direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * pred_noise
            y_t = torch.sqrt(alpha_prev) * y_0_hat + direction + sigma * noise
            samples.append(self.decode_target(y_t, condition).detach().cpu().clone())
            timesteps.append(int(t_prev.item()))
        return {
            "final": self.decode_target(y_t, condition),
            "samples": samples,
            "predictions": predictions,
            "timesteps": timesteps,
        }

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        side_labels: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> torch.Tensor:
        batch, *_ = target.shape
        target_model_space = self.encode_target(target, condition)
        t = torch.randint(1, self.num_timesteps, (batch,), device=target.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((batch, 1), device=target.device) + gamma_t1
        noise = torch.randn_like(target_model_space)
        y_noisy = self.q_sample(target_model_space, sample_gammas.view(-1, 1, 1, 1), noise)
        model_output = self.denoise_fn(torch.cat([condition, y_noisy], dim=1), sample_gammas.view(-1), side=side_labels)
        objective_target = self._training_objective(target_model_space, sample_gammas, noise)
        sample_weights = self._min_snr_weights(sample_gammas)
        diff_loss, diff_breakdown = self._compute_training_loss(model_output, objective_target, sample_weights)
        y_0_hat = self.predict_start_from_output_gamma(y_noisy, sample_gammas.view(-1), model_output)
        y_0_hat_for_losses = y_0_hat.clamp(-1.0, 1.0) if self.x0_reconstruction_clamp else y_0_hat
        decoded_y_0_hat: torch.Tensor | None = None
        consistency_loss = torch.zeros((), device=target.device, dtype=diff_loss.dtype)
        consistency_stats: dict[str, torch.Tensor] = {}
        consistency_lambda, consistency_warmup = self._scheduled_lambda(
            self.consistency_lambda,
            self.consistency_start_step,
            self.consistency_warmup_steps,
            global_step,
        )
        consistency_active = self.consistency_enabled and consistency_lambda > 0.0
        if consistency_active:
            decoded_y_0_hat = self.decode_target(y_0_hat_for_losses, condition)
            consistency_loss, consistency_stats = self._compute_consistency_loss(condition, sample_gammas, decoded_y_0_hat)
        raw_consistency_term = consistency_loss * consistency_lambda
        consistency_term = raw_consistency_term
        consistency_clip_scale = torch.ones_like(raw_consistency_term)
        consistency_term_cap = raw_consistency_term.detach()
        if self.consistency_max_ratio is not None:
            consistency_term_cap = diff_loss.detach().clamp_min(0.0) * self.consistency_max_ratio
            consistency_clip_scale = torch.clamp(
                consistency_term_cap / raw_consistency_term.detach().clamp_min(self.consistency_eps),
                max=1.0,
            )
            consistency_term = raw_consistency_term * consistency_clip_scale
        x0_reconstruction_loss = torch.zeros((), device=target.device, dtype=diff_loss.dtype)
        x0_breakdown: dict[str, torch.Tensor] = {}
        x0_reconstruction_active = self.x0_reconstruction_enabled and (
            global_step is None or global_step >= self.x0_reconstruction_start_step
        )
        if x0_reconstruction_active:
            x0_reconstruction_loss, x0_breakdown = self._compute_x0_reconstruction_loss(y_0_hat_for_losses, target_model_space)
        raw_x0_reconstruction_term = self.x0_reconstruction_lambda * x0_reconstruction_loss
        x0_reconstruction_term = raw_x0_reconstruction_term
        x0_reconstruction_clip_scale = torch.ones_like(raw_x0_reconstruction_term)
        x0_reconstruction_term_cap = raw_x0_reconstruction_term.detach()
        if self.x0_reconstruction_max_ratio is not None:
            x0_reconstruction_term_cap = diff_loss.detach().clamp_min(0.0) * self.x0_reconstruction_max_ratio
            x0_reconstruction_clip_scale = torch.clamp(
                x0_reconstruction_term_cap / raw_x0_reconstruction_term.detach().clamp_min(self.consistency_eps),
                max=1.0,
            )
            x0_reconstruction_term = raw_x0_reconstruction_term * x0_reconstruction_clip_scale
        difference_loss = torch.zeros((), device=target.device, dtype=diff_loss.dtype)
        if self.difference_enabled:
            if decoded_y_0_hat is None:
                decoded_y_0_hat = self.decode_target(y_0_hat_for_losses, condition)
            difference_loss = self._compute_difference_loss(decoded_y_0_hat, target)
        difference_term = self.difference_lambda * difference_loss

        difference_edge_loss = torch.zeros((), device=target.device, dtype=diff_loss.dtype)
        if self.difference_edge_enabled:
            if decoded_y_0_hat is None:
                decoded_y_0_hat = self.decode_target(y_0_hat_for_losses, condition)
            difference_edge_loss = self._compute_difference_edge_loss(decoded_y_0_hat, target)
        difference_edge_term = self.difference_edge_lambda * difference_edge_loss

        total_loss = diff_loss + consistency_term + x0_reconstruction_term + difference_term + difference_edge_term
        diff_loss_value = float(diff_loss.detach().item())
        raw_consistency_term_value = float(raw_consistency_term.detach().item())
        consistency_term_value = float(consistency_term.detach().item())
        consistency_clip_scale_value = float(consistency_clip_scale.detach().item())
        raw_x0_reconstruction_term_value = float(raw_x0_reconstruction_term.detach().item())
        x0_reconstruction_term_value = float(x0_reconstruction_term.detach().item())
        x0_reconstruction_clip_scale_value = float(x0_reconstruction_clip_scale.detach().item())
        difference_term_value = float(difference_term.detach().item())
        difference_edge_term_value = float(difference_edge_term.detach().item())
        first_name, second_name = self._target_channel_names()
        self.last_loss_breakdown = {
            "diff_loss": diff_loss_value,
            f"{first_name}_diff_loss": float(diff_breakdown[f"{first_name}_diff_loss"].detach().item()) if f"{first_name}_diff_loss" in diff_breakdown else diff_loss_value,
            f"{second_name}_diff_loss": float(diff_breakdown[f"{second_name}_diff_loss"].detach().item()) if f"{second_name}_diff_loss" in diff_breakdown else diff_loss_value,
            "min_snr_active": 1.0 if self.min_snr_enabled else 0.0,
            "min_snr_weight_mean": float(sample_weights.detach().mean().item()),
            "min_snr_weight_min": float(sample_weights.detach().min().item()),
            "min_snr_weight_max": float(sample_weights.detach().max().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "consistency_term_raw": raw_consistency_term_value,
            "consistency_term": consistency_term_value,
            "consistency_term_cap": float(consistency_term_cap.detach().item()),
            "consistency_max_ratio": float(self.consistency_max_ratio) if self.consistency_max_ratio is not None else -1.0,
            "consistency_clip_scale": consistency_clip_scale_value,
            "consistency_capped": 1.0 if consistency_clip_scale_value < 0.999999 else 0.0,
            "consistency_ratio_raw": raw_consistency_term_value / max(diff_loss_value, 1e-8),
            "consistency_ratio": consistency_term_value / max(diff_loss_value, 1e-8),
            "consistency_active": 1.0 if consistency_active else 0.0,
            "consistency_warmup": float(consistency_warmup),
            "consistency_active_ratio": float(consistency_stats.get("consistency_active_ratio", torch.zeros((), device=target.device)).detach().item()),
            "consistency_mask_ratio": float(consistency_stats.get("consistency_mask_ratio", torch.zeros((), device=target.device)).detach().item()),
            "x0_reconstruction_loss": float(x0_reconstruction_loss.detach().item()),
            f"{first_name}_x0_loss": float(x0_breakdown[f"{first_name}_x0_loss"].detach().item()) if f"{first_name}_x0_loss" in x0_breakdown else float(x0_reconstruction_loss.detach().item()),
            f"{second_name}_x0_loss": float(x0_breakdown[f"{second_name}_x0_loss"].detach().item()) if f"{second_name}_x0_loss" in x0_breakdown else float(x0_reconstruction_loss.detach().item()),
            "x0_reconstruction_term_raw": raw_x0_reconstruction_term_value,
            "x0_reconstruction_term": x0_reconstruction_term_value,
            "x0_reconstruction_term_cap": float(x0_reconstruction_term_cap.detach().item()),
            "x0_reconstruction_max_ratio": float(self.x0_reconstruction_max_ratio) if self.x0_reconstruction_max_ratio is not None else -1.0,
            "x0_reconstruction_clip_scale": x0_reconstruction_clip_scale_value,
            "x0_reconstruction_capped": 1.0 if x0_reconstruction_clip_scale_value < 0.999999 else 0.0,
            "x0_reconstruction_ratio_raw": raw_x0_reconstruction_term_value / max(diff_loss_value, 1e-8),
            "x0_reconstruction_ratio": x0_reconstruction_term_value / max(diff_loss_value, 1e-8),
            "x0_reconstruction_active": 1.0 if x0_reconstruction_active else 0.0,
            "difference_loss": float(difference_loss.detach().item()),
            "difference_term": difference_term_value,
            "difference_ratio": difference_term_value / max(diff_loss_value, 1e-8),
            "difference_active": 1.0 if self.difference_enabled else 0.0,
            "difference_edge_loss": float(difference_edge_loss.detach().item()),
            "difference_edge_term": difference_edge_term_value,
            "difference_edge_ratio": difference_edge_term_value / max(diff_loss_value, 1e-8),
            "difference_edge_active": 1.0 if self.difference_edge_enabled else 0.0,
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss
