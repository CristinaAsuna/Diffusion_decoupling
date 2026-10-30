from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .utils import save_tensor_png
except ImportError:
    from utils import save_tensor_png


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse_value = mse(pred, target)
    if mse_value <= 1e-12:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse_value))


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pred = pred.float()
    target = target.float()
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return float(score.mean().item())


def compute_fid_for_directories(pred_dir: str | Path, gt_dir: str | Path) -> float:
    from cleanfid import fid

    return float(fid.compute_fid(str(pred_dir), str(gt_dir)))


def compute_channelwise_fid(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    scores: list[float] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for channel_idx in range(pred.shape[1]):
            pred_dir = tmp_root / f"pred_{channel_idx}"
            gt_dir = tmp_root / f"gt_{channel_idx}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            for batch_idx in range(pred.shape[0]):
                save_tensor_png(pred[batch_idx, channel_idx : channel_idx + 1], pred_dir / f"{batch_idx:05d}.png")
                save_tensor_png(target[batch_idx, channel_idx : channel_idx + 1], gt_dir / f"{batch_idx:05d}.png")
            scores.append(compute_fid_for_directories(pred_dir, gt_dir))
    return scores
