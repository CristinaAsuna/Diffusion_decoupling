from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _denormalize_array(array: np.ndarray, data_norm_cfg: dict | None = None) -> np.ndarray:
    cfg = data_norm_cfg or {}
    mode = str(cfg.get("normalize", "none"))
    value_range = cfg.get("value_range")
    if mode == "fixed_range_m11":
        if value_range is None:
            return array
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return ((array + 1.0) * 0.5) * (max_val - min_val) + min_val
    if mode == "fixed_range_01":
        if value_range is None:
            return array
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return array * (max_val - min_val) + min_val
    return array












def _single_channel(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3:
        return array[0].astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D or CHW array, got shape {array.shape}")


def _branch_arrays(
    full: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor | None = None,
    data_norm_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    full_np = _single_channel(_denormalize_array(_to_numpy(full), data_norm_cfg))
    pred_np = _denormalize_array(_to_numpy(pred), data_norm_cfg)
    if pred_np.ndim != 3 or pred_np.shape[0] < 2:
        raise ValueError(f"Branch prediction expects at least 2 channels, got shape {pred_np.shape}")
    left_np = pred_np[0].astype(np.float32, copy=False)
    right_np = pred_np[1].astype(np.float32, copy=False)
    target_left = None
    target_right = None
    if target is not None:
        target_np = _denormalize_array(_to_numpy(target), data_norm_cfg)
        if target_np.ndim == 3 and target_np.shape[0] >= 2:
            target_left = target_np[0].astype(np.float32, copy=False)
            target_right = target_np[1].astype(np.float32, copy=False)
    return full_np, left_np, right_np, target_left, target_right


def _sobel_np(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(array.astype(np.float32, copy=False), ((1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    ) / 4.0
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 4.0
    return gx, gy


def compute_branch_physics_metrics(
    full: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor | None = None,
    data_norm_cfg: dict | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    full_np, left_np, right_np, target_left, target_right = _branch_arrays(full, pred, target, data_norm_cfg)
    pred_sum = left_np + right_np
    residual = full_np - pred_sum
    abs_residual = np.abs(residual)
    mask = full_np > np.percentile(full_np, 20)
    if not np.any(mask):
        mask = np.ones_like(full_np, dtype=bool)

    metrics = {
        "physics_residual_mae": float(abs_residual.mean()),
        "physics_residual_rmse": float(np.sqrt(np.mean(residual * residual))),
        "physics_residual_rel_mae": float(np.mean(abs_residual / (np.abs(full_np) + eps))),
        "physics_residual_rel_mae_masked": float(np.mean(abs_residual[mask] / (np.abs(full_np[mask]) + eps))),
    }
    if np.std(pred_sum[mask]) > eps and np.std(full_np[mask]) > eps:
        metrics["physics_corr_masked"] = float(np.corrcoef(pred_sum[mask].ravel(), full_np[mask].ravel())[0, 1])
    else:
        metrics["physics_corr_masked"] = 0.0

    if target_left is not None and target_right is not None:
        left_error = left_np - target_left
        right_error = right_np - target_right
        pred_diff = left_np - right_np
        target_diff = target_left - target_right
        diff_error = pred_diff - target_diff
        pred_edge_x, pred_edge_y = _sobel_np(pred_diff)
        target_edge_x, target_edge_y = _sobel_np(target_diff)
        metrics.update(
            {
                "left_loss": float(np.mean(left_error * left_error)),
                "right_loss": float(np.mean(right_error * right_error)),
                "left_mae": float(np.mean(np.abs(left_error))),
                "right_mae": float(np.mean(np.abs(right_error))),
                "difference_mae": float(np.mean(np.abs(diff_error))),
                "difference_rmse": float(np.sqrt(np.mean(diff_error * diff_error))),
                "difference_edge_mae": float(
                    0.5 * (np.mean(np.abs(pred_edge_x - target_edge_x)) + np.mean(np.abs(pred_edge_y - target_edge_y)))
                ),
            }
        )
    return metrics


def save_physics_visualization(
    full: torch.Tensor,
    pred: torch.Tensor,
    save_path: str | Path,
    title: str = "",
    data_norm_cfg: dict | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_np, left_np, right_np, _, _ = _branch_arrays(full, pred, data_norm_cfg=data_norm_cfg)
    pred_sum = left_np + right_np
    residual = full_np - pred_sum

    mask = full_np > np.percentile(full_np, 20)
    if not np.any(mask):
        mask = np.ones_like(full_np, dtype=bool)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = float(np.percentile(full_np, 1))
    vmax = float(np.percentile(full_np, 99))
    if abs(vmax - vmin) < 1e-8:
        vmin = float(full_np.min())
        vmax = float(full_np.max() + 1e-8)

    axes[0, 0].imshow(full_np, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Full")
    axes[0, 1].imshow(left_np, cmap="gray")
    axes[0, 1].set_title("Pred Left")
    axes[0, 2].imshow(right_np, cmap="gray")
    axes[0, 2].set_title("Pred Right")
    axes[1, 0].imshow(pred_sum, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Pred Left + Right")

    res_abs_max = float(np.percentile(np.abs(residual), 99))
    if res_abs_max < 1e-8:
        res_abs_max = 1e-8
    axes[1, 1].imshow(residual, cmap="bwr", vmin=-res_abs_max, vmax=res_abs_max)
    axes[1, 1].set_title("Residual: Full - (Pred Left + Pred Right)")

    y = full_np[mask].ravel()
    x = pred_sum[mask].ravel()
    if len(x) > 20000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), 20000, replace=False)
        x = x[idx]
        y = y[idx]
    axes[1, 2].scatter(x, y, s=1, alpha=0.25)
    if len(x) > 0 and len(y) > 0:
        min_v = float(min(x.min(), y.min()))
        max_v = float(max(x.max(), y.max()))
        axes[1, 2].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    axes[1, 2].set_xlabel("Pred Left + Right")
    axes[1, 2].set_ylabel("Full")
    axes[1, 2].set_title("Scatter")

    for ax in axes.ravel():
        if ax is not axes[1, 2]:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
