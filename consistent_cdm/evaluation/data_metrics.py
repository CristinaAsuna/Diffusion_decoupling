"""Raw D and branch image metrics, including the Full/2 reference."""

from __future__ import annotations

import math

from typing import Any, Mapping

import numpy as np

from scipy.ndimage import uniform_filter

def _coerce_same_arrays(pred: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred_array = np.asarray(pred, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred_array.shape} vs {target_array.shape}"
        )
    if pred_array.ndim < 2:
        raise ValueError("Image arrays must have at least two dimensions")
    if not np.all(np.isfinite(pred_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("Metric inputs must contain only finite values")
    return pred_array, target_array

def _ssim_2d(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """Dependency-light local SSIM using a 7x7 uniform window."""

    height, width = pred.shape
    window = min(7, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        # Well-defined global fallback for tiny test/edge patches.
        mu_x, mu_y = float(np.mean(pred)), float(np.mean(target))
        var_x, var_y = float(np.var(pred)), float(np.var(target))
        covariance = float(np.mean((pred - mu_x) * (target - mu_y)))
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        return float(
            ((2 * mu_x * mu_y + c1) * (2 * covariance + c2))
            / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
        )

    ux = uniform_filter(pred, size=window, mode="reflect")
    uy = uniform_filter(target, size=window, mode="reflect")
    correction = window * window / (window * window - 1)
    vx = correction * (uniform_filter(pred * pred, window, mode="reflect") - ux * ux)
    vy = correction * (uniform_filter(target * target, window, mode="reflect") - uy * uy)
    vxy = correction * (
        uniform_filter(pred * target, window, mode="reflect") - ux * uy
    )
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    score = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=np.abs(denominator) > np.finfo(np.float64).eps,
    )
    return float(np.mean(score))

def structural_similarity(pred: Any, target: Any, *, data_range: float = 6.0) -> float:
    """Return mean SSIM over all leading dimensions."""

    pred_array, target_array = _coerce_same_arrays(pred, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    batches_pred = pred_array.reshape((-1,) + pred_array.shape[-2:])
    batches_target = target_array.reshape((-1,) + target_array.shape[-2:])
    return float(
        np.mean(
            [
                _ssim_2d(pred_item, target_item, data_range)
                for pred_item, target_item in zip(batches_pred, batches_target)
            ]
        )
    )

def gradient_mae(pred: Any, target: Any) -> float:
    """Mean absolute error of row and column finite-difference gradients."""

    pred_array, target_array = _coerce_same_arrays(pred, target)
    if min(pred_array.shape[-2:]) < 2:
        return 0.0
    pred_grad = np.gradient(pred_array, axis=(-2, -1))
    target_grad = np.gradient(target_array, axis=(-2, -1))
    return float(
        0.5
        * (
            np.mean(np.abs(pred_grad[0] - target_grad[0]))
            + np.mean(np.abs(pred_grad[1] - target_grad[1]))
        )
    )

def image_metrics(pred: Any, target: Any, *, data_range: float = 6.0) -> dict[str, float]:
    """Compute fixed-range MAE, RMSE, PSNR, SSIM, and gradient MAE."""

    pred_array, target_array = _coerce_same_arrays(pred, target)
    residual = pred_array - target_array
    mae_value = float(np.mean(np.abs(residual)))
    mse_value = float(np.mean(residual * residual))
    rmse_value = math.sqrt(mse_value)
    psnr_value = math.inf if mse_value == 0 else 20 * math.log10(data_range / rmse_value)
    return {
        "mae": mae_value,
        "rmse": rmse_value,
        "psnr": float(psnr_value),
        "ssim": structural_similarity(pred_array, target_array, data_range=data_range),
        "gradient_mae": gradient_mae(pred_array, target_array),
    }

def asymmetry_metrics(
    pred_difference: Any,
    target_difference: Any,
    *,
    mask: Any | None = None,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Measure signed left/right asymmetry without common anatomy dominating.

    Inputs are ``D=(left-right)/2`` (or the unscaled left-right difference;
    every normalized quantity below is invariant to that common scale).  The
    identity score is +1 for exact direct pairing, 0 for a symmetric collapse,
    and -1 for an exact left/right swap.
    """

    pred, target = _coerce_same_arrays(pred_difference, target_difference)
    if mask is None:
        keep = np.ones(pred.shape, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != pred.shape:
            raise ValueError(f"Asymmetry mask shape {keep.shape} != image shape {pred.shape}")
    if not np.any(keep):
        return {
            "mae": math.nan,
            "nmae": math.nan,
            "identity": math.nan,
            "signed_gain": math.nan,
            "energy_ratio": math.nan,
            "gradient_mae": math.nan,
            "count": 0.0,
        }

    pred_values = pred[keep]
    target_values = target[keep]
    direct = float(np.mean(np.abs(pred_values - target_values)))
    swapped = float(np.mean(np.abs(pred_values + target_values)))
    target_l1 = float(np.mean(np.abs(target_values)))
    target_l2 = float(np.mean(target_values * target_values))
    pred_l2 = float(np.mean(pred_values * pred_values))
    return {
        "mae": direct,
        "nmae": float(direct / max(target_l1, epsilon)),
        "identity": float((swapped - direct) / max(swapped + direct, epsilon)),
        "signed_gain": float(np.mean(pred_values * target_values) / max(target_l2, epsilon)),
        "energy_ratio": float(math.sqrt(pred_l2 / max(target_l2, epsilon))),
        "gradient_mae": gradient_mae(pred, target) if mask is None else math.nan,
        "count": float(np.count_nonzero(keep)),
    }

def asymmetry_rich_mask(target_difference: Any, *, quantile: float = 0.75) -> np.ndarray:
    """Return the GT-|D| top-quantile mask for one image.

    The threshold is defined per image so a large patient/view cannot dominate
    the cohort.  Degenerate all-zero differences intentionally select no
    pixels: identity is undefined when no side-specific signal exists.
    """

    target = np.asarray(target_difference, dtype=np.float64)
    if target.ndim < 2 or not np.all(np.isfinite(target)):
        raise ValueError("target_difference must be a finite image")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must lie in [0, 1)")
    magnitude = np.abs(target)
    nonzero = magnitude[magnitude > 0]
    if not len(nonzero):
        return np.zeros_like(target, dtype=bool)
    threshold = float(np.quantile(nonzero, quantile))
    return magnitude >= threshold

def _mean_metric_dict(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, float]:
    return {name: float(0.5 * (first[name] + second[name])) for name in first}

def branch_metrics(
    pred_left: Any,
    pred_right: Any,
    gt_left: Any,
    gt_right: Any,
    *,
    full: Any | None = None,
    data_range: float = 6.0,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Score a two-branch prediction, including identity/swap and local errors."""

    pred_l, gt_l = _coerce_same_arrays(pred_left, gt_left)
    pred_r, gt_r = _coerce_same_arrays(pred_right, gt_right)
    if pred_l.shape != pred_r.shape or gt_l.shape != gt_r.shape:
        raise ValueError("All four branch images must have the same shape")

    left_scores = image_metrics(pred_l, gt_l, data_range=data_range)
    right_scores = image_metrics(pred_r, gt_r, data_range=data_range)
    gt_sum = gt_l + gt_r
    target_full = gt_sum if full is None else np.asarray(full, dtype=np.float64)
    if target_full.shape != gt_sum.shape:
        raise ValueError(f"Full image shape {target_full.shape} != branch shape {gt_sum.shape}")
    sum_scores = image_metrics(pred_l + pred_r, target_full, data_range=data_range)
    sum_scores["nmae"] = float(
        np.mean(np.abs(pred_l + pred_r - target_full))
        / max(float(np.mean(np.abs(target_full))), epsilon)
    )
    # Keep the historical left-right image metrics, and add D-space metrics
    # whose normalization makes full/2 collapse and a swap interpretable.
    pred_difference = (pred_l - pred_r) / 2.0
    gt_difference = (gt_l - gt_r) / 2.0
    difference_scores = image_metrics(pred_difference, gt_difference, data_range=data_range)
    asymmetry_scores = asymmetry_metrics(pred_difference, gt_difference, epsilon=epsilon)
    asymmetry_rich_scores = asymmetry_metrics(
        pred_difference,
        gt_difference,
        mask=asymmetry_rich_mask(gt_difference),
        epsilon=epsilon,
    )

    correct_error = 0.5 * (left_scores["mae"] + right_scores["mae"])
    swapped_left = image_metrics(pred_l, gt_r, data_range=data_range)
    swapped_right = image_metrics(pred_r, gt_l, data_range=data_range)
    swapped_error = 0.5 * (swapped_left["mae"] + swapped_right["mae"])
    swap = {
        "correct_mae": float(correct_error),
        "swapped_mae": float(swapped_error),
        "margin": float(swapped_error - correct_error),
        "prefers_swapped": bool(swapped_error < correct_error),
        "swap_rate": float(swapped_error < correct_error),
    }

    result: dict[str, Any] = {
        "left": left_scores,
        "right": right_scores,
        "side_average": _mean_metric_dict(left_scores, right_scores),
        "sum": sum_scores,
        "difference": difference_scores,
        "asymmetry": asymmetry_scores,
        "asymmetry_rich": asymmetry_rich_scores,
        "swap": swap,
    }

    if full is not None:
        half = target_full / 2.0
        baseline_left = image_metrics(half, gt_l, data_range=data_range)
        baseline_right = image_metrics(half, gt_r, data_range=data_range)
        result["full_half_baseline"] = {
            "left": baseline_left,
            "right": baseline_right,
            "side_average": _mean_metric_dict(baseline_left, baseline_right),
            "sum_nmae": float(
                np.mean(np.abs(2 * half - target_full))
                / max(float(np.mean(np.abs(target_full))), epsilon)
            ),
            "asymmetry": asymmetry_metrics(
                np.zeros_like(gt_difference), gt_difference, epsilon=epsilon
            ),
            "asymmetry_rich": asymmetry_metrics(
                np.zeros_like(gt_difference),
                gt_difference,
                mask=asymmetry_rich_mask(gt_difference),
                epsilon=epsilon,
            ),
        }

    return result


def flatten_metrics(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dictionaries using ``.``-separated keys."""

    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(flatten_metrics(item, name))
        else:
            output[name] = item
    return output
