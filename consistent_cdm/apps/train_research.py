
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from consistent_cdm.data.dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from consistent_cdm.baseline.d_only import DNormalization, DOnlyDataset, decode_standardized_d, hard_decoded_branch_metrics
from consistent_cdm.model.flow_matching import ConditionalFlowMatching
from consistent_cdm.contracts import TargetContract
from consistent_cdm.factories import generative_run_group
from consistent_cdm.trainers import prepare_generative_training
from consistent_cdm.utils.checkpointing import load_checkpoint_file, require_training_checkpoint
from consistent_cdm.utils.config import ensure_dir, load_config
from consistent_cdm.utils.metrics import compute_channelwise_fid, mae, mse, psnr, ssim
from consistent_cdm.utils.paths import apply_run_layout
from consistent_cdm.utils.utils import pick_device, save_tensor_npy, save_tensor_png, set_seed, tensor_to_uint8_image
from consistent_cdm.visual.physics import compute_branch_physics_metrics, safe_filename, save_physics_visualization


LOSS_BREAKDOWN_KEYS = (
    "flow_matching_loss",
    "diff_loss",
    "left_diff_loss",
    "right_diff_loss",
    "r_diff_loss",
    "d_diff_loss",
    "min_snr_active",
    "min_snr_weight_mean",
    "min_snr_weight_min",
    "min_snr_weight_max",
    "consistency_loss",
    "consistency_term_raw",
    "consistency_term",
    "consistency_term_cap",
    "consistency_max_ratio",
    "consistency_clip_scale",
    "consistency_capped",
    "consistency_ratio_raw",
    "consistency_ratio",
    "consistency_active",
    "consistency_warmup",
    "consistency_active_ratio",
    "consistency_mask_ratio",
    "x0_reconstruction_loss",
    "left_x0_loss",
    "right_x0_loss",
    "r_x0_loss",
    "d_x0_loss",
    "x0_reconstruction_term_raw",
    "x0_reconstruction_term",
    "x0_reconstruction_term_cap",
    "x0_reconstruction_max_ratio",
    "x0_reconstruction_clip_scale",
    "x0_reconstruction_capped",
    "x0_reconstruction_ratio_raw",
    "x0_reconstruction_ratio",
    "x0_reconstruction_active",
    "difference_loss",
    "difference_term",
    "difference_ratio",
    "difference_active",
    "difference_edge_loss",
    "difference_edge_term",
    "difference_edge_ratio",
    "difference_edge_active",
    "total_loss",
)


def build_dataset(dataset_cfg: dict):
    dataset_type = dataset_cfg.get("type", "paired_dirs")
    common_kwargs = {
        "image_size": dataset_cfg.get("image_size"),
        "normalize": dataset_cfg.get("normalize", "range_m11"),
        "value_range": dataset_cfg.get("value_range"),
        "clip_range": dataset_cfg.get("clip_range"),
        "target_mode": dataset_cfg.get("target_mode"),
        "target_side": dataset_cfg.get("target_side"),
        "target_sides": dataset_cfg.get("target_sides"),
        "side_labels": dataset_cfg.get("side_labels"),
    }
    if dataset_type == "paired_dirs":
        dataset = NpyConditionTargetDataset(
            condition_dir=dataset_cfg["condition_dir"],
            target_dirs=dataset_cfg["target_dirs"],
            names_file=dataset_cfg.get("names_file"),
            **common_kwargs,
        )
    elif dataset_type == "case_folders":
        dataset = CaseFolderNpyDataset(
            case_root=dataset_cfg["case_root"],
            condition_file=dataset_cfg.get("condition_file"),
            target_files=dataset_cfg.get("target_files"),
            case_names_file=dataset_cfg.get("case_names_file"),
            case_ids=dataset_cfg.get("case_ids"),
            include_patterns=dataset_cfg.get("include_patterns"),
            variants=dataset_cfg.get("variants"),
            condition_template=dataset_cfg.get("condition_template"),
            target_templates=dataset_cfg.get("target_templates"),
            target_template=dataset_cfg.get("target_template"),
            split=dataset_cfg.get("split"),
            split_seed=dataset_cfg.get("split_seed", 1234),
            train_ratio=dataset_cfg.get("train_ratio", 0.9),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    d_norm_cfg = dataset_cfg.get("d_only_normalization")
    if d_norm_cfg is None:
        return dataset
    # The scale is precomputed from the fixed training-patient protocol; this
    # wrapper never estimates statistics from validation data.
    normalization = DNormalization(
        mean=float(d_norm_cfg.get("mean", 0.0)), std=float(d_norm_cfg.get("std", 1.0)),
        scale=float(d_norm_cfg["scale"]), scale_method=str(d_norm_cfg.get("scale_method", "fixed")),
        scale_quantile=d_norm_cfg.get("scale_quantile"), scale_sample_count=int(d_norm_cfg.get("scale_sample_count", 0)),
        count=int(d_norm_cfg.get("count", 0)), raw_min=float(d_norm_cfg.get("raw_min", 0.0)), raw_max=float(d_norm_cfg.get("raw_max", 0.0)),
    )
    if normalization.scale <= 0.0:
        raise ValueError("d_only_normalization.scale must be positive")
    return DOnlyDataset(dataset, dataset_cfg, normalization)


def build_loader(
    dataset_cfg: dict,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    dataset = build_dataset(dataset_cfg)
    if shuffle and len(dataset) < batch_size * world_size:
        raise ValueError(f"Training dataset has {len(dataset)} samples for batch_size={batch_size}, world_size={world_size}; reduce batch_size")
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return dataset, loader, sampler


def assert_data_governance(dataset, governance: dict) -> None:
    """Fail closed before a development trainer can see frozen/non-Bone IDs."""
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("Data governance requires a case-folder dataset with sample metadata")
    case_ids = {str(sample.name).split("__", 1)[0] for sample in samples}
    forbidden = set(map(str, governance.get("forbidden_case_ids", [])))
    overlap = sorted(case_ids & forbidden)
    if overlap:
        raise ValueError("Frozen cases selected for training/evaluation: " + ", ".join(overlap))
    prefixes = tuple(map(str, governance.get("required_case_prefixes", [])))
    if prefixes:
        invalid = sorted(case_id for case_id in case_ids if not case_id.startswith(prefixes))
        if invalid:
            raise ValueError("Unexpected non-development case IDs: " + ", ".join(invalid[:10]))


def setup_distributed(device_cfg: str | None = None) -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        device = pick_device(device_cfg)
        return {
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": device,
            "is_main": True,
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend)
    return {
        "distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def maybe_init_swanlab(cfg: dict):
    if not cfg.get("use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed, training will continue without it.")
        return None
    run_id = os.environ.get("SWANLAB_RUN_ID", cfg.get("run_id"))
    resume = os.environ.get("SWANLAB_RESUME", cfg.get("resume"))
    mode = os.environ.get("SWANLAB_MODE", cfg.get("mode"))
    logdir = os.environ.get("SWANLAB_LOGDIR", cfg.get("logdir"))
    settings = {
        "project": cfg.get("project", "palette_decoupling"),
        "experiment_name": cfg.get("experiment_name"),
        "config": cfg,
    }
    if run_id:
        settings["id"] = run_id
    if resume is not None:
        settings["resume"] = resume
    if mode:
        settings["mode"] = mode
    if logdir:
        settings["logdir"] = logdir
    print(
        {
            "swanlab_project": settings.get("project"),
            "swanlab_run_id": settings.get("id"),
            "swanlab_resume": settings.get("resume"),
            "swanlab_mode": settings.get("mode"),
            "swanlab_logdir": settings.get("logdir"),
        }
    )
    return swanlab.init(**settings)


def maybe_init_tensorboard(cfg: dict, output_dir: Path):
    if not cfg.get("use_tensorboard", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard is not installed, training will continue without TensorBoard logging.")
        return None

    logdir = os.environ.get("TENSORBOARD_LOGDIR", cfg.get("tensorboard_logdir"))
    if logdir:
        logdir = ensure_dir(logdir)
    else:
        logdir = ensure_dir(output_dir / "tensorboard")

    print({"tensorboard_logdir": str(logdir)})
    return SummaryWriter(log_dir=str(logdir))


def log_monitoring_run(run, payload: dict, context: str):
    if run is None:
        return None
    try:
        run.log(payload)
        return run
    except Exception as exc:
        print({"warning": f"swanlab logging disabled after {context} failure", "error": repr(exc)})
        return None


def tensorboard_add_scalars(writer, namespace: str, values: dict, step: int, skip_keys: tuple[str, ...] = ("step",)):
    if writer is None:
        return
    for key, value in values.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            writer.add_scalar(f"{namespace}/{key}", int(value), step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(f"{namespace}/{key}", value, step)


def tensorboard_prepare_image(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    image = (image + 1.0) / 2.0
    return image


def tensorboard_add_preview_images(
    writer,
    step: int,
    condition: torch.Tensor,
    target: torch.Tensor | None,
    prediction: torch.Tensor,
    name: str,
):
    if writer is None:
        return
    writer.add_image(f"preview/{name}/condition", tensorboard_prepare_image(condition), step)
    if target is not None:
        for channel_idx in range(target.shape[0]):
            writer.add_image(
                f"preview/{name}/target_ch{channel_idx}",
                tensorboard_prepare_image(target[channel_idx : channel_idx + 1]),
                step,
            )
    for channel_idx in range(prediction.shape[0]):
        writer.add_image(
            f"preview/{name}/pred_ch{channel_idx}",
            tensorboard_prepare_image(prediction[channel_idx : channel_idx + 1]),
            step,
        )


def resolve_preview_channels(channels_cfg, num_channels: int) -> list[int]:
    if channels_cfg is None:
        return list(range(num_channels))
    channels = [int(channel_idx) for channel_idx in channels_cfg]
    return [channel_idx for channel_idx in channels if 0 <= channel_idx < num_channels]


def swanlab_log_preview_images(
    run,
    step: int,
    prediction: torch.Tensor,
    name: str,
    channels: list[int],
    labels: list[str] | None = None,
) -> bool:
    if run is None or not channels:
        return False
    try:
        import swanlab
        from PIL import Image

        payload = {"step": step}
        for channel_idx in channels:
            label = labels[channel_idx] if labels is not None and channel_idx < len(labels) else f"pred_ch{channel_idx}"
            image = Image.fromarray(tensor_to_uint8_image(prediction[channel_idx : channel_idx + 1]))
            payload[f"preview/{label}"] = swanlab.Image(
                image,
                caption=f"step {step} | {name} | {label}",
            )
        run.log(payload)
        return True
    except Exception as exc:
        print({"warning": "swanlab preview image logging disabled", "error": repr(exc)})
        return False


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: dict | None) -> None:
    if not rng_state:
        return
    python_state = rng_state.get("python")
    numpy_state = rng_state.get("numpy")
    torch_state = rng_state.get("torch")
    cuda_state = rng_state.get("cuda")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def capture_rng_state_by_rank(dist_ctx: dict) -> dict[str, dict]:
    local_state = capture_rng_state()
    if not dist_ctx["distributed"]:
        return {"0": local_state}

    gathered_states = [None for _ in range(dist_ctx["world_size"])]
    dist.all_gather_object(gathered_states, local_state)
    return {str(rank): state for rank, state in enumerate(gathered_states) if state is not None}


def restore_rng_state_for_rank(training_state: dict | None, dist_ctx: dict, is_main: bool) -> bool:
    if not training_state:
        return False

    rng_state_by_rank = training_state.get("rng_state_by_rank")
    if rng_state_by_rank is not None:
        saved_world_size = int(training_state.get("rng_world_size", len(rng_state_by_rank)))
        if saved_world_size != dist_ctx["world_size"]:
            if is_main:
                print(
                    {
                        "warning": "Skipping RNG restore because world size changed.",
                        "saved_world_size": saved_world_size,
                        "current_world_size": dist_ctx["world_size"],
                    }
                )
            return False

        rank_rng_state = rng_state_by_rank.get(str(dist_ctx["rank"]))
        if rank_rng_state is None:
            if is_main:
                print(
                    {
                        "warning": "Skipping RNG restore because this rank has no saved RNG state.",
                        "rank": dist_ctx["rank"],
                    }
                )
            return False

        restore_rng_state(rank_rng_state)
        return True

    legacy_rng_state = training_state.get("rng_state")
    if legacy_rng_state is not None and not dist_ctx["distributed"]:
        restore_rng_state(legacy_rng_state)
        return True

    if legacy_rng_state is not None and dist_ctx["distributed"] and is_main:
        print({"warning": "Skipping legacy single-state RNG restore for DDP checkpoint."})
    return False


def normalize_training_progress(train_epoch: int, batches_seen_in_epoch: int, batches_per_epoch: int) -> tuple[int, int]:
    if batches_per_epoch <= 0:
        return train_epoch, batches_seen_in_epoch
    normalized_epoch = train_epoch + batches_seen_in_epoch // batches_per_epoch
    normalized_batches = batches_seen_in_epoch % batches_per_epoch
    return normalized_epoch, normalized_batches


def restore_train_iterator(train_loader, train_sampler, train_epoch: int, batches_seen_in_epoch: int):
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)
    for _ in range(batches_seen_in_epoch):
        try:
            next(train_iter)
        except StopIteration:
            break
    return train_iter


def build_training_state(
    early_stop_best_score: float,
    early_stop_bad_validations: int,
    train_epoch: int,
    batches_seen_in_epoch: int,
    batches_per_epoch: int,
    recent_losses: list[float],
    dist_ctx: dict,
) -> dict:
    normalized_epoch, normalized_batches = normalize_training_progress(train_epoch, batches_seen_in_epoch, batches_per_epoch)
    rng_state_by_rank = capture_rng_state_by_rank(dist_ctx)
    training_state = {
        "early_stop_best_score": early_stop_best_score,
        "early_stop_bad_validations": early_stop_bad_validations,
        "train_epoch": normalized_epoch,
        "batches_seen_in_epoch": normalized_batches,
        "recent_losses": list(recent_losses),
        "rng_state_by_rank": rng_state_by_rank,
        "rng_world_size": dist_ctx["world_size"],
    }
    if not dist_ctx["distributed"]:
        training_state["rng_state"] = rng_state_by_rank.get("0")
    return training_state


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            if not torch.is_floating_point(value):
                value.copy_(model_state[key])
            else:
                value.mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)


def run_sampler(
    model,
    condition: torch.Tensor,
    sample_cfg: dict,
    side_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    sampler = sample_cfg.get("sampler", "ddim")
    steps = sample_cfg.get("steps", 50)
    if sampler in {"flow_euler", "flow_heun"}:
        if not isinstance(model, ConditionalFlowMatching):
            raise ValueError(f"Sampler {sampler} requires a ConditionalFlowMatching model")
        return model.sample_flow(condition, sample_steps=steps, solver=sampler.removeprefix("flow_"), side_labels=side_labels)
    if sampler == "ddim":
        return model.sample_ddim(condition, sample_steps=steps, eta=sample_cfg.get("eta", 0.0), side_labels=side_labels)
    if sampler == "ddpm":
        return model.sample(condition, sample_steps=steps, side_labels=side_labels)
    if sampler in {"dpm_solver_pp", "dpm++", "dpmsolver++"}:
        return model.sample_dpm_solver_pp(
            condition,
            sample_steps=steps,
            solver_order=int(sample_cfg.get("solver_order", 2)),
            side_labels=side_labels,
        )
    raise ValueError(f"Unsupported sampler: {sampler}")


def is_better_metric(metric_name: str, current: float, best: float) -> bool:
    if metric_name in {"val_ssim", "psnr", "val_psnr"}:
        return current > best
    return current < best


def resolve_step_lr(train_cfg: dict, step: int) -> float:
    base_lr = float(train_cfg["lr"])
    schedule_cfg = train_cfg.get("lr_schedule", {})
    if not schedule_cfg or not schedule_cfg.get("enabled", False):
        return base_lr
    schedule_type = schedule_cfg.get("type", "step")
    if schedule_type == "step":
        current_lr = base_lr
        schedule_steps = schedule_cfg.get("steps", [])
        for item in sorted(schedule_steps, key=lambda entry: int(entry["step"])):
            if step >= int(item["step"]):
                current_lr = float(item["lr"])
            else:
                break
        return current_lr
    if schedule_type in {"cosine", "cosine_with_warmup"}:
        max_steps = int(train_cfg["max_steps"])
        warmup_steps = int(schedule_cfg.get("warmup_steps", 0))
        warmup_start_lr = float(schedule_cfg.get("warmup_start_lr", 0.0))
        min_lr = float(schedule_cfg.get("min_lr", 0.0))
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_progress = step / max(warmup_steps, 1)
            return warmup_start_lr + warmup_progress * (base_lr - warmup_start_lr)
        cosine_total = max(1, max_steps - warmup_steps)
        cosine_progress = min(max((step - warmup_steps) / cosine_total, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return min_lr + (base_lr - min_lr) * cosine_decay
    current_lr = base_lr
    raise ValueError(f"Unsupported lr_schedule type: {schedule_type}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def compute_early_stop_score(val_metrics: dict[str, float], early_stop_cfg: dict) -> float:
    score_cfg = early_stop_cfg.get("score", {})
    metric_name = score_cfg.get("metric")
    if metric_name:
        # Lower error metrics are converted to a maximize score; this makes
        # early stop agree exactly with a D-NMAE selected checkpoint.
        return -float(val_metrics[str(metric_name)])
    mae_weight = float(score_cfg.get("mae_weight", 1.0))
    ssim_weight = float(score_cfg.get("ssim_weight", 1.0))
    return ssim_weight * float(val_metrics["val_ssim"]) - mae_weight * float(val_metrics["val_mae"])


def get_batch_side_labels(batch: dict, device: torch.device) -> torch.Tensor | None:
    side = batch.get("side")
    if side is None:
        return None
    if not isinstance(side, torch.Tensor):
        raise TypeError("batch['side'] must be a torch.Tensor")
    return side.to(device=device, dtype=torch.long)


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    sample_cfg: dict,
    max_batches: int | None = None,
    compute_fid: bool = False,
    global_step: int | None = None,
    data_norm_cfg: dict | None = None,
    physics_visual_dir: Path | None = None,
    physics_visual_limit: int = 0,
    physics_visual_prefix: str = "",
    sampling_seed: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    mses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    breakdown_values: dict[str, list[float]] = {}
    branch_values: dict[str, list[float]] = {}
    physics_visuals_saved = 0
    physics_visual_warning_printed = False
    d_nmae_numer = 0.0
    d_nmae_denom = 0.0
    is_d_only = bool(getattr(model, "is_d_only", False))
    if sampling_seed is not None:
        # Validation is deliberately a fixed stochastic experiment.  This is
        # invoked only by rank zero after the DDP barrier.
        torch.manual_seed(int(sampling_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sampling_seed))
    is_branch_model = bool(getattr(getattr(model, "denoise_fn", None), "branch_decoder", False))
    fid_preds = []
    fid_targets = []
    progress_total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch_idx, batch in enumerate(tqdm(loader, desc="validate", leave=False, total=progress_total)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        side_labels = get_batch_side_labels(batch, device)
        loss = model(target, condition, side_labels=side_labels, global_step=global_step)
        loss_breakdown = dict(getattr(model, "last_loss_breakdown", {}))
        pred = run_sampler(model, condition, sample_cfg, side_labels=side_labels)
        losses.append(float(loss.item()))
        metric_target = model.encode_target(target, condition) if is_d_only else target
        maes.append(mae(pred, metric_target))
        mses.append(mse(pred, metric_target))
        if is_d_only:
            pred_raw_d = model.raw_d_from_model_space(pred)
            target_raw_d = model.raw_d_from_target(target)
            d_nmae_numer += float((pred_raw_d - target_raw_d).abs().sum().item())
            d_nmae_denom += float(target_raw_d.abs().sum().item())
            d_norm = DNormalization.from_dict({
                "mean": 0.0, "std": 1.0, "scale": float(model.difference_scale),
                "scale_method": "checkpoint", "scale_quantile": None,
                "scale_sample_count": 0, "count": 0, "raw_min": 0.0, "raw_max": 0.0,
            })
            # DOnlyDataset retains the paired target solely for these raw
            # hard-decoded branch metrics; it is never supplied to the model.
            for key, value in hard_decoded_branch_metrics(
                pred, condition, batch["branch_target"].to(device), data_norm_cfg, d_norm,
                raw_condition=batch["raw_condition"].to(device) if "raw_condition" in batch else None,
                raw_target=batch["raw_target"].to(device) if "raw_target" in batch else None,
            ).items():
                branch_values.setdefault(key, []).append(value)
        else:
            psnrs.append(psnr(pred, metric_target))
            ssims.append(ssim(pred, metric_target))
        for key, value in loss_breakdown.items():
            breakdown_values.setdefault(key, []).append(float(value))
        if is_branch_model and pred.shape[1] >= 2 and target.shape[1] >= 2:
            batch_names = batch.get("name", [f"batch{batch_idx}_idx{idx}" for idx in range(pred.shape[0])])
            for item_idx in range(pred.shape[0]):
                branch_metrics = compute_branch_physics_metrics(
                    condition[item_idx],
                    pred[item_idx],
                    target[item_idx],
                    data_norm_cfg=data_norm_cfg,
                )
                for key, value in branch_metrics.items():
                    branch_values.setdefault(key, []).append(float(value))
                if physics_visual_dir is not None and physics_visuals_saved < physics_visual_limit:
                    sample_name = str(batch_names[item_idx])
                    title = (
                        f"{physics_visual_prefix}{sample_name} | "
                        f"L loss={branch_metrics.get('left_loss', 0.0):.6g}, "
                        f"R loss={branch_metrics.get('right_loss', 0.0):.6g}, "
                        f"res MAE={branch_metrics['physics_residual_mae']:.6g}"
                    )
                    try:
                        save_physics_visualization(
                            condition[item_idx],
                            pred[item_idx],
                            Path(physics_visual_dir) / f"{physics_visual_prefix}{safe_filename(sample_name)}_physics.png",
                            title=title,
                            data_norm_cfg=data_norm_cfg,
                        )
                        physics_visuals_saved += 1
                    except ImportError as exc:
                        if not physics_visual_warning_printed:
                            print(
                                {
                                    "warning": "Skipping physics visualizations because an optional dependency is missing.",
                                    "error": str(exc),
                                }
                            )
                            physics_visual_warning_printed = True
                        physics_visual_limit = 0
        if compute_fid:
            fid_preds.append(pred.detach().cpu())
            fid_targets.append(target.detach().cpu())

    metrics = {
        "val_loss": sum(losses) / max(len(losses), 1),
        "val_d_model_l1" if is_d_only else "val_mae": sum(maes) / max(len(maes), 1),
        "val_d_model_mse" if is_d_only else "val_mse": sum(mses) / max(len(mses), 1),
    }
    if not is_d_only:
        metrics["val_psnr"] = sum(psnrs) / max(len(psnrs), 1)
        metrics["val_ssim"] = sum(ssims) / max(len(ssims), 1)
    if is_d_only:
        metrics["val_d_nmae"] = d_nmae_numer / max(d_nmae_denom, 1e-8)
    for key in LOSS_BREAKDOWN_KEYS:
        if key in breakdown_values and breakdown_values[key]:
            metrics[f"val_{key}"] = sum(breakdown_values[key]) / len(breakdown_values[key])
    for key, values in branch_values.items():
        if values:
            metrics[f"val_branch_{key}"] = sum(values) / len(values)
    if compute_fid and fid_preds:
        pred_tensor = torch.cat(fid_preds, dim=0)
        target_tensor = torch.cat(fid_targets, dim=0)
        try:
            fid_per_channel = compute_channelwise_fid(pred_tensor, target_tensor)
            metrics["fid_mean"] = float(sum(fid_per_channel) / len(fid_per_channel))
        except Exception as exc:
            print(f"Skipping FID due to error: {exc}")
    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    ema: EMA,
    step: int,
    cfg: dict,
    best_metric: float | None = None,
    best_metrics: dict[str, float] | None = None,
    training_state: dict | None = None,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": step,
            "best_metric": best_metric,
            "best_metrics": best_metrics or {},
            "training_state": training_state or {},
        },
        path,
    )


def load_checkpoint(path, model, optimizer, ema: EMA, device: torch.device) -> tuple[int, float, dict[str, float], dict]:
    state = require_training_checkpoint(load_checkpoint_file(path, device))

    unwrap_model(model).load_state_dict(state["model"], strict=True)

    if "ema_model" in state:
        ema.ema_model.load_state_dict(state["ema_model"], strict=True)
    else:
        ema.ema_model.load_state_dict(state["model"], strict=True)

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    else:
        raise ValueError("Resume checkpoint is missing optimizer state; use last.pt or step_xxxxxx.pt for continuation.")

    start_step = int(state.get("step", 0))
    best_metric = float(state.get("best_metric", float("inf")))
    saved_best_metrics = state.get("best_metrics", {})
    best_metrics = {str(key): float(value) for key, value in saved_best_metrics.items()}
    training_state = dict(state.get("training_state", {}))
    return start_step, best_metric, best_metrics, training_state

def main() -> None:
    parser = argparse.ArgumentParser(description="Research-style step-based trainer for palette_decoupling.")
    parser.add_argument("--config", required=True, help="Path to yaml/json config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--case-root", default=None, help="Optional override for dataset.train/val case_root.")
    parser.add_argument("--experiment-name", default=None, help="Override logging.experiment_name and unified output folder name.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.case_root is not None:
        for split_name in ("train", "val", "test"):
            split_cfg = cfg.get("dataset", {}).get(split_name)
            if isinstance(split_cfg, dict) and "case_root" in split_cfg:
                split_cfg["case_root"] = args.case_root
    run_group = generative_run_group(cfg)
    cfg = apply_run_layout(cfg, group=run_group, default_experiment=Path(args.config).stem, experiment_name=args.experiment_name)
    set_seed(int(cfg.get("seed", 1234)))
    dist_ctx = setup_distributed(cfg.get("device"))
    device = dist_ctx["device"]
    is_main = dist_ctx["is_main"]

    output_dir = ensure_dir(cfg["output"]["root"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    sample_dir = ensure_dir(output_dir / "samples")

    train_dataset, train_loader, train_sampler = build_loader(
        cfg["dataset"]["train"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        distributed=dist_ctx["distributed"],
        rank=dist_ctx["rank"],
        world_size=dist_ctx["world_size"],
    )
    val_dataset, val_loader, _ = build_loader(
        cfg["dataset"]["val"],
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        distributed=False,
    )
    governance = cfg.get("data_governance", {})
    assert_data_governance(train_dataset, governance)
    assert_data_governance(val_dataset, governance)
    target_parameterization_cfg = cfg["diffusion"].get("target_parameterization") or cfg.get("target_parameterization")
    target_contract = TargetContract.from_config(
        target_parameterization_cfg,
        raw_value_scale=float(cfg["dataset"]["train"].get("value_range", [0.0, 1.0])[1] - cfg["dataset"]["train"].get("value_range", [0.0, 1.0])[0]),
    )
    if is_main and target_contract.is_d_only:
        d_norm = cfg["dataset"]["train"].get("d_only_normalization")
        if d_norm is None:
            raise ValueError("D-only diffusion requires dataset.train.d_only_normalization")
        (output_dir / "d_normalization.json").write_text(json.dumps(d_norm, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if is_main:
        print({"run_layout": cfg.get("run_layout", {})})
        print(
            {
                "device": str(device),
                "rank": dist_ctx["rank"],
                "world_size": dist_ctx["world_size"],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "condition_channels": train_dataset.condition_channels,
                "target_channels": train_dataset.target_channels,
            }
        )

    setup = prepare_generative_training(cfg, train_dataset)
    image_size = setup.image_size
    d_only_training = setup.contract.is_d_only
    if setup.contract != target_contract:
        raise RuntimeError("Factory target contract differs from validated training contract")
    model = setup.model.to(device)
    if is_main:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    if dist_ctx["distributed"]:
        model = DDP(model, device_ids=[dist_ctx["local_rank"]] if device.type == "cuda" else None, output_device=dist_ctx["local_rank"] if device.type == "cuda" else None)
    run = maybe_init_swanlab(cfg.get("logging", {})) if is_main else None
    tb_writer = maybe_init_tensorboard(cfg.get("logging", {}), output_dir) if is_main else None

    max_steps = int(cfg["train"]["max_steps"])
    validate_every = int(cfg["train"]["validate_every_steps"])
    save_every = int(cfg["train"].get("save_every_steps", validate_every))
    log_every = int(cfg["train"].get("log_every_steps", 50))
    ema_start = int(cfg["train"].get("ema_start_step", 0))
    fid_every = int(cfg["validation"].get("fid_every_steps", validate_every))
    physics_visual_limit = int(cfg["validation"].get("physics_visual_limit", 1))
    preview_cfg = cfg["validation"].get("preview", {})
    preview_channels_cfg = preview_cfg.get("channels", cfg["validation"].get("preview_channels", [0, 1]))
    preview_save_png = bool(preview_cfg.get("save_png", True))
    preview_save_npy = bool(preview_cfg.get("save_npy", False))
    preview_tensorboard = bool(preview_cfg.get("tensorboard", True))
    swanlab_preview_enabled = bool(preview_cfg.get("swanlab", True))
    best_metric_name = cfg["train"].get("best_metric", "val_mae")
    best_metric = float("-inf") if best_metric_name in {"val_ssim", "psnr", "val_psnr"} else float("inf")
    best_metrics = ({"val_d_model_l1": float("inf"), "val_branch_side_ssim": float("-inf")}
                    if d_only_training else {"val_mae": float("inf"), "val_ssim": float("-inf")})
    early_stop_cfg = cfg["train"].get("early_stop", {})
    early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
    early_stop_patience = int(early_stop_cfg.get("patience_validations", 20))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))
    early_stop_best_score = float("-inf")
    early_stop_bad_validations = 0
    start_step = 0
    recent_losses: list[float] = []
    train_epoch = 0
    batches_seen_in_epoch = 0
    training_state: dict = {}

    resume_path = args.resume or cfg["train"].get("resume_checkpoint")
    if resume_path:
        start_step, best_metric, loaded_best_metrics, training_state = load_checkpoint(resume_path, model, optimizer, ema, device)
        best_metrics.update(loaded_best_metrics)
        # Checkpoints written before hard-decoded branch monitoring used the
        # generic D-space MAE/SSIM keys.  Resume them safely while allowing the
        # new branch image checkpoints to start fresh.
        for key, default in (("val_d_model_l1", float("inf")), ("val_branch_side_ssim", float("-inf"))) if d_only_training else ():
            best_metrics.setdefault(key, default)
        early_stop_best_score = float(training_state.get("early_stop_best_score", early_stop_best_score))
        early_stop_bad_validations = int(training_state.get("early_stop_bad_validations", early_stop_bad_validations))
        train_epoch = int(training_state.get("train_epoch", train_epoch))
        batches_seen_in_epoch = int(training_state.get("batches_seen_in_epoch", batches_seen_in_epoch))
        saved_recent_losses = training_state.get("recent_losses", recent_losses)
        recent_losses = [float(item) for item in saved_recent_losses][-log_every:]
        if is_main:
            print(
                {
                    "resumed_from": resume_path,
                    "start_step": start_step,
                    "best_metric": best_metric,
                    "best_metrics": best_metrics,
                    "early_stop_best_score": early_stop_best_score,
                    "early_stop_bad_validations": early_stop_bad_validations,
                    "train_epoch": train_epoch,
                    "batches_seen_in_epoch": batches_seen_in_epoch,
                }
            )
        if dist_ctx["distributed"]:
            dist.barrier()

    progress = tqdm(
        range(start_step + 1, max_steps + 1),
        desc="train steps",
        leave=True,
        initial=start_step,
        total=max_steps,
        dynamic_ncols=True,
        mininterval=1.0,
        disable=not is_main,
    )
    train_epoch, batches_seen_in_epoch = normalize_training_progress(train_epoch, batches_seen_in_epoch, len(train_loader))
    train_iter = restore_train_iterator(train_loader, train_sampler, train_epoch, batches_seen_in_epoch)
    rng_restored = restore_rng_state_for_rank(training_state, dist_ctx, is_main) if resume_path else False
    if is_main and resume_path:
        print({"rng_restored": rng_restored})
    if dist_ctx["distributed"]:
        dist.barrier()

    try:
        for step in progress:
            current_lr = resolve_step_lr(cfg["train"], step)
            set_optimizer_lr(optimizer, current_lr)
            try:
                batch = next(train_iter)
                batches_seen_in_epoch += 1
            except StopIteration:
                train_epoch += 1
                batches_seen_in_epoch = 0
                train_iter = restore_train_iterator(train_loader, train_sampler, train_epoch, batches_seen_in_epoch)
                batch = next(train_iter)
                batches_seen_in_epoch = 1

            model.train()
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            side_labels = get_batch_side_labels(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(target, condition, side_labels=side_labels, global_step=step)
            loss.backward()
            optimizer.step()
            if step >= ema_start:
                ema.update(unwrap_model(model))

            loss_value = float(loss.item())
            loss_breakdown = dict(getattr(unwrap_model(model), "last_loss_breakdown", {}))
            recent_losses.append(loss_value)
            if len(recent_losses) > log_every:
                recent_losses.pop(0)
            if is_main:
                postfix = {"loss": f"{loss_value:.6f}", "avg": f"{sum(recent_losses)/len(recent_losses):.6f}"}
                if "left_diff_loss" in loss_breakdown and "right_diff_loss" in loss_breakdown:
                    postfix["left"] = f"{loss_breakdown['left_diff_loss']:.6f}"
                    postfix["right"] = f"{loss_breakdown['right_diff_loss']:.6f}"
                if "r_diff_loss" in loss_breakdown and "d_diff_loss" in loss_breakdown:
                    postfix["r"] = f"{loss_breakdown['r_diff_loss']:.6f}"
                    postfix["d"] = f"{loss_breakdown['d_diff_loss']:.6f}"
                if "consistency_ratio" in loss_breakdown:
                    postfix["cons"] = f"{loss_breakdown.get('consistency_term', 0.0):.6f}"
                    postfix["cons_r"] = f"{loss_breakdown['consistency_ratio']:.3f}"
                    postfix["act"] = f"{loss_breakdown.get('consistency_active_ratio', 0.0):.2f}"
                    if loss_breakdown.get("consistency_capped", 0.0) > 0.0:
                        postfix["clip"] = f"{loss_breakdown.get('consistency_clip_scale', 1.0):.2f}"
                if "x0_reconstruction_ratio" in loss_breakdown:
                    postfix["x0"] = f"{loss_breakdown.get('x0_reconstruction_term', 0.0):.6f}"
                    postfix["x0_r"] = f"{loss_breakdown['x0_reconstruction_ratio']:.3f}"
                    if loss_breakdown.get("x0_reconstruction_capped", 0.0) > 0.0:
                        postfix["x0_clip"] = f"{loss_breakdown.get('x0_reconstruction_clip_scale', 1.0):.2f}"
                if loss_breakdown.get("difference_active", 0.0) > 0.0:
                    postfix["diff"] = f"{loss_breakdown.get('difference_term', 0.0):.6f}"
                if loss_breakdown.get("difference_edge_active", 0.0) > 0.0:
                    postfix["edge"] = f"{loss_breakdown.get('difference_edge_term', 0.0):.6f}"
                progress.set_postfix(**postfix)

            if run is not None and step % log_every == 0:
                train_log = {"step": step, "train_loss": sum(recent_losses) / len(recent_losses), "lr": get_optimizer_lr(optimizer)}
                for key in LOSS_BREAKDOWN_KEYS:
                    if key in loss_breakdown:
                        train_log[key] = loss_breakdown[key]
                run = log_monitoring_run(run, train_log, "train_log")
                tensorboard_add_scalars(tb_writer, "train", train_log, step)

            should_stop = False
            if step % validate_every == 0 or step == max_steps:
                if dist_ctx["distributed"]:
                    dist.barrier()
                validation_training_state = build_training_state(
                    early_stop_best_score,
                    early_stop_bad_validations,
                    train_epoch,
                    batches_seen_in_epoch,
                    len(train_loader),
                    recent_losses,
                    dist_ctx,
                )
                if is_main:
                    eval_model = ema.ema_model
                    val_metrics = validate(
                        eval_model,
                        val_loader,
                        device,
                        sample_cfg=cfg["validation"]["sampler"],
                        max_batches=cfg["validation"].get("max_batches"),
                        compute_fid=(fid_every > 0 and step % fid_every == 0),
                        global_step=step,
                        data_norm_cfg=cfg["dataset"]["val"],
                        physics_visual_dir=sample_dir / "physics",
                        physics_visual_limit=physics_visual_limit,
                        physics_visual_prefix=f"step_{step:06d}_",
                        sampling_seed=cfg["validation"].get("sampling_seed"),
                    )
                    val_metrics["lr"] = get_optimizer_lr(optimizer)
                    val_metrics["step"] = step
                    print(val_metrics)
                    if "val_branch_left_loss" in val_metrics and "val_branch_right_loss" in val_metrics:
                        print(
                            {
                                "step": step,
                                "val_branch_left_loss": val_metrics["val_branch_left_loss"],
                                "val_branch_right_loss": val_metrics["val_branch_right_loss"],
                                "val_branch_left_mae": val_metrics.get("val_branch_left_mae"),
                                "val_branch_right_mae": val_metrics.get("val_branch_right_mae"),
                                "val_branch_difference_mae": val_metrics.get("val_branch_difference_mae"),
                                "val_branch_difference_edge_mae": val_metrics.get("val_branch_difference_edge_mae"),
                                "val_branch_physics_residual_mae": val_metrics.get("val_branch_physics_residual_mae"),
                                "physics_visual_dir": str(sample_dir / "physics") if physics_visual_limit > 0 else None,
                            }
                        )
                    if run is not None:
                        run = log_monitoring_run(run, val_metrics, "val_metrics")
                    tensorboard_add_scalars(tb_writer, "val", val_metrics, step)

                    mae_metric = "val_d_model_l1" if d_only_training else "val_mae"
                    current_mae = float(val_metrics[mae_metric])
                    if current_mae < best_metrics[mae_metric]:
                        best_metrics[mae_metric] = current_mae
                        save_checkpoint(
                            checkpoint_dir / ("best_ema_d_l1.pt" if d_only_training else "best_ema_mae.pt"),
                            unwrap_model(model),
                            optimizer,
                            ema,
                            step,
                            cfg,
                            best_metric=current_mae,
                            best_metrics=best_metrics,
                            training_state=validation_training_state,
                        )

                    ssim_metric = "val_branch_side_ssim" if d_only_training else "val_ssim"
                    current_ssim = float(val_metrics[ssim_metric])
                    if current_ssim > best_metrics[ssim_metric]:
                        best_metrics[ssim_metric] = current_ssim
                        save_checkpoint(
                            checkpoint_dir / ("best_ema_side_ssim.pt" if d_only_training else "best_ema_ssim.pt"),
                            unwrap_model(model),
                            optimizer,
                            ema,
                            step,
                            cfg,
                            best_metric=current_ssim,
                            best_metrics=best_metrics,
                            training_state=validation_training_state,
                        )

                    metric_value = float(val_metrics.get(best_metric_name, val_metrics[mae_metric]))
                    if is_better_metric(best_metric_name, metric_value, best_metric):
                        best_metric = metric_value
                        save_checkpoint(
                            checkpoint_dir / "best_ema.pt",
                            unwrap_model(model),
                            optimizer,
                            ema,
                            step,
                            cfg,
                            best_metric=best_metric,
                            best_metrics=best_metrics,
                            training_state=validation_training_state,
                        )

                    if early_stop_enabled:
                        current_score = compute_early_stop_score(val_metrics, early_stop_cfg)
                        val_metrics["early_stop_score"] = current_score
                        if current_score > early_stop_best_score + early_stop_min_delta:
                            early_stop_best_score = current_score
                            early_stop_bad_validations = 0
                        else:
                            early_stop_bad_validations += 1
                        print(
                            {
                                "early_stop_score": current_score,
                                "early_stop_best_score": early_stop_best_score,
                                "early_stop_bad_validations": early_stop_bad_validations,
                                "early_stop_patience": early_stop_patience,
                            }
                        )
                        if run is not None:
                            run = log_monitoring_run(
                                run,
                                {
                                    "step": step,
                                    "early_stop_score": current_score,
                                    "early_stop_best_score": early_stop_best_score,
                                    "early_stop_bad_validations": early_stop_bad_validations,
                                },
                                "early_stop",
                            )
                        tensorboard_add_scalars(
                            tb_writer,
                            "early_stop",
                            {
                                "early_stop_score": current_score,
                                "early_stop_best_score": early_stop_best_score,
                                "early_stop_bad_validations": early_stop_bad_validations,
                            },
                            step,
                        )

                    preview_batch = next(iter(val_loader))
                    preview_condition = preview_batch["condition"].to(device)
                    preview_target = preview_batch["target"]
                    preview_side_labels = get_batch_side_labels(preview_batch, device)
                    preview_pred = run_sampler(eval_model, preview_condition, cfg["validation"]["sampler"], side_labels=preview_side_labels)
                    preview_name = str(preview_batch["name"][0])
                    if d_only_training:
                        d_norm = DNormalization.from_dict({
                            "mean": 0.0, "std": 1.0, "scale": float(eval_model.difference_scale),
                            "scale_method": "checkpoint", "scale_quantile": None,
                            "scale_sample_count": 0, "count": 0, "raw_min": 0.0, "raw_max": 0.0,
                        })
                        raw_d, raw_branches, normalized_branches = decode_standardized_d(
                            preview_pred, preview_condition, cfg["dataset"]["val"], d_norm,
                            raw_condition=preview_batch["raw_condition"].to(device) if "raw_condition" in preview_batch else None,
                        )
                        if preview_save_npy:
                            save_tensor_npy(raw_d[0], sample_dir / f"step_{step:06d}_{preview_name}_d.npy")
                            save_tensor_npy(raw_branches[0], sample_dir / f"step_{step:06d}_{preview_name}_branches_raw.npy")
                        if preview_save_png:
                            save_tensor_png(preview_pred[0], sample_dir / f"step_{step:06d}_{preview_name}_d_normalized.png")
                            save_tensor_png(normalized_branches[0, 0:1], sample_dir / f"step_{step:06d}_{preview_name}_left.png")
                            save_tensor_png(normalized_branches[0, 1:2], sample_dir / f"step_{step:06d}_{preview_name}_right.png")
                        if swanlab_preview_enabled and run is not None:
                            swanlab_preview_enabled = swanlab_log_preview_images(
                                run, step, torch.cat((preview_pred[0], normalized_branches[0]), dim=0),
                                preview_name, [0, 1, 2], labels=["D_normalized", "left", "right"],
                            )
                    else:
                        preview_channels = resolve_preview_channels(preview_channels_cfg, preview_pred.shape[1])
                        if preview_save_npy:
                            save_tensor_npy(preview_pred[0], sample_dir / f"step_{step:06d}_{preview_name}_pred.npy")
                        if preview_save_png:
                            for channel_idx in preview_channels:
                                save_tensor_png(preview_pred[0, channel_idx : channel_idx + 1], sample_dir / f"step_{step:06d}_{preview_name}_pred_ch{channel_idx}.png")
                        if swanlab_preview_enabled and run is not None:
                            swanlab_preview_enabled = swanlab_log_preview_images(run, step, preview_pred[0], preview_name, preview_channels)
                    if preview_tensorboard:
                        tensorboard_add_preview_images(
                            tb_writer,
                            step,
                            preview_condition[0],
                            preview_target[0] if preview_target is not None else None,
                            preview_pred[0],
                            preview_name,
                        )

                    if early_stop_enabled and early_stop_bad_validations >= early_stop_patience:
                        print(
                            {
                                "early_stop_triggered": True,
                                "step": step,
                                "early_stop_best_score": early_stop_best_score,
                                "early_stop_bad_validations": early_stop_bad_validations,
                            }
                        )
                        should_stop = True

                if dist_ctx["distributed"]:
                    stop_tensor = torch.tensor(1 if (should_stop and is_main) else 0, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
                    dist.barrier()

            periodic_training_state = None
            if step % save_every == 0 or step == max_steps:
                periodic_training_state = build_training_state(
                    early_stop_best_score,
                    early_stop_bad_validations,
                    train_epoch,
                    batches_seen_in_epoch,
                    len(train_loader),
                    recent_losses,
                    dist_ctx,
                )
            if is_main and (step % save_every == 0 or step == max_steps):
                save_checkpoint(
                    checkpoint_dir / f"step_{step:06d}.pt",
                    unwrap_model(model),
                    optimizer,
                    ema,
                    step,
                    cfg,
                    best_metric=best_metric,
                    best_metrics=best_metrics,
                    training_state=periodic_training_state,
                )
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    ema,
                    step,
                    cfg,
                    best_metric=best_metric,
                    best_metrics=best_metrics,
                    training_state=periodic_training_state,
                )
            if dist_ctx["distributed"] and (step % save_every == 0 or step == max_steps):
                dist.barrier()

            if should_stop:
                break

        if run is not None:
            run.finish()
    finally:
        if tb_writer is not None:
            tb_writer.close()
        cleanup_distributed(dist_ctx["distributed"])


if __name__ == "__main__":
    main()
