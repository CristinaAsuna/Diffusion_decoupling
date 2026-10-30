from __future__ import annotations

import copy

import csv

import json

import math

import os

import random

from pathlib import Path

from typing import Any

import numpy as np

import torch

import torch.distributed as dist

import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image, ImageDraw

from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from consistent_cdm.data.dataset import CaseFolderNpyDataset, NpyConditionTargetDataset

from consistent_cdm.utils.config import ensure_dir, load_config

from consistent_cdm.utils.metrics import compute_channelwise_fid, mae, mse, psnr, ssim

from consistent_cdm.utils.utils import pick_device, save_tensor_npy, save_tensor_png, set_seed, tensor_to_uint8_image

def build_dataset(dataset_cfg: dict[str, Any]):
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
        return NpyConditionTargetDataset(
            condition_dir=dataset_cfg["condition_dir"],
            target_dirs=dataset_cfg["target_dirs"],
            names_file=dataset_cfg.get("names_file"),
            **common_kwargs,
        )
    if dataset_type == "case_folders":
        return CaseFolderNpyDataset(
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
    raise ValueError(f"Unsupported dataset type: {dataset_type}")

def setup_distributed(device_cfg: str | None = None) -> dict[str, Any]:
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

def denormalize_tensor(tensor: torch.Tensor, data_norm_cfg: dict[str, Any] | None = None) -> torch.Tensor:
    cfg = data_norm_cfg or {}
    mode = str(cfg.get("normalize", "none"))
    value_range = cfg.get("value_range")
    if mode == "fixed_range_m11" and value_range is not None:
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return ((tensor + 1.0) * 0.5) * (max_val - min_val) + min_val
    if mode == "fixed_range_01" and value_range is not None:
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return tensor * (max_val - min_val) + min_val
    return tensor

def normalize_tensor(tensor: torch.Tensor, data_norm_cfg: dict[str, Any] | None = None) -> torch.Tensor:
    cfg = data_norm_cfg or {}
    mode = str(cfg.get("normalize", "none"))
    value_range = cfg.get("value_range")
    if mode == "fixed_range_m11" and value_range is not None:
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return ((tensor - min_val) / max(max_val - min_val, 1e-8)) * 2.0 - 1.0
    if mode == "fixed_range_01" and value_range is not None:
        min_val = float(value_range[0])
        max_val = float(value_range[1])
        return (tensor - min_val) / max(max_val - min_val, 1e-8)
    return tensor

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

def resolve_step_lr(train_cfg: dict[str, Any], step: int) -> float:
    base_lr = float(train_cfg["lr"])
    sched = train_cfg.get("lr_schedule", {})
    if not sched or not bool(sched.get("enabled", False)):
        return base_lr
    sched_type = str(sched.get("type", "cosine_with_warmup"))
    if sched_type != "cosine_with_warmup":
        raise ValueError(f"Unsupported lr_schedule.type: {sched_type}")
    warmup_steps = int(sched.get("warmup_steps", 0))
    warmup_start_lr = float(sched.get("warmup_start_lr", 0.0))
    min_lr = float(sched.get("min_lr", 0.0))
    max_steps = int(train_cfg["max_steps"])
    if warmup_steps > 0 and step <= warmup_steps:
        ratio = step / max(warmup_steps, 1)
        return warmup_start_lr + ratio * (base_lr - warmup_start_lr)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr + cosine * (base_lr - min_lr)

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

def init_tensorboard(logging_cfg: dict[str, Any], output_dir: Path):
    if not bool(logging_cfg.get("use_tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard is not installed; training continues without TensorBoard.")
        return None
    logdir = os.environ.get("TENSORBOARD_LOGDIR", logging_cfg.get("tensorboard_logdir")) or str(output_dir / "tensorboard")
    print({"tensorboard_logdir": str(logdir)})
    return SummaryWriter(log_dir=str(logdir))

def maybe_init_swanlab(logging_cfg: dict[str, Any], full_cfg: dict[str, Any] | None = None):
    if not bool(logging_cfg.get("use_swanlab", False)):
        return None
    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed; training continues without SwanLab.")
        return None

    run_id = os.environ.get("SWANLAB_RUN_ID", logging_cfg.get("run_id"))
    resume = os.environ.get("SWANLAB_RESUME", logging_cfg.get("resume"))
    mode = os.environ.get("SWANLAB_MODE", logging_cfg.get("mode"))
    logdir = os.environ.get("SWANLAB_LOGDIR", logging_cfg.get("logdir"))
    settings = {
        "project": logging_cfg.get("project", "palette_decoupling_cbct_baselines"),
        "experiment_name": logging_cfg.get("experiment_name"),
        "config": full_cfg or logging_cfg,
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
            "swanlab_experiment_name": settings.get("experiment_name"),
            "swanlab_run_id": settings.get("id"),
            "swanlab_resume": settings.get("resume"),
            "swanlab_mode": settings.get("mode"),
            "swanlab_logdir": settings.get("logdir"),
        }
    )
    return swanlab.init(**settings)

def log_scalars(writer, namespace: str, scalars: dict[str, float | int], step: int) -> None:
    if writer is None:
        return
    for key, value in scalars.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"{namespace}/{key}", float(value), step)

def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)

def save_checkpoint(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from consistent_cdm.utils.checkpointing import capture_rng_state
    payload = dict(payload, rng_state=capture_rng_state())
    torch.save(payload, path)

def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
