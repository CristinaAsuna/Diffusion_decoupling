"""Train the minimal globally scaled-D + L1-only ResUNet probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from consistent_cdm.evaluation.data_metrics import asymmetry_metrics
from consistent_cdm.utils.config import ensure_dir, load_config
from consistent_cdm.utils.paths import apply_run_layout
from consistent_cdm.utils.checkpointing import load_checkpoint_file, restore_rng_state

from .common import (
    EMA,
    build_dataset,
    cleanup_distributed,
    log_scalars,
    maybe_init_swanlab,
    resolve_step_lr,
    save_checkpoint,
    seed_everything,
    set_optimizer_lr,
    setup_distributed,
    tensor_to_float,
    unwrap_model,
)
from .d_only import (
    DOnlyDataset,
    assert_allowed_case_prefixes,
    assert_no_forbidden_cases,
    resolve_d_normalization,
    decode_standardized_d,
)
from consistent_cdm.factories import build_deterministic_generator



def _loader(dataset, batch_size: int, *, shuffle: bool, workers: int, distributed: bool, rank: int, world_size: int):
    if shuffle and len(dataset) < batch_size * world_size:
        raise ValueError(f"Training dataset has {len(dataset)} samples for batch_size={batch_size}, world_size={world_size}; reduce batch_size")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=shuffle), sampler


@torch.no_grad()
def _validate(model, loader, device, data_cfg, normalization) -> dict[str, float]:
    model.eval()
    total, l1_total = 0, 0.0
    nmae_values: list[float] = []
    identity_values: list[float] = []
    signed_gain_values: list[float] = []
    full_half_nmae_values: list[float] = []
    full_half_identity_values: list[float] = []
    full_half_gain_values: list[float] = []
    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        prediction = model(condition)
        l1_total += float(F.l1_loss(prediction, target, reduction="sum"))
        total += int(target.numel())
        raw_d, _, _ = decode_standardized_d(prediction, condition, data_cfg, normalization, raw_condition=batch["raw_condition"].to(device) if "raw_condition" in batch else None)
        target_raw_d = target * normalization.scale
        for index, (pred_item, target_item) in enumerate(zip(raw_d.detach().cpu().numpy(), target_raw_d.detach().cpu().numpy())):
            scores = asymmetry_metrics(pred_item[0], target_item[0])
            nmae_values.append(scores["nmae"])
            identity_values.append(scores["identity"])
            signed_gain_values.append(scores["signed_gain"])
            full_half = asymmetry_metrics(np.zeros_like(target_item[0]), target_item[0])
            full_half_nmae_values.append(full_half["nmae"])
            full_half_identity_values.append(full_half["identity"])
            full_half_gain_values.append(full_half["signed_gain"])
    if total == 0:
        raise ValueError("Validation loader produced no samples")
    result = {
        "val_d_l1": l1_total / total,
        "val_d_nmae": float(sum(nmae_values) / len(nmae_values)),
        "val_d_identity": float(sum(identity_values) / len(identity_values)),
        # G_D: projection of predicted signed D onto GT D.  Positive values
        # mean the model has more direct than swapped orientation evidence.
        "val_d_G": float(sum(signed_gain_values) / len(signed_gain_values)),
        "val_d_signed_gain": float(sum(signed_gain_values) / len(signed_gain_values)),
        "val_full_half_d_nmae": float(sum(full_half_nmae_values) / len(full_half_nmae_values)),
        "val_full_half_d_identity": float(sum(full_half_identity_values) / len(full_half_identity_values)),
        "val_full_half_d_G": float(sum(full_half_gain_values) / len(full_half_gain_values)),
        "num_samples": float(len(nmae_values)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_run_layout(cfg, group="baseline", default_experiment=Path(args.config).stem, experiment_name=args.experiment_name)
    governance = cfg.get("data_governance", {})
    if governance.get("require_case_names_file") and not cfg["dataset"]["train"].get("case_names_file"):
        raise ValueError("D-only development config requires an explicit non-test case_names_file")
    seed_everything(int(cfg.get("seed", 1234)))
    dist_ctx = setup_distributed(cfg.get("device"))
    device, is_main = dist_ctx["device"], bool(dist_ctx["is_main"])
    output = ensure_dir(cfg["output"]["root"])

    train_base = build_dataset(cfg["dataset"]["train"])
    val_base = build_dataset(cfg["dataset"]["val"])
    forbidden = governance.get("forbidden_case_ids", [])
    assert_no_forbidden_cases(train_base, forbidden)
    assert_no_forbidden_cases(val_base, forbidden)
    assert_allowed_case_prefixes(train_base, governance.get("required_case_prefixes", []))
    assert_allowed_case_prefixes(val_base, governance.get("required_case_prefixes", []))
    normalization = resolve_d_normalization(
        train_base,
        cfg["dataset"]["train"],
        cfg.get("d_normalization", {}),
    )
    if is_main:
        (output / "d_normalization.json").write_text(json.dumps(normalization.to_dict(), indent=2) + "\n", encoding="utf-8")
    if dist_ctx["distributed"]:
        dist.barrier()
    train_set = DOnlyDataset(train_base, cfg["dataset"]["train"], normalization)
    val_set = DOnlyDataset(val_base, cfg["dataset"]["val"], normalization)
    train_loader, train_sampler = _loader(train_set, int(cfg["train"]["batch_size"]), shuffle=True,
                                          workers=int(cfg["train"].get("num_workers", 0)), distributed=dist_ctx["distributed"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"])
    val_loader, _ = _loader(val_set, int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])), shuffle=False,
                            workers=int(cfg["train"].get("num_workers", 0)), distributed=False, rank=0, world_size=1)
    model_cfg = dict(cfg["model"])
    model_cfg["final_activation"] = "identity"
    model = build_deterministic_generator(model_cfg, condition_channels=train_set.condition_channels, target_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), betas=tuple(cfg["train"].get("betas", [0.9, 0.999])), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_metric_name = "val_d_l1"
    start_step, best = 0, float("inf")
    resume = args.resume or cfg["train"].get("resume_checkpoint")
    if resume:
        state = load_checkpoint_file(resume, device)
        restore_rng_state(state.get("rng_state"))
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); ema.ema_model.load_state_dict(state["ema_model"])
        start_step, best = int(state["step"]), float(state.get("best_metric", best))
        if state.get("d_normalization") != normalization.to_dict():
            raise ValueError("Resume checkpoint D normalization does not match the current train patients")
        if state.get("best_metric_name", best_metric_name) != best_metric_name:
            raise ValueError("Resume checkpoint selection metric does not match this D-only experiment")
    if dist_ctx["distributed"]:
        model = DDP(model, device_ids=[dist_ctx["local_rank"]] if device.type == "cuda" else None)
    run = maybe_init_swanlab(cfg.get("logging", {}), cfg) if is_main else None
    iterator = iter(train_loader); epoch = 0
    if train_sampler is not None: train_sampler.set_epoch(epoch)
    for step in tqdm(range(start_step + 1, int(cfg["train"]["max_steps"]) + 1), initial=start_step, total=int(cfg["train"]["max_steps"]), disable=not is_main, desc="d-only steps"):
        try: batch = next(iterator)
        except StopIteration:
            epoch += 1
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            iterator = iter(train_loader); batch = next(iterator)
        set_optimizer_lr(optimizer, resolve_step_lr(cfg["train"], step))
        condition, target = batch["condition"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(condition)
            full_resolution_l1 = F.l1_loss(prediction, target)
            loss = full_resolution_l1
        scaler.scale(loss).backward()
        if float(cfg["train"].get("grad_clip", 0.0)) > 0:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), float(cfg["train"]["grad_clip"]))
        scaler.step(optimizer); scaler.update()
        if step >= int(cfg["train"].get("ema_start_step", 0)): ema.update(unwrap_model(model))
        if is_main and step % int(cfg["train"].get("log_every_steps", 50)) == 0:
            payload = {
                "step": step,
                "d_l1": tensor_to_float(full_resolution_l1),
                "loss": tensor_to_float(loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            print(payload); log_scalars(None, "train", payload, step)
            if run is not None: run.log(payload)
        if step % int(cfg["train"]["validate_every_steps"]) == 0 or step == int(cfg["train"]["max_steps"]):
            if dist_ctx["distributed"]: dist.barrier()
            if is_main:
                metrics = _validate(ema.ema_model, val_loader, device, cfg["dataset"]["val"], normalization)
                metrics["step"] = float(step); print(metrics)
                if run is not None: run.log(metrics)
                improved = metrics[best_metric_name] < best
                if improved:
                    best = metrics[best_metric_name]
                checkpoint = {
                    "step": step, "model": unwrap_model(model).state_dict(),
                    "ema_model": ema.ema_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "d_normalization": normalization.to_dict(),
                    "config": cfg, "best_metric": best, "best_metric_name": best_metric_name,
                    "val_metrics": metrics,
                }
                if improved:
                    save_checkpoint(output / "checkpoints" / "best.pt", checkpoint)
                save_checkpoint(output / "checkpoints" / "last.pt", checkpoint)
            if dist_ctx["distributed"]: dist.barrier()
    cleanup_distributed(dist_ctx["distributed"])


if __name__ == "__main__":
    main()
