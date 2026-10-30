"""Train the development-only one-channel, hard-additive D-only Pix2Pix baseline.

This deliberately does *not* share the branch Pix2Pix objective: the generator and
PatchGAN see a single globally scaled D target, and the only generator losses are
hinge cGAN and L1(D').  Branches are reconstructed only after prediction.
"""

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
from consistent_cdm.utils.utils import save_tensor_npy, save_tensor_png, tensor_to_uint8_image

from .common import (EMA, build_dataset, cleanup_distributed, maybe_init_swanlab,
                     resolve_step_lr, save_checkpoint, seed_everything,
                     set_optimizer_lr, setup_distributed, tensor_to_float,
                     unwrap_model)
from .d_only import (DOnlyDataset, assert_allowed_case_prefixes,
                     assert_no_forbidden_cases, compute_d_normalization, resolve_d_normalization,
                     decode_standardized_d, hard_decoded_branch_metrics)
from consistent_cdm.factories import build_deterministic_discriminator, build_deterministic_generator
def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def discriminator_hinge_loss(real_logits, fake_logits):
    return sum(F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
               for real, fake in zip(real_logits, fake_logits)) / len(real_logits)


def generator_hinge_loss(fake_logits):
    return -sum(logit.mean() for logit in fake_logits) / len(fake_logits)


def _loader(dataset, batch_size, *, shuffle, workers, distributed, rank, world_size):
    if shuffle and len(dataset) < batch_size * world_size:
        raise ValueError(f"Training dataset has {len(dataset)} samples for batch_size={batch_size}, world_size={world_size}; reduce batch_size")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=shuffle), sampler


@torch.no_grad()
def validate_d_only(generator, loader, device, data_cfg, normalization):
    """Raw-domain per-view D metrics. Checkpoint selection uses raw D-nMAE."""
    generator.eval()
    l1_sum, pixels = 0.0, 0
    nmae, identity, gain = [], [], []
    branch_values: dict[str, list[float]] = {}
    for batch in loader:
        condition, target = batch["condition"].to(device), batch["target"].to(device)
        prediction = generator(condition)
        l1_sum += float(F.l1_loss(prediction, target, reduction="sum")); pixels += target.numel()
        raw_pred, _, _ = decode_standardized_d(prediction, condition, data_cfg, normalization, raw_condition=batch["raw_condition"].to(device) if "raw_condition" in batch else None)
        raw_target = target * normalization.scale
        for pred_item, target_item in zip(raw_pred.detach().cpu().numpy(), raw_target.detach().cpu().numpy()):
            scores = asymmetry_metrics(pred_item[0], target_item[0])
            nmae.append(float(scores["nmae"])); identity.append(float(scores["identity"])); gain.append(float(scores["signed_gain"]))
        for key, value in hard_decoded_branch_metrics(
            prediction, condition, batch["branch_target"].to(device), data_cfg, normalization,
            raw_condition=batch["raw_condition"].to(device) if "raw_condition" in batch else None,
            raw_target=batch["raw_target"].to(device) if "raw_target" in batch else None,
        ).items():
            branch_values.setdefault(key, []).append(value)
    if not nmae:
        raise ValueError("Validation loader produced no samples")
    result = {"val_d_l1": l1_sum / pixels, "val_d_nmae": float(np.mean(nmae)),
            "val_d_identity": float(np.mean(identity)), "val_d_G": float(np.mean(gain)),
            "val_d_signed_gain": float(np.mean(gain)), "num_samples": float(len(nmae))}
    result.update({f"val_{key}": float(np.mean(values)) for key, values in branch_values.items()})
    return result


def _swanlab_preview(run, step, name, d_prediction, decoded_branches) -> None:
    if run is None:
        return
    try:
        import swanlab
        from PIL import Image
        panels = {"D_normalized": d_prediction, "left": decoded_branches[0:1], "right": decoded_branches[1:2]}
        run.log({"step": step, **{
            f"preview/{label}": swanlab.Image(Image.fromarray(tensor_to_uint8_image(value)), caption=f"step {step} | {name} | {label}")
            for label, value in panels.items()
        }})
    except Exception as exc:
        print({"warning": "SwanLab preview logging disabled", "error": repr(exc)})


def _resume(path, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, ema, normalization, device):
    state = load_checkpoint_file(path, device)
    if state.get("d_normalization") != normalization.to_dict():
        raise ValueError("Resume checkpoint D normalization does not match current train patients")
    if state.get("loss_contract") != "hinge_cgan_plus_100_l1_d_only":
        raise ValueError("Resume checkpoint is not the strict D-only Pix2Pix loss contract")
    generator.load_state_dict(state["model"]); discriminator.load_state_dict(state["discriminator"])
    ema.ema_model.load_state_dict(state["ema_model"])
    opt_g.load_state_dict(state["optimizer_g"]); opt_d.load_state_dict(state["optimizer_d"])
    scaler_g.load_state_dict(state["scaler_g"]); scaler_d.load_state_dict(state["scaler_d"])
    restore_rng_state(state.get("rng_state"))
    return int(state["step"]), float(state.get("best_metric", float("inf")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--resume"); parser.add_argument("--experiment-name")
    args = parser.parse_args()
    cfg = apply_run_layout(load_config(args.config), group="baseline", default_experiment=Path(args.config).stem, experiment_name=args.experiment_name)
    governance = cfg.get("data_governance", {})
    if governance.get("require_case_names_file") and not cfg["dataset"]["train"].get("case_names_file"):
        raise ValueError("D-only development config requires an explicit non-test case_names_file")
    # Explicitly reject accidental reintroduction of branch/physical auxiliary losses.
    expected_loss = {"adversarial_weight": 1.0, "image_weight": 100.0, "loss_contract": "hinge_cgan_plus_100_l1_d_only"}
    if any(cfg.get("loss", {}).get(k) != value for k, value in expected_loss.items()):
        raise ValueError("D-only Pix2Pix requires exactly hinge cGAN + 100*L1(D')")
    forbidden_keys = {"physics_weight", "feature_matching_weight", "r1_gamma", "edge_weight", "perceptual_weight"}
    present = forbidden_keys & set(cfg.get("loss", {}))
    if present:
        raise ValueError("D-only Pix2Pix forbids auxiliary losses: " + ", ".join(sorted(present)))
    seed_everything(int(cfg.get("seed", 1234)))
    dist_ctx = setup_distributed(cfg.get("device")); device, is_main = dist_ctx["device"], bool(dist_ctx["is_main"])
    output = ensure_dir(cfg["output"]["root"]); checkpoint_dir = ensure_dir(output / "checkpoints"); sample_dir = ensure_dir(output / "samples")
    train_base, val_base = build_dataset(cfg["dataset"]["train"]), build_dataset(cfg["dataset"]["val"])
    for dataset in (train_base, val_base):
        assert_no_forbidden_cases(dataset, governance.get("forbidden_case_ids", [])); assert_allowed_case_prefixes(dataset, governance.get("required_case_prefixes", []))
    normalization = resolve_d_normalization(train_base, cfg["dataset"]["train"], cfg.get("d_normalization", {}))
    if is_main:
        (output / "d_normalization.json").write_text(json.dumps(normalization.to_dict(), indent=2) + "\n", encoding="utf-8")
    if dist_ctx["distributed"]: dist.barrier()
    train_set, val_set = DOnlyDataset(train_base, cfg["dataset"]["train"], normalization), DOnlyDataset(val_base, cfg["dataset"]["val"], normalization)
    train_loader, sampler = _loader(train_set, int(cfg["train"]["batch_size"]), shuffle=True, workers=int(cfg["train"].get("num_workers", 0)), distributed=dist_ctx["distributed"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"])
    val_loader, _ = _loader(val_set, int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])), shuffle=False, workers=int(cfg["train"].get("num_workers", 0)), distributed=False, rank=0, world_size=1)
    generator_cfg = dict(cfg["model"].get("generator", cfg["model"])); generator_cfg["final_activation"] = "identity"
    generator = build_deterministic_generator(generator_cfg, condition_channels=train_set.condition_channels, target_channels=1).to(device)
    discriminator = build_deterministic_discriminator(cfg["model"], condition_channels=train_set.condition_channels, target_channels=1).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=float(cfg["train"]["lr_g"]), betas=tuple(cfg["train"].get("betas_g", [0.5, .999])), weight_decay=float(cfg["train"].get("weight_decay_g", 0.0)))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=float(cfg["train"].get("lr_d", cfg["train"]["lr_g"])), betas=tuple(cfg["train"].get("betas_d", [0.5, .999])), weight_decay=float(cfg["train"].get("weight_decay_d", 0.0)))
    ema = EMA(generator, decay=float(cfg["train"].get("ema_decay", .999)))
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"; scaler_g = torch.cuda.amp.GradScaler(enabled=amp); scaler_d = torch.cuda.amp.GradScaler(enabled=amp)
    step, best = 0, float("inf")
    if args.resume or cfg["train"].get("resume_checkpoint"):
        step, best = _resume(args.resume or cfg["train"]["resume_checkpoint"], generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, ema, normalization, device)
    if dist_ctx["distributed"]:
        generator = DDP(generator, device_ids=[dist_ctx["local_rank"]] if device.type == "cuda" else None); discriminator = DDP(discriminator, device_ids=[dist_ctx["local_rank"]] if device.type == "cuda" else None)
    run = maybe_init_swanlab(cfg.get("logging", {}), cfg) if is_main else None
    epoch, iterator = 0, iter(train_loader)
    if sampler is not None: sampler.set_epoch(epoch)
    max_steps, validate_every, log_every = int(cfg["train"]["max_steps"]), int(cfg["train"]["validate_every_steps"]), int(cfg["train"].get("log_every_steps", 50))
    save_every = int(cfg["train"].get("save_every_steps", validate_every))
    try:
        for current_step in tqdm(range(step + 1, max_steps + 1), initial=step, total=max_steps, disable=not is_main, desc="d-only pix2pix"):
            try: batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None: sampler.set_epoch(epoch)
                iterator = iter(train_loader); batch = next(iterator)
            condition, target = batch["condition"].to(device), batch["target"].to(device)
            set_optimizer_lr(opt_g, resolve_step_lr({**cfg["train"], "lr": cfg["train"]["lr_g"]}, current_step)); set_optimizer_lr(opt_d, resolve_step_lr({**cfg["train"], "lr": cfg["train"].get("lr_d", cfg["train"]["lr_g"])}, current_step))
            set_requires_grad(generator, False); set_requires_grad(discriminator, True); opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                with torch.no_grad(): fake_for_d = unwrap_model(generator)(condition)
                d_loss = discriminator_hinge_loss(discriminator(condition, target), discriminator(condition, fake_for_d.detach()))
            scaler_d.scale(d_loss).backward(); scaler_d.step(opt_d); scaler_d.update()
            set_requires_grad(generator, True); set_requires_grad(discriminator, False); opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                fake = generator(condition); adv = generator_hinge_loss(unwrap_model(discriminator)(condition, fake)); d_l1 = F.l1_loss(fake, target); g_loss = adv + 100.0 * d_l1
            scaler_g.scale(g_loss).backward(); scaler_g.step(opt_g); scaler_g.update(); set_requires_grad(discriminator, True)
            if current_step >= int(cfg["train"].get("ema_start_step", 0)): ema.update(unwrap_model(generator))
            if is_main and current_step % log_every == 0:
                payload = {"step": current_step, "g_loss": tensor_to_float(g_loss), "d_loss": tensor_to_float(d_loss), "adversarial_hinge": tensor_to_float(adv), "d_l1": tensor_to_float(d_l1), "lr_g": float(opt_g.param_groups[0]["lr"]), "lr_d": float(opt_d.param_groups[0]["lr"])}
                print(payload)
                if run is not None: run.log(payload)
            if current_step % validate_every == 0 or current_step == max_steps:
                if dist_ctx["distributed"]: dist.barrier()
                if is_main:
                    metrics = validate_d_only(ema.ema_model, val_loader, device, cfg["dataset"]["val"], normalization); metrics["step"] = float(current_step); print(metrics)
                    if run is not None: run.log(metrics)
                    if metrics["val_d_nmae"] < best:
                        best = metrics["val_d_nmae"]
                        save_checkpoint(checkpoint_dir / "best.pt", {"step": current_step, "model": unwrap_model(generator).state_dict(), "ema_model": ema.ema_model.state_dict(), "discriminator": unwrap_model(discriminator).state_dict(), "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(), "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(), "d_normalization": normalization.to_dict(), "loss_contract": "hinge_cgan_plus_100_l1_d_only", "config": cfg, "best_metric": best, "best_metric_name": "val_d_nmae", "val_metrics": metrics})
                    checkpoint = {"step": current_step, "model": unwrap_model(generator).state_dict(), "ema_model": ema.ema_model.state_dict(), "discriminator": unwrap_model(discriminator).state_dict(), "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(), "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(), "d_normalization": normalization.to_dict(), "loss_contract": "hinge_cgan_plus_100_l1_d_only", "config": cfg, "best_metric": best, "best_metric_name": "val_d_nmae", "val_metrics": metrics}
                    save_checkpoint(checkpoint_dir / "last.pt", checkpoint)
                    if current_step % save_every == 0 or current_step == max_steps:
                        save_checkpoint(checkpoint_dir / f"step_{current_step:06d}.pt", checkpoint)
                    preview = next(iter(val_loader))
                    preview_condition = preview["condition"][0:1].to(device)
                    preview_prediction = ema.ema_model(preview_condition)
                    raw_d, raw_branches, normalized_branches = decode_standardized_d(preview_prediction, preview_condition, cfg["dataset"]["val"], normalization)
                    preview_name = str(preview["name"][0])
                    save_tensor_npy(raw_d[0], sample_dir / f"step_{current_step:06d}_{preview_name}_d.npy")
                    save_tensor_npy(raw_branches[0], sample_dir / f"step_{current_step:06d}_{preview_name}_branches_raw.npy")
                    save_tensor_png(preview_prediction[0], sample_dir / f"step_{current_step:06d}_{preview_name}_d_normalized.png")
                    save_tensor_png(normalized_branches[0, 0:1], sample_dir / f"step_{current_step:06d}_{preview_name}_left.png")
                    save_tensor_png(normalized_branches[0, 1:2], sample_dir / f"step_{current_step:06d}_{preview_name}_right.png")
                    _swanlab_preview(run, current_step, preview_name, preview_prediction[0], normalized_branches[0])
                if dist_ctx["distributed"]: dist.barrier()
    finally:
        cleanup_distributed(dist_ctx["distributed"])


if __name__ == "__main__":
    main()
