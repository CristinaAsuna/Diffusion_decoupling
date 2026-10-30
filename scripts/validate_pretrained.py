#!/usr/bin/env python3
"""Validate a trusted D-only training checkpoint without updating its weights.

Inputs are explicitly supplied single images. This command performs no cohort
search, selection, optimization, or paired scientific evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from consistent_cdm import runtime
from consistent_cdm.utils.checkpointing import load_checkpoint_file


def _optimizer_summary(state):
    optimizer = state.get("optimizer")
    if optimizer is None:
        return {"present": False}
    if not isinstance(optimizer, dict) or not {"state", "param_groups"} <= optimizer.keys():
        raise ValueError("Checkpoint optimizer state has an unsupported structure")
    tensors = [value for entry in optimizer["state"].values() for value in entry.values()
               if isinstance(value, torch.Tensor)]
    finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    if not finite:
        raise ValueError("Checkpoint optimizer contains nonfinite tensors")
    return {"present": True, "parameter_states": len(optimizer["state"]),
            "parameter_groups": len(optimizer["param_groups"]), "tensor_count": len(tensors),
            "all_tensors_finite": finite, "restoration_tested": False}


def _resume_smoke(model, cfg, checkpoint, device):
    """Exercise the native training loader without executing an optimizer step."""
    import copy
    from consistent_cdm.apps.train_research import EMA, load_checkpoint
    resumed = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=float(cfg["train"].get("lr", 1e-4)))
    ema = EMA(resumed, decay=float(cfg["train"].get("ema_decay", .9999)))
    step, _, _, training_state = load_checkpoint(checkpoint, resumed, optimizer, ema, torch.device(device))
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if name in state and state[name].shape != parameter.shape:
                raise ValueError(f"Restored optimizer {name} shape disagrees with parameter")
    return {"restoration_tested":True, "native_loader":"apps.train_research.load_checkpoint",
            "restored_step":step, "optimizer_parameter_states":len(optimizer.state),
            "ema_restored":True, "optimizer_updated":False,
            "rng_state_available":bool(training_state.get("rng_state") or training_state.get("rng_state_by_rank")),
            "scope":"state_restore_only_no_training_or_bitwise_continuation_claim"}


@torch.no_grad()
def _synthetic_loss(model, family, full, device, seed):
    """Check the native forward objective using synthetic standardized D only."""
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    was_training = model.training
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        condition = torch.from_numpy(np.clip(full, 0, 6).copy())[None,None].to(device)/3 - 1
        target = torch.randn_like(condition)*.1
        try:
            model.eval()
            if family in ("diffusion", "flow_matching"):
                loss = model(target, condition)
            else:
                loss = torch.nn.functional.l1_loss(model(condition), target)
        finally:
            model.train(was_training)
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise ValueError("Native forward objective did not return a finite scalar")
    return {"value":float(loss), "target":"synthetic_standardized_D_gaussian_std_0.1",
            "gradients_computed":False, "optimizer_updated":False,
            "purpose":"execution_smoke_only_no_accuracy_claim"}


def validate(args):
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    source = Path(args.input)
    if not source.is_file() or source.suffix.lower() != ".npy":
        raise ValueError("--input must be one raw Full NPY file")
    print(json.dumps({"phase":"check_checkpoint", "checkpoint":str(checkpoint)}),flush=True)
    metadata = load_checkpoint_file(checkpoint, torch.device("cpu"))
    if "ema_model" not in metadata:
        raise ValueError("Validation requires a full checkpoint containing EMA weights")
    training_state = metadata.get("training_state", {})
    summary = {"step":metadata.get("step"), "optimizer":_optimizer_summary(metadata),
               "rng_state_present":bool(metadata.get("rng_state") or training_state.get("rng_state") or training_state.get("rng_state_by_rank"))}
    del metadata
    started = time.perf_counter()
    model,cfg,family,scale,weights = runtime.load_predictor(checkpoint, device=args.device)
    print(json.dumps({"phase":"strict_ema_loaded", "family":family,"step":summary["step"]}),flush=True)
    if weights != "ema_model":
        raise ValueError("Validation requires strict EMA checkpoint loading")
    size = int(cfg["dataset"]["train"]["image_size"][0])
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else cfg.get("calibration")
    inputs = [("synthetic", source, "raw")]
    if args.real_input:
        inputs.append(("real", Path(args.real_input), "quantile"))
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for name,path,mapping in inputs:
        print(json.dumps({"phase":"sampling", "input_kind":name,"sampler":cfg.get("validation",{}).get("sampler")}),flush=True)
        full,transform = runtime.load_input(path, size=size, mapping=mapping, calibration=calibration)
        begin = time.perf_counter()
        d,left,right = runtime.predict(model,family,cfg,scale,full,seed=args.seed)
        if any(a.shape != full.shape or not np.isfinite(a).all() for a in (d,left,right)):
            raise ValueError("Predictions must match Full shape and contain finite values")
        sum_error = float(np.max(np.abs(left+right-full)))
        diff_error = float(np.max(np.abs((left-right)/2-d)))
        tolerance = max(1.,float(np.max(np.abs(full))),float(np.max(np.abs(d))))*1e-6
        if max(sum_error,diff_error) > tolerance:
            raise ValueError("Hard decoder identities failed")
        row = {"name":name, "input":str(path.resolve()), "input_sha256":runtime.sha256(path),
               "transform":transform, "shape":list(full.shape), "all_finite":True,
               "sum_max_abs_error":sum_error, "difference_max_abs_error":diff_error,
               "decoder_tolerance":tolerance, "inference_seconds":time.perf_counter()-begin}
        if args.repeat:
            print(json.dumps({"phase":"repeat_sampling","input_kind":name}),flush=True)
            repeated = runtime.predict(model,family,cfg,scale,full,seed=args.seed)
            errors = [float(np.max(np.abs(a-b))) for a,b in zip((d,left,right), repeated)]
            row["same_seed_repeat_max_abs_error"] = max(errors)
            if max(errors) > tolerance:
                raise ValueError("Repeated same-seed inference exceeded numerical tolerance")
        if args.check_loss and name == "synthetic":
            row["native_loss_smoke"] = _synthetic_loss(model,family,full,args.device,args.seed)
        runtime.save_prediction(output/name,full,d,left,right,d_cap=scale)
        runtime.write_json(output/name/"manifest.json",row)
        rows.append(row)
        print(json.dumps({"phase":"image_passed","input_kind":name,"sum_max_abs_error":sum_error}),flush=True)
    if getattr(args,"check_resume",False):
        print(json.dumps({"phase":"native_resume_check"}),flush=True)
        summary["optimizer"]["resume_check"] = _resume_smoke(model,cfg,checkpoint,args.device)
        summary["optimizer"]["restoration_tested"] = True
    report = {"schema_version":1, "checkpoint":str(checkpoint.resolve()),
              "checkpoint_sha256":runtime.sha256(checkpoint), "checkpoint_step":summary["step"],
              "family":family, "weights":weights, "strict_state_dict_load":True,
              "model_parameter_count":sum(p.numel() for p in model.parameters()),
              "device":args.device, "seed":args.seed, "d_scale":scale,
              "solver":cfg.get("validation",{}).get("sampler") if family in ("diffusion","flow_matching") else None,
              "flow_matching":cfg.get("flow_matching") if family == "flow_matching" else None,
              "checkpoint_optimizer":summary["optimizer"], "rng_state_present":summary["rng_state_present"],
              "side_contract":cfg.get("side_contract", "historical_stored_order_unverified"), "decoder":"L=F_raw/2+D; R=F_raw/2-D",
              "scope":"functional_smoke_no_training_or_paired_accuracy_claim", "images":rows,
              "elapsed_seconds":time.perf_counter()-started}
    runtime.write_json(output/"validation.json",report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--input",required=True,help="One raw Full NPY development image")
    parser.add_argument("--output",required=True)
    parser.add_argument("--real-input",help="Optional single real PNG/JPEG/TIFF")
    parser.add_argument("--calibration",help="Train-only intensity reference JSON for real input")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--repeat",action="store_true",help="Repeat each prediction with the same seed")
    parser.add_argument("--check-loss",action="store_true",help="Finite native forward loss on synthetic D; no backward/update")
    parser.add_argument("--check-resume",action="store_true",help="Dry-run native optimizer/model/EMA restoration without training")
    args = parser.parse_args(argv)
    try:
        report = validate(args)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"family":report["family"],"checkpoint_step":report["checkpoint_step"],
                      "images":len(report["images"]),"report":str(Path(args.output)/"validation.json")},indent=2))


if __name__ == "__main__":
    main()
