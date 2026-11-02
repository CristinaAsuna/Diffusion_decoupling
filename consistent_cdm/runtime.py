"""Portable data preparation, family dispatch, and condition-only inference."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.resources
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

FAMILIES = ("diffusion", "flow_matching", "resunet", "pix2pix")
SIDE_CONTRACT = "anatomical_lps_label8_left_v2"
SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_raw(array, size):
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Expected a finite 2D grayscale array")
    return F.interpolate(torch.from_numpy(value.copy())[None, None], size=(size, size),
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def prepare_data(data_root, output, *, seed=1234, train_ratio=.9, split_file=None,
                 image_size=256, variants=("std", "aug"), max_samples=4_000_000):
    """Freeze patient partitions, scale and intensity references using train only."""
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared output already contains files: {output}")
    if not 0 < train_ratio < 1 or image_size < 8 or max_samples < 1:
        raise ValueError("Require 0<train_ratio<1, image_size>=8 and max_samples>0")
    if not variants or len(set(variants)) != len(variants) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in variants):
        raise ValueError("Variants must be unique, nonempty simple names")
    cases = sorted(p.name for p in data_root.iterdir() if p.is_dir() and (p / "projections").is_dir())
    if len(cases) < 2:
        raise ValueError("At least two patient folders with projections/ are required")
    if split_file:
        split = json.loads(Path(split_file).read_text())
        if not {"train", "val"} <= split.keys() or set(split) - {"train", "val", "test"}:
            raise ValueError("Split JSON requires train/val and optional test ID lists")
        selected = [case for ids in split.values() for case in ids]
        if len(selected) != len(set(selected)) or set(selected) != set(cases):
            raise ValueError("Split must assign every patient exactly once without overlap")
    else:
        shuffled = list(cases)
        random.Random(seed).shuffle(shuffled)
        count = max(1, min(len(cases)-1, int(len(cases) * train_ratio)))
        split = {"train": sorted(shuffled[:count]), "val": sorted(shuffled[count:])}
    if not split["train"] or not split["val"]:
        raise ValueError("Train and validation partitions must both be nonempty")
    train_ids = set(split["train"])
    per_view = max(1, max_samples // (len(train_ids) * len(variants)))
    d_samples, f_samples, audit, sources = [], [], [], []
    count = 0
    total = square = 0.
    minimum, maximum = math.inf, -math.inf
    for case in cases:
        manifest = data_root / case / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing anatomical provenance: {manifest}; generate data with cbct-preprocess")
        metadata = json.loads(manifest.read_text())
        contract = metadata.get("side_contract", {})
        version = contract.get("version") if isinstance(contract, dict) else contract
        if version != SIDE_CONTRACT or metadata.get("status", "complete") != "complete":
            raise ValueError(f"Incomplete case or unsupported anatomical side contract in {manifest}: {version}")
        sources.append({"path": str(manifest.relative_to(data_root)), "sha256": sha256(manifest)})
        for variant in variants:
            paths = [data_root / case / "projections" / f"{variant}_{branch}.npy"
                     for branch in ("full", "left", "right")]
            if not all(p.is_file() for p in paths):
                raise FileNotFoundError(f"Incomplete F/L/R triplet: {case}/{variant}")
            original = [np.load(p, allow_pickle=False) for p in paths]
            if len({a.shape for a in original}) != 1:
                raise ValueError(f"Mismatched projection shapes: {case}/{variant}")
            full, left, right = [resize_raw(a, image_size) for a in original]
            residual = float(np.mean(np.abs(full-left-right)) / max(float(np.mean(np.abs(full))), 1e-8))
            if residual > 1e-5:
                raise ValueError(f"F != L+R for {case}/{variant}: sum NMAE={residual:g}")
            audit.append({"case": case, "variant": variant, "sum_nmae": residual,
                          "condition_saturation_fraction": float(np.mean((full < 0) | (full > 6)))})
            sources.extend({"path": str(p.relative_to(data_root)), "sha256": sha256(p)} for p in paths)
            if case not in train_ids:
                continue
            d = ((left-right)/2).astype(np.float64)
            count += d.size
            total += float(d.sum()); square += float(np.square(d).sum())
            minimum = min(minimum, float(d.min())); maximum = max(maximum, float(d.max()))
            stride = max(1, int(math.ceil(d.size/per_view)))
            d_samples.append(np.abs(d).ravel()[::stride][:per_view])
            f_samples.append(full.ravel()[::stride][:per_view])
    values = np.concatenate(d_samples)
    scale = float(np.quantile(values, .95))
    std = math.sqrt(max(square/count - (total/count)**2, 0.))
    if scale <= 1e-8 or std <= 1e-8:
        raise ValueError("Training D has near-zero scale/variance")
    normalization = dict(mean=total/count, std=std, scale=scale, scale_method="abs_percentile",
                         scale_quantile=.95, scale_sample_count=len(values), count=count,
                         raw_min=minimum, raw_max=maximum)
    calibration = {"raw_full_quantiles": np.percentile(np.concatenate(f_samples), [1,50,99]).tolist(),
                   "source": "train_patients_only", "train_case_ids": sorted(train_ids),
                   "mapping": "piecewise_linear_p1_p50_p99", "physical_calibration": False}
    identity = dict(split=split, normalization=normalization, sources=sources,
                    image_size=image_size, variants=list(variants), side_contract=SIDE_CONTRACT)
    prepared_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    prepared = dict(schema_version=1, prepared_id=prepared_id, data_root=str(data_root),
                    seed=seed, image_size=image_size, variants=list(variants), split=split,
                    d_normalization=normalization, calibration=calibration,
                    side_contract=SIDE_CONTRACT, sources=sources, audit=audit)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prepared.json", prepared)
    write_json(output / "split.json", split)
    write_json(output / "d_normalization.json", normalization)
    write_json(output / "calibration.json", calibration)
    return prepared


def read_prepared(path):
    path = Path(path)
    return json.loads((path / "prepared.json" if path.is_dir() else path).read_text())


def verify_prepared(prepared):
    """Reject stale statistics if source triplets changed after preparation."""
    if prepared.get("provenance_mode") == HISTORICAL_CONTRACT:
        return _verify_historical(prepared)
    if prepared.get("side_contract") != SIDE_CONTRACT:
        raise ValueError("Prepared data has an unsupported anatomical side contract")
    identity = dict(split=prepared["split"], normalization=prepared["d_normalization"],
                    sources=prepared["sources"], image_size=prepared["image_size"],
                    variants=prepared["variants"], side_contract=prepared["side_contract"])
    actual_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if actual_id != prepared["prepared_id"]:
        raise ValueError("Prepared metadata changed; prepare a new dataset version")
    root = Path(prepared["data_root"])
    for source in prepared["sources"]:
        path = root / source["path"]
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Prepared source changed; prepare a new dataset version: {path}")



HISTORICAL_CONTRACT = "historical_stored_order_unverified"

def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _historical_contract(cfg):
    """Resolve the *actual* embedded dataset enumeration without loading arrays."""
    from consistent_cdm.baseline.common import build_dataset
    result = {"split": {}, "samples": {}}
    for partition in ("train", "val"):
        dcfg = cfg["dataset"][partition]
        if dcfg.get("type") != "case_folders":
            raise ValueError("Historical diagnostics require case-folder provenance")
        if dcfg.get("normalize") != "fixed_range_m11" or list(dcfg.get("value_range", [])) != [0, 6]:
            raise ValueError("Historical diagnostic normalization must be fixed [0,6] to [-1,1]")
        if list(dcfg.get("clip_range", [])) != [0, 6]:
            raise ValueError("Historical diagnostics require the embedded [0,6] clipping contract")
        dataset = build_dataset(dcfg)
        cases = sorted({sample.name.split("__")[0] for sample in dataset.samples})
        if not cases or any(not re.fullmatch(r"Bone_[0-9]+", case) for case in cases):
            raise ValueError("Historical diagnostics accept Bone patients only; frozen SY is forbidden")
        variants = list(dcfg.get("variants") or [])
        if not variants or set(sample.name for sample in dataset.samples) != {c+"__"+v for c in cases for v in variants}:
            raise ValueError("Incomplete historical patient/variant enumeration")
        result["split"][partition] = cases
        result["samples"][partition] = dataset.samples
        size = dcfg.get("image_size")
        if not isinstance(size, (list,tuple)) or len(size) != 2 or size[0] != size[1]:
            raise ValueError("Historical diagnostics require square image_size")
        if partition == "train":
            result.update(image_size=int(size[0]), variants=variants,
                          data_root=str(Path(dcfg["case_root"]).resolve()))
        elif (result["image_size"] != size[0] or result["variants"] != variants or
              result["data_root"] != str(Path(dcfg["case_root"]).resolve())):
            raise ValueError("Embedded train/val interfaces disagree")
    if set(result["split"]["train"]) & set(result["split"]["val"]):
        raise ValueError("Historical patient partitions overlap")
    return result


def _historical_id(prepared):
    return _json_hash({k: v for k, v in prepared.items() if k != "prepared_id"})


def prepare_historical_diagnostics(checkpoint, calibration_path, output):
    """Bind a read-only development adapter to one immutable native checkpoint.

    Does not relabel anatomy, fit statistics, copy data or authorize training.
    Only held-out projection arrays are hashed; embedded train enumeration is
    inspected to bind the historical split, without loading training arrays.
    """
    from consistent_cdm.utils.checkpointing import load_checkpoint_file
    state = load_checkpoint_file(checkpoint, torch.device("cpu"))
    cfg = state["config"]
    if cfg.get("prepared_id"):
        raise ValueError("Use the original prepared dataset for modern checkpoints")
    historical = _historical_contract(cfg)
    normalization = state.get("d_normalization") or cfg["dataset"]["train"].get("d_only_normalization")
    if not normalization or not math.isfinite(float(normalization["scale"])) or float(normalization["scale"]) <= 0:
        raise ValueError("Missing embedded train-only D scale")
    calibration_path = Path(calibration_path).resolve()
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("scope") != "Bone_train_only" and calibration.get("source") != "train_patients_only":
        raise ValueError("Calibration must have explicit train-only provenance")
    if "train_case_ids" in calibration and sorted(calibration["train_case_ids"]) != historical["split"]["train"]:
        raise ValueError("Calibration patient provenance differs from embedded train split")
    ref = np.asarray(calibration.get("raw_full_quantiles", []))
    if ref.shape != (3,) or not np.isfinite(ref).all() or np.any(np.diff(ref) <= 1e-8):
        raise ValueError("Invalid train-only calibration")
    root = Path(historical["data_root"])
    samples = historical.pop("samples")
    sources, projection_templates = [], None
    for sample in samples["val"]:
        case,variant = sample.name.split("__")
        paths = [sample.condition_path, *sample.target_paths]
        if len(paths) != 3:
            raise ValueError("Historical adapter requires ordered Full/left/right")
        expected = [f"{variant}_{name}.npy" for name in ("full", "left", "right")]
        if [path.name for path in paths] != expected:
            raise ValueError("Historical stored target order is not explicit Full/left/right")
        templates = [str(path.relative_to(root / case)).replace(variant,"{variant}",1) for path in paths]
        if projection_templates is None: projection_templates = templates
        if projection_templates != templates: raise ValueError("Nonuniform historical projection layout")
        sources.extend(dict(path=str(path.relative_to(root)), sha256=sha256(path)) for path in paths)
    prepared = dict(schema_version=2, provenance_mode=HISTORICAL_CONTRACT, side_contract=HISTORICAL_CONTRACT,
                    **historical, projection_templates=projection_templates,
                    preprocessing="historical_normalize_before_resize", d_normalization=normalization,
                    calibration=calibration, sources=sources,
                    source_checkpoint=dict(path=str(Path(checkpoint).resolve()), sha256=sha256(checkpoint)),
                    embedded_dataset_sha256=_json_hash(cfg["dataset"]),
                    metadata_sources=[dict(path=str(calibration_path), sha256=sha256(calibration_path))],
                    scope="diagnostics_only_no_training_no_frozen_test",
                    quantile_protocol="raw_resize_then_per_image_quantile_mapping_not_historical_raw")
    prepared["prepared_id"] = _historical_id(prepared)
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Prepared output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prepared.json", prepared)
    return prepared


def _verify_historical(prepared):
    if prepared.get("schema_version") != 2 or prepared.get("side_contract") != HISTORICAL_CONTRACT:
        raise ValueError("Unsupported historical diagnostic schema")
    if prepared.get("prepared_id") != _historical_id(prepared):
        raise ValueError("Historical prepared metadata changed")
    if prepared.get("scope") != "diagnostics_only_no_training_no_frozen_test":
        raise ValueError("Historical adapter is diagnostics only")
    split = prepared["split"]
    if set(split) != {"train", "val"} or any(not split[k] for k in split):
        raise ValueError("Historical train/val partitions must be explicit")
    ids = split["train"] + split["val"]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"Bone_[0-9]+", c) for c in ids):
        raise ValueError("Overlapping or frozen/non-Bone patients in historical adapter")
    root = Path(prepared["data_root"])
    expected = {str(path.relative_to(root)) for case in split["val"] for variant in prepared["variants"]
                for path in projection_paths(prepared, case, variant)}
    if expected != {entry["path"] for entry in prepared["sources"]}:
        raise ValueError("Historical source manifest does not cover the validation cohort")
    for item in prepared["sources"]:
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Historical prepared source changed: {path}")
    for item in prepared["metadata_sources"]:
        if not Path(item["path"]).is_file() or sha256(item["path"]) != item["sha256"]:
            raise ValueError("Historical calibration source changed")


def verify_diagnostic_checkpoint(prepared, cfg, checkpoint):
    if prepared.get("provenance_mode") != HISTORICAL_CONTRACT:
        if cfg.get("prepared_id") != prepared["prepared_id"]:
            raise ValueError("Checkpoint and prepared dataset identifiers differ")
        return
    binding = prepared["source_checkpoint"]
    if sha256(checkpoint) != binding["sha256"] or _json_hash(cfg["dataset"]) != prepared["embedded_dataset_sha256"]:
        raise ValueError("Historical checkpoint or embedded dataset binding differs")
    actual = _historical_contract(cfg)
    for key in ("split", "image_size", "variants", "data_root"):
        if actual[key] != prepared[key]: raise ValueError(f"Historical checkpoint disagrees with {key}")
    norm = cfg["dataset"]["train"].get("d_only_normalization")
    if not norm or float(norm["scale"]) != float(prepared["d_normalization"]["scale"]):
        raise ValueError("Historical checkpoint disagrees with train-only scale")


def projection_paths(prepared, case, variant):
    if case not in prepared["split"]["val"]:
        raise ValueError("Diagnostic projection access requires a declared validation patient")
    templates = prepared.get("projection_templates", ["projections/{variant}_full.npy",
                           "projections/{variant}_left.npy", "projections/{variant}_right.npy"])
    return [Path(prepared["data_root"])/case/t.format(variant=variant) for t in templates]


def diagnostic_predict(model, family, cfg, scale, full_source, prepared, *, mapping="raw", seed=1234):
    size = prepared["image_size"]
    raw = resize_raw(full_source, size)
    mapped,_ = map_intensity(raw, mapping, prepared["calibration"])
    normalized = None
    if mapping == "raw" and prepared.get("preprocessing") == "historical_normalize_before_resize":
        # Reproduce the historical normalized tensor exactly; avoid an inverse
        # affine followed by another affine, which adds rounding differences.
        normalized = resize_raw(np.clip(np.asarray(full_source, np.float32),0,6)/3-1, size)
    return predict(model, family, cfg, scale, mapped, seed=seed, normalized_condition=normalized)


def recipe(family, path=None):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if path:
        from consistent_cdm.utils.config import load_config
        return load_config(path)
    resource = importlib.resources.files("consistent_cdm").joinpath("recipes", f"{family}.yaml")
    return yaml.safe_load(resource.read_text())


def training_config(family, prepared, *, config=None, tiny=False):
    if prepared.get("provenance_mode") == HISTORICAL_CONTRACT:
        raise ValueError("Historical diagnostic prepared data cannot be used for training")
    cfg = copy.deepcopy(recipe(family, config))
    cfg["family"] = family
    cfg["prepared_id"] = prepared["prepared_id"]
    cfg["side_contract"] = SIDE_CONTRACT
    cfg["calibration"] = prepared["calibration"]
    cfg["seed"] = prepared["seed"]
    size = prepared["image_size"]
    for partition in ("train", "val"):
        cfg.setdefault("dataset", {})[partition] = dict(
            type="case_folders", case_root=prepared["data_root"], case_ids=prepared["split"][partition],
            split=None, variants=prepared["variants"], condition_template="projections/{variant}_full.npy",
            target_templates=["projections/{variant}_left.npy", "projections/{variant}_right.npy"],
            target_mode="dual", image_size=[size,size], normalize="fixed_range_m11",
            value_range=[0.,6.], clip_range=[0.,6.], d_only_normalization=prepared["d_normalization"])
    cfg["d_normalization"] = prepared["d_normalization"]
    cfg.setdefault("output", {})["group"] = family
    cfg["data_governance"] = {}
    if family in ("diffusion", "flow_matching"):
        cfg["model"]["image_size"] = size
        cfg["diffusion"]["target_parameterization"] = dict(type="d_only", difference_scale=prepared["d_normalization"]["scale"],
                                                         already_standardized=True, clamp_decoded=False)
        cfg["generative"] = {"type": family}
    else:
        generator = cfg["model"].get("generator", cfg["model"])
        generator["image_size"] = size
    if tiny:
        if family in ("diffusion", "flow_matching"):
            cfg["model"].update(inner_channel=32, channel_mults=[1,2], res_blocks=1, attn_res=[],
                                num_head_channels=8, dropout=0.)
            cfg["diffusion"]["beta_schedule"]["n_timestep"] = 32
            cfg["validation"]["sampler"]["steps"] = 2
            if family == "flow_matching": cfg["flow_matching"]["steps"] = 2
        else:
            generator.update(base_channels=8, channel_mults=[1,2], blocks_per_level=1,
                             attention_resolutions=[], attention_heads=1, norm_groups=4, use_se=False, dropout=0.)
            if family == "pix2pix":
                cfg["model"]["discriminator"].update(base_channels=8, num_layers=2, num_scales=1, spectral_norm=False)
        cfg["train"].update(batch_size=1, val_batch_size=1, num_workers=0, validate_every_steps=1,
                            save_every_steps=1, log_every_steps=1, ema_start_step=0, amp=False)
        cfg["train"]["lr_schedule"] = {"enabled": False}
        cfg["train"]["early_stop"] = {"enabled": False}
        cfg.setdefault("validation", {})["max_batches"] = 1
    return cfg


def train(args):
    prepared = read_prepared(args.prepared)
    verify_prepared(prepared)
    cfg = training_config(args.family, prepared, config=args.config, tiny=args.tiny)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run-id must be a single safe directory name")
    runs_root = Path(os.environ.get("CONSISTENT_CDM_RUNS_ROOT", "runs")).resolve()
    run = runs_root / args.family / args.run_id
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError(f"Run ID already exists; use --resume or a new ID: {run}")
    if args.resume:
        from consistent_cdm.utils.checkpointing import load_checkpoint_file
        saved = load_checkpoint_file(args.resume, torch.device("cpu"))["config"]
        for field in ("prepared_id", "family", "model"):
            if saved.get(field) != cfg.get(field):
                raise ValueError(f"Resume checkpoint disagrees with requested {field}")
    if args.max_steps is not None: cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None: cfg["train"]["batch_size"] = args.batch_size
    if cfg["train"]["max_steps"] < 1 or cfg["train"]["batch_size"] < 1:
        raise ValueError("max-steps and batch-size must be positive")
    if args.device: cfg["device"] = args.device
    cfg.setdefault("logging", {})["experiment_name"] = args.run_id
    cfg["logging"].update(use_swanlab=False, use_tensorboard=False) if args.tiny else None
    run.mkdir(parents=True, exist_ok=True)
    resolved = run / "recipe.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(run / "prepared.json", prepared)
    modules = {"diffusion":"consistent_cdm.apps.train_research", "flow_matching":"consistent_cdm.apps.train_research",
               "resunet":"consistent_cdm.baseline.train_d_only", "pix2pix":"consistent_cdm.baseline.train_pix2pix_d_only"}
    command = [sys.executable, "-m", modules[args.family], "--config", str(resolved), "--experiment-name", args.run_id]
    if args.resume: command.extend(["--resume", args.resume])
    env = dict(os.environ, CONSISTENT_CDM_RUNS_ROOT=str(runs_root))
    telemetry = run / "telemetry"
    telemetry.mkdir(exist_ok=True)
    log_path = telemetry / "train.log"
    print(f"Training log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "resume": args.resume}) + "\n")
        log.flush()
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    return run


def load_predictor(checkpoint, *, device="cpu", family=None, config=None):
    from consistent_cdm.utils.checkpointing import load_checkpoint_file, select_model_state
    from consistent_cdm.factories import build_generative_model, build_deterministic_generator
    state = load_checkpoint_file(checkpoint, torch.device(device))
    cfg = state.get("config")
    if config:
        from consistent_cdm.utils.config import load_config
        cfg = load_config(config)
    if cfg is None:
        raise ValueError("Checkpoint must embed config, or supply --config")
    declared_family = cfg.get("family")
    if declared_family is None and "diffusion" in cfg:
        # Original native checkpoints predate the unified family field. The
        # model factory already interprets this embedded generative setting;
        # derive dispatch from the same setting without rewriting the recipe.
        from consistent_cdm.contracts import TargetContract
        from consistent_cdm.factories import generative_run_group
        target_cfg = cfg["diffusion"].get("target_parameterization") or cfg.get("target_parameterization")
        contract = TargetContract.from_config(target_cfg, raw_value_scale=1.0)
        if contract.is_d_only:
            declared_family = generative_run_group(cfg)
    selected_family = family or declared_family
    if selected_family not in FAMILIES:
        raise ValueError("Checkpoint lacks a supported family; supply --family explicitly")
    if declared_family and selected_family != declared_family:
        raise ValueError("Requested family disagrees with checkpoint config")
    norm = state.get("d_normalization") or cfg["dataset"]["train"].get("d_only_normalization")
    if not norm or not math.isfinite(float(norm["scale"])) or float(norm["scale"]) <= 0:
        raise ValueError("Checkpoint is missing a valid train-only D scale")
    if selected_family in ("diffusion", "flow_matching"):
        model, contract = build_generative_model(cfg, condition_channels=1, dataset_target_channels=1,
                                                 image_size=int(cfg["model"]["image_size"]))
        if not contract.is_d_only: raise ValueError("Only D-only checkpoints are supported")
        if not math.isclose(contract.difference_scale, float(norm["scale"]), rel_tol=1e-6):
            raise ValueError("Checkpoint target scale disagrees with normalization")
    else:
        generator_cfg = cfg["model"].get("generator", cfg["model"])
        model = build_deterministic_generator(generator_cfg, condition_channels=1, target_channels=1)
    weights, weight_name = select_model_state(state, "ema")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model, cfg, selected_family, float(norm["scale"]), weight_name


@torch.no_grad()
def predict(model, family, cfg, scale, full_raw, *, seed=1234, normalized_condition=None):
    device = next(model.parameters()).device
    condition = (torch.from_numpy(np.clip(full_raw, 0, 6).copy())[None,None].to(device) / 3 - 1
                 if normalized_condition is None else
                 torch.from_numpy(np.asarray(normalized_condition, np.float32).copy())[None,None].to(device))
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    if family in ("diffusion", "flow_matching"):
        from consistent_cdm.apps.train_research import run_sampler
        value = run_sampler(model, condition, cfg["validation"]["sampler"])
    else:
        value = model(condition)
    d = value[0,0].detach().cpu().numpy().astype(np.float32) * scale
    if not np.isfinite(d).all(): raise ValueError("Model produced nonfinite D")
    return d, full_raw / 2 + d, full_raw / 2 - d


def map_intensity(full, mapping, calibration):
    """Apply the raw or train-only p1/p50/p99 condition mapping."""
    if mapping == "raw": return full, None
    if mapping != "quantile": raise ValueError(f"Unsupported mapping {mapping}")
    if not calibration: raise ValueError("Quantile mapping needs train-only calibration")
    reference = np.asarray(calibration["raw_full_quantiles"], dtype=np.float32)
    source_quantiles = np.percentile(full, [1,50,99])
    if reference.shape != (3,) or not np.isfinite(reference).all() or np.any(np.diff(reference) <= 1e-8):
        raise ValueError("Calibration requires increasing finite p1/p50/p99")
    if np.any(np.diff(source_quantiles) <= 1e-8):
        raise ValueError("Input has degenerate intensity quantiles")
    return np.interp(full, source_quantiles, reference).astype(np.float32), source_quantiles


def load_input(path, *, size, mapping=None, calibration=None, nose_direction="right", invert=False, crop=None):
    path = Path(path)
    raw_npy = path.suffix.lower() == ".npy"
    if raw_npy:
        source = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            source = np.asarray(image)
        if source.ndim == 3:
            source = np.asarray(source, dtype=np.float32)[..., :3] @ np.array([.299,.587,.114], dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError(f"Expected finite grayscale input: {path}")
    original_shape = list(source.shape)
    if crop:
        x0,y0,x1,y1 = crop
        if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
            raise ValueError("Crop must lie inside input image")
        source = source[y0:y1,x0:x1]
    if nose_direction == "left": source = source[:, ::-1]
    if invert: source = float(source.max()) + float(source.min()) - source
    full = resize_raw(source, size)
    mapping = mapping or ("raw" if raw_npy else "quantile")
    full, source_quantiles = map_intensity(full, mapping, calibration)
    return full, dict(original_shape=original_shape, output_shape=list(full.shape), mapping=mapping,
                      source_quantiles=None if source_quantiles is None else source_quantiles.tolist(),
                      nose_direction=nose_direction, invert=invert, crop=crop,
                      condition_saturation_fraction=float(np.mean((full < 0) | (full > 6))))


def save_prediction(output, full, d, left, right, *, d_cap):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    tiles = []
    for name,array,maximum in (("full",full,6.),("half",full/2,3.),("left",left,3.),("right",right,3.)):
        if name != "half": np.save(output / f"{name}.npy", array.astype(np.float32))
        tiles.append((name,Image.fromarray(np.clip(array/maximum*255,0,255).astype(np.uint8)).convert("RGB")))
    np.save(output / "d.npy", d.astype(np.float32))
    normalized = np.clip(d/max(d_cap,1e-8), -1, 1)
    color = np.stack([np.clip(normalized,0,1),1-np.abs(normalized),np.clip(-normalized,0,1)],-1)
    tiles.append(("D",Image.fromarray((color*255).astype(np.uint8))))
    canvas = Image.new("RGB", (256*len(tiles),280), "white")
    draw = ImageDraw.Draw(canvas)
    for index,(name,tile) in enumerate(tiles):
        canvas.paste(tile.resize((256,256)),(index*256,0)); draw.text((index*256+8,260),name,fill="black")
    canvas.save(output / "preview.png")


def infer(args):
    model,cfg,family,scale,weights = load_predictor(args.checkpoint, device=args.device, family=args.family, config=args.config)
    size = int(cfg["dataset"]["train"]["image_size"][0])
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else cfg.get("calibration")
    input_path, output = Path(args.input), Path(args.output)
    paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in SUFFIXES) if input_path.is_dir() else [input_path]
    if not paths: raise FileNotFoundError("No supported input images")
    # Include suffix in output IDs so foo.npy and foo.png cannot collide.
    rows = []
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        full,transform = load_input(path,size=size,mapping=args.mapping,calibration=calibration,
                                   nose_direction=args.nose_direction,invert=args.invert,crop=args.crop)
        d,left,right = predict(model,family,cfg,scale,full,seed=args.seed)
        name = path.name
        save_prediction(output/name,full,d,left,right,d_cap=scale)
        rows.append(dict(input=str(path.resolve()),input_sha256=sha256(path),output=name,transform=transform))
    manifest = dict(schema_version=1,family=family,checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=sha256(args.checkpoint),weights=weights,seed=args.seed,d_scale=scale,
                    sampler=cfg.get("validation",{}).get("sampler") if family in ("diffusion","flow_matching") else None,
                    side_contract=cfg.get("side_contract", "historical_stored_order_unverified"),decoder="L=F_raw/2+D; R=F_raw/2-D; no clipping",
                    calibration=calibration,display="F:[0,6]; half/L/R:[0,3]; D:symmetric train scale",
                    scope="condition_only_no_paired_accuracy_metrics",images=rows)
    write_json(output/"manifest.json",manifest)
    return manifest


def evaluate(args):
    from consistent_cdm.evaluation.data_metrics import branch_metrics, flatten_metrics
    prepared = read_prepared(args.prepared)
    verify_prepared(prepared)
    model,cfg,family,scale,weights = load_predictor(args.checkpoint, device=args.device)
    verify_diagnostic_checkpoint(prepared, cfg, args.checkpoint)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; size=prepared["image_size"]
    for case in prepared["split"]["val"]:
        for variant in prepared["variants"]:
            originals=[np.load(path,allow_pickle=False) for path in projection_paths(prepared,case,variant)]
            full,left,right=[resize_raw(value,size) for value in originals]
            d,_,_=diagnostic_predict(model,family,cfg,scale,originals[0],prepared,seed=args.seed)
            lp,rp=full/2+d,full/2-d
            print(f"evaluate {case}/{variant}",flush=True)
            save_prediction(output/f"{case}__{variant}",full,d,lp,rp,d_cap=scale)
            for method,l,r in ((family,lp,rp),("full_half",full/2,full/2)):
                metrics=flatten_metrics(branch_metrics(l,r,left,right,full=full,data_range=6.))
                rows.append(dict(case=case,variant=variant,method=method,**metrics))
    with (output/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={}
    for method in (family,"full_half"):
        selected=[r for r in rows if r["method"]==method]
        # Average variants within each patient before averaging patients.
        summary[method]={key:float(np.mean([np.mean([r[key] for r in selected if r["case"]==case])
                                            for case in prepared["split"]["val"]]))
                         for key in selected[0] if key not in ("case","variant","method") and isinstance(selected[0][key],(float,int))}
    # Exact predictions may have +inf PSNR; JSON represents those as explicit strings.
    def finite(value):
        if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value,float) and not math.isfinite(value): return str(value)
        return value
    write_json(output/"summary.json",finite(dict(patient_macro=summary,prepared_id=prepared["prepared_id"],
               checkpoint_sha256=sha256(args.checkpoint),weights=weights,seed=args.seed,cohort="validation_only",
               side_contract=prepared["side_contract"],preprocessing=prepared.get("preprocessing","raw_resize_then_normalize"))))
    return summary


STRESS_SUMMARY_KEYS = ("asymmetry.nmae", "asymmetry.identity", "asymmetry.signed_gain", "asymmetry.energy_ratio",
                       "asymmetry_rich.nmae", "asymmetry_rich.identity", "asymmetry_rich.energy_ratio",
                       "side_average.psnr", "side_average.ssim", "d_rms_over_scale", "condition_saturation_fraction")


def stress_test(args):
    """Score D on validation patients under controlled condition degradations.

    L/R are decoded with the clean (oracle) line integral so the metrics isolate
    how each degradation changes the D estimate; ``d_rms_over_scale`` needs no
    target and is directly comparable with real-image inference.
    """
    from consistent_cdm.evaluation.data_metrics import branch_metrics, flatten_metrics
    from consistent_cdm.evaluation.degradations import DEGRADATIONS, degrade
    prepared = read_prepared(args.prepared)
    verify_prepared(prepared)
    model,cfg,family,scale,weights = load_predictor(args.checkpoint, device=args.device)
    verify_diagnostic_checkpoint(prepared, cfg, args.checkpoint)
    unknown = set(args.operators or ()) - set(DEGRADATIONS)
    if unknown: raise ValueError(f"Unknown degradations: {sorted(unknown)}")
    operators = args.operators or list(DEGRADATIONS)
    calibration = prepared["calibration"]
    if args.max_cases is not None and args.max_cases < 1: raise ValueError("max-cases must be positive")
    cases = prepared["split"]["val"][:args.max_cases] if args.max_cases else prepared["split"]["val"]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output is not empty: {output}")
    (output/"conditions").mkdir(parents=True)
    size = prepared["image_size"]
    settings = [("none", 0.)] + [(name, level) for name in operators for level in DEGRADATIONS[name].levels]
    rows = []; skipped = []
    for index, case in enumerate(cases):
        for variant in prepared["variants"]:
            raw = [np.load(path, allow_pickle=False) for path in projection_paths(prepared,case,variant)]
            for name, level in settings:
                if name == "none":
                    full, left, right = raw
                else:
                    full, left, right = degrade(name, level, *raw, model_size=size, seed=args.seed)
                left, right = resize_raw(left, size), resize_raw(right, size)
                oracle = left + right
                try:
                    resized = resize_raw(full,size)
                    if args.mapping == "quantile" and float(np.ptp(resized)) <= max(1e-7, float(np.max(np.abs(resized)))*1e-6):
                        raise ValueError("Input has degenerate intensity quantiles (information erased)")
                    condition, _ = map_intensity(resized, args.mapping, calibration)
                except ValueError as error:
                    if "degenerate intensity quantiles" not in str(error): raise
                    skipped.append(dict(case=case,variant=variant,operator=name,level=level,reason=str(error)))
                    continue
                d,_,_ = diagnostic_predict(model,family,cfg,scale,full,prepared,mapping=args.mapping,seed=args.seed)
                if getattr(args,"save_arrays",False):
                    item=output/"arrays"/case/variant/f"{name}_{level:g}"
                    item.mkdir(parents=True)
                    for key,value in (("condition",condition),("d",d),("gt_d",(left-right)/2),("oracle_full",oracle)):
                        np.save(item/f"{key}.npy",np.asarray(value,np.float32))
                if index == 0 and variant == prepared["variants"][0]:
                    Image.fromarray(np.clip(condition/6*255, 0, 255).astype(np.uint8)).save(
                        output/"conditions"/f"{name}_{level:g}.png")
                predictions = [(family, oracle/2+d, oracle/2-d)]
                if name == "none": predictions.append(("full_half", oracle/2, oracle/2))
                for method, lp, rp in predictions:
                    metrics = flatten_metrics(branch_metrics(lp, rp, left, right, full=oracle, data_range=6.))
                    metrics["d_rms_over_scale"] = float(np.sqrt(np.mean(np.square((lp-rp)/2)))/scale)
                    metrics["condition_saturation_fraction"] = float(np.mean((condition < 0) | (condition > 6)))
                    rows.append(dict(case=case, variant=variant, method=method, operator=name, level=level,
                                     **{k: v for k, v in metrics.items() if not k.startswith("full_half_baseline")}))
            print(f"stress {index+1}/{len(cases)} {case}/{variant} mapping={args.mapping}",flush=True)
    write_json(output/"skipped.json",skipped)
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for method, name, level in dict.fromkeys((r["method"], r["operator"], r["level"]) for r in rows):
        selected = [r for r in rows if (r["method"], r["operator"], r["level"]) == (method, name, level)]
        # Average variants within each patient before averaging patients.
        means = {key: float(np.nanmean([np.mean([r[key] for r in selected if r["case"] == case]) for case in cases]))
                 for key in STRESS_SUMMARY_KEYS}
        means = {key: value if math.isfinite(value) else str(value) for key, value in means.items()}
        spec = DEGRADATIONS.get(name)
        summary.append(dict(method=method, operator=name, level=level, kind=spec.kind if spec else None,
                            description=spec.description if spec else "clean condition", **means))
    write_json(output/"summary.json", dict(rows=summary, prepared_id=prepared["prepared_id"], mapping=args.mapping,
               checkpoint_sha256=sha256(args.checkpoint), weights=weights, seed=args.seed, d_scale=scale,
               decoder="oracle F (geometrically transformed for geometric controls): L=F/2+D; R=F/2-D", cases=cases, cohort="validation_only",
               side_contract=prepared["side_contract"],skipped_count=len(skipped),save_arrays=bool(getattr(args,"save_arrays",False)),
               preprocessing=(prepared.get("preprocessing","raw_resize_then_normalize") if args.mapping=="raw" else "raw_resize_then_per_image_quantile_mapping")))
    return summary
