from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import SimpleITK as sitk

from .io import strip_nii_suffix


def preflight(config: dict) -> tuple[str, Path]:
    """Check external nnU-Net resources before starting any inference."""
    if not config.get("enabled", False):
        raise ValueError("No segmentation supplied: select the automatic recipe or provide --segmentation")
    executable = shutil.which(str(config.get("executable", "nnUNetv2_predict")))
    if executable is None:
        raise FileNotFoundError("nnUNetv2_predict is unavailable; install the segmentation extra (nnunetv2==2.6.4)")
    root = config.get("results_dir") or os.environ.get("nnUNet_results")
    if not root or not Path(root).is_dir():
        raise FileNotFoundError("Set nnunet.results_dir or nnUNet_results to your external trained-model root")
    dataset = str(config.get("dataset", "015"))
    candidates = list(Path(root).glob(f"Dataset{int(dataset):03d}_*")) if dataset.isdigit() else [Path(root) / dataset]
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise FileNotFoundError(f"Expected one Dataset{dataset}_* directory below {root}; found {len(candidates)}")
    model = candidates[0] / (str(config.get("trainer", "nnUNetTrainerNoMirroring")) + "__" + str(config.get("plans", "nnUNetPlans")) + "__" + str(config.get("configuration", "3d_fullres")))
    required = [model / "plans.json", model / "dataset.json"]
    required += [model / f"fold_{fold}" / str(config.get("checkpoint", "checkpoint_final.pth")) for fold in config.get("folds", ["all"])]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing external nnU-Net model resources: " + ", ".join(missing))
    # Read, but never rewrite, the trained checkpoint's normalization/plans.
    for path in required[:2]:
        with path.open() as stream:
            json.load(stream)
    return executable, model


def model_hashes(model: Path, config: dict) -> dict:
    folds = [str(value) for value in config.get("folds", ["all"])]
    hashes = {}
    for path in [model / "plans.json", model / "dataset.json", *[model / f"fold_{fold}" / str(config.get("checkpoint", "checkpoint_final.pth")) for fold in folds]]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def predict_segmentation(image_path: str | Path, work_dir: str | Path, config: dict) -> Path:
    executable, model = preflight(config)
    image_path, work_dir = Path(image_path).resolve(), Path(work_dir)
    case_id = strip_nii_suffix(image_path).removesuffix("_0000")
    input_dir, output_dir = work_dir / "input", work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_input = input_dir / f"{case_id}_0000.nii.gz"
    # Write a real gzip NIfTI even when the caller supplies an uncompressed .nii.
    sitk.WriteImage(sitk.ReadImage(str(image_path)), str(staged_input), True)
    folds = [str(value) for value in config.get("folds", ["all"])]
    command = [executable, "-i", str(input_dir), "-o", str(output_dir),
        "-d", model.parent.name, "-c", str(config.get("configuration", "3d_fullres")),
        "-tr", str(config.get("trainer", "nnUNetTrainerNoMirroring")),
        "-p", str(config.get("plans", "nnUNetPlans")), "-f", *folds,
        "-chk", str(config.get("checkpoint", "checkpoint_final.pth")),
        *[str(value) for value in config.get("extra_args", [])]]
    env = os.environ.copy()
    env["nnUNet_results"] = str(model.parent.parent.resolve())
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
    (work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (work_dir / "stdout.log").write_text(completed.stdout)
    (work_dir / "stderr.log").write_text(completed.stderr)
    hashes = model_hashes(model, config)
    (work_dir / "resources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError(f"nnU-Net failed with exit code {completed.returncode}; see {work_dir / 'stderr.log'}")
    result = output_dir / f"{case_id}.nii.gz"
    if not result.exists():
        raise FileNotFoundError(f"nnU-Net returned success but did not create {result}")
    return result
