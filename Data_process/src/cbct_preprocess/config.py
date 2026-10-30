from __future__ import annotations

from pathlib import Path
from importlib.resources import files

import yaml

from .side_contract import validate as validate_side_contract


def load_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path is not None else files("cbct_preprocess").joinpath("recipes", "automatic.yaml")
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a YAML mapping")
    required = {
        "side_contract", "labels", "alignment", "split", "attenuation",
        "projection", "nnunet", "runtime",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    projection = config["projection"]
    for key in ("sad_mm", "sid_mm", "detector_pixel_spacing_mm", "output_size", "crop", "variants"):
        if key not in projection:
            raise ValueError(f"Missing projection.{key}")
    if float(projection["sad_mm"]) <= 0 or float(projection["sid_mm"]) <= float(projection["sad_mm"]):
        raise ValueError("projection requires sid_mm > sad_mm > 0")
    for key in ("detector_pixel_spacing_mm", "output_size"):
        values = projection[key]
        if len(values) != 2 or any(float(value) <= 0 for value in values):
            raise ValueError(f"projection.{key} must contain two positive values")
    if projection.get("raw_size") is not None:
        values = projection["raw_size"]
        if len(values) != 2 or any(int(value) <= 0 for value in values):
            raise ValueError("projection.raw_size must be null or two positive integers [H,W]")
    crop = projection["crop"]
    if crop.get("mode") not in {"content", "none", "fixed"}:
        raise ValueError("projection.crop.mode must be content, none, or fixed")
    if crop.get("mode") == "fixed" and len(crop.get("bbox_yxyx", [])) != 4:
        raise ValueError("fixed crop requires projection.crop.bbox_yxyx")
    flip_lr = projection.get("flip_lr", True)
    if not isinstance(flip_lr, bool) and flip_lr != "anatomical_anterior_right":
        raise ValueError(
            "projection.flip_lr must be true, false, or anatomical_anterior_right"
        )
    for key in ("padding_pixels", "trim_pixels"):
        values = crop.get(key, [0, 0, 0, 0])
        if len(values) != 4 or any(int(value) < 0 for value in values):
            raise ValueError(f"projection.crop.{key} must contain four non-negative integers")
    validate_side_contract(config)
    return config
