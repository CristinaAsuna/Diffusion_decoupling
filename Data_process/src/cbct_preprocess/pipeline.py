from __future__ import annotations

import hashlib
import shutil
import traceback
from pathlib import Path

import numpy as np

from .alignment import build_alignment, resample_pair_with_grid, residual_angle_deg
from .drr_geometry import (
    PixelTransform,
    detector_shape_for_volume,
    generate_rays,
    make_camera,
    resolve_horizontal_flip,
)
from .drr_renderer import additivity_metrics, apply_pixel_transform, content_bbox, render_line_integral
from .geometry import ImageGeometry, validate_same_geometry
from .io import image_array, read_image, strip_nii_suffix, write_image
from .manifest import write_json
from .nnunet import predict_segmentation
from .split import complementary_masks, hu_to_mu, split_hu_image
from .side_contract import provenance as side_contract_provenance, validate as validate_side_contract


def _case_seed(case_id: str, global_seed: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def resolve_variants(case_id: str, projection_config: dict) -> list[dict]:
    rng = np.random.default_rng(_case_seed(case_id, int(projection_config.get("global_seed", 1234))))
    resolved = []
    for variant in projection_config["variants"]:
        if "primary_angle_deg" in variant:
            angle = float(variant["primary_angle_deg"])
        else:
            distribution = variant["sample_normal_deg"]
            magnitude = abs(float(rng.normal(distribution["mean"], distribution["std"])))
            angle = magnitude * (float(rng.choice([-1.0, 1.0])) if distribution.get("random_sign", True) else 1.0)
            angle = float(np.clip(angle, -distribution["max_abs"], distribution["max_abs"]))
        resolved.append({"name": str(variant["name"]), "primary_angle_deg": angle})
    return resolved


def _render_variant(
    variant, geometry, arrays_mu, projection_config, case_dir,
):
    spacing_uv = tuple(float(v) for v in projection_config["detector_pixel_spacing_mm"])
    camera = make_camera(
        variant["primary_angle_deg"], 0.0, projection_config["sad_mm"], projection_config["sid_mm"], spacing_uv, (1, 1)
    )
    if projection_config.get("raw_size") is None:
        raw_shape = detector_shape_for_volume(
            geometry, camera, projection_config.get("detector_padding_pixels", [32, 32])
        )
    else:
        raw_shape = tuple(int(v) for v in projection_config["raw_size"])
    camera = make_camera(
        variant["primary_angle_deg"], 0.0, projection_config["sad_mm"], projection_config["sid_mm"],
        spacing_uv, raw_shape,
    )
    flip_lr, orientation = resolve_horizontal_flip(
        projection_config.get("flip_lr", True), geometry, camera
    )
    rays = generate_rays(camera)
    raw = {
        name: render_line_integral(
            volume, rays, geometry.spacing_xyz, raw_shape,
            step_size_mm=float(projection_config["step_size_mm"]),
            chunk_size=int(projection_config["chunk_size"]),
            device=str(projection_config.get("device", "auto")),
        )
        for name, volume in arrays_mu.items()
    }
    crop_config = projection_config["crop"]
    displayed_full = np.fliplr(raw["full"]) if flip_lr else raw["full"]
    if crop_config["mode"] == "content":
        bbox = content_bbox(
            displayed_full,
            crop_config["threshold_fraction"],
            crop_config.get("padding_pixels", [0, 0, 0, 0]),
            crop_config.get("trim_pixels", [0, 0, 0, 0]),
        )
    elif crop_config["mode"] == "none":
        bbox = (0, raw_shape[0], 0, raw_shape[1])
    else:
        bbox = tuple(int(v) for v in crop_config["bbox_yxyx"])
    output_shape = tuple(int(v) for v in projection_config["output_size"])
    pixel_transform = PixelTransform(*raw_shape, bbox, *output_shape, flip_lr)
    final = {name: apply_pixel_transform(value, pixel_transform) for name, value in raw.items()}
    for name, value in final.items():
        np.save(case_dir / "projections" / f"{variant['name']}_{name}.npy", value.astype(np.float32))
    return {
        "camera": camera.as_dict(),
        "display_orientation": orientation,
        "pixel_transform": pixel_transform.as_dict(),
        "projection_files": {name: f"projections/{variant['name']}_{name}.npy" for name in final},
        "physics_additivity": additivity_metrics(final["full"], final["left"], final["right"]),
    }


def _process_case_impl(image_path, segmentation_path, output_dir, config, case_id):
    case_dir = Path(output_dir) / case_id
    (case_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (case_dir / "projections").mkdir(parents=True, exist_ok=True)
    if segmentation_path is None:
        segmentation_path = predict_segmentation(image_path, case_dir / "nnunet", config["nnunet"])
    image, label = read_image(image_path), read_image(segmentation_path)
    validate_same_geometry(image, label)
    labels, alignment_config = config["labels"], config["alignment"]
    mapping, original_plane, aligned_plane, centroid_left, centroid_right, angle_before = build_alignment(
        label, int(labels["left"]), int(labels["right"])
    )
    if angle_before > float(alignment_config["maximum_input_angle_deg"]):
        raise ValueError(f"Input left-right angle {angle_before:.3f} exceeds configured maximum")
    preserve_input_grid = bool(alignment_config.get("preserve_input_grid", True))
    aligned_image, aligned_label = resample_pair_with_grid(
        image, label, mapping, alignment_config["image_fill_hu"], preserve_input_grid
    )
    angle_after = residual_angle_deg(aligned_label, int(labels["left"]), int(labels["right"]))
    if angle_after > float(alignment_config["maximum_residual_angle_deg"]):
        raise ValueError(f"Residual alignment angle {angle_after:.3f} exceeds configured maximum")
    geometry = ImageGeometry.from_sitk(aligned_image)
    left_mask, right_mask, distance = complementary_masks(
        aligned_image, aligned_plane, config["split"]["boundary_goes_to"]
    )
    left_image, right_image = split_hu_image(
        aligned_image, left_mask, right_mask, config["attenuation"]["air_hu"]
    )
    write_image(aligned_image, case_dir / "volumes" / "full_aligned.nii.gz")
    write_image(aligned_label, case_dir / "volumes" / "segmentation_aligned.nii.gz")
    write_image(left_image, case_dir / "volumes" / "left.nii.gz")
    write_image(right_image, case_dir / "volumes" / "right.nii.gz")
    if config["split"].get("save_masks", True):
        np.savez_compressed(case_dir / "volumes" / "split_masks.npz", left=left_mask, right=right_mask)
    attenuation = config["attenuation"]
    arrays_mu = {
        "full": hu_to_mu(image_array(aligned_image), attenuation["water_mu_per_mm"]),
        "left": hu_to_mu(image_array(left_image), attenuation["water_mu_per_mm"]),
        "right": hu_to_mu(image_array(right_image), attenuation["water_mu_per_mm"]),
    }
    mu_error = np.abs(arrays_mu["full"] - arrays_mu["left"] - arrays_mu["right"])
    variants = {}
    for variant in resolve_variants(case_id, config["projection"]):
        variants[variant["name"]] = {
            "primary_angle_deg": variant["primary_angle_deg"],
            **_render_variant(
                variant, geometry, arrays_mu, config["projection"], case_dir,
            ),
        }
    metadata_json = Path(image_path).with_name(strip_nii_suffix(image_path).removesuffix("_0000") + ".json")
    return {
        "schema_version": "2.0",
        "status": "complete",
        "case_id": case_id,
        "side_contract": side_contract_provenance(),
        "inputs": {
            "image": str(Path(image_path).resolve()),
            "image_json": str(metadata_json.resolve()) if metadata_json.exists() else None,
            "segmentation": str(Path(segmentation_path).resolve()),
        },
        "coordinate_conventions": {
            "physical": "DICOM LPS millimetres", "array": "zyx", "continuous_index": "xyz",
        },
        "original_geometry": ImageGeometry.from_sitk(image).as_dict(),
        "aligned_geometry": geometry.as_dict(),
        "midplane_original": original_plane.as_dict(),
        "midplane_aligned": aligned_plane.as_dict(),
        "label_centroids_original_lps_mm": {
            "left": {"label": int(labels["left"]), "point_lps_mm": centroid_left},
            "right": {"label": int(labels["right"]), "point_lps_mm": centroid_right},
        },
        "alignment": {
            **mapping.as_dict(), "angle_before_deg": angle_before, "angle_after_resampled_deg": angle_after,
            "reference_grid": "input image grid" if preserve_input_grid else "expanded rotated voxel-centre bounds",
            "preserve_input_grid": preserve_input_grid,
            "image_fill_hu": alignment_config["image_fill_hu"],
        },
        "split": {
            "rule": "left iff signed_distance >= 0" if config["split"]["boundary_goes_to"] == "left" else "left iff signed_distance > 0",
            "positive_side": "patient left / label 8",
            "negative_side": "patient right / label 9",
            "left_voxels": int(left_mask.sum()), "right_voxels": int(right_mask.sum()),
            "partition_voxels": int(left_mask.size), "signed_distance_range_mm": [float(distance.min()), float(distance.max())],
            "mu_mean_abs_partition_error": float(mu_error.mean()), "mu_max_abs_partition_error": float(mu_error.max()),
        },
        "variants": variants,
    }


def process_case(image_path, segmentation_path, output_dir, config, case_id=None):
    validate_side_contract(config)
    case_id = case_id or strip_nii_suffix(image_path).removesuffix("_0000")
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError(f"Unsafe case_id: {case_id!r}")
    case_dir = Path(output_dir) / case_id
    manifest_path = case_dir / "manifest.json"
    if manifest_path.exists():
        if not config.get("runtime", {}).get("overwrite", False):
            raise FileExistsError(f"Output exists for {case_id}; set runtime.overwrite=true to replace")
        # Remove only pipeline-owned outputs. Unknown case files are preserved.
        for name in ("volumes", "projections", "nnunet"):
            owned_path = case_dir / name
            if owned_path.is_symlink() or owned_path.is_file():
                owned_path.unlink()
            elif owned_path.is_dir():
                shutil.rmtree(owned_path)
        manifest_path.unlink()
    try:
        manifest = _process_case_impl(image_path, segmentation_path, output_dir, config, case_id)
    except Exception as error:
        manifest = {
            "schema_version": "2.0", "status": "error", "case_id": case_id,
            "side_contract": side_contract_provenance(),
            "inputs": {"image": str(image_path), "segmentation": str(segmentation_path) if segmentation_path else None},
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        }
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return manifest
