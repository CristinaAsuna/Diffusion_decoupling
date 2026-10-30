from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .geometry import ImageGeometry


def strip_nii_suffix(path: str | Path) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def read_image(path: str | Path):
    return sitk.ReadImage(str(path))


def image_array(image, dtype=np.float32) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(dtype, copy=False)


def write_image(image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def image_from_array(array_zyx: np.ndarray, reference):
    image = sitk.GetImageFromArray(array_zyx)
    image.CopyInformation(reference)
    return image


def discover_cases(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    files = list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii"))
    # Prefer .nii.gz if both suffixes are present for one case.
    by_case: dict[str, Path] = {}
    for path in sorted(files, key=lambda item: (not item.name.endswith(".nii.gz"), item.name)):
        by_case.setdefault(strip_nii_suffix(path).removesuffix("_0000"), path)
    return list(by_case.values())


def find_case_file(folder: str | Path | None, case_id: str, suffixes=(".nii.gz", ".nii")) -> Path | None:
    if folder is None:
        return None
    folder = Path(folder)
    for candidate_id in (case_id, f"{case_id}_0000"):
        for suffix in suffixes:
            candidate = folder / f"{candidate_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def centroid_for_label(label_image, label_value: int) -> np.ndarray:
    array = sitk.GetArrayViewFromImage(label_image)
    coordinates_zyx = np.argwhere(array == int(label_value))
    if coordinates_zyx.size == 0:
        raise ValueError(f"Required label {label_value} is absent")
    center_index_xyz = coordinates_zyx[:, ::-1].mean(axis=0)
    return ImageGeometry.from_sitk(label_image).index_to_physical(center_index_xyz)
