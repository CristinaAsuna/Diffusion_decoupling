from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _stem_to_path_map(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.glob("*.npy"))}


def _to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")
    if array.shape[0] <= 8:
        return array
    if array.shape[-1] <= 8:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"Unable to infer channel dimension from shape {array.shape}")


def _normalize(
    array: np.ndarray,
    mode: str,
    value_range: list[float] | tuple[float, float] | None = None,
    clip_range: list[float] | tuple[float, float] | None = None,
) -> np.ndarray:
    array = array.astype(np.float32)
    if clip_range is not None:
        clip_min, clip_max = float(clip_range[0]), float(clip_range[1])
        array = np.clip(array, clip_min, clip_max)
    if mode == "none":
        return array
    if mode == "fixed_range_01":
        if value_range is None:
            raise ValueError("value_range is required for fixed_range_01")
        min_val, max_val = float(value_range[0]), float(value_range[1])
        if max_val - min_val < 1e-8:
            raise ValueError("Invalid value_range for fixed_range_01")
        return np.clip((array - min_val) / (max_val - min_val), 0.0, 1.0)
    if mode == "fixed_range_m11":
        return _normalize(array, "fixed_range_01", value_range=value_range) * 2.0 - 1.0
    if mode == "minmax_01":
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)
    if mode == "range_m11":
        return _normalize(array, "minmax_01") * 2.0 - 1.0
    if mode == "zscore":
        mean = float(array.mean())
        std = float(array.std())
        if std < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - mean) / std
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _resize_tensor(tensor: torch.Tensor, image_size: list[int] | None) -> torch.Tensor:
    if image_size is None:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=tuple(image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


@dataclass
class SampleSpec:
    name: str
    condition_path: Path
    target_paths: list[Path]
    side_name: str | None = None
    side_id: int | None = None


class NpyConditionTargetDataset(Dataset):
    def __init__(
        self,
        condition_dir: str,
        target_dirs: Iterable[str],
        image_size: list[int] | None = None,
        normalize: str = "range_m11",
        names_file: str | None = None,
        value_range: list[float] | None = None,
        clip_range: list[float] | None = None,
        target_mode: str | None = None,
        target_side: str | None = None,
        target_sides: Iterable[str] | None = None,
        side_labels: dict[str, int] | None = None,
    ) -> None:
        self.condition_dir = Path(condition_dir)
        self.target_dirs = [Path(item) for item in target_dirs]
        self.image_size = image_size
        self.normalize = normalize
        self.value_range = value_range
        self.clip_range = clip_range
        resolved_target_mode = "dual" if target_mode == "multi_channel" else target_mode
        self.target_mode = resolved_target_mode or ("dual" if len(self.target_dirs) > 1 else "single")
        self.target_side = target_side
        self.target_sides = list(target_sides) if target_sides is not None else ["left", "right"]
        self.side_labels = side_labels or {side: idx for idx, side in enumerate(self.target_sides)}
        self.samples = self._build_samples(names_file)
        if not self.samples:
            raise ValueError("No paired .npy samples were found.")

    def _build_samples(self, names_file: str | None) -> list[SampleSpec]:
        condition_map = _stem_to_path_map(self.condition_dir)
        target_maps = [_stem_to_path_map(directory) for directory in self.target_dirs]
        if names_file:
            names = [
                line.strip()
                for line in Path(names_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            names = sorted(condition_map.keys())

        samples: list[SampleSpec] = []
        for name in names:
            if name not in condition_map:
                continue
            if self.target_mode == "side_cond":
                if len(target_maps) != len(self.target_sides):
                    raise ValueError("target_mode=side_cond expects target_dirs and target_sides to have the same length")
                for side_name, target_map in zip(self.target_sides, target_maps):
                    target_path = target_map.get(name)
                    if target_path is None:
                        continue
                    samples.append(
                        SampleSpec(
                            name=f"{name}__{side_name}",
                            condition_path=condition_map[name],
                            target_paths=[target_path],
                            side_name=side_name,
                            side_id=self.side_labels[side_name],
                        )
                    )
                continue

            if self.target_mode == "single":
                if self.target_side is not None:
                    if self.target_side not in self.target_sides:
                        raise ValueError(f"Unsupported target_side: {self.target_side}")
                    side_index = self.target_sides.index(self.target_side)
                    target_maps_to_use = [target_maps[side_index]]
                else:
                    target_maps_to_use = target_maps[:1]
            else:
                target_maps_to_use = target_maps

            target_paths: list[Path] = []
            for target_map in target_maps_to_use:
                target_path = target_map.get(name)
                if target_path is None:
                    target_paths = []
                    break
                target_paths.append(target_path)
            if target_paths:
                samples.append(SampleSpec(name=name, condition_path=condition_map[name], target_paths=target_paths))
        return samples

    def _load_raw(self, path: Path) -> torch.Tensor:
        return _resize_tensor(torch.from_numpy(_to_chw(np.load(path)).astype(np.float32)), self.image_size)

    def _load_single(self, path: Path) -> torch.Tensor:
        # Match prepared-data/inference order: resize raw values first, then
        # apply the one fixed input mapping. Raw fields remain unconstrained.
        raw = self._load_raw(path)
        array = _normalize(raw.numpy(), self.normalize, value_range=self.value_range, clip_range=self.clip_range)
        return torch.from_numpy(array).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        raw_condition = self._load_raw(sample.condition_path)
        raw_target = torch.cat([self._load_raw(path) for path in sample.target_paths], dim=0)
        condition = torch.from_numpy(_normalize(raw_condition.numpy(), self.normalize,
                         value_range=self.value_range, clip_range=self.clip_range))
        target = torch.from_numpy(_normalize(raw_target.numpy(), self.normalize,
                         value_range=self.value_range, clip_range=self.clip_range))
        result: dict[str, torch.Tensor | str] = {"name": sample.name, "condition": condition, "target": target,
            "raw_condition": raw_condition, "raw_target": raw_target}
        if sample.side_name is not None and sample.side_id is not None:
            result["side_name"] = sample.side_name
            result["side"] = torch.tensor(sample.side_id, dtype=torch.long)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def condition_channels(self) -> int:
        first = self[0]["condition"]
        assert isinstance(first, torch.Tensor)
        return int(first.shape[0])

    @property
    def target_channels(self) -> int:
        first = self[0]["target"]
        assert isinstance(first, torch.Tensor)
        return int(first.shape[0])

    @property
    def num_side_classes(self) -> int | None:
        if not any(sample.side_id is not None for sample in self.samples):
            return None
        return len(self.side_labels)


class CaseFolderNpyDataset(Dataset):
    def __init__(
        self,
        case_root: str,
        condition_file: str | None = None,
        target_files: Iterable[str] | None = None,
        image_size: list[int] | None = None,
        normalize: str = "range_m11",
        case_names_file: str | None = None,
        case_ids: Iterable[str] | None = None,
        include_patterns: Iterable[str] | None = None,
        variants: Iterable[str] | None = None,
        condition_template: str | None = None,
        target_templates: Iterable[str] | None = None,
        split: str | None = None,
        split_seed: int = 1234,
        train_ratio: float = 0.9,
        value_range: list[float] | None = None,
        clip_range: list[float] | None = None,
        target_mode: str | None = None,
        target_side: str | None = None,
        target_sides: Iterable[str] | None = None,
        target_template: str | None = None,
        side_labels: dict[str, int] | None = None,
    ) -> None:
        self.case_root = Path(case_root)
        self.case_ids = None if case_ids is None else set(map(str, case_ids))
        self.condition_file = condition_file
        self.target_files = list(target_files) if target_files is not None else []
        self.image_size = image_size
        self.normalize = normalize
        self.include_patterns = list(include_patterns) if include_patterns is not None else [condition_file]
        self.variants = list(variants) if variants is not None else None
        self.condition_template = condition_template
        self.target_templates = list(target_templates) if target_templates is not None else None
        self.split = split
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.value_range = value_range
        self.clip_range = clip_range
        inferred_mode = "single"
        if target_mode is not None:
            inferred_mode = "dual" if target_mode == "multi_channel" else target_mode
        elif self.target_templates is not None and len(self.target_templates) > 1:
            inferred_mode = "dual"
        self.target_mode = inferred_mode
        self.target_side = target_side
        self.target_sides = list(target_sides) if target_sides is not None else ["left", "right"]
        self.target_template = target_template
        self.side_labels = side_labels or {side: idx for idx, side in enumerate(self.target_sides)}
        self.samples = self._build_samples(case_names_file)
        if not self.samples:
            raise ValueError("No case-folder .npy samples were found.")

    def _resolve_case_names(self, case_names_file: str | None) -> list[str]:
        if case_names_file:
            case_names = [
                line.strip()
                for line in Path(case_names_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            case_names = sorted(path.name for path in self.case_root.iterdir() if path.is_dir())
        if self.case_ids is not None:
            missing = self.case_ids - set(case_names)
            if missing:
                raise ValueError(f"Requested case IDs not found: {sorted(missing)}")
            case_names = [name for name in case_names if name in self.case_ids]
        if self.split is None:
            return case_names
        rng = random.Random(self.split_seed)
        case_names = case_names[:]
        rng.shuffle(case_names)
        split_index = max(1, min(len(case_names) - 1, int(round(len(case_names) * self.train_ratio))))
        if self.split == "train":
            return sorted(case_names[:split_index])
        if self.split in {"val", "test"}:
            return sorted(case_names[split_index:])
        raise ValueError(f"Unsupported split: {self.split}")

    def _resolve_target_paths_for_variant(self, case_dir: Path, variant: str) -> list[tuple[str | None, list[Path], int | None]]:
        if self.target_mode == "side_cond":
            outputs: list[tuple[str | None, list[Path], int | None]] = []
            for side_name in self.target_sides:
                if self.target_template is not None:
                    target_path = case_dir / self.target_template.format(variant=variant, side=side_name)
                else:
                    target_path = case_dir / f"{variant}_{side_name}.npy"
                if target_path.exists():
                    outputs.append((side_name, [target_path], self.side_labels[side_name]))
            return outputs

        if self.target_mode == "single":
            if self.target_templates is not None:
                if len(self.target_templates) == 1:
                    return [(None, [case_dir / self.target_templates[0].format(variant=variant)], None)]
                raise ValueError("target_mode=single expects exactly one target_templates entry, or use target_template/target_side")
            side_name = self.target_side or self.target_sides[0]
            if self.target_template is not None:
                return [(None, [case_dir / self.target_template.format(variant=variant, side=side_name)], None)]
            return [(None, [case_dir / f"{variant}_{side_name}.npy"], None)]

        if self.target_mode == "dual":
            if self.target_templates is not None:
                target_paths = [case_dir / template.format(variant=variant) for template in self.target_templates]
            else:
                if self.target_template is not None:
                    target_paths = [case_dir / self.target_template.format(variant=variant, side=side_name) for side_name in self.target_sides]
                else:
                    target_paths = [case_dir / f"{variant}_{side_name}.npy" for side_name in self.target_sides]
            return [(None, target_paths, None)]

        raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def _resolve_target_paths_for_pattern(self, case_dir: Path, pattern: str) -> list[tuple[str | None, list[Path], int | None]]:
        stem = Path(pattern).stem
        if self.target_mode == "side_cond":
            outputs: list[tuple[str | None, list[Path], int | None]] = []
            for side_name in self.target_sides:
                if self.target_template is not None:
                    target_path = case_dir / self.target_template.format(pattern_stem=stem, side=side_name)
                else:
                    target_path = case_dir / f"{side_name}.npy"
                if target_path.exists():
                    outputs.append((side_name, [target_path], self.side_labels[side_name]))
            return outputs

        if self.target_mode == "single":
            if self.target_files:
                if len(self.target_files) == 1:
                    return [(None, [case_dir / self.target_files[0]], None)]
                raise ValueError("target_mode=single expects exactly one target_files entry")
            side_name = self.target_side or self.target_sides[0]
            if self.target_template is not None:
                return [(None, [case_dir / self.target_template.format(pattern_stem=stem, side=side_name)], None)]
            return [(None, [case_dir / f"{side_name}.npy"], None)]

        if self.target_mode == "dual":
            if self.target_files:
                return [(None, [case_dir / target_file for target_file in self.target_files], None)]
            if self.target_template is not None:
                return [
                    (
                        None,
                        [case_dir / self.target_template.format(pattern_stem=stem, side=side_name) for side_name in self.target_sides],
                        None,
                    )
                ]
            return [(None, [case_dir / f"{side_name}.npy" for side_name in self.target_sides], None)]

        raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def _build_samples(self, case_names_file: str | None) -> list[SampleSpec]:
        case_names = self._resolve_case_names(case_names_file)
        samples: list[SampleSpec] = []
        for case_name in case_names:
            case_dir = self.case_root / case_name
            if not case_dir.is_dir():
                continue
            if self.variants is not None and self.condition_template is not None:
                for variant in self.variants:
                    condition_path = case_dir / self.condition_template.format(variant=variant)
                    if not condition_path.exists():
                        continue
                    targets = self._resolve_target_paths_for_variant(case_dir, variant)
                    for side_name, target_paths, side_id in targets:
                        valid = all(path.exists() for path in target_paths)
                        if not valid:
                            continue
                        sample_name = f"{case_name}__{variant}" if side_name is None else f"{case_name}__{variant}__{side_name}"
                        samples.append(
                            SampleSpec(
                                name=sample_name,
                                condition_path=condition_path,
                                target_paths=target_paths,
                                side_name=side_name,
                                side_id=side_id,
                            )
                        )
                continue
            for pattern in self.include_patterns:
                condition_path = case_dir / pattern
                if not condition_path.exists():
                    continue
                targets = self._resolve_target_paths_for_pattern(case_dir, pattern)
                for side_name, target_paths, side_id in targets:
                    valid = all(path.exists() for path in target_paths)
                    if not valid:
                        continue
                    sample_name = f"{case_name}__{Path(pattern).stem}" if side_name is None else f"{case_name}__{Path(pattern).stem}__{side_name}"
                    samples.append(
                        SampleSpec(
                            name=sample_name,
                            condition_path=condition_path,
                            target_paths=target_paths,
                            side_name=side_name,
                            side_id=side_id,
                        )
                    )
        return samples

    def _load_raw(self, path: Path) -> torch.Tensor:
        return _resize_tensor(torch.from_numpy(_to_chw(np.load(path)).astype(np.float32)), self.image_size)

    def _load_single(self, path: Path) -> torch.Tensor:
        # Match prepared-data/inference order: resize raw values first, then
        # apply the one fixed input mapping. Raw fields remain unconstrained.
        raw = self._load_raw(path)
        array = _normalize(raw.numpy(), self.normalize, value_range=self.value_range, clip_range=self.clip_range)
        return torch.from_numpy(array).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        condition = self._load_single(sample.condition_path)
        target = torch.cat([self._load_single(path) for path in sample.target_paths], dim=0)
        result: dict[str, torch.Tensor | str] = {"name": sample.name, "condition": condition, "target": target,
            "raw_condition": self._load_raw(sample.condition_path),
            "raw_target": torch.cat([self._load_raw(path) for path in sample.target_paths], dim=0)}
        if sample.side_name is not None and sample.side_id is not None:
            result["side_name"] = sample.side_name
            result["side"] = torch.tensor(sample.side_id, dtype=torch.long)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def condition_channels(self) -> int:
        first = self[0]["condition"]
        assert isinstance(first, torch.Tensor)
        return int(first.shape[0])

    @property
    def target_channels(self) -> int:
        first = self[0]["target"]
        assert isinstance(first, torch.Tensor)
        return int(first.shape[0])

    @property
    def num_side_classes(self) -> int | None:
        if not any(sample.side_id is not None for sample in self.samples):
            return None
        return len(self.side_labels)
