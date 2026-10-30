from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _load_config_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported config format: {path}")
    return data or {}


def _resolve_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Config inheritance cycle detected: {chain}")

    cfg = _load_config_file(path)
    base_entry = cfg.get("_base_")
    if base_entry is None:
        return cfg

    base_paths = base_entry if isinstance(base_entry, list) else [base_entry]
    merged: dict[str, Any] = {}
    for base_path in base_paths:
        resolved_base = Path(base_path)
        if not resolved_base.is_absolute():
            resolved_base = path.parent / resolved_base
        merged = _deep_merge(merged, _resolve_config(resolved_base, (*seen, path)))
    return _deep_merge(merged, cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    return _expand_env_vars(_resolve_config(Path(path)))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
