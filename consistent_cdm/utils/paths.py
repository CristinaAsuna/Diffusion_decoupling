from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(os.environ.get("CONSISTENT_CDM_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def sanitize_path_name(name: str | None, default: str = "default") -> str:
    raw = str(name or default).strip() or default
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-") or default


def _base_path(env_name: str, cfg_value: str | None, default_name: str) -> Path:
    root = project_root()
    value = os.environ.get(env_name) or cfg_value or default_name
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def apply_run_layout(
    cfg: dict[str, Any],
    group: str,
    default_experiment: str,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Normalize output, TensorBoard, and SwanLab paths for a training/infer run.

    Default layout:
      runs/<group>/<experiment>/
        checkpoints/, samples/, infer/, telemetry/{tensorboard,swanlab}/

    Set output.use_unified_layout=false in a config to preserve explicit paths.
    """
    output_cfg = cfg.setdefault("output", {})
    logging_cfg = cfg.setdefault("logging", {})

    use_unified = bool(output_cfg.get("use_unified_layout", True))
    group_name = sanitize_path_name(output_cfg.get("group") or logging_cfg.get("group") or group, default=group)
    exp_name = sanitize_path_name(
        experiment_name or logging_cfg.get("experiment_name") or output_cfg.get("experiment_name") or default_experiment,
        default=default_experiment,
    )

    logging_cfg["experiment_name"] = str(experiment_name or logging_cfg.get("experiment_name") or exp_name)
    output_cfg["group"] = group_name

    if use_unified:
        runs_base = _base_path("CONSISTENT_CDM_RUNS_ROOT", output_cfg.get("base_dir"), "runs")
        output_root = runs_base / group_name / exp_name
        tensorboard_root = output_root / "telemetry" / "tensorboard"
        swanlab_root = output_root / "telemetry" / "swanlab"

        output_cfg["root"] = str(output_root)
        output_cfg["infer_root"] = str(output_root / "infer")
        logging_cfg["tensorboard_logdir"] = str(tensorboard_root)
        logging_cfg["logdir"] = str(swanlab_root)
    else:
        if output_cfg.get("root") is not None:
            output_cfg["root"] = str(_resolve_path(output_cfg["root"]))
        if output_cfg.get("infer_root") is not None:
            output_cfg["infer_root"] = str(_resolve_path(output_cfg["infer_root"]))
        if logging_cfg.get("tensorboard_logdir") is not None:
            logging_cfg["tensorboard_logdir"] = str(_resolve_path(logging_cfg["tensorboard_logdir"]))
        if logging_cfg.get("logdir") is not None:
            logging_cfg["logdir"] = str(_resolve_path(logging_cfg["logdir"]))

    cfg["run_layout"] = {
        "group": group_name,
        "experiment_name": exp_name,
        "output_root": output_cfg.get("root"),
        "infer_root": output_cfg.get("infer_root"),
        "tensorboard_logdir": logging_cfg.get("tensorboard_logdir"),
        "swanlab_logdir": logging_cfg.get("logdir"),
    }
    return cfg
