from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from importlib.resources import files

from .config import load_config
from .io import discover_cases, find_case_file, strip_nii_suffix
from .manifest import write_json
from .pipeline import process_case
from .nnunet import preflight, model_hashes


DEFAULT_CONFIG = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbct-preprocess",
        description="nnU-Net → physical midplane alignment → split → consistent DRR",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-config", help="Export an editable packaged recipe")
    export.add_argument("--recipe", choices=["automatic", "supplied_segmentation"], default="automatic")
    export.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("check-model", help="Check external nnU-Net weights and plans")
    one = subparsers.add_parser("case", help="Process one CBCT")
    one.add_argument("--image", type=Path, required=True)
    one.add_argument("--segmentation", type=Path)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--case-id")
    dataset = subparsers.add_parser("dataset", help="Pair and process a directory of cases")
    dataset.add_argument("--images-dir", type=Path, required=True)
    dataset.add_argument("--segmentations-dir", type=Path)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--case-glob", default="*")
    inventory = subparsers.add_parser("inventory", help="Report image/segmentation pairing without processing")
    inventory.add_argument("--images-dir", type=Path, required=True)
    inventory.add_argument("--segmentations-dir", type=Path)
    inventory.add_argument("--output", type=Path)
    return parser


def _pairs(args) -> list[dict]:
    pairs = []
    for image in discover_cases(args.images_dir):
        case_id = strip_nii_suffix(image).removesuffix("_0000")
        if not Path(case_id).match(args.case_glob if hasattr(args, "case_glob") else "*"):
            continue
        segmentation = find_case_file(args.segmentations_dir, case_id) if args.segmentations_dir else None
        pairs.append({"case_id": case_id, "image": image, "segmentation": segmentation})
    return pairs


def _inventory(args, config) -> dict:
    pairs = _pairs(args)
    return {
        "count_images": len(pairs),
        "count_with_segmentation": sum(item["segmentation"] is not None for item in pairs),
        "cases": [{key: str(value) if isinstance(value, Path) else value for key, value in item.items()} for item in pairs],
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export-config":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to replace {args.output}")
        args.output.write_text(files("cbct_preprocess").joinpath("recipes", args.recipe + ".yaml").read_text())
        return 0
    config = load_config(args.config)
    if args.command == "check-model":
        executable, model = preflight(config["nnunet"])
        print(json.dumps({"executable": executable, "model": str(model), "sha256": model_hashes(model, config["nnunet"])}, indent=2))
        return 0
    if args.command == "inventory":
        report = _inventory(args, config)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "case":
        manifest = process_case(
            args.image, args.segmentation, args.output, config, args.case_id
        )
        print(f"complete: {manifest['case_id']} -> {args.output / manifest['case_id']}")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(args)
    if not pairs:
        raise ValueError(f"No NIfTI cases match {args.images_dir} / {args.case_glob}")
    summary = {"status": "running", "inventory": _inventory(args, config), "complete": [], "errors": []}
    write_json(args.output / "dataset_manifest.json", summary)
    for item in pairs:
        try:
            process_case(
                item["image"], item["segmentation"], args.output, config, item["case_id"]
            )
            summary["complete"].append(item["case_id"])
        except Exception as error:
            summary["errors"].append(
                {"case_id": item["case_id"], "type": type(error).__name__, "message": str(error)}
            )
            if not config.get("runtime", {}).get("continue_on_error", True):
                summary["status"] = "stopped_on_error"
                write_json(args.output / "dataset_manifest.json", summary)
                raise
        write_json(args.output / "dataset_manifest.json", summary)
    summary["status"] = "complete" if not summary["errors"] else "complete_with_errors"
    write_json(args.output / "dataset_manifest.json", summary)
    print(f"{summary['status']}: {len(summary['complete'])} complete, {len(summary['errors'])} errors")
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
