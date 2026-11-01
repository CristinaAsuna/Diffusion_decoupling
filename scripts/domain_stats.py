"""Compare real radiograph conditions with validation DRR conditions.

Computes target-free statistics of the model-resolution condition (radial power
spectrum, gradient percentiles, foreground fraction, histogram) for real images
loaded exactly as ``decouple infer`` loads them and for validation DRRs.  With
``--checkpoint`` it also reports predicted D energy relative to the train-only
scale, and optionally sweeps inversion, nose direction and central zoom on the
real images to see which input convention restores nontrivial D.  Uses
validation patients only; it reports no accuracy for real images.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from consistent_cdm import runtime
from consistent_cdm.evaluation.degradations import scale as zoom

SPECTRUM_BINS = 16


def condition_stats(condition):
    centred = condition - condition.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centred))) ** 2
    y, x = np.indices(power.shape)
    radius = np.hypot(y - power.shape[0] // 2, x - power.shape[1] // 2) / (min(power.shape) / 2)
    edges = np.linspace(0, 1, SPECTRUM_BINS + 1)
    total = max(float(power[radius <= 1].sum()), 1e-12)
    spectrum = [float(power[(radius >= a) & (radius < b)].sum() / total) for a, b in zip(edges[:-1], edges[1:])]
    gradient = np.hypot(*np.gradient(condition))
    histogram = np.histogram(np.clip(condition, 0, 6), bins=12, range=(0, 6))[0] / condition.size
    stats = {f"spectrum_{i:02d}": v for i, v in enumerate(spectrum)}
    stats.update(high_frequency_fraction=float(sum(spectrum[SPECTRUM_BINS // 2:])),
                 gradient_p50=float(np.percentile(gradient, 50)), gradient_p90=float(np.percentile(gradient, 90)),
                 gradient_p99=float(np.percentile(gradient, 99)), foreground_fraction=float(np.mean(condition > .3)))
    stats.update({f"histogram_{i:02d}": float(v) for i, v in enumerate(histogram)})
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--real", required=True, help="Real image file or flat folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--nose-direction", choices=["left", "right"], default="right")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Also try invert x nose direction x zoom on real images")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    prepared = runtime.read_prepared(args.prepared)
    size, calibration = prepared["image_size"], prepared["calibration"]
    predictor = None
    if args.checkpoint:
        model, cfg, family, scale, _ = runtime.load_predictor(args.checkpoint, device=args.device)
        if cfg.get("prepared_id") != prepared["prepared_id"]:
            raise SystemExit("Checkpoint and prepared dataset identifiers differ")
        predictor = lambda condition: float(np.sqrt(np.mean(np.square(
            runtime.predict(model, family, cfg, scale, condition, seed=args.seed)[0]))) / scale)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise SystemExit(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = prepared["split"]["val"][:args.max_cases] if args.max_cases else prepared["split"]["val"]
    for case in cases:
        for variant in prepared["variants"]:
            root = Path(prepared["data_root"]) / case / "projections"
            full, left, right = [runtime.resize_raw(np.load(root / f"{variant}_{n}.npy", allow_pickle=False), size)
                                 for n in ("full", "left", "right")]
            row = dict(domain="drr_val", source=f"{case}/{variant}", setting="raw", **condition_stats(full))
            row["gt_d_rms_over_scale"] = float(np.sqrt(np.mean(np.square((left - right) / 2)))) / prepared["d_normalization"]["scale"]
            if predictor: row["d_rms_over_scale"] = predictor(full)
            rows.append(row)

    real = Path(args.real)
    paths = sorted(p for p in real.iterdir() if p.suffix.lower() in runtime.SUFFIXES) if real.is_dir() else [real]
    settings = [(args.invert, args.nose_direction, 1.)]
    if args.sweep:
        settings = list(itertools.product((False, True), ("right", "left"), (.8, 1., 1.25)))
    for path in paths:
        for invert, nose, factor in settings:
            condition, _ = runtime.load_input(path, size=size, calibration=calibration,
                                              nose_direction=nose, invert=invert)
            condition = zoom(condition, factor)
            row = dict(domain="real", source=path.name, setting=f"invert={invert};nose={nose};zoom={factor:g}",
                       **condition_stats(condition))
            if predictor: row["d_rms_over_scale"] = predictor(condition)
            rows.append(row)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "domain_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for domain, setting in dict.fromkeys((r["domain"], r["setting"]) for r in rows):
        selected = [r for r in rows if (r["domain"], r["setting"]) == (domain, setting)]
        summary[f"{domain}|{setting}"] = {key: float(np.mean([r[key] for r in selected if key in r]))
                                          for key in fields if key not in ("domain", "source", "setting")
                                          and any(key in r for r in selected)}
    runtime.write_json(output / "summary.json", dict(groups=summary, prepared_id=prepared["prepared_id"],
                       cases=cases, real_count=len(paths), cohort="validation_only_and_unpaired_real"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, values in summary.items():
        if not (name.startswith("drr_val") or name.endswith("zoom=1")): continue
        axes[0].semilogy([values[f"spectrum_{i:02d}"] for i in range(SPECTRUM_BINS)], label=name)
        axes[1].plot(np.arange(12) * .5 + .25, [values[f"histogram_{i:02d}"] for i in range(12)], label=name)
    axes[0].set(title="Radial power spectrum", xlabel="radius bin (0=DC, 15=Nyquist)")
    axes[1].set(title="Condition histogram", xlabel="mapped line integral")
    axes[1].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output / "domain_stats.png", dpi=120)


if __name__ == "__main__":
    main()
