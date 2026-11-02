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



def model_condition_from_source(source, size, preprocessing="raw_resize_then_normalize"):
    """Return [0,6]-mapped and [-1,1] exact model tensors, before any model.

    Historical checkpoints clip/normalize native source pixels before resizing.
    Raw Full stays separate because its negative Lanczos tails are meaningful
    provenance even though the model clips them. Returning the inverse affine
    of the actual normalized tensor avoids silently swapping clip/resize order.
    """
    source=np.asarray(source,np.float32)
    if preprocessing == "historical_normalize_before_resize":
        normalized=runtime.resize_raw(np.clip(source,0,6)/3-1,size)
    else:
        normalized=np.clip(runtime.resize_raw(source,size),0,6)/3-1
    return ((normalized+1)*3).astype(np.float32), normalized.astype(np.float32)

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
    parser.add_argument("--save-arrays", action="store_true", help="Export numerical arrays; disabled by default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.max_cases is not None and args.max_cases < 1: parser.error("max-cases must be positive")
    prepared = runtime.read_prepared(args.prepared)
    runtime.verify_prepared(prepared)
    size, calibration = prepared["image_size"], prepared["calibration"]
    predictor = None
    checkpoint_meta = None
    if args.checkpoint:
        model, cfg, family, scale, weights = runtime.load_predictor(args.checkpoint, device=args.device)
        runtime.verify_diagnostic_checkpoint(prepared,cfg,args.checkpoint)
        predictor = lambda condition: runtime.predict(model,family,cfg,scale,condition,seed=args.seed)[0]
        checkpoint_meta = dict(sha256=runtime.sha256(args.checkpoint),family=family,weights=weights,
                               d_scale=scale,seed=args.seed,sampler=cfg.get("validation",{}).get("sampler"))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise SystemExit(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = prepared["split"]["val"][:args.max_cases] if args.max_cases else prepared["split"]["val"]
    for case in cases:
        for variant in prepared["variants"]:
            raw=[np.load(path,allow_pickle=False) for path in runtime.projection_paths(prepared,case,variant)]
            full,left,right=[runtime.resize_raw(value,size) for value in raw]
            model_condition,normalized_condition=model_condition_from_source(raw[0],size,prepared.get("preprocessing","raw_resize_then_normalize"))
            row = dict(domain="drr_val", source=f"{case}/{variant}", setting="raw", **condition_stats(model_condition))
            row["condition_saturation_fraction"]=float(np.mean((full<0)|(full>6)))
            row["gt_d_rms_over_scale"] = float(np.sqrt(np.mean(np.square((left - right) / 2)))) / prepared["d_normalization"]["scale"]
            if predictor:
                d=runtime.diagnostic_predict(model,family,cfg,scale,raw[0],prepared,seed=args.seed)[0]
                row["d_rms_over_scale"]=float(np.sqrt(np.mean(np.square(d)))/scale)
            if args.save_arrays:
                item=output/"arrays"/"drr_val"/case/variant
                item.mkdir(parents=True)
                for key,value in (("condition",full),("model_condition",model_condition),
                                  ("normalized_condition",normalized_condition),("gt_d",(left-right)/2)):
                    np.save(item/f"{key}.npy",np.asarray(value,np.float32))
                if predictor: np.save(item/"d.npy",np.asarray(d,np.float32))
            rows.append(row)
        print(f"domain drr {case}",flush=True)

    real = Path(args.real)
    paths = sorted(p for p in real.iterdir() if p.suffix.lower() in runtime.SUFFIXES) if real.is_dir() else [real]
    if args.max_cases: paths=paths[:args.max_cases]
    if not paths: raise ValueError("No real input images")
    real_sources=[dict(path=str(path.resolve()),sha256=runtime.sha256(path)) for path in paths]
    settings = [(args.invert, args.nose_direction, 1.)]
    if args.sweep:
        settings = list(itertools.product((False, True), ("right", "left"), (.8, 1., 1.25)))
    for path in paths:
        for invert, nose, factor in settings:
            condition, _ = runtime.load_input(path, size=size, calibration=calibration,
                                              nose_direction=nose, invert=invert)
            condition = zoom(condition, factor)
            row = dict(domain="real", source=path.name, setting=f"invert={invert};nose={nose};zoom={factor:g}",
                       **condition_stats(np.clip(condition,0,6)))
            row["condition_saturation_fraction"]=float(np.mean((condition<0)|(condition>6)))
            if args.save_arrays:
                item=output/"arrays"/path.name/f"invert{int(invert)}_nose{nose}_zoom{factor:g}"
                item.mkdir(parents=True)
                np.save(item/"condition.npy",condition.astype(np.float32))
            if predictor:
                d=predictor(condition)
                row["d_rms_over_scale"]=float(np.sqrt(np.mean(np.square(d)))/scale)
                if args.save_arrays: np.save(item/"d.npy",d.astype(np.float32))
            rows.append(row)
        print(f"domain real {path.name}",flush=True)

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
                       cases=cases, real_count=len(paths), cohort="validation_only_and_unpaired_real",
                       checkpoint=checkpoint_meta, real_sources=real_sources, seed=args.seed,save_arrays=args.save_arrays,
                       side_contract=prepared["side_contract"], calibration=calibration,
                       drr_preprocessing=prepared.get("preprocessing","raw_resize_then_normalize"),
                       real_preprocessing="raw_resize_then_quantile_then_zoom_then_model_clip",
                       condition_stats_domain="DRR inverse-affine of exact normalized model tensor; real clipped mapped input [0,6]; pre-clipping saturation reported",
                       array_semantics={"drr_condition.npy":"raw resized Full, not the model input after native clipping",
                                        "drr_model_condition.npy":"inverse affine (normalized_condition+1)*3",
                                        "drr_normalized_condition.npy":"exact [-1,1] tensor supplied to model",
                                        "real_condition.npy":"mapped Full after optional zoom, before clipping"},
                       inference_selection="all settings diagnostic only, no paired real accuracy"))
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
