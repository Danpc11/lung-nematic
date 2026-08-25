#!/usr/bin/env python3
"""Cross-map gel time lapses onto fixed histology. Terminal, parallel.

    python run_crossmap.py \\
        --histology data --gel-frames frames \\
        --timelapse-results resultados_timelapse \\
        --output resultados_crossmap --n-jobs -1

Re-measures both systems at a matched physical scale and uses the gel series to
infer things about the histology defects that a fixed section cannot yield on
its own: the sign of the stiffness response, an effective stiffness by inverting
the calibration, and whether a region has equilibrated.

Why matched scale is not optional
---------------------------------
The packaged configurations detect at 8.0 um in histology (sigma 70 px at
0.1147 um/px) and 28.2 um on a gel (sigma 40 px at 0.7042 um/px). Defect density
falls roughly as ``1/sigma^2``, so that 3.5x mismatch is about a 12x offset
before any biology enters. Sweeps here are specified in micrometres and
converted per system, and the comparison uses the dimensionless ``rho * xi^2``.

Detection parameters also differ per system and both are set explicitly. A
confluent monolayer spans density 0.69-0.91, so the histology default quantile
of 0.45 discards nearly half the gel field; histology keeps it, because there
low nuclear density really does mean empty tissue.

Parallelism
-----------
Images are independent, so the sweep runs in a process pool. Workers receive a
path and open their own image; nothing large is pickled.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lung_nematic.config import load_default_config
from lung_nematic.crossmap import (
    calibration_curve,
    gel_scale_sweep,
    histology_scale_sweep,
    infer_stiffness,
    sigmas_for_microns,
    steady_state_window,
)
from lung_nematic.io_utils import read_rgb

SUPPORTED = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def stiffness_from_name(name: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*k?pa", name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def resolve_workers(n_jobs: int, n_tasks: int) -> int:
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("--n-jobs must be -1 (all cores) or a positive integer")
    requested = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    return max(1, min(int(requested), max(int(n_tasks), 1)))


def sweep_one(args):
    """Worker: sweep one image. Returns (key, table) or (key, None) on failure."""
    kind, key, path, config, sigmas_um, microns_per_pixel = args
    image = read_rgb(Path(path))
    sweep = histology_scale_sweep if kind == "histology" else gel_scale_sweep
    try:
        table = sweep(image, config, sigmas_um, microns_per_pixel)
    except ValueError as error:
        return key, None, str(error)
    return key, table, None


def run_sweeps(tasks, n_jobs: int, label: str) -> list:
    workers = resolve_workers(n_jobs, len(tasks))
    results = []
    if workers == 1:
        for task in tqdm(tasks, desc=label):
            results.append(sweep_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(sweep_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{label} [{workers}p]"):
                results.append(future.result())
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--histology", required=True, type=Path)
    parser.add_argument("--gel-frames", required=True, type=Path,
                        help="Directory of per-stiffness subdirectories.")
    parser.add_argument("--timelapse-results", required=True, type=Path,
                        help="Output directory of run_timelapse.py, for "
                             "defect_kinetics.tsv.")
    parser.add_argument("--histology-results", type=Path, default=None,
                        help="Optional: summary_metrics_nuclear.csv, for the "
                             "sectioning diagnostic.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)

    scale = parser.add_argument_group("calibration")
    scale.add_argument("--histology-mpp", type=float, default=0.114679)
    scale.add_argument("--gel-mpp", type=float, default=1 / 1.42)
    scale.add_argument("--sigmas-um", type=float, nargs="+",
                       default=[8.0, 12.0, 15.0, 20.0, 28.0, 40.0])
    scale.add_argument("--reference-sigma-um", type=float, default=15.0,
                       help="Scale at which the calibration is built. 15 um is "
                            "where the gel detector was validated against an "
                            "independent winding count.")

    detect = parser.add_argument_group("detection")
    detect.add_argument("--gel-density-quantile", type=float, default=0.10)
    detect.add_argument("--gel-grid-step-px", type=int, default=12)
    detect.add_argument("--gel-cluster-radius-px", type=float, default=45.0)
    detect.add_argument("--histology-density-quantile", type=float, default=None,
                        help="Defaults to the packaged value, which is correct "
                             "for tissue.")

    limits = parser.add_argument_group("sampling")
    limits.add_argument("--max-histology-images", type=int, default=None)
    limits.add_argument("--max-gel-frames", type=int, default=5,
                        help="Steady-state frames per stiffness (default 5).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    for sigma in args.sigmas_um:
        try:
            sigmas_for_microns((sigma,), args.gel_mpp)
            sigmas_for_microns((sigma,), args.histology_mpp)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1

    print(f"{'um':>6} {'histology px':>14} {'gel px':>10}")
    for sigma in args.sigmas_um:
        print(f"{sigma:>6.0f} "
              f"{sigmas_for_microns((sigma,), args.histology_mpp)[0]:>14.1f} "
              f"{sigmas_for_microns((sigma,), args.gel_mpp)[0]:>10.1f}")

    # --- gate: only steady series can be compared to fixed tissue ------------
    kinetics_path = args.timelapse_results / "defect_kinetics.tsv"
    if not kinetics_path.exists():
        print(f"\nmissing {kinetics_path}; run run_timelapse.py first",
              file=sys.stderr)
        return 1
    kinetics = pd.read_csv(kinetics_path, sep="\t")

    print("\nsteady-state gate:")
    usable = []
    for stiffness, group in kinetics.groupby("stiffness_kPa"):
        verdict = steady_state_window(group.sort_values("frame"))
        flag = "STEADY" if verdict["reached_steady_state"] else "NOT STEADY"
        print(f"  {stiffness:5g} kPa  {flag:11s}  "
              f"{verdict.get('reason', '')}")
        if verdict["reached_steady_state"]:
            usable.append((stiffness, verdict))

    if len(usable) < 2:
        print("\nfewer than two steady stiffnesses: no calibration is possible, "
              "and any inferred stiffness would be fabricated.", file=sys.stderr)
        return 1

    # --- sweeps --------------------------------------------------------------
    histology_config = load_default_config()
    if args.histology_density_quantile is not None:
        histology_config = replace(
            histology_config, density_quantile=args.histology_density_quantile
        )
    gel_config = replace(
        load_default_config(),
        density_quantile=args.gel_density_quantile,
        defect_grid_step_px=args.gel_grid_step_px,
        defect_cluster_radius_px=args.gel_cluster_radius_px,
    )

    histology_paths = sorted(p for p in args.histology.rglob("*")
                             if p.suffix.lower() in SUPPORTED)
    if args.max_histology_images:
        by_group: dict[str, list[Path]] = {}
        for path in histology_paths:
            by_group.setdefault(path.name.split("_")[0], []).append(path)
        histology_paths = [p for group in by_group.values()
                           for p in group[: args.max_histology_images]]

    tasks = [("histology", path.stem, str(path), histology_config,
              tuple(args.sigmas_um), args.histology_mpp)
             for path in histology_paths]

    for stiffness, window in usable:
        folder = next(
            (p for p in args.gel_frames.iterdir()
             if p.is_dir() and stiffness_from_name(p.name) == stiffness), None
        )
        if folder is None:
            print(f"  no frame directory for {stiffness:g} kPa")
            continue
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in SUPPORTED)
        paths = paths[window["first_frame"]: window["last_frame"] + 1]
        for path in paths[: args.max_gel_frames]:
            tasks.append(("gel", f"{stiffness}|{path.name}", str(path),
                          gel_config, tuple(args.sigmas_um), args.gel_mpp))

    print(f"\nsweeping {len(tasks)} images at {len(args.sigmas_um)} scales")
    results = run_sweeps(tasks, args.n_jobs, "sweep")

    histology_rows, gel_rows, skipped = [], [], 0
    for key, table, error in results:
        if table is None:
            skipped += 1
            continue
        if "|" in key:
            stiffness, filename = key.split("|", 1)
            table["stiffness_kPa"] = float(stiffness)
            table["filename"] = filename
            gel_rows.append(table)
        else:
            table["image_id"] = key
            table["dx"] = key.split("_")[0]
            table["patient"] = key.split("_")[1] if "_" in key else None
            histology_rows.append(table)
    if skipped:
        print(f"  {skipped} images skipped (no oriented nuclei)")

    # Every image failing is a real outcome, not an edge case: nuclei rounder
    # than min_aspect_ratio_for_orientation carry no direction, so a whole
    # cohort can yield nothing. Concatenating an empty list raises a pandas
    # error that says nothing about the cause.
    if not gel_rows:
        print("\nno gel frame produced a sweep; check --gel-density-quantile "
              "and --sigmas-um", file=sys.stderr)
        return 1
    if not histology_rows:
        print("\nno histology image produced a sweep: every one lacked "
              "oriented nuclei. Nuclei rounder than "
              "min_aspect_ratio_for_orientation carry no direction, so check "
              "the segmentation on a single image before running the cohort.",
              file=sys.stderr)
        return 1

    histology_sweep = pd.concat(histology_rows, ignore_index=True)
    gel_sweep = pd.concat(gel_rows, ignore_index=True)

    # --- calibration and inversion ------------------------------------------
    reference = args.reference_sigma_um
    gel_points = (gel_sweep[gel_sweep.sigma_um == reference]
                  .groupby("stiffness_kPa", as_index=False).rho_xi2.median())
    print(f"\ngel response at {reference:g} um:")
    print(gel_points.round(4).to_string(index=False))

    try:
        curve = calibration_curve(gel_points)
    except ValueError as error:
        print(f"\ncalibration failed: {error}", file=sys.stderr)
        return 1

    direction = "DECREASES" if curve["log_slope"] < 0 else "INCREASES"
    print(f"\nrho*xi^2 {direction} with stiffness: exponent "
          f"{curve['log_slope']:+.3f} over "
          f"{curve['stiffness_range_kPa'][0]:g}-"
          f"{curve['stiffness_range_kPa'][1]:g} kPa "
          f"({curve['n_points']} points)")
    print("  negative -> elasticity outpaces activity: stiffer tissue is more")
    print("              ordered with fewer defects")
    print("  positive -> activity outpaces elasticity: stiffer tissue is more")
    print("              defective")
    print("  Three points leave one degree of freedom after fitting two")
    print("  parameters. This is a description, not a test.")

    histology_at_ref = histology_sweep[
        histology_sweep.sigma_um == reference
    ].dropna(subset=["rho_xi2"])

    inferred = pd.DataFrame()
    if not histology_at_ref.empty:
        try:
            inferred = infer_stiffness(histology_at_ref.rho_xi2.to_numpy(), curve)
            inferred["image_id"] = histology_at_ref.image_id.to_numpy()
            inferred["dx"] = histology_at_ref.dx.to_numpy()
            inferred["patient"] = histology_at_ref.patient.to_numpy()

            print("\ninferred stiffness, median per patient:")
            within = inferred[inferred.within_calibrated_range]
            if within.empty:
                print("  every image fell outside the calibrated range")
            else:
                print(within.groupby(["dx", "patient"])
                      .inferred_stiffness_kPa.median().round(1).to_string())
            outside = int((~inferred.within_calibrated_range).sum())
            print(f"\n  {outside}/{len(inferred)} images outside the "
                  f"calibrated range, excluded above")
        except ValueError as error:
            print(f"\ninversion refused: {error}")

    # --- sectioning diagnostic ----------------------------------------------
    if args.histology_results:
        summary_path = args.histology_results / "summary_metrics_nuclear.csv"
        tracks_path = args.timelapse_results / "defect_tracks.tsv"
        if summary_path.exists() and tracks_path.exists():
            hist = pd.read_csv(summary_path)
            denominator = (hist.n_plus_half + hist.n_minus_half).replace(0, np.nan)
            hist_imbalance = (hist.n_plus_half - hist.n_minus_half).abs() / denominator

            tracks = pd.read_csv(tracks_path, sep="\t")
            gel_imbalance = []
            for _, group in tracks.groupby(["stiffness_kPa", "frame"]):
                plus = int((group.charge > 0).sum())
                minus = int((group.charge < 0).sum())
                if plus + minus:
                    gel_imbalance.append(abs(plus - minus) / (plus + minus))
            gel_imbalance = np.array(gel_imbalance)

            if gel_imbalance.size:
                limit = float(np.percentile(gel_imbalance, 95))
                exceeds = float((hist_imbalance > limit).mean())
                print("\nsectioning diagnostic:")
                print(f"  gel imbalance       median "
                      f"{np.median(gel_imbalance):.3f}, 95th pct {limit:.3f}")
                print(f"  histology imbalance median "
                      f"{hist_imbalance.median():.3f}")
                print(f"  {exceeds:.0%} of sections exceed the gel 95th pct")
                if exceeds > 0.3:
                    print("  ! most sections are not clean 2D slices: a section "
                          "cuts disclination LINES, not points, so its density "
                          "is not directly comparable to a gel's. Treat the "
                          "inferred stiffnesses as provisional.")

    # --- write ---------------------------------------------------------------
    tables = {
        "histology_scale_sweep.tsv": histology_sweep,
        "gel_scale_sweep.tsv": gel_sweep,
        "calibration_curve.tsv": pd.DataFrame([curve]),
    }
    if not inferred.empty:
        tables["inferred_stiffness.tsv"] = inferred
    print()
    for name, table in tables.items():
        table.to_csv(args.output / name, sep="\t", index=False)
        print(f"  wrote {args.output / name}  ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
