#!/usr/bin/env python3
"""Analyse stiffness-controlled time lapses and track defects. Terminal, parallel.

    python run_timelapse.py --frames frames --output resultados --n-jobs -1

``--frames`` is a directory holding one subdirectory per stiffness, named so the
number can be read from it (``5kPa``, ``10kPa``, ``23kPa``). Frames are read in
sorted filename order, so zero-padded names are required: ``t1, t2, ... t10``
sorts ``t10`` before ``t2`` and silently scrambles time.

Parallelism
-----------
Per-frame analysis is embarrassingly parallel and runs in a process pool. The
package's null model uses threads because SciPy's Gaussian filters release the
GIL, but defect detection walks a grid in pure Python and does not, so threads
would serialise on exactly the expensive part. Workers receive a path rather
than an array, so each opens its own image and nothing large is pickled.

Results are reordered by frame index before tracking. Completion order from a
pool is arbitrary, and a time series assembled in that order would be silently
shuffled - every velocity wrong, nothing raised.

Detection defaults
------------------
The packaged configuration is calibrated for histology and does not transfer.
``density_quantile = 0.45`` puts the threshold at 0.889 while a confluent
monolayer spans 0.69-0.91 with a median of 0.905, so it discards nearly half the
field as low density when coverage is actually 82%. Measured on real frames, the
packaged default returned 0-4 defects where an independent plaquette-winding
count finds 13; the defaults below recover 10-11.
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
from lung_nematic.io_utils import read_rgb
from lung_nematic.lic import lic_rgb
from lung_nematic.phase_contrast import analyze_phase_contrast
from lung_nematic.tracking import (
    calibrate,
    defect_kinetics,
    estimate_drift,
    motility_by_charge,
    subtract_drift,
    track_defects,
    track_summary,
)

SUPPORTED = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

# Above this fractional imbalance between +1/2 and -1/2 detections, the median
# step is dominated by one class's own motion rather than by stage drift.
DRIFT_IMBALANCE_LIMIT = 0.5
ARRAY_KEYS = {"field", "mask", "coverage_mask", "defects"}


def stiffness_from_name(name: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*k?pa", name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def frame_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)


def analyse_one(args) -> tuple[int, dict, pd.DataFrame, np.ndarray | None]:
    """Worker: analyse a single frame. Returns its index so order can be restored."""
    index, path, config, render = args
    image = read_rgb(Path(path))
    result = analyze_phase_contrast(image, config)

    summary = {k: v for k, v in result.items() if k not in ARRAY_KEYS}
    summary.update(frame=index, filename=Path(path).name,
                   height_px=image.shape[0], width_px=image.shape[1])

    found = result["defects"]
    defects = (
        found[["x_px", "y_px", "charge"]].copy()
        if len(found) else
        pd.DataFrame(columns=["x_px", "y_px", "charge"], dtype=float)
    )
    texture = None
    if render:
        field = result["field"]
        # Rendered in the worker: the field arrays are large and pickling them
        # back to the parent would cost more than the render itself.
        texture = lic_rgb(field["theta"], field["order"],
                          result["coverage_mask"], seed=index)
    return index, summary, defects, texture


def resolve_workers(n_jobs: int, n_tasks: int) -> int:
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("--n-jobs must be -1 (all cores) or a positive integer")
    requested = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    return max(1, min(int(requested), int(n_tasks)))


def analyse_series(paths, config, n_jobs: int, label: str, render: bool = False):
    """Analyse every frame of one series, in parallel, in frame order."""
    tasks = [(index, str(path), config, render)
             for index, path in enumerate(paths)]
    workers = resolve_workers(n_jobs, len(tasks))

    collected: dict[int, tuple] = {}
    if workers == 1:
        for task in tqdm(tasks, desc=label):
            index, summary, defects, texture = analyse_one(task)
            collected[index] = (summary, defects, texture)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(analyse_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{label} [{workers}p]"):
                index, summary, defects, texture = future.result()
                collected[index] = (summary, defects, texture)

    # Restore frame order. Pool completion order is arbitrary and a series
    # assembled in it would be shuffled without raising anything.
    ordered = [collected[index] for index in sorted(collected)]
    return (
        pd.DataFrame([summary for summary, _, _ in ordered]),
        [defects for _, defects, _ in ordered],
        [texture for _, _, texture in ordered],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frames", required=True, type=Path,
                        help="Directory of per-stiffness subdirectories.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Worker processes; -1 uses all cores (default).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames per series, for a trial run.")

    scale = parser.add_argument_group("calibration")
    scale.add_argument("--microns-per-pixel", type=float, default=1 / 1.42,
                       help="Default 0.70423 (1/1.42 px per um).")
    scale.add_argument(
        "--seconds-per-frame", type=float, default=None,
        help=(
            "Acquisition interval. Without it, speeds stay in px/frame - still "
            "comparable across stiffness, just not physical. The frame rate in "
            "an exported video is a playback rate and must not be used here."
        ),
    )

    detect = parser.add_argument_group("detection")
    detect.add_argument("--sigma-um", type=float, default=15.0,
                        help="Smoothing scale in um (default 15).")
    detect.add_argument("--sigma-band-ratio", type=float, default=1.4,
                        help="Second scale, as a multiple of the first. The "
                             "multiscale persistence filter needs two scales; "
                             "one sigma disables it and finds nothing.")
    detect.add_argument("--density-quantile", type=float, default=0.10,
                        help="Default 0.10 for phase contrast (0.45 is the "
                             "histology value and discards half a monolayer).")
    detect.add_argument("--grid-step-px", type=int, default=12)
    detect.add_argument("--cluster-radius-px", type=float, default=45.0)

    track = parser.add_argument_group("tracking")
    track.add_argument("--max-displacement-px", type=float, default=25.0,
                       help="Linking gate. Check track_summary afterwards: a "
                            "median track length of 1-2 frames means this is "
                            "too tight and real trajectories are being broken.")
    track.add_argument("--force-drift-correction", action="store_true",
                       help="Subtract drift even when the charge classes are "
                            "badly imbalanced. Off by default, because there "
                            "the median step is the dominant class's own "
                            "motion and subtracting it destroys the signal.")
    track.add_argument("--no-drift-correction", action="store_true",
                       help="Keep raw coordinates. Drift adds a common velocity "
                            "VECTOR, so with independent propulsion directions "
                            "it inflates both charge classes and compresses the "
                            "contrast between them - measured at 41% loss. "
                            "Leave correction on unless the frames are already "
                            "registered.")
    render = parser.add_argument_group("rendering")
    render.add_argument("--render-mp4", action="store_true",
                        help="Write an MP4 per stiffness: line integral "
                             "convolution of the director field, with tracked "
                             "defects and their velocity vectors overlaid.")
    render.add_argument("--fps", type=int, default=6)
    render.add_argument("--trail-frames", type=int, default=8,
                        help="Length of the trail drawn behind each defect.")
    render.add_argument("--velocity-gain", type=float, default=6.0,
                        help="Arrow length per px/frame of speed. Purely "
                             "visual; the numbers are in the TSVs.")
    return parser


def render_series(textures, tracks, output_path, fps, trail_frames,
                  velocity_gain):
    """Write the MP4: LIC background, defect markers, trails, velocity arrows.

    Colour encodes charge, not speed: +1/2 and -1/2 are physically different
    objects and only the first self-propels, so the eye should separate them
    first. Speed is shown by arrow length.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    plus_colour, minus_colour = (255, 96, 64), (64, 176, 255)
    writer = imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)
    try:
        for index, texture in enumerate(textures):
            if texture is None:
                continue
            canvas = Image.fromarray(texture)
            draw = ImageDraw.Draw(canvas)

            present = tracks[tracks.frame == index]
            for _, defect in present.iterrows():
                colour = plus_colour if defect.charge > 0 else minus_colour
                x, y = float(defect.x_px), float(defect.y_px)

                history = tracks[
                    (tracks.track_id == defect.track_id)
                    & (tracks.frame <= index)
                    & (tracks.frame > index - trail_frames)
                ].sort_values("frame")
                if len(history) > 1:
                    draw.line(
                        [(float(r.x_px), float(r.y_px))
                         for r in history.itertuples()],
                        fill=colour, width=2,
                    )

                if len(history) > 1:
                    previous = history.iloc[-2]
                    dx = x - float(previous.x_px)
                    dy = y - float(previous.y_px)
                    draw.line([(x, y), (x + dx * velocity_gain,
                                        y + dy * velocity_gain)],
                              fill=colour, width=3)

                radius = 7
                if defect.charge > 0:
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                                 outline=colour, width=3)
                else:
                    draw.polygon([(x, y - radius), (x - radius, y + radius),
                                  (x + radius, y + radius)],
                                 outline=colour)

            draw.text((10, 10), f"frame {index}", fill=(255, 255, 255))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.frames.is_dir():
        print(f"not a directory: {args.frames}", file=sys.stderr)
        return 1

    sigma_px = args.sigma_um / args.microns_per_pixel
    if sigma_px < 1.0:
        print(f"sigma {args.sigma_um} um is below one pixel at "
              f"{args.microns_per_pixel} um/px", file=sys.stderr)
        return 1

    config = replace(
        load_default_config(),
        sigmas_px=(sigma_px, sigma_px * args.sigma_band_ratio),
        density_quantile=args.density_quantile,
        defect_grid_step_px=args.grid_step_px,
        defect_cluster_radius_px=args.cluster_radius_px,
    )

    series = {}
    for folder in sorted(p for p in args.frames.iterdir() if p.is_dir()):
        stiffness = stiffness_from_name(folder.name)
        if stiffness is None:
            print(f"  skipping {folder.name}: no stiffness in the name")
            continue
        paths = frame_paths(folder)
        if args.max_frames:
            paths = paths[: args.max_frames]
        if len(paths) < 3:
            print(f"  skipping {folder.name}: {len(paths)} frames, need 3")
            continue
        series[stiffness] = paths

    if not series:
        print("no usable series found", file=sys.stderr)
        return 1

    print(f"scale {args.microns_per_pixel:.5f} um/px "
          f"({1 / args.microns_per_pixel:.2f} px/um)")
    print(f"sigma {args.sigma_um:g} um = "
          f"({sigma_px:.1f}, {sigma_px * args.sigma_band_ratio:.1f}) px\n")

    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics, all_tracks, all_motility, all_kinetics, all_summaries = (
        [], [], [], [], []
    )

    for stiffness, paths in series.items():
        metrics, defect_frames, textures = analyse_series(
            paths, config, args.n_jobs, f"{stiffness:g} kPa",
            render=args.render_mp4,
        )
        metrics["stiffness_kPa"] = stiffness

        raw = track_defects(defect_frames,
                            max_displacement_px=args.max_displacement_px)

        n_plus = int((raw.charge > 0).sum()) if len(raw) else 0
        n_minus = int((raw.charge < 0).sum()) if len(raw) else 0
        total = n_plus + n_minus
        imbalance = abs(n_plus - n_minus) / total if total else 1.0

        # A common component that differs between charge classes is not drift -
        # the sheet is flowing, and subtracting it removes real signal.
        for charge in sorted(raw.charge.unique()):
            per_class = estimate_drift(raw[raw.charge == charge])
            if not per_class.empty:
                print(f"    charge {charge:+.1f}  median step "
                      f"({per_class.drift_x_px.median():+.2f}, "
                      f"{per_class.drift_y_px.median():+.2f}) px/frame")

        # The median step estimates drift only when defect motion is otherwise
        # uncorrelated. If one charge class dominates the detections, the median
        # IS that class's motion: subtracting it would zero the self-propulsion
        # this whole analysis exists to measure, and the result would look like
        # a clean null rather than a destroyed signal.
        drift_unreliable = imbalance > DRIFT_IMBALANCE_LIMIT
        if drift_unreliable and not args.no_drift_correction:
            print(f"    ! charge imbalance {imbalance:.2f} exceeds "
                  f"{DRIFT_IMBALANCE_LIMIT}: skipping drift correction, since "
                  f"the median step would be the dominant class's own motion")
            if not args.force_drift_correction:
                tracks = raw
            else:
                tracks = subtract_drift(raw, estimate_drift(raw))
        elif args.no_drift_correction:
            tracks = raw
        else:
            tracks = subtract_drift(raw, estimate_drift(raw))

        if imbalance > DRIFT_IMBALANCE_LIMIT:
            print(f"    ! detections are {n_plus}/+{n_minus}- : in a closed 2D "
                  f"nematic defects are born and die in pairs, so this means "
                  f"the detector is biased. Re-check --sigma-um and "
                  f"--density-quantile before reading any result below.")

        motility = motility_by_charge(tracks)
        motility["stiffness_kPa"] = stiffness

        areas = dict(zip(
            metrics.frame,
            metrics.coverage_fraction * metrics.width_px * metrics.height_px
            * (args.microns_per_pixel / 1000.0) ** 2,
        ))
        kinetics = defect_kinetics(tracks, areas)
        kinetics["stiffness_kPa"] = stiffness

        summary = track_summary(tracks)
        summary["stiffness_kPa"] = stiffness

        plus = int((tracks.charge > 0).sum()) if len(tracks) else 0
        minus = int((tracks.charge < 0).sum()) if len(tracks) else 0
        median_length = (
            summary.n_frames.median() if len(summary) else float("nan")
        )
        print(f"    {tracks.track_id.nunique() if len(tracks) else 0} tracks, "
              f"{plus + minus} detections (+{plus} / -{minus}), "
              f"median track {median_length:.0f} frames, "
              f"mean S {metrics.global_nematic_order_S.mean():.3f}")
        if median_length <= 2:
            print("    ! median track length <= 2 frames: raise "
                  "--max-displacement-px, tracks are being broken")

        if args.render_mp4:
            video = args.output / f"director_{stiffness:g}kPa.mp4"
            render_series(textures, tracks, video, args.fps,
                          args.trail_frames, args.velocity_gain)
            print(f"    wrote {video}")

        all_metrics.append(metrics)
        all_tracks.append(tracks.assign(stiffness_kPa=stiffness))
        all_motility.append(motility)
        all_kinetics.append(kinetics)
        all_summaries.append(summary)
        print()

    motility = pd.concat(all_motility, ignore_index=True)
    kinetics = pd.concat(all_kinetics, ignore_index=True)

    if args.seconds_per_frame:
        motility = calibrate(motility, args.microns_per_pixel,
                             args.seconds_per_frame)
        kinetics = calibrate(kinetics, args.microns_per_pixel,
                             args.seconds_per_frame)
    else:
        print("! --seconds-per-frame not given: speeds are px/frame, "
              "comparable across stiffness but not physical\n")

    tables = {
        "per_frame_metrics.tsv": pd.concat(all_metrics, ignore_index=True),
        "defect_tracks.tsv": pd.concat(all_tracks, ignore_index=True),
        "motility_by_charge.tsv": motility,
        "defect_kinetics.tsv": kinetics,
        "track_summary.tsv": pd.concat(all_summaries, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, sep="\t", index=False)
        print(f"  wrote {args.output / name}  ({len(table)} rows)")

    speed_column = (
        "mean_speed_um_per_s" if "mean_speed_um_per_s" in motility.columns
        else "mean_speed_px_per_frame"
    )
    wide = motility.pivot_table(index="stiffness_kPa", columns="charge",
                                values=speed_column)
    if 0.5 in wide.columns and -0.5 in wide.columns:
        wide["contrast"] = wide[0.5] - wide[-0.5]
        print(f"\nactivity signature ({speed_column}):")
        print(wide.round(4).to_string())
        print("\n  speed(+1/2) > speed(-1/2) means the nematic is active;")
        print("  the contrast scales with activity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
