#!/usr/bin/env python3
"""Generate LIC, collective-flow, speed, and directionality maps for gels."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lung_nematic.flow import collective_flow, flow_summary
from lung_nematic.io_utils import read_rgb
from lung_nematic.lic import lic_rgb

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

SUPPORTED = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def stiffness_from_name(name: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*k?pa", name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def frame_paths(folder: Path) -> list[Path]:
    """Return encoded chronology; acquisition names must be zero-padded."""
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in SUPPORTED)


def analyse_pair(task):
    index, first_path, second_path, settings = task
    first = read_rgb(first_path)
    second = read_rgb(second_path)
    result = collective_flow(first, second, **settings["flow"])
    summary = flow_summary(result, settings["mpp"], settings["spf"])
    summary.update(frame=index, next_frame=index + 1,
                   first_filename=first_path.name,
                   second_filename=second_path.name)
    return index, summary, result


def render_panel(result, output: Path, title: str, microns_per_pixel: float,
                 seconds_per_frame: float, vector_step: int) -> None:
    theta = result["director_theta_rad"]
    order = result["local_order"]
    mask = result["mask"]
    lic = lic_rgb(theta, order, mask, n_steps=20)
    speed = (result["speed_px_per_frame"] * microns_per_pixel * 60
             / seconds_per_frame)
    alignment = result["flow_director_alignment"]
    u, v = result["u_px_per_frame"], result["v_px_per_frame"]
    yy, xx = np.mgrid[0:speed.shape[0]:vector_step,
                      0:speed.shape[1]:vector_step]

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(lic)
    axes[0].set_title("LIC: director nemático")
    speed_map = axes[1].imshow(speed, cmap="magma", vmin=0,
                               vmax=np.nanpercentile(speed, 95))
    axes[1].quiver(xx, yy, u[::vector_step, ::vector_step],
                   v[::vector_step, ::vector_step], color="cyan",
                   angles="xy", scale_units="xy", scale=1, width=0.003)
    axes[1].set_title("Flujo colectivo (µm/min)")
    figure.colorbar(speed_map, ax=axes[1], fraction=0.046)
    align_map = axes[2].imshow(alignment, cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title("Alineamiento flujo–director")
    figure.colorbar(align_map, ax=axes[2], fraction=0.046,
                    label="cos[2(θv−θn)]")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stiffness", nargs="+", type=float, default=[5, 23])
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--microns-per-pixel", type=float, default=0.70423)
    parser.add_argument("--seconds-per-frame", type=float, required=True)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--flow-smoothing-px", type=float, default=12.0)
    parser.add_argument("--director-sigma-px", type=float, default=18.0)
    parser.add_argument("--keep-global-translation", action="store_true")
    parser.add_argument("--render-stride", type=int, default=10,
                        help="Render every Nth interval; metrics use all pairs.")
    parser.add_argument("--vector-step", type=int, default=16,
                        help="Quiver spacing on the downsampled map.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    wanted = set(args.stiffness)
    folders = {}
    for folder in args.frames.iterdir():
        if folder.is_dir():
            stiffness = stiffness_from_name(folder.name)
            if stiffness in wanted:
                folders[stiffness] = folder
    missing = wanted - set(folders)
    if missing:
        raise SystemExit(f"missing stiffness folders: {sorted(missing)}")

    settings = {
        "flow": {
            "downsample": args.downsample,
            "smoothing_px": args.flow_smoothing_px,
            "director_sigma_px": args.director_sigma_px,
            "subtract_median_translation": not args.keep_global_translation,
        },
        "mpp": args.microns_per_pixel,
        "spf": args.seconds_per_frame,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for stiffness in sorted(folders):
        paths = frame_paths(folders[stiffness])
        if args.max_frames:
            paths = paths[:args.max_frames]
        if len(paths) < 2:
            raise SystemExit(f"{stiffness:g} kPa has fewer than two frames")
        out = args.output / f"{stiffness:g}kPa"
        panels = out / "panels"
        panels.mkdir(parents=True, exist_ok=True)
        tasks = [(index, paths[index], paths[index + 1], settings)
                 for index in range(len(paths) - 1)]
        workers = max(1, min(args.n_jobs, len(tasks), os.cpu_count() or 1))
        summaries, panel_paths = [], []
        sums = {}
        count = 0
        executor = (ProcessPoolExecutor(max_workers=workers)
                    if workers > 1 else nullcontext())
        with executor as pool:
            # map releases completed results as they are consumed; retaining a
            # Future per dense field would otherwise keep roughly a gigabyte of
            # arrays alive over a 287-frame series. The serial path is useful
            # on login nodes where process semaphores are intentionally denied.
            results = (pool.map(analyse_pair, tasks, chunksize=1)
                       if pool is not None else map(analyse_pair, tasks))
            for index, summary, result in tqdm(
                results, total=len(tasks),
                desc=f"flow {stiffness:g} kPa [{workers}p]",
            ):
                summary["stiffness_kPa"] = stiffness
                summaries.append(summary)
                for key in ("u_px_per_frame", "v_px_per_frame",
                            "speed_px_per_frame", "flow_director_alignment",
                            "local_order"):
                    values = np.nan_to_num(result[key], nan=0.0).astype(float)
                    sums[key] = sums.get(key, np.zeros_like(values)) + values
                theta = result["director_theta_rad"]
                for key, values in (
                    ("director_cos2", np.cos(2 * theta)),
                    ("director_sin2", np.sin(2 * theta)),
                ):
                    sums[key] = sums.get(key, np.zeros_like(values)) + values
                count += 1
                if index % args.render_stride == 0:
                    panel = panels / f"flow_{index:06d}.png"
                    render_panel(result, panel, f"{stiffness:g} kPa, frame {index}",
                                 args.microns_per_pixel, args.seconds_per_frame,
                                 args.vector_step)
                    panel_paths.append((index, panel))

        table = pd.DataFrame(summaries).sort_values("frame")
        table.to_csv(out / "flow_metrics.tsv", sep="\t", index=False)
        all_summaries.append(table)
        means = {key: value / count for key, value in sums.items()}
        means["director_theta_rad"] = 0.5 * np.arctan2(
            means.pop("director_sin2"), means.pop("director_cos2")
        )
        means["mask"] = np.ones_like(means["local_order"], dtype=bool)
        render_panel(means, out / "mean_flow_panel.png",
                     f"{stiffness:g} kPa, temporal mean",
                     args.microns_per_pixel, args.seconds_per_frame,
                     args.vector_step)
        np.savez_compressed(out / "mean_flow_maps.npz", **means)
        if panel_paths:
            with imageio.get_writer(out / "flow_maps.mp4", fps=args.fps,
                                    # H.264 requires even dimensions. A block
                                    # size of two pads the rendered panel by at
                                    # most one pixel instead of failing late.
                                    macro_block_size=2) as writer:
                for _, panel in sorted(panel_paths):
                    writer.append_data(imageio.imread(panel))
        print(f"  wrote {out}")
    pd.concat(all_summaries).to_csv(args.output / "flow_metrics_all.tsv",
                                    sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
