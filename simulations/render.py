"""Rendering of the fibroblastic-focus simulation to frames, GIF and MP4.

Each frame shows three coupled layers:
  * the substrate stiffness field E (background heat map, kPa);
  * the cells as elongated segments, coloured by phenotype
    (quiescent fibroblast vs activated myofibroblast);
  * the candidate +/-1/2 nematic defects detected in the coarse-grained
    director field, using the same winding criterion applied to histology.

Colours follow the Wong colour-blind-safe palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

from .model import FocusConfig, FocusSimulation

FIBROBLAST = "#56B4E9"
MYOFIBROBLAST = "#D55E00"
DEFECT_PLUS = "#0072B2"
DEFECT_MINUS = "#E69F00"

STIFFNESS_CMAP = LinearSegmentedColormap.from_list(
    "stiffness", ["#f7f7f7", "#e8dcc8", "#c9a227", "#8c4a1e", "#3b1f14"]
)


def _rod_segments(sim: FocusSimulation) -> np.ndarray:
    """End points of each cell's long axis (kept for diagnostics)."""
    cfg = sim.cfg
    half = 0.5 * cfg.rod_length_um
    ux, uy = np.cos(sim.theta), np.sin(sim.theta)
    x0, y0 = sim.x - half * ux, sim.y - half * uy
    x1, y1 = sim.x + half * ux, sim.y + half * uy
    return np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)


def _cell_collection(sim: FocusSimulation, mask: np.ndarray, axis,
                     facecolor: str, edgecolor: str, ghosts: bool = True):
    """Cells drawn as filled ellipses with their true footprint in microns.

    ``units='xy'`` makes the width and height data coordinates, so the cells
    occupy real area on the substrate instead of a fixed number of screen
    points. Cells near a boundary are duplicated across the periodic edges so
    the tissue looks continuous.
    """
    cfg = sim.cfg
    if not mask.any():
        return None

    x, y, theta = sim.x[mask], sim.y[mask], sim.theta[mask]

    if ghosts:
        margin = 0.5 * cfg.rod_length_um
        shifts = [(0.0, 0.0)]
        for dx in (-cfg.width_um, 0.0, cfg.width_um):
            for dy in (-cfg.height_um, 0.0, cfg.height_um):
                if dx == 0.0 and dy == 0.0:
                    continue
                shifts.append((dx, dy))
        xs, ys, ts = [x], [y], [theta]
        for dx, dy in shifts[1:]:
            gx, gy = x + dx, y + dy
            near = (
                (gx > -margin) & (gx < cfg.width_um + margin)
                & (gy > -margin) & (gy < cfg.height_um + margin)
            )
            if near.any():
                xs.append(gx[near])
                ys.append(gy[near])
                ts.append(theta[near])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        theta = np.concatenate(ts)

    n = x.size
    collection = EllipseCollection(
        widths=np.full(n, cfg.rod_length_um),
        heights=np.full(n, cfg.rod_width_um),
        angles=np.degrees(theta),
        units="xy",
        offsets=np.column_stack([x, y]),
        offset_transform=axis.transData,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=0.4,
        alpha=0.80,
    )
    return collection


def detect_defects(sim: FocusSimulation, sigma_um: float = 25.0,
                   grid_step: int = 3, min_distance_um: float = 30.0,
                   min_packing_fraction: float = 0.30) -> dict:
    """Locate +/-1/2 defects in the simulated director field.

    Uses the same construction as the histology pipeline: the winding of the
    doubled phase around a plaquette of the coarse-grained director.

    A defect is only meaningful where a nematic phase actually exists, so
    detection is gated on the *absolute* local packing fraction rather than on
    a relative quantile. Below ``min_packing_fraction`` the rods are too dilute
    to be orientationally ordered and any winding is noise.
    """
    cfg = sim.cfg
    field = sim.director_field(sigma_um=sigma_um)
    theta = field["theta"]
    density = field["density"]
    ny, nx = theta.shape

    ys = np.arange(0, ny - grid_step, grid_step)
    xs = np.arange(0, nx - grid_step, grid_step)
    if ys.size < 2 or xs.size < 2:
        return {"plus": np.zeros((0, 2)), "minus": np.zeros((0, 2))}

    corners = [
        theta[np.ix_(ys, xs)],
        theta[np.ix_(ys, xs + grid_step)],
        theta[np.ix_(ys + grid_step, xs + grid_step)],
        theta[np.ix_(ys + grid_step, xs)],
    ]
    phases = [2 * c for c in corners]
    winding = np.zeros_like(phases[0])
    for k in range(4):
        step = phases[(k + 1) % 4] - phases[k]
        winding += np.arctan2(np.sin(step), np.cos(step))
    charge = winding / (4 * np.pi)

    dens_here = density[np.ix_(ys, xs)]
    # Absolute gate: local packing fraction must exceed min_packing_fraction.
    # `density` counts cells per grid cell, so convert to a packing fraction.
    packing = dens_here * cfg.cell_area_um2 / (cfg.grid_step_um ** 2)
    valid = packing >= min_packing_fraction
    out = {}
    for key, target in (("plus", 0.5), ("minus", -0.5)):
        hit = valid & (np.abs(charge - target) < 0.2)
        row, col = np.nonzero(hit)
        px = (xs[col] + grid_step / 2) * cfg.grid_step_um
        py = (ys[row] + grid_step / 2) * cfg.grid_step_um
        points = np.column_stack([px, py])
        out[key] = _thin(points, min_distance_um)
    return out


def _thin(points: np.ndarray, min_distance: float) -> np.ndarray:
    """Greedy spatial thinning so one defect is not counted many times."""
    if len(points) == 0:
        return points
    kept: list[np.ndarray] = []
    for point in points:
        if all(np.hypot(*(point - other)) >= min_distance for other in kept):
            kept.append(point)
    return np.array(kept)


def draw_frame(sim: FocusSimulation, path: str | Path,
               show_defects: bool = True, dpi: int = 110) -> dict:
    """Render one frame; returns the defect counts for bookkeeping."""
    cfg = sim.cfg
    figure, axis = plt.subplots(figsize=(7.2, 7.4))

    norm = Normalize(vmin=cfg.E_healthy_kPa, vmax=max(cfg.E_act_kPa * 2.5, 40))
    image = axis.imshow(
        sim.E, origin="lower", cmap=STIFFNESS_CMAP, norm=norm,
        extent=[0, cfg.width_um, 0, cfg.height_um], interpolation="bilinear",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("substrate stiffness E (kPa)")

    # threshold contours: the two thresholds that define the switch
    axis.contour(sim.grid_x, sim.grid_y, sim.E, levels=[cfg.E_tgfb_kPa],
                 colors=["#999999"], linewidths=0.8, linestyles="dotted")
    axis.contour(sim.grid_x, sim.grid_y, sim.E, levels=[cfg.E_act_kPa],
                 colors=["#000000"], linewidths=1.2, linestyles="dashed")

    for mask, face, edge in (
        (~sim.myo, FIBROBLAST, "#2a6f97"),
        (sim.myo, MYOFIBROBLAST, "#7f2704"),
    ):
        collection = _cell_collection(sim, mask, axis, face, edge)
        if collection is not None:
            axis.add_collection(collection)

    counts = {"n_plus": 0, "n_minus": 0}
    if show_defects:
        defects = detect_defects(sim)
        if len(defects["plus"]):
            axis.scatter(defects["plus"][:, 0], defects["plus"][:, 1], marker="+",
                         s=150, linewidths=2.2, color=DEFECT_PLUS, zorder=5)
        if len(defects["minus"]):
            axis.scatter(defects["minus"][:, 0], defects["minus"][:, 1], marker="x",
                         s=120, linewidths=2.2, color=DEFECT_MINUS, zorder=5)
        counts = {"n_plus": len(defects["plus"]), "n_minus": len(defects["minus"])}

    metrics = sim.focus_metrics()
    injury_on = sim.time_h < cfg.injury_duration_h
    state = "ATII injury ACTIVE" if injury_on else "injury withdrawn"
    axis.set_title(
        f"day {metrics['time_d']:.1f}   |   {state}\n"
        f"cells {metrics['n_cells']} (area {metrics['area_fraction']*100:.0f}%)   "
        f"myofibroblasts {metrics['myo_fraction']*100:.0f}%   "
        f"E(focus) {metrics['E_focus_kPa']:.1f} kPa   "
        f"defects +{counts['n_plus']}/-{counts['n_minus']}",
        fontsize=10,
    )
    axis.set_xlim(0, cfg.width_um)
    axis.set_ylim(0, cfg.height_um)
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_aspect("equal")

    handles = [
        Patch(facecolor=FIBROBLAST, edgecolor="#2a6f97", label="fibroblast"),
        Patch(facecolor=MYOFIBROBLAST, edgecolor="#7f2704", label="myofibroblast"),
        plt.Line2D([], [], color="black", lw=1.2, ls="--",
                   label=f"E = {cfg.E_act_kPa:.0f} kPa (phenotype threshold)"),
        plt.Line2D([], [], color=DEFECT_PLUS, marker="+", ls="", label="+1/2 defect"),
        plt.Line2D([], [], color=DEFECT_MINUS, marker="x", ls="", label="-1/2 defect"),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return counts


def run_and_record(config: FocusConfig, output_dir: str | Path,
                   frame_every_h: float = 4.0, fps: int = 12,
                   make_gif: bool = True, make_mp4: bool = True,
                   show_defects: bool = True) -> dict:
    """Run a simulation, write every frame, then assemble GIF and/or MP4."""
    import imageio.v2 as imageio

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    sim = FocusSimulation(config)
    n_steps = int(round(config.total_time_h / config.dt_h))
    every = max(1, int(round(frame_every_h / config.dt_h)))

    frame_paths: list[Path] = []
    records: list[dict] = []

    def snapshot() -> None:
        path = frames_dir / f"frame_{len(frame_paths):04d}.png"
        counts = draw_frame(sim, path, show_defects=show_defects)
        frame_paths.append(path)
        record = sim.focus_metrics()
        record.update(counts)
        records.append(record)

    snapshot()
    for step_index in range(1, n_steps + 1):
        sim.step()
        if step_index % every == 0:
            snapshot()

    images = [imageio.imread(p) for p in frame_paths]
    outputs: dict[str, str] = {}
    if make_gif:
        gif_path = output_dir / "focus_simulation.gif"
        imageio.mimsave(gif_path, images, fps=fps, loop=0)
        outputs["gif"] = str(gif_path)
    if make_mp4:
        mp4_path = output_dir / "focus_simulation.mp4"
        imageio.mimsave(mp4_path, images, fps=fps, macro_block_size=None)
        outputs["mp4"] = str(mp4_path)

    import pandas as pd
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "timeseries.csv", index=False)
    outputs["timeseries"] = str(output_dir / "timeseries.csv")
    outputs["n_frames"] = len(frame_paths)
    outputs["final"] = records[-1]
    return outputs
