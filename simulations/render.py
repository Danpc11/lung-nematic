"""Rendering of the alveolar epithelium simulation to frames, GIF and MP4.

Each frame shows the alveolar architecture and the state of the epithelium
lining it:

  * alveolar polygons filled by mechanical state - open, collapsed
    (derecruited but recoverable) or indurated (irreversibly lost);
  * septal segments coloured by epithelial state, so the loss of AT1 coverage
    and the accumulation of aberrant basaloid cells are directly visible;
  * the injury region, where AT2 cells cannot complete differentiation.

Colours follow the Wong colour-blind-safe palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch

from .model import (
    ABERRANT, AT1, AT2, COLLAPSED, EMPTY, INDURATED, KRT8, OPEN,
    AlveolarConfig, AlveolarSimulation,
)

STATE_COLOUR = {
    AT1: "#009E73",       # green: intact barrier
    AT2: "#56B4E9",       # light blue: progenitor
    KRT8: "#E69F00",      # amber: transitional
    ABERRANT: "#D55E00",  # vermillion: stuck / aberrant basaloid
    EMPTY: "#BBBBBB",     # grey: denuded basement membrane
}
ALVEOLAR_FILL = {
    OPEN: "#FFFFFF",
    COLLAPSED: "#CFCFCF",
    INDURATED: "#8C6D5B",
}


def draw_frame(sim: AlveolarSimulation, path: str | Path, dpi: int = 110) -> None:
    cfg = sim.cfg
    g = sim.geometry
    figure, axis = plt.subplots(figsize=(7.6, 7.9))

    polygons = [a.vertices for a in g.alveoli]
    colours = [ALVEOLAR_FILL[int(s)] for s in sim.alveolar_state]
    axis.add_collection(
        PolyCollection(polygons, facecolors=colours, edgecolors="#DDDDDD",
                       linewidths=0.5, zorder=1)
    )

    half = 0.5 * cfg.segment_length_um
    starts, ends = [], []
    for index in range(g.n_segments):
        centre = g.segment_centre[index]
        septum = g.septa[g.segment_septum[index]]
        direction = septum.end - septum.start
        unit = direction / max(np.linalg.norm(direction), 1e-9)
        starts.append(centre - half * unit)
        ends.append(centre + half * unit)
    segments = np.stack([np.array(starts), np.array(ends)], axis=1)
    seg_colours = [STATE_COLOUR[int(s)] for s in sim.state]
    axis.add_collection(
        LineCollection(segments, colors=seg_colours, linewidths=3.4,
                       capstyle="round", zorder=3)
    )

    injured = sim.time_h < cfg.injury_duration_h
    circle = plt.Circle(
        sim.injury_centre, cfg.injury_radius_um, fill=False,
        edgecolor="#CC79A7", linewidth=1.6,
        linestyle="--" if injured else ":",
        alpha=0.9 if injured else 0.35, zorder=4,
    )
    axis.add_patch(circle)

    metrics = sim.metrics()
    lesion = ("AT2->AT1 block ACTIVE" if injured else "block lifted")
    axis.set_title(
        f"day {metrics['time_d']:.0f}   |   {lesion}\n"
        f"AT1 {metrics['frac_AT1']*100:.0f}%   aberrant {metrics['frac_aberrant']*100:.0f}%   "
        f"denuded {metrics['frac_denuded']*100:.0f}%   |   "
        f"alveoli open {metrics['frac_open']*100:.0f}%   indurated {metrics['frac_indurated']*100:.0f}%\n"
        f"surfactant {metrics['mean_surfactant']:.2f}   EMT released {metrics['mesenchymal_released']}",
        fontsize=10,
    )
    axis.set_xlim(0, cfg.width_um)
    axis.set_ylim(0, cfg.height_um)
    axis.set_aspect("equal")
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")

    handles = [
        plt.Line2D([], [], color=STATE_COLOUR[AT1], lw=3, label="AT1"),
        plt.Line2D([], [], color=STATE_COLOUR[AT2], lw=3, label="AT2"),
        plt.Line2D([], [], color=STATE_COLOUR[KRT8], lw=3, label="KRT8+ transitional"),
        plt.Line2D([], [], color=STATE_COLOUR[ABERRANT], lw=3, label="aberrant basaloid"),
        plt.Line2D([], [], color=STATE_COLOUR[EMPTY], lw=3, label="denuded"),
        Patch(facecolor=ALVEOLAR_FILL[COLLAPSED], edgecolor="#999999", label="collapsed"),
        Patch(facecolor=ALVEOLAR_FILL[INDURATED], edgecolor="#999999", label="indurated"),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)

    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _even_dimensions(image: np.ndarray) -> np.ndarray:
    """Trim to even width and height; H.264 rejects odd frame sizes."""
    height, width = image.shape[:2]
    return image[: height - (height % 2), : width - (width % 2)]


def run_and_record(config: AlveolarConfig, output_dir: str | Path,
                   frame_every_h: float = 24.0, fps: int = 10,
                   make_gif: bool = True, make_mp4: bool = True) -> dict:
    """Run the epithelial simulation and assemble a movie of it."""
    import imageio.v2 as imageio
    import pandas as pd

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    sim = AlveolarSimulation(config)
    n_steps = int(round(config.total_time_h / config.dt_h))
    every = max(1, int(round(frame_every_h / config.dt_h)))

    paths: list[Path] = []
    records: list[dict] = []

    def snapshot() -> None:
        path = frames_dir / f"frame_{len(paths):04d}.png"
        draw_frame(sim, path)
        paths.append(path)
        records.append(sim.metrics())

    snapshot()
    for index in range(1, n_steps + 1):
        sim.step()
        if index % every == 0:
            snapshot()

    images = [imageio.imread(p) for p in paths]
    images = [_even_dimensions(image) for image in images]
    outputs: dict = {"n_frames": len(paths)}
    if make_gif:
        gif_path = output_dir / "alveolar_simulation.gif"
        imageio.mimsave(gif_path, images, fps=fps, loop=0)
        outputs["gif"] = str(gif_path)
    if make_mp4:
        mp4_path = output_dir / "alveolar_simulation.mp4"
        imageio.mimsave(mp4_path, images, fps=fps, macro_block_size=None)
        outputs["mp4"] = str(mp4_path)

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "timeseries.csv", index=False)
    outputs["timeseries"] = str(output_dir / "timeseries.csv")
    outputs["final"] = records[-1]
    return outputs
