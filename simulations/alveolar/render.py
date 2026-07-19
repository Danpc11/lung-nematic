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


def draw_coupled_frame(coupled, path: str | Path, dpi: int = 110,
                       show_strain_panel: bool = True,
                       show_defects: bool = True,
                       stiffness_cmap: str = "pink_r",
                       strain_cmap: str = "RdBu_r",
                       cell_alpha: float = 0.55) -> dict:
    """Frame of the coupled model: alveoli, epithelium and mesenchyme together."""
    from matplotlib.collections import EllipseCollection

    cfg = coupled.cfg
    sim = coupled.epithelium
    mes = coupled.mesenchyme
    g = sim.geometry
    if show_strain_panel:
        figure, (axis, strain_axis) = plt.subplots(1, 2, figsize=(14.6, 7.8))
    else:
        figure, axis = plt.subplots(figsize=(8.0, 8.4))
        strain_axis = None

    axis.imshow(mes.stiffness_kPa, origin="lower", cmap=stiffness_cmap,
                extent=[0, cfg.width_um, 0, cfg.height_um],
                vmin=cfg.E_healthy_kPa, vmax=max(cfg.E_act_kPa * 2.5, 40),
                interpolation="bilinear", zorder=0)

    polygons = [a.vertices for a in g.alveoli]
    colours = [ALVEOLAR_FILL[int(s)] for s in sim.alveolar_state]
    alphas = [0.0 if int(s) == OPEN else 0.45 for s in sim.alveolar_state]
    collection = PolyCollection(polygons, facecolors=colours, edgecolors="#BBBBBB",
                                linewidths=0.6, zorder=1)
    collection.set_alpha(alphas)
    axis.add_collection(collection)

    if mes.n_cells:
        n = mes.n_cells
        face = np.where(mes.myo, "#D55E00", "#0072B2")
        axis.add_collection(EllipseCollection(
            widths=np.full(n, cfg.cell_length_um),
            heights=np.full(n, cfg.cell_width_um),
            angles=np.degrees(mes.theta), units="xy",
            offsets=np.column_stack([mes.x, mes.y]),
            offset_transform=axis.transData,
            facecolors=list(face), edgecolors="none", alpha=cell_alpha, zorder=2,
        ))

    half = 0.5 * cfg.segment_length_um
    starts, ends = [], []
    for index in range(g.n_segments):
        centre = g.segment_centre[index]
        septum = g.septa[g.segment_septum[index]]
        direction = septum.end - septum.start
        unit = direction / max(np.linalg.norm(direction), 1e-9)
        starts.append(centre - half * unit)
        ends.append(centre + half * unit)
    axis.add_collection(LineCollection(
        np.stack([np.array(starts), np.array(ends)], axis=1),
        colors=[STATE_COLOUR[int(s)] for s in sim.state],
        linewidths=2.2, capstyle="round", zorder=3,
    ))

    defects = mes.detect_defects()
    for key, marker, colour in (("plus", "+", "#000000"), ("minus", "x", "#7F3B08")):
        pts = defects[key] if show_defects else np.zeros((0, 2))
        if len(pts):
            axis.scatter(pts[:, 0], pts[:, 1], marker=marker, s=110,
                         linewidths=2.0, color=colour, zorder=5)

    m = coupled.metrics()
    axis.set_title(
        f"day {m['time_d']:.0f}   |   aberrant {m['frac_aberrant']*100:.1f}%   "
        f"indurated {m['frac_indurated']*100:.0f}%\n"
        f"mesenchymal {m['n_mesenchymal']} ({m['n_myofibroblast']} myofibroblast)   "
        f"packing {m['packing_in_permitted']:.2f}   E_max {m['max_stiffness_kPa']:.0f} kPa\n"
        f"septum {m['mean_septal_thickness_um']:.1f} um   "
        f"defects +{m['n_defect_plus']}/-{m['n_defect_minus']}",
        fontsize=10,
    )
    axis.set_xlim(0, cfg.width_um); axis.set_ylim(0, cfg.height_um)
    axis.set_aspect("equal"); axis.set_xlabel("x (um)"); axis.set_ylabel("y (um)")

    handles = [
        plt.Line2D([], [], color="#0072B2", lw=4, label="fibroblast"),
        plt.Line2D([], [], color="#D55E00", lw=4, label="myofibroblast"),
        plt.Line2D([], [], color=STATE_COLOUR[AT1], lw=2.5, label="AT1"),
        plt.Line2D([], [], color=STATE_COLOUR[ABERRANT], lw=2.5, label="aberrant basaloid"),
        Patch(facecolor=ALVEOLAR_FILL[INDURATED], alpha=0.45, label="indurated alveolus"),
        plt.Line2D([], [], color="k", marker="+", ls="", label="+1/2 defect"),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)

    # --- second panel: where the breath actually goes ---
    if strain_axis is None:
        figure.tight_layout(); figure.savefig(path, dpi=dpi); plt.close(figure)
        return m
    strain = mes.strain
    image = strain_axis.imshow(
        strain / cfg.tidal_strain, origin="lower", cmap=strain_cmap,
        extent=[0, cfg.width_um, 0, cfg.height_um],
        vmin=0.0, vmax=2.0, interpolation="bilinear",
    )
    bar = figure.colorbar(image, ax=strain_axis, fraction=0.046, pad=0.02)
    bar.set_label("tidal strain / normal")
    strain_axis.add_collection(PolyCollection(
        polygons, facecolors="none", edgecolors="#444444", linewidths=0.5,
    ))
    strain_axis.set_xlim(0, cfg.width_um); strain_axis.set_ylim(0, cfg.height_um)
    strain_axis.set_aspect("equal"); strain_axis.set_xlabel("x (um)")
    strain_axis.set_title(
        "Breathing: fixed tidal volume, redistributed\n"
        f"ventilated tissue now strains {m['strain_amplification']:.2f}x normal   "
        f"(blue = shielded, red = overstretched)\n"
        f"injury threshold {cfg.overstrain_threshold / cfg.tidal_strain:.2f}x",
        fontsize=10,
    )
    figure.tight_layout(); figure.savefig(path, dpi=dpi); plt.close(figure)
    return m


def run_and_record_coupled(config, output_dir: str | Path,
                           frame_every_h: float = 24.0, fps: int = 8,
                           dpi: int = 110, show_strain_panel: bool = True,
                           show_defects: bool = True,
                           stiffness_cmap: str = "pink_r",
                           strain_cmap: str = "RdBu_r",
                           cell_alpha: float = 0.55,
                           make_gif: bool = True, make_mp4: bool = True,
                           progress=None) -> dict:
    """Run the coupled epithelium + mesenchyme model and assemble a movie."""
    import imageio.v2 as imageio
    import pandas as pd

    from .mesenchyme import CoupledSimulation

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    coupled = CoupledSimulation(config)
    n_steps = int(round(config.total_time_h / config.dt_h))
    every = max(1, int(round(frame_every_h / config.dt_h)))
    paths: list[Path] = []
    records: list[dict] = []

    def snapshot() -> None:
        path = frames_dir / f"frame_{len(paths):04d}.png"
        records.append(draw_coupled_frame(
            coupled, path, dpi=dpi, show_strain_panel=show_strain_panel,
            show_defects=show_defects, stiffness_cmap=stiffness_cmap,
            strain_cmap=strain_cmap, cell_alpha=cell_alpha,
        ))
        paths.append(path)
        if progress is not None:
            progress(len(paths))

    snapshot()
    for index in range(1, n_steps + 1):
        coupled.step()
        if index % every == 0:
            snapshot()

    images = [_even_dimensions(imageio.imread(p)) for p in paths]
    outputs = {"n_frames": len(paths)}
    if make_gif:
        gif_path = output_dir / "coupled_simulation.gif"
        imageio.mimsave(gif_path, images, fps=fps, loop=0)
        outputs["gif"] = str(gif_path)
    if make_mp4:
        mp4_path = output_dir / "coupled_simulation.mp4"
        imageio.mimsave(mp4_path, images, fps=fps, macro_block_size=None)
        outputs["mp4"] = str(mp4_path)
    pd.DataFrame(records).to_csv(output_dir / "timeseries.csv", index=False)
    outputs["timeseries"] = str(output_dir / "timeseries.csv")
    outputs["final"] = records[-1]
    return outputs
