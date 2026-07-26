"""Particle-based 2.5D view of the coupled alveolar fibrosis model.

This module does not replace the biological model.  It projects the existing
coupled epithelial/mesenchymal state into a particle representation:

* epithelial surface patches are aggregated into biologically proportioned
  AT1 and AT2 cell particles;
* every fibroblast or myofibroblast remains one elongated particle;
* open, collapsed and indurated alveoli are translucent domes;
* the evolving stiffness field is drawn beneath the cells.

The result is deliberately called 2.5D.  The biological dynamics still evolve
on the validated two-dimensional alveolar tessellation; the third coordinate is
a stable visual embedding that separates cells and makes the multicellular
architecture legible.  This keeps the numerical model auditable while providing
a stepping stone toward a true three-dimensional tissue simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .mesenchyme import CoupledSimulation
from .model import (
    ABERRANT,
    AT1,
    AT2,
    COLLAPSED,
    EMPTY,
    INDURATED,
    KRT8,
    AlveolarConfig,
)


EPITHELIAL_COLOURS = {
    AT1: "#28C76F",
    AT2: "#4DB8FF",
    KRT8: "#FFD23F",
    ABERRANT: "#FF7A00",
}

EPITHELIAL_LABELS = {
    AT1: "AT1",
    AT2: "AT2",
    KRT8: "KRT8+ transitional",
    ABERRANT: "aberrant basaloid",
}

ALVEOLAR_COLOURS = {
    0: "#A8DADC",         # open
    COLLAPSED: "#FFB000",
    INDURATED: "#E63946",
}


@dataclass(frozen=True)
class ParticleRenderConfig:
    """Visual parameters for the 2.5D particle rendering."""

    figure_width: float = 12.0
    figure_height: float = 8.0
    dpi: int = 110
    elevation_deg: float = 30.0
    azimuth_deg: float = -58.0
    orbit_deg_per_frame: float = 0.0
    alveolus_height_um: float = 82.0
    dome_rings: int = 5
    open_shell_alpha: float = 0.045
    collapsed_shell_alpha: float = 0.14
    indurated_shell_alpha: float = 0.20
    epithelial_marker_size: float = 22.0
    mesenchymal_marker_size: float = 8.0
    mesenchymal_linewidth: float = 1.5
    matrix_alpha: float = 0.30
    matrix_vertical_gain_um_per_kpa: float = 0.22
    matrix_stride: int = 2
    breathing_frames_per_cycle: int = 6
    # The footprint stays in the histological plane, so the roughly threefold
    # conversion from linear to volumetric strain is shown in dome height.
    breathing_height_gain: float = 3.0
    show_breathing: bool = True
    crop_to_tissue: bool = True
    tissue_padding_fraction: float = 0.035
    show_matrix: bool = True
    show_injury_region: bool = True
    show_legend: bool = True
    show_colourbar: bool = True
    black_background: bool = True

    def validate(self) -> None:
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise ValueError("Figure dimensions must be positive.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.alveolus_height_um <= 0:
            raise ValueError("alveolus_height_um must be positive.")
        if self.dome_rings < 2:
            raise ValueError("dome_rings must be at least 2.")
        if self.matrix_stride < 1:
            raise ValueError("matrix_stride must be at least 1.")
        if self.breathing_frames_per_cycle < 2:
            raise ValueError("breathing_frames_per_cycle must be at least 2.")
        if self.breathing_height_gain < 0:
            raise ValueError("breathing_height_gain must be non-negative.")
        if self.tissue_padding_fraction < 0:
            raise ValueError("tissue_padding_fraction must be non-negative.")
        for name in (
            "open_shell_alpha",
            "collapsed_shell_alpha",
            "indurated_shell_alpha",
            "matrix_alpha",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")


@dataclass(frozen=True)
class ParticleSnapshot:
    """Arrays needed to draw one frame."""

    epithelial_xyz: np.ndarray
    epithelial_state: np.ndarray
    mesenchymal_xyz: np.ndarray
    mesenchymal_theta: np.ndarray
    mesenchymal_myo: np.ndarray
    mesenchymal_stiffness_kpa: np.ndarray

    @property
    def n_epithelial(self) -> int:
        return int(self.epithelial_xyz.shape[0])

    @property
    def n_mesenchymal(self) -> int:
        return int(self.mesenchymal_xyz.shape[0])

    @property
    def n_myofibroblast(self) -> int:
        return int(self.mesenchymal_myo.sum())


def _alveolus_heights(
    coupled: CoupledSimulation,
    render: ParticleRenderConfig,
    frame_index: int = 0,
) -> np.ndarray:
    ep = coupled.epithelium
    ratio = ep.radius_um / np.maximum(ep.open_radius_um, 1e-9)
    heights = render.alveolus_height_um * np.clip(ratio, 0.12, 1.0)
    heights = np.where(
        ep.alveolar_state == INDURATED,
        np.minimum(heights, render.alveolus_height_um * 0.16),
        heights,
    )
    if render.show_breathing:
        inspiration = breathing_inspiration_fraction(render, frame_index)
        heights *= 1.0 + (
            render.breathing_height_gain
            * np.clip(_alveolar_tidal_strain(coupled), 0.0, 0.25)
            * inspiration
        )
    return heights


def breathing_inspiration_fraction(
    render: ParticleRenderConfig,
    frame_index: int,
) -> float:
    """Smooth respiratory phase: zero at expiration and one at inspiration."""

    phase = (
        2.0
        * np.pi
        * (frame_index % render.breathing_frames_per_cycle)
        / render.breathing_frames_per_cycle
    )
    return float(0.5 * (1.0 - np.cos(phase)))


def _alveolar_tidal_strain(coupled: CoupledSimulation) -> np.ndarray:
    """Mean local tidal strain amplitude for each alveolus."""

    ep = coupled.epithelium
    mes = coupled.mesenchyme
    values = np.zeros(ep.geometry.n_alveoli, dtype=float)
    for index in range(ep.geometry.n_alveoli):
        local = mes.strain[mes.alveolus_label == index]
        if local.size:
            values[index] = float(local.mean())
    return values


def _segment_heights(
    coupled: CoupledSimulation,
    alveolus_heights: np.ndarray,
) -> np.ndarray:
    geometry = coupled.epithelium.geometry
    values = np.zeros(geometry.n_segments, dtype=float)
    weights = np.zeros(geometry.n_segments, dtype=float)
    for column in (0, 1):
        owners = geometry.segment_alveoli[:, column]
        valid = owners >= 0
        values[valid] += alveolus_heights[owners[valid]]
        weights[valid] += 1.0
    return values / np.maximum(weights, 1.0)


def _epithelial_particle_indices(coupled: CoupledSimulation) -> np.ndarray:
    """Aggregate surface patches into stereologically proportioned cells.

    Each compact AT2, KRT8+ or aberrant patch remains one cell. AT1 patches are
    aggregated because one flat AT1 cell covers far more surface. At healthy
    initialization the visual count is approximately 1.7 AT2 cells per AT1
    cell while the state lattice retains 6% AT2 surface coverage.
    """

    ep = coupled.epithelium
    cfg = coupled.cfg
    state = ep.state
    compact = np.flatnonzero((state != EMPTY) & (state != AT1))
    at1 = np.flatnonzero(state == AT1)
    if at1.size == 0:
        return compact

    n_segments = ep.geometry.n_segments
    baseline_at2 = max(
        1,
        int(round(cfg.healthy_at2_surface_fraction * n_segments)),
    )
    baseline_at1_cells = max(
        1,
        int(round(
            baseline_at2 / cfg.healthy_at2_to_at1_number_ratio
        )),
    )
    baseline_at1_surface = max(n_segments - baseline_at2, 1)
    target_at1 = max(
        1,
        int(round(
            baseline_at1_cells * at1.size / baseline_at1_surface
        )),
    )
    positions = np.linspace(
        0,
        at1.size - 1,
        min(target_at1, at1.size),
        dtype=int,
    )
    return np.sort(np.concatenate([compact, at1[np.unique(positions)]]))


def build_particle_snapshot(
    coupled: CoupledSimulation,
    render_config: ParticleRenderConfig | None = None,
    frame_index: int = 0,
) -> ParticleSnapshot:
    """Project one coupled state into explicit epithelial and mesenchymal cells."""

    render = render_config or ParticleRenderConfig()
    render.validate()
    ep = coupled.epithelium
    mes = coupled.mesenchyme
    geometry = ep.geometry

    alveolus_heights = _alveolus_heights(coupled, render, frame_index)
    segment_heights = _segment_heights(coupled, alveolus_heights)
    segment_index = np.arange(geometry.n_segments)

    # A deterministic golden-ratio phase distributes cells through the shallow
    # septal height without making them jump between frames.
    phase = np.mod(segment_index * 0.6180339887498949, 1.0)
    epithelial_z = (0.14 + 0.72 * phase) * segment_heights
    epithelial_index = _epithelial_particle_indices(coupled)
    epithelial_xyz = np.column_stack(
        [
            geometry.segment_centre[epithelial_index, 0],
            geometry.segment_centre[epithelial_index, 1],
            epithelial_z[epithelial_index],
        ]
    )

    if mes.n_cells:
        ix = np.clip(
            (mes.x / mes.grid_step).astype(int),
            0,
            mes.nx - 1,
        )
        iy = np.clip(
            (mes.y / mes.grid_step).astype(int),
            0,
            mes.ny - 1,
        )
        owner = mes.alveolus_label[iy, ix]
        inside_alveolus = owner >= 0
        owner_state = np.full(mes.n_cells, -1, dtype=int)
        owner_state[inside_alveolus] = ep.alveolar_state[owner[inside_alveolus]]

        local_height = np.full(mes.n_cells, render.alveolus_height_um * 0.16)
        fillable = inside_alveolus & (owner_state != 0)
        local_height[fillable] = alveolus_heights[owner[fillable]]

        # Stable pseudo-depth derived from position: migration changes depth
        # smoothly, while surviving cells do not flicker randomly.
        depth_phase = 0.5 + 0.5 * np.sin(0.021 * mes.x + 0.017 * mes.y)
        mesenchymal_z = 3.0 + depth_phase * np.maximum(local_height * 0.78, 4.0)
        mesenchymal_xyz = np.column_stack([mes.x, mes.y, mesenchymal_z])
        local_stiffness = mes.stiffness_kPa[iy, ix]
    else:
        mesenchymal_xyz = np.zeros((0, 3), dtype=float)
        local_stiffness = np.zeros(0, dtype=float)

    return ParticleSnapshot(
        epithelial_xyz=epithelial_xyz,
        epithelial_state=ep.state[epithelial_index].copy(),
        mesenchymal_xyz=mesenchymal_xyz,
        mesenchymal_theta=mes.theta.copy(),
        mesenchymal_myo=mes.myo.copy(),
        mesenchymal_stiffness_kpa=local_stiffness,
    )


def _dome_faces(
    vertices: np.ndarray,
    centroid: np.ndarray,
    height: float,
    n_rings: int,
) -> list[list[tuple[float, float, float]]]:
    """Faceted dome over one alveolar polygon."""

    faces: list[list[tuple[float, float, float]]] = []
    n_vertices = len(vertices)
    ring_fraction = np.linspace(0.0, 1.0, n_rings)
    rings: list[np.ndarray] = []
    for fraction in ring_fraction:
        xy = centroid + fraction * (vertices - centroid)
        z = height * np.sqrt(max(0.0, 1.0 - fraction**2))
        rings.append(np.column_stack([xy, np.full(n_vertices, z)]))

    centre = (float(centroid[0]), float(centroid[1]), float(height))
    first = rings[1]
    for index in range(n_vertices):
        next_index = (index + 1) % n_vertices
        faces.append([centre, tuple(first[index]), tuple(first[next_index])])

    for ring_index in range(1, n_rings - 1):
        inner = rings[ring_index]
        outer = rings[ring_index + 1]
        for index in range(n_vertices):
            next_index = (index + 1) % n_vertices
            faces.append(
                [
                    tuple(inner[index]),
                    tuple(outer[index]),
                    tuple(outer[next_index]),
                    tuple(inner[next_index]),
                ]
            )
    return faces


def _draw_matrix_surface(axis, coupled, render):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    mes = coupled.mesenchyme
    cfg = coupled.cfg
    stiffness = mes.stiffness_kPa
    upper = max(cfg.E_act_kPa * 2.5, float(np.percentile(stiffness, 99)), 10.0)
    normalise = Normalize(vmin=cfg.E_healthy_kPa, vmax=upper, clip=True)
    colourmap = plt.get_cmap("magma")
    colours = colourmap(normalise(stiffness))

    activation = np.clip(
        (stiffness - cfg.E_healthy_kPa)
        / max(cfg.E_act_kPa - cfg.E_healthy_kPa, 1e-9),
        0.0,
        1.0,
    )
    colours[..., 3] = render.matrix_alpha * (0.18 + 0.82 * activation)
    z = -3.0 + render.matrix_vertical_gain_um_per_kpa * (
        stiffness - cfg.E_healthy_kPa
    )
    axis.plot_surface(
        mes.grid_x,
        mes.grid_y,
        z,
        rstride=render.matrix_stride,
        cstride=render.matrix_stride,
        facecolors=colours,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=0,
    )
    return normalise, colourmap


def _draw_alveolar_shells(axis, coupled, render, frame_index):
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ep = coupled.epithelium
    heights = _alveolus_heights(coupled, render, frame_index)
    faces = []
    face_colours = []
    for alveolus, state, height in zip(
        ep.geometry.alveoli,
        ep.alveolar_state,
        heights,
    ):
        dome = _dome_faces(
            alveolus.vertices,
            alveolus.centroid,
            float(height),
            render.dome_rings,
        )
        if state == COLLAPSED:
            alpha = render.collapsed_shell_alpha
        elif state == INDURATED:
            alpha = render.indurated_shell_alpha
        else:
            alpha = render.open_shell_alpha
        rgba = to_rgba(ALVEOLAR_COLOURS[int(state)], alpha)
        faces.extend(dome)
        face_colours.extend([rgba] * len(dome))

    collection = Poly3DCollection(
        faces,
        facecolors=face_colours,
        edgecolors=(0.75, 0.82, 0.88, 0.10),
        linewidths=0.25,
        zorder=1,
    )
    axis.add_collection3d(collection)


def _draw_epithelium(axis, snapshot, render):
    for state in (AT1, AT2, KRT8, ABERRANT):
        selected = snapshot.epithelial_state == state
        if not selected.any():
            continue
        xyz = snapshot.epithelial_xyz[selected]
        if state == AT1:
            # A broad horizontal glyph represents the very thin AT1 plate.
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=render.epithelial_marker_size * 8.0,
                c=EPITHELIAL_COLOURS[state],
                marker="_",
                linewidths=2.4,
                alpha=0.72,
                depthshade=True,
                zorder=3,
            )
        else:
            scale = {AT2: 1.0, KRT8: 1.35, ABERRANT: 1.5}[state]
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=render.epithelial_marker_size * scale,
                c=EPITHELIAL_COLOURS[state],
                edgecolors="none",
                alpha=0.94,
                depthshade=True,
                zorder=3,
            )


def _draw_mesenchyme(axis, coupled, snapshot, render):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if snapshot.n_mesenchymal == 0:
        return
    half_length = 0.42 * coupled.cfg.cell_length_um
    dx = half_length * np.cos(snapshot.mesenchymal_theta)
    dy = half_length * np.sin(snapshot.mesenchymal_theta)
    xyz = snapshot.mesenchymal_xyz
    segments = np.stack(
        [
            np.column_stack([xyz[:, 0] - dx, xyz[:, 1] - dy, xyz[:, 2]]),
            np.column_stack([xyz[:, 0] + dx, xyz[:, 1] + dy, xyz[:, 2]]),
        ],
        axis=1,
    )
    colours = np.where(snapshot.mesenchymal_myo, "#FF3B30", "#2F80ED")
    axis.add_collection3d(
        Line3DCollection(
            segments,
            colors=list(colours),
            linewidths=render.mesenchymal_linewidth,
            alpha=0.86,
            zorder=4,
        )
    )
    axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=render.mesenchymal_marker_size,
        c=list(colours),
        edgecolors="none",
        alpha=0.92,
        depthshade=True,
        zorder=5,
    )


def _style_axis(axis, coupled, render, frame_index):
    cfg = coupled.cfg
    foreground = "#E8EEF2" if render.black_background else "#1D2730"
    background = "#000000" if render.black_background else "#FFFFFF"
    axis.set_facecolor(background)
    if render.crop_to_tissue:
        vertices = np.vstack(
            [alveolus.vertices for alveolus in coupled.epithelium.geometry.alveoli]
        )
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        pad = render.tissue_padding_fraction * float(np.max(upper - lower))
        x_limits = (lower[0] - pad, upper[0] + pad)
        y_limits = (lower[1] - pad, upper[1] + pad)
    else:
        x_limits = (0.0, cfg.width_um)
        y_limits = (0.0, cfg.height_um)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(-5, render.alveolus_height_um * 1.35)
    aspect_y = (y_limits[1] - y_limits[0]) / max(
        x_limits[1] - x_limits[0],
        1e-9,
    )
    axis.set_box_aspect((1.0, aspect_y, 0.38))
    axis.view_init(
        elev=render.elevation_deg,
        azim=render.azimuth_deg + frame_index * render.orbit_deg_per_frame,
    )
    axis.set_xlabel("x (µm)", color=foreground, labelpad=8)
    axis.set_ylabel("y (µm)", color=foreground, labelpad=8)
    axis.set_zlabel("2.5D tissue depth", color=foreground, labelpad=8)
    axis.tick_params(colors=foreground, labelsize=7)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0, 0, 0, 0))
        pane.set_edgecolor((0.6, 0.68, 0.72, 0.25))
    axis.grid(False)


def _draw_injury_region(axis, coupled):
    ep = coupled.epithelium
    cfg = coupled.cfg
    angle = np.linspace(0, 2 * np.pi, 160)
    x = ep.injury_centre[0] + cfg.injury_radius_um * np.cos(angle)
    y = ep.injury_centre[1] + cfg.injury_radius_um * np.sin(angle)
    axis.plot(x, y, np.full_like(x, -1.5), "--", color="#F72585", lw=1.2, alpha=0.8)


def _add_legend(axis):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], marker="o", linestyle="", color=EPITHELIAL_COLOURS[state],
               label=EPITHELIAL_LABELS[state], markersize=6)
        for state in (AT1, AT2, KRT8, ABERRANT)
    ]
    handles.extend(
        [
            Line2D([], [], color="#2F80ED", lw=2, label="fibroblast"),
            Line2D([], [], color="#FF3B30", lw=2, label="myofibroblast"),
            Patch(facecolor=ALVEOLAR_COLOURS[COLLAPSED], alpha=0.5,
                  label="collapsed alveolus"),
            Patch(facecolor=ALVEOLAR_COLOURS[INDURATED], alpha=0.5,
                  label="indurated alveolus"),
        ]
    )
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=7,
        framealpha=0.55,
        facecolor="#111111",
        labelcolor="white",
        ncol=2,
    )


def draw_particle_frame(
    coupled: CoupledSimulation,
    output_path: str | Path,
    render_config: ParticleRenderConfig | None = None,
    frame_index: int = 0,
) -> dict:
    """Render one coupled simulation state as a 2.5D particle scene."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    render = render_config or ParticleRenderConfig()
    render.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    background = "#000000" if render.black_background else "#FFFFFF"
    foreground = "#F4F7F8" if render.black_background else "#17212B"
    figure = plt.figure(
        figsize=(render.figure_width, render.figure_height),
        facecolor=background,
    )
    axis = figure.add_subplot(111, projection="3d")
    snapshot = build_particle_snapshot(coupled, render, frame_index)

    normalise = colourmap = None
    if render.show_matrix:
        normalise, colourmap = _draw_matrix_surface(axis, coupled, render)
    _draw_alveolar_shells(axis, coupled, render, frame_index)
    _draw_epithelium(axis, snapshot, render)
    _draw_mesenchyme(axis, coupled, snapshot, render)
    if render.show_injury_region:
        _draw_injury_region(axis, coupled)
    _style_axis(axis, coupled, render, frame_index)
    if render.show_legend:
        _add_legend(axis)

    metrics = coupled.metrics()
    inspiration = (
        breathing_inspiration_fraction(render, frame_index)
        if render.show_breathing
        else 0.0
    )
    breathing_title = (
        f"breath {inspiration * 100:.0f}% inspired  |  "
        if render.show_breathing
        else ""
    )
    title = (
        f"day {metrics['time_d']:.1f}  |  {snapshot.n_epithelial} epithelial cells  "
        f"|  {snapshot.n_mesenchymal} mesenchymal  "
        f"|  {snapshot.n_myofibroblast} myofibroblasts\n"
        f"{breathing_title}"
        f"collapsed {metrics['frac_collapsed'] * 100:.0f}%  "
        f"indurated {metrics['frac_indurated'] * 100:.0f}%  "
        f"max stiffness {metrics['max_stiffness_kPa']:.1f} kPa"
    )
    axis.set_title(title, color=foreground, fontsize=11, pad=10)

    if render.show_colourbar and normalise is not None and colourmap is not None:
        scalar = ScalarMappable(norm=normalise, cmap=colourmap)
        scalar.set_array([])
        colourbar = figure.colorbar(scalar, ax=axis, fraction=0.025, pad=0.04)
        colourbar.set_label("ECM stiffness (kPa)", color=foreground)
        colourbar.ax.tick_params(colors=foreground, labelsize=7)

    figure.subplots_adjust(left=0.01, right=0.94, bottom=0.02, top=0.91)
    figure.savefig(
        output_path,
        dpi=render.dpi,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    metrics = dict(metrics)
    metrics.update(
        n_epithelial_particles=snapshot.n_epithelial,
        n_mesenchymal_particles=snapshot.n_mesenchymal,
        n_myofibroblast_particles=snapshot.n_myofibroblast,
        breathing_inspiration_fraction=inspiration,
    )
    return metrics


def _even_dimensions(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return image[: height - (height % 2), : width - (width % 2)]


def run_and_record_particles(
    config: AlveolarConfig,
    output_dir: str | Path,
    *,
    frame_every_h: float = 24.0,
    fps: int = 8,
    breathing_subframes: int = 1,
    render_config: ParticleRenderConfig | None = None,
    make_gif: bool = True,
    make_mp4: bool = True,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Run the coupled model and save particle frames, TSV, GIF and MP4."""

    if frame_every_h <= 0:
        raise ValueError("frame_every_h must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if breathing_subframes < 1:
        raise ValueError("breathing_subframes must be at least 1.")

    import imageio.v2 as imageio
    import pandas as pd

    render = render_config or ParticleRenderConfig()
    render.validate()
    output_dir = Path(output_dir)
    frames_dir = output_dir / "particle_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frames_dir.glob("frame_*.png"):
        stale_frame.unlink()
    config.to_json(output_dir / "particle_model_config.json")
    (output_dir / "particle_render_config.json").write_text(
        json.dumps(asdict(render), indent=2),
        encoding="utf-8",
    )

    coupled = CoupledSimulation(config)
    n_steps = int(round(config.total_time_h / config.dt_h))
    every = max(1, int(round(frame_every_h / config.dt_h)))
    frame_paths: list[Path] = []
    records: list[dict] = []

    def snapshot() -> None:
        for _ in range(breathing_subframes):
            frame_path = frames_dir / f"frame_{len(frame_paths):04d}.png"
            records.append(
                draw_particle_frame(
                    coupled,
                    frame_path,
                    render_config=render,
                    frame_index=len(frame_paths),
                )
            )
            frame_paths.append(frame_path)
            if progress is not None:
                progress(len(frame_paths))

    snapshot()
    for step in range(1, n_steps + 1):
        coupled.step()
        if step % every == 0 or step == n_steps:
            snapshot()

    images = [_even_dimensions(imageio.imread(path)) for path in frame_paths]
    outputs: dict = {"n_frames": len(frame_paths)}
    if make_gif:
        gif_path = output_dir / "alveolar_particles.gif"
        imageio.mimsave(gif_path, images, duration=1000.0 / fps, loop=0)
        outputs["gif"] = str(gif_path)
    if make_mp4:
        mp4_path = output_dir / "alveolar_particles.mp4"
        if mp4_path.exists():
            mp4_path.unlink()
        try:
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()
            imageio.mimsave(
                mp4_path,
                images,
                format="FFMPEG",
                fps=fps,
                macro_block_size=None,
            )
            outputs["mp4"] = str(mp4_path)
        except Exception as error:
            if mp4_path.exists():
                mp4_path.unlink()
            outputs["mp4_error"] = (
                "MP4 was skipped because FFmpeg is unavailable: "
                f"{type(error).__name__}: {error}"
            )

    timeseries_path = output_dir / "particle_timeseries.tsv"
    pd.DataFrame(records).to_csv(timeseries_path, sep="\t", index=False)
    outputs.update(
        timeseries=str(timeseries_path),
        final=records[-1],
        simulation=coupled,
    )
    return outputs


def accelerated_particle_demo_config(seed: int = 4) -> AlveolarConfig:
    """Small, accelerated scenario for visual testing.

    Rates are intentionally accelerated and are not a biological calibration.
    Use ``AlveolarConfig`` directly for scientific runs.
    """

    return AlveolarConfig(
        width_um=780.0,
        height_um=620.0,
        alveolar_diameter_um=145.0,
        segment_length_um=10.0,
        dt_h=0.5,
        total_time_h=360.0,
        injury_radius_um=190.0,
        injury_duration_h=300.0,
        repair_failure_factor=0.025,
        injury_activation_boost=18.0,
        krt8_to_aberrant_rate=0.008,
        aberrant_emt_rate=0.012,
        aberrant_clearance_rate=0.001,
        stall_promotion_strength=8.0,
        repair_inhibition_strength=5.0,
        induration_time_h=180.0,
        mesenchyme_grid_step_um=5.0,
        septal_thickness_um=5.0,
        n_resident_fibroblasts=100,
        E_act_kPa=8.0,
        activation_rate_per_h=0.12,
        deposition_rate_kPa_per_h=0.35,
        degradation_rate_per_h=0.002,
        myofibroblast_death_rate=0.00025,
        rate_scale=0.55,
        seed=seed,
    )
