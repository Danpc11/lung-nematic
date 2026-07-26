"""Rendering and animation for the genuine 3D alveolar model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .model import (
    ABERRANT,
    AT1,
    AT2,
    COLLAPSED,
    INDURATED,
    KRT8,
    OPEN,
    Alveolar3DConfig,
    Alveolar3DSimulation,
)

EPITHELIAL_COLOURS = {
    AT1: "#2ECC71",
    AT2: "#4DB8FF",
    KRT8: "#FFD23F",
    ABERRANT: "#FF7A00",
}

ALVEOLAR_COLOURS = {
    OPEN: "#8ECAE6",
    COLLAPSED: "#FFB000",
    INDURATED: "#E63946",
}


def format_calendar_time(days: float) -> str:
    """Human-readable calendar time without hiding the exact day."""

    if days >= 365.25:
        return f"year {days / 365.25:.2f} (day {days:.0f})"
    if days >= 60.0:
        return f"month {days / 30.4375:.1f} (day {days:.0f})"
    return f"day {days:.1f}"


@dataclass(frozen=True)
class Alveolar3DRenderConfig:
    """Visual parameters independent from the biological configuration."""

    figure_width: float = 12.0
    figure_height: float = 9.0
    dpi: int = 105
    elevation_deg: float = 24.0
    azimuth_deg: float = -58.0
    orbit_deg_per_frame: float = 0.35
    breathing_frames_per_cycle: int = 6
    sphere_latitudes: int = 14
    sphere_longitudes: int = 22
    open_alpha: float = 0.055
    collapsed_alpha: float = 0.19
    indurated_alpha: float = 0.24
    fibroblast_linewidth: float = 1.5
    fibroblast_marker_size: float = 7.0
    matrix_points_per_alveolus: int = 80
    show_matrix: bool = True
    show_legend: bool = True
    black_background: bool = True

    # Presentation mode strips the scientific scaffolding (3D axes, panes,
    # colourbar, multi-line technical title) and shows a clean, solid-shaded
    # tissue over black with a compact corner legend, closer to a biomedical
    # illustration. It stays fully deterministic and driven by the model - it is
    # a display style, not a generative step. The diagnostic look is the default
    # so nothing downstream changes unless this is turned on.
    presentation: bool = False
    presentation_title: str = ""

    def validate(self) -> None:
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise ValueError("Figure dimensions must be positive.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.breathing_frames_per_cycle < 2:
            raise ValueError("breathing_frames_per_cycle must be at least 2.")
        if self.sphere_latitudes < 6 or self.sphere_longitudes < 8:
            raise ValueError("Sphere resolution is too low.")
        if self.matrix_points_per_alveolus < 0:
            raise ValueError("matrix_points_per_alveolus cannot be negative.")


def inspiration_fraction(
    render: Alveolar3DRenderConfig,
    frame_index: int,
) -> float:
    """Sinusoidal breath from end-expiration (0) to inspiration (1)."""

    phase = (
        2.0
        * np.pi
        * (frame_index % render.breathing_frames_per_cycle)
        / render.breathing_frames_per_cycle
    )
    return float(0.5 * (1.0 - np.cos(phase)))


def _sphere_mesh(
    centre: np.ndarray,
    radius: float,
    render: Alveolar3DRenderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.linspace(0.0, 2.0 * np.pi, render.sphere_longitudes)
    latitude = np.linspace(0.0, np.pi, render.sphere_latitudes)
    lon, lat = np.meshgrid(longitude, latitude)
    x = centre[0] + radius * np.sin(lat) * np.cos(lon)
    y = centre[1] + radius * np.sin(lat) * np.sin(lon)
    z = centre[2] + radius * np.cos(lat)
    return x, y, z


def _alveolar_mesh(
    simulation: Alveolar3DSimulation,
    alveolus: int,
    render: Alveolar3DRenderConfig,
    breath: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface mesh after breathing and directional interdependence."""

    longitude = np.linspace(
        0.0,
        2.0 * np.pi,
        render.sphere_longitudes,
    )
    latitude = np.linspace(0.0, np.pi, render.sphere_latitudes)
    lon, lat = np.meshgrid(longitude, latitude)
    normals = np.stack(
        [
            np.sin(lat) * np.cos(lon),
            np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ],
        axis=-1,
    )
    xyz = simulation.alveolar_surface_xyz(
        alveolus,
        normals,
        breath,
    )
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _draw_alveoli(
    axis,
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
    breath: float,
) -> None:
    from matplotlib.colors import to_rgba

    for index in range(simulation.n_alveoli):
        state = int(simulation.alveolar_state[index])
        x, y, z = _alveolar_mesh(
            simulation,
            index,
            render,
            breath,
        )
        if state == OPEN:
            alpha = render.open_alpha
        elif state == COLLAPSED:
            alpha = render.collapsed_alpha
        else:
            alpha = render.indurated_alpha

        if render.presentation:
            # Solid, shaded shells with a warm septal tint read as tissue rather
            # than a translucent wireframe. Alpha is raised so the walls have
            # body; shade=True gives the spheres volume under a light source.
            presentation_alpha = min(1.0, alpha + 0.32) if state == OPEN else min(1.0, alpha + 0.45)
            colour = to_rgba(ALVEOLAR_COLOURS[state], presentation_alpha)
            axis.plot_surface(
                x,
                y,
                z,
                color=colour,
                edgecolor="none",
                antialiased=True,
                shade=True,
                zorder=1,
            )
            continue

        colour = to_rgba(ALVEOLAR_COLOURS[state], alpha)
        axis.plot_surface(
            x,
            y,
            z,
            color=colour,
            edgecolor=(0.72, 0.82, 0.88, 0.11),
            linewidth=0.22,
            antialiased=True,
            shade=False,
            zorder=1,
        )

        # The faint original boundary makes loss of aerated volume explicit.
        if state != OPEN:
            x0, y0, z0 = _sphere_mesh(
                simulation.centres[index],
                float(simulation.open_radius_um[index]),
                render,
            )
            axis.plot_wireframe(
                x0,
                y0,
                z0,
                rstride=3,
                cstride=4,
                color=ALVEOLAR_COLOURS[state],
                linewidth=0.25,
                alpha=0.14,
                zorder=0,
            )


def _draw_epithelium(
    axis,
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
    breath: float,
) -> None:
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xyz = simulation.epithelial_xyz(breath)
    normals = simulation.epithelial_normal
    owner = simulation.epithelial_owner
    respiratory_scale = simulation.respiratory_scale(breath)
    surface_factor = simulation.surface_radial_factor(owner, normals)
    alveolar_linear_scale = (
        simulation.radius_um[owner]
        / simulation.open_radius_um[owner]
        * respiratory_scale[owner]
        * np.sqrt(surface_factor)
    )

    # AT1 cells are thin, broad plates; AT2 cells are compact and cuboidal.
    # Tangential polygon area is derived from measured human apical area.
    area_by_state = {
        AT1: simulation.cfg.at1_apical_surface_um2,
        AT2: simulation.cfg.at2_apical_surface_um2,
        KRT8: 1.35 * simulation.cfg.at2_apical_surface_um2,
        ABERRANT: 1.65 * simulation.cfg.at2_apical_surface_um2,
    }
    angles = np.linspace(0.0, 2.0 * np.pi, 15, endpoint=False)
    for state in (AT1, AT2, KRT8, ABERRANT):
        selected = simulation.epithelial_state == state
        if not np.any(selected):
            continue
        polygons: list[np.ndarray] = []
        for centre, normal, local_scale in zip(
            xyz[selected],
            normals[selected],
            alveolar_linear_scale[selected],
        ):
            reference = (
                np.array([0.0, 0.0, 1.0])
                if abs(normal[2]) < 0.90
                else np.array([0.0, 1.0, 0.0])
            )
            tangent_u = np.cross(normal, reference)
            tangent_u /= max(np.linalg.norm(tangent_u), 1e-12)
            tangent_v = np.cross(normal, tangent_u)
            radius = (
                np.sqrt(area_by_state[state] / np.pi)
                * local_scale
            )
            # Offset slightly into the lumen to keep the plate visible over
            # the translucent alveolar shell.
            visible_centre = centre + 0.45 * normal
            polygons.append(
                visible_centre
                + radius
                * (
                    np.cos(angles)[:, None] * tangent_u
                    + np.sin(angles)[:, None] * tangent_v
                )
            )

        is_at1 = state == AT1
        face_alpha = 0.12 if is_at1 else 0.88
        edge_alpha = 0.52 if is_at1 else 0.92
        collection = Poly3DCollection(
            polygons,
            facecolors=to_rgba(EPITHELIAL_COLOURS[state], face_alpha),
            edgecolors=to_rgba(EPITHELIAL_COLOURS[state], edge_alpha),
            linewidths=0.65 if is_at1 else 0.35,
            zorder=4,
        )
        axis.add_collection3d(collection)


def _draw_mesenchyme(
    axis,
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
    breath: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if simulation.n_mesenchymal == 0:
        return
    xyz = simulation.displayed_fibroblast_xyz(breath)
    half = 0.5 * simulation.cfg.fibroblast_length_um
    orientation = simulation.fibroblast_orientation
    segments = np.stack(
        [
            xyz - half * orientation,
            xyz + half * orientation,
        ],
        axis=1,
    )
    colours = np.where(
        simulation.myofibroblast,
        "#FF3B30",
        "#2F80ED",
    )
    axis.add_collection3d(
        Line3DCollection(
            segments,
            colors=list(colours),
            linewidths=render.fibroblast_linewidth,
            alpha=0.90,
            zorder=6,
        )
    )
    axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=render.fibroblast_marker_size,
        c=list(colours),
        edgecolors="none",
        alpha=0.94,
        depthshade=True,
        zorder=7,
    )


def _matrix_cloud(
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    values: list[np.ndarray] = []
    cfg = simulation.cfg
    for owner in range(simulation.n_alveoli):
        excess = (
            simulation.stiffness_kpa[owner]
            - cfg.healthy_stiffness_kpa
        )
        if excess <= 0.05 and simulation.alveolar_state[owner] == OPEN:
            continue
        count = render.matrix_points_per_alveolus
        if count == 0:
            continue
        rng = np.random.default_rng(cfg.seed + 3001 + owner)
        direction = rng.normal(size=(count, 3))
        direction /= np.maximum(
            np.linalg.norm(direction, axis=1, keepdims=True),
            1e-12,
        )
        radial = rng.random(count) ** (1.0 / 3.0)
        radius = (
            0.86
            * simulation.open_radius_um[owner]
            * radial
        )
        cloud = (
            simulation.centres[owner]
            + direction * radius[:, None]
        )
        points.append(cloud)
        values.append(
            np.full(count, simulation.stiffness_kpa[owner])
        )
    if not points:
        return np.zeros((0, 3)), np.zeros(0)
    return np.vstack(points), np.concatenate(values)


def _draw_matrix(
    axis,
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    xyz, stiffness = _matrix_cloud(simulation, render)
    normalise = Normalize(
        vmin=simulation.cfg.healthy_stiffness_kpa,
        vmax=max(
            simulation.cfg.activation_stiffness_kpa * 2.0,
            float(np.max(simulation.stiffness_kpa)),
        ),
        clip=True,
    )
    colourmap = plt.get_cmap("magma")
    if len(xyz):
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=4.0,
            c=stiffness,
            cmap=colourmap,
            norm=normalise,
            edgecolors="none",
            alpha=0.22,
            depthshade=False,
            zorder=2,
        )
    return normalise, colourmap


def _style_axis(
    axis,
    simulation: Alveolar3DSimulation,
    render: Alveolar3DRenderConfig,
    frame_index: int,
) -> None:
    foreground = "#E8EEF2" if render.black_background else "#1D2730"
    background = "#000000" if render.black_background else "#FFFFFF"
    axis.set_facecolor(background)
    radius = simulation.cfg.open_radius_um
    lower = simulation.centres.min(axis=0) - 1.30 * radius
    upper = simulation.centres.max(axis=0) + 1.30 * radius
    spans = upper - lower
    maximum = float(np.max(spans))
    midpoint = 0.5 * (upper + lower)
    lower = midpoint - 0.5 * maximum
    upper = midpoint + 0.5 * maximum
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(
        elev=render.elevation_deg,
        azim=render.azimuth_deg + render.orbit_deg_per_frame * frame_index,
    )

    if render.presentation:
        # Strip every piece of scaffolding so only the tissue remains, the way
        # a biomedical illustration is presented: no axes, ticks, panes or grid.
        axis.set_axis_off()
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((0, 0, 0, 0))
            pane.set_edgecolor((0, 0, 0, 0))
        axis.grid(False)
        return

    axis.set_xlabel("x (µm)", color=foreground, labelpad=7)
    axis.set_ylabel("y (µm)", color=foreground, labelpad=7)
    axis.set_zlabel("z (µm)", color=foreground, labelpad=7)
    axis.tick_params(colors=foreground, labelsize=7)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0, 0, 0, 0))
        pane.set_edgecolor((0.6, 0.68, 0.72, 0.22))
    axis.grid(False)


def _add_legend(axis) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D(
            [],
            [],
            marker="_",
            markeredgewidth=2.5,
            linestyle="",
            color=EPITHELIAL_COLOURS[AT1],
            label="AT1 plate",
            markersize=11,
        ),
    ]
    handles.extend(
        [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=EPITHELIAL_COLOURS[state],
            label=label,
            markersize=6,
        )
        for state, label in (
            (AT2, "AT2"),
            (KRT8, "KRT8+"),
            (ABERRANT, "aberrant"),
        )
        ]
    )
    handles.extend(
        [
            Line2D([], [], color="#2F80ED", lw=2, label="fibroblast"),
            Line2D([], [], color="#FF3B30", lw=2, label="myofibroblast"),
            Patch(
                facecolor=ALVEOLAR_COLOURS[COLLAPSED],
                alpha=0.5,
                label="collapsed",
            ),
            Patch(
                facecolor=ALVEOLAR_COLOURS[INDURATED],
                alpha=0.5,
                label="indurated",
            ),
        ]
    )
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=7,
        framealpha=0.58,
        facecolor="#111111",
        labelcolor="white",
        ncol=2,
    )


def _stage_caption(metrics: dict) -> str:
    """A short stage title driven by the tissue state, not the clock.

    The stage a viewer reads must match what is actually on screen, so the
    wording is chosen from the real quantities the model reports - transitional
    fraction, collapse and induration counts, myofibroblast count and matrix
    stiffness - with calendar time used only as the timestamp, never to decide
    the biology. This avoids captions like "Month 0 - early injury" when no
    injury has occurred yet.
    """
    days = metrics["time_d"]
    when = _timestamp(days)

    healthy_stiffness = 2.0
    stiffened = metrics.get("max_stiffness_kpa", healthy_stiffness) > 3.0
    has_myofib = metrics.get("n_myofibroblast", 0) > 0
    has_indurated = metrics.get("n_indurated", 0) > 0
    has_collapsed = metrics.get("n_collapsed", 0) > 0
    has_transitional = (
        metrics.get("frac_KRT8", 0.0) + metrics.get("frac_aberrant", 0.0)
    ) > 0.01

    if has_indurated and stiffened:
        stage = "chronic fibroblastic focus"
    elif has_myofib and stiffened:
        stage = "fibroblastic focus"
    elif has_collapsed:
        stage = "alveolar collapse"
    elif has_transitional:
        stage = "early epithelial injury"
    else:
        stage = "healthy alveoli"
    return f"{when} — {stage}"


def _timestamp(days: float) -> str:
    """Calendar label for a caption: Day 0, Month N, or Year N.N."""
    if days < 1.0:
        return "Day 0"
    if days < 365.25:
        return f"Month {days / 30.4375:.0f}"
    return f"Year {days / 365.25:.1f}"


# Which cell types to list in the presentation legend at each stage, mirroring
# the reference frames: healthy shows only the resident cells, later stages add
# the injury and fibrosis markers as they appear.
def _presentation_legend(axis, metrics: dict) -> None:
    """Legend rows chosen from the real tissue state, not from time.

    A cell type is listed only when the model actually reports it, so the key
    never claims a population that is not on screen. Colours are taken from the
    same dictionaries the render uses, so a swatch always matches its object.
    """
    from matplotlib.patches import Patch

    healthy_stiffness = 2.0
    rows = [
        (EPITHELIAL_COLOURS[AT1], "AT1 — squamous epithelial plate"),
        (EPITHELIAL_COLOURS[AT2], "AT2 — cuboidal epithelial cell"),
    ]
    # KRT8+ only when transitional cells actually exist
    if metrics.get("frac_KRT8", 0.0) > 0.0:
        rows.append((EPITHELIAL_COLOURS[KRT8], "KRT8+ — transitional epithelial cell"))
    if metrics.get("frac_aberrant", 0.0) > 0.0:
        rows.append((EPITHELIAL_COLOURS[ABERRANT], "Aberrant basaloid — transitional cell"))

    # fibroblasts are present throughout; myofibroblasts only once they appear
    rows.append(("#2F80ED", "Fibroblast — spindle-shaped stromal cell"))
    if metrics.get("n_myofibroblast", 0) > 0:
        rows.append(("#FF3B30", "Myofibroblast — contractile stromal cell"))
    # collagen only when the matrix has actually stiffened past healthy
    if metrics.get("max_stiffness_kpa", healthy_stiffness) > 3.0:
        rows.append((ALVEOLAR_COLOURS[COLLAPSED], "Collagen — amber fibers"))

    # alveolar shell rows reflect what is drawn, with the true colours
    rows.append((ALVEOLAR_COLOURS[OPEN], "Open alveolus — aerated lumen"))
    if metrics.get("n_collapsed", 0) > 0:
        rows.append((ALVEOLAR_COLOURS[COLLAPSED], "Collapsed alveolus — lost volume"))
    if metrics.get("n_indurated", 0) > 0:
        rows.append((ALVEOLAR_COLOURS[INDURATED], "Indurated alveolus — scarred"))

    handles = [Patch(facecolor=colour, edgecolor="none", label=label)
               for colour, label in rows]
    legend = axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=9,
        framealpha=0.55,
        facecolor="#0A0A0A",
        edgecolor="#3A3A3A",
        labelcolor="white",
        borderpad=0.9,
        labelspacing=0.6,
        handlelength=1.1,
    )
    legend.set_zorder(20)


def draw_3d_frame(
    simulation: Alveolar3DSimulation,
    output_path: str | Path,
    *,
    render_config: Alveolar3DRenderConfig | None = None,
    frame_index: int = 0,
) -> dict:
    """Render one biological state at one phase of the respiratory cycle."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    render = render_config or Alveolar3DRenderConfig()
    render.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    breath = inspiration_fraction(render, frame_index)

    background = "#000000" if render.black_background else "#FFFFFF"
    foreground = "#F4F7F8" if render.black_background else "#17212B"
    figure = plt.figure(
        figsize=(render.figure_width, render.figure_height),
        facecolor=background,
    )
    axis = figure.add_subplot(111, projection="3d")

    normalise = colourmap = None
    if render.show_matrix:
        normalise, colourmap = _draw_matrix(axis, simulation, render)
    _draw_alveoli(axis, simulation, render, breath)
    _draw_epithelium(axis, simulation, render, breath)
    _draw_mesenchyme(axis, simulation, render, breath)
    _style_axis(axis, simulation, render, frame_index)
    if render.show_legend:
        if render.presentation:
            _presentation_legend(axis, simulation.metrics())
        else:
            _add_legend(axis)

    metrics = simulation.metrics()

    if render.presentation:
        # A single clean caption instead of the technical multi-line readout.
        # Caller can supply an explicit stage title; otherwise derive a short one.
        if render.presentation_title:
            caption = render.presentation_title
        else:
            caption = _stage_caption(metrics)
        figure.text(
            0.5, 0.055, caption,
            color=foreground, fontsize=15, ha="center", va="center",
            fontweight="bold",
        )
        # no colourbar, no axes: fill the frame with the tissue
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    else:
        if metrics["temporal_calibration"] == "human_chronic_disease_scale":
            clock = f"{format_calendar_time(metrics['time_d'])}  |  chronic calibration"
        else:
            clock = (
                f"{format_calendar_time(metrics['time_d'])}  |  "
                f"kinetics {metrics['kinetic_rate_scale']:.2g}x"
            )
        title = (
            f"true 3D  |  {clock}  |  "
            f"breath {breath * 100:.0f}% inspired\n"
            f"{metrics['n_open']} open, {metrics['n_collapsed']} collapsed, "
            f"{metrics['n_indurated']} indurated  |  "
            f"{metrics['n_mesenchymal']} mesenchymal, "
            f"{metrics['n_myofibroblast']} myofibroblasts\n"
            f"E max {metrics['max_stiffness_kpa']:.1f} kPa  |  "
            f"septal strain max "
            f"{100 * metrics['max_local_septal_strain']:.1f}%"
        )
        axis.set_title(title, color=foreground, fontsize=11, pad=9)

        if normalise is not None and colourmap is not None:
            scalar = ScalarMappable(norm=normalise, cmap=colourmap)
            scalar.set_array([])
            colourbar = figure.colorbar(
                scalar,
                ax=axis,
                fraction=0.025,
                pad=0.03,
            )
            colourbar.set_label("ECM stiffness (kPa)", color=foreground)
            colourbar.ax.tick_params(colors=foreground, labelsize=7)

        figure.subplots_adjust(left=0.01, right=0.94, bottom=0.02, top=0.91)
    figure.savefig(
        output_path,
        dpi=render.dpi,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    record = dict(metrics)
    record["breathing_inspiration_fraction"] = breath
    return record


def _even_dimensions(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return image[: height - height % 2, : width - width % 2]


def _write_mp4(
    imageio,
    images: list[np.ndarray],
    frames_dir: Path,
    output_path: Path,
    fps: int,
) -> str:
    """Use imageio-ffmpeg when available, with a native macOS fallback."""

    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
        imageio.mimsave(
            output_path,
            images,
            format="FFMPEG",
            fps=fps,
            macro_block_size=None,
        )
        return "imageio-ffmpeg"
    except ImportError:
        pass

    ffmpeg = os.environ.get("LUNG_NEMATIC_FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg and Path(ffmpeg).is_file():
        encoder_listing = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "libx264" in encoder_listing:
            encoder = "libx264"
        elif "libopenh264" in encoder_listing:
            encoder = "libopenh264"
        elif "h264_videotoolbox" in encoder_listing:
            encoder = "h264_videotoolbox"
        else:
            encoder = "mpeg4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                encoder,
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"FFmpeg ({encoder})"

    raise RuntimeError(
        "No MP4 encoder found. Install imageio-ffmpeg or set "
        "LUNG_NEMATIC_FFMPEG to an ffmpeg executable."
    )


def run_and_record_3d(
    config: Alveolar3DConfig,
    output_dir: str | Path,
    *,
    state_every_h: float = 24.0,
    breathing_subframes: int = 6,
    fps: int = 8,
    render_config: Alveolar3DRenderConfig | None = None,
    make_gif: bool = True,
    make_mp4: bool = True,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Run the true-3D model and save frames, animation and time series."""

    if state_every_h <= 0:
        raise ValueError("state_every_h must be positive.")
    if breathing_subframes < 2:
        raise ValueError("breathing_subframes must be at least 2.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    config.validate()

    import imageio.v2 as imageio
    import pandas as pd

    render = render_config or Alveolar3DRenderConfig(
        breathing_frames_per_cycle=breathing_subframes,
    )
    render.validate()
    if render.breathing_frames_per_cycle != breathing_subframes:
        raise ValueError(
            "breathing_frames_per_cycle must equal breathing_subframes."
        )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames_3d"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()
    config.to_json(output_dir / "model_3d_config.json")
    (output_dir / "render_3d_config.json").write_text(
        json.dumps(asdict(render), indent=2),
        encoding="utf-8",
    )

    simulation = Alveolar3DSimulation(config)
    total_steps = round(config.total_time_h / config.dt_h)
    every = max(1, round(state_every_h / config.dt_h))
    frame_paths: list[Path] = []
    records: list[dict] = []

    def record_breath() -> None:
        for _ in range(breathing_subframes):
            path = frames_dir / f"frame_{len(frame_paths):04d}.png"
            records.append(
                draw_3d_frame(
                    simulation,
                    path,
                    render_config=render,
                    frame_index=len(frame_paths),
                )
            )
            frame_paths.append(path)
            if progress is not None:
                progress(len(frame_paths))

    record_breath()
    for step in range(1, total_steps + 1):
        simulation.step()
        if step % every == 0 or step == total_steps:
            record_breath()

    images = [_even_dimensions(imageio.imread(path)) for path in frame_paths]
    outputs: dict = {"n_frames": len(images)}
    if make_gif:
        gif_path = output_dir / "alveolar_fibrosis_3d.gif"
        imageio.mimsave(
            gif_path,
            images,
            duration=1000.0 / fps,
            loop=0,
        )
        outputs["gif"] = str(gif_path)
    if make_mp4:
        mp4_path = output_dir / "alveolar_fibrosis_3d.mp4"
        if mp4_path.exists():
            mp4_path.unlink()
        try:
            outputs["mp4_encoder"] = _write_mp4(
                imageio,
                images,
                frames_dir,
                mp4_path,
                fps,
            )
            outputs["mp4"] = str(mp4_path)
        except Exception as error:  # noqa: BLE001 - encoder can fail many ways; clean up and continue
            if mp4_path.exists():
                mp4_path.unlink()
            outputs["mp4_error"] = (
                "MP4 encoding failed: "
                f"{type(error).__name__}: {error}"
            )

    timeseries_path = output_dir / "timeseries_3d.tsv"
    pd.DataFrame(records).to_csv(timeseries_path, sep="\t", index=False)
    outputs.update(
        timeseries=str(timeseries_path),
        final=records[-1],
        simulation=simulation,
    )
    return outputs
