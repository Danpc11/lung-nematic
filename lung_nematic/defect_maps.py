"""
Per-defect director maps.

For every detected defect this renders a crop of the H&E image with the local
**director field** drawn as streamlines (headless: no arrowheads), coloured by
the local nematic order S. This is the director analogue of the top row of an
active-nematic defect figure. It is NOT a velocity/flow map: fixed H&E gives no
time-resolved motion, so |v|, radial velocity and the flow angle theta_v are not
computable here. The colour is order S, not speed.

For an integer (+/-1) defect the source field is taken from the collagen
(structure-tensor) layer only, as requested: nuclear +1 asters/vortices are
rare and unstable, whereas the fibre architecture is where an aster/vortex/
spiral around a focus or lumen would live.

For +1 defects the spiral angle theta0 is estimated from the director itself:
theta(phi) = phi + theta0 for a +1 defect, so theta0 = <theta - phi> (circular
mean, period pi, wrapped to (-pi/2, pi/2]). theta0 ~ 0 is an aster (radial),
theta0 ~ +/-pi/2 a vortex (circular), in between a spiral. This is exactly the
x-axis quantity of the paper's panel i, and unlike theta_v it is a pure
director quantity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig


def _crop_bounds(cx, cy, half, width, height):
    x0 = int(max(0, round(cx - half)))
    x1 = int(min(width, round(cx + half)))
    y0 = int(max(0, round(cy - half)))
    y1 = int(min(height, round(cy + half)))
    return x0, x1, y0, y1


def estimate_spiral_angle(
    theta: np.ndarray,
    cx: float,
    cy: float,
    inner_frac: float = 0.25,
    outer_frac: float = 0.95,
) -> float:
    """Spiral angle theta0 of a +1 defect from the director field.

    Averaged over an annulus around the core (the core itself is singular). The
    result is in radians, wrapped to (-pi/2, pi/2]: 0 = aster, +/-pi/2 = vortex.
    """
    height, width = theta.shape
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    r_max = min(cx, cy, width - cx, height - cy)
    ring = (radius >= inner_frac * r_max) & (radius <= outer_frac * r_max)
    if not ring.any():
        return float("nan")

    phi = np.arctan2(dy[ring], dx[ring])
    delta = theta[ring] - phi
    # Director has period pi; take the circular mean on the doubled angle.
    mean = np.angle(np.mean(np.exp(2j * delta))) / 2.0
    # Wrap to (-pi/2, pi/2].
    mean = (mean + np.pi / 2) % np.pi - np.pi / 2
    return float(mean)


def _classify_spiral(theta0: float) -> str:
    if not np.isfinite(theta0):
        return "undefined"
    magnitude = abs(theta0)
    if magnitude < np.pi / 8:
        return "aster"
    if magnitude > 3 * np.pi / 8:
        return "vortex"
    return "spiral"


def render_defect_map(
    rgb: np.ndarray,
    source_field: dict[str, np.ndarray],
    defect: pd.Series,
    output_path: str | Path,
    window_px: int,
    title_prefix: str = "",
) -> dict:
    """Render one defect's director streamlines over the H&E crop.

    Returns a small dict of the defect's map metadata (including theta0 for
    integer defects).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    height, width = source_field["theta"].shape
    cx = float(defect["x_px"])
    cy = float(defect["y_px"])
    charge = float(defect["charge"])
    half = window_px // 2
    x0, x1, y0, y1 = _crop_bounds(cx, cy, half, width, height)

    theta = source_field["theta"][y0:y1, x0:x1]
    order = source_field["order"][y0:y1, x0:x1]
    rgb_crop = rgb[y0:y1, x0:x1]

    local_cx = cx - x0
    local_cy = cy - y0

    theta0 = float("nan")
    spiral = ""
    if charge == 1.0:
        theta0 = estimate_spiral_angle(theta, local_cx, local_cy)
        spiral = _classify_spiral(theta0)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(rgb_crop, extent=[0, theta.shape[1], theta.shape[0], 0])

    ys = np.arange(theta.shape[0])
    xs = np.arange(theta.shape[1])
    u = np.cos(theta)
    v = np.sin(theta)
    speed = np.clip(order, 0, 1)
    stream = axis.streamplot(
        xs, ys, u, v,
        color=speed, cmap="viridis", density=1.4, linewidth=1.1,
        arrowstyle="-",
    )
    stream.lines.set_clim(0, 1)
    colorbar = figure.colorbar(stream.lines, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("Local order S")

    marker = {0.5: "+", -0.5: "x", 1.0: "*", -1.0: "s"}.get(charge, "o")
    axis.scatter([local_cx], [local_cy], marker=marker, s=260,
                 facecolors="none", edgecolors="white", linewidths=2.2, zorder=5)

    charge_label = {0.5: "+1/2", -0.5: "-1/2", 1.0: "+1", -1.0: "-1"}.get(
        charge, f"{charge:+g}"
    )
    title = f"{title_prefix}{charge_label} defect"
    if charge == 1.0 and np.isfinite(theta0):
        title += f"  |  theta0 = {np.degrees(theta0):+.0f} deg ({spiral})"
    axis.set_title(title)
    axis.set_xlim(0, theta.shape[1])
    axis.set_ylim(theta.shape[0], 0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)

    return {
        "defect_id": int(defect.get("defect_id", -1)),
        "charge": charge,
        "x_px": cx,
        "y_px": cy,
        "theta0_rad": theta0,
        "spiral_class": spiral,
        "map_path": str(output),
    }


def render_defect_maps(
    rgb: np.ndarray,
    main_field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    output_dir: str | Path,
    tag: str,
    config: AnalysisConfig,
    integer_field: dict[str, np.ndarray] | None = None,
    window_px: int | None = None,
) -> pd.DataFrame:
    """Render a director map for every defect.

    Half-integer defects use ``main_field``. Integer defects use
    ``integer_field`` (the collagen layer) when provided; if it is None they are
    skipped, since the request is to source integer-defect maps from collagen
    only.
    """
    if defects.empty:
        return pd.DataFrame()

    output_dir = Path(output_dir)
    window = window_px or config.defect_map_window_px
    records: list[dict] = []

    for _, defect in defects.iterrows():
        charge = float(defect["charge"])
        if abs(charge) == 1.0:
            if integer_field is None:
                continue
            field = integer_field
            prefix = "collagen | "
        else:
            field = main_field
            prefix = ""

        did = int(defect.get("defect_id", len(records) + 1))
        charge_tag = f"{charge:+g}".replace("+", "p").replace("-", "m").replace(
            ".", "_"
        )
        path = output_dir / f"{tag}_defect{did:03d}_q{charge_tag}.png"
        records.append(
            render_defect_map(rgb, field, defect, path, window, prefix)
        )

    return pd.DataFrame(records)
