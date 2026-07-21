from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color

from .config import AnalysisConfig
from .nematic import get_density_threshold


def save_overlay(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    output_path: str | Path,
    config: AnalysisConfig,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(14, 10))
    axis.imshow(rgb)

    step = config.field_grid_step_px
    height, width = tissue_mask.shape
    density_cutoff = get_density_threshold(
        field["density"],
        tissue_mask,
        config.density_quantile,
    )

    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if not (
                tissue_mask[y, x]
                and field["density"][y, x] > density_cutoff
                and field["order"][y, x]
                >= config.min_local_order_for_display
            ):
                continue

            half_length = 5 + 25 * field["order"][y, x]
            dx = half_length * np.cos(field["theta"][y, x])
            dy = half_length * np.sin(field["theta"][y, x])
            axis.plot(
                [x - dx, x + dx],
                [y - dy, y + dy],
                linewidth=1.3,
            )

    if not defects.empty:
        positive = defects.loc[defects["charge"] == 0.5]
        negative = defects.loc[defects["charge"] == -0.5]
        plus_one = defects.loc[defects["charge"] == 1.0]
        minus_one = defects.loc[defects["charge"] == -1.0]

        if not positive.empty:
            axis.scatter(
                positive["x_px"],
                positive["y_px"],
                marker="+",
                s=180,
                linewidths=2.5,
                color="#0072B2",
                label="Candidate +1/2",
            )

        if not negative.empty:
            axis.scatter(
                negative["x_px"],
                negative["y_px"],
                marker="x",
                s=150,
                linewidths=2.5,
                color="#D55E00",
                label="Candidate -1/2",
            )

        if not plus_one.empty:
            axis.scatter(
                plus_one["x_px"],
                plus_one["y_px"],
                marker="*",
                s=320,
                linewidths=1.5,
                facecolors="none",
                edgecolors="#009E73",
                label="Candidate +1",
            )

        if not minus_one.empty:
            axis.scatter(
                minus_one["x_px"],
                minus_one["y_px"],
                marker="s",
                s=220,
                linewidths=2.0,
                facecolors="none",
                edgecolors="#CC79A7",
                label="Candidate -1",
            )

        axis.legend(loc="lower left")

    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_diagnostic_panel(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    labels: np.ndarray,
    field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original image")

    axes[0, 1].imshow(tissue_mask, cmap="gray")
    axes[0, 1].set_title("Tissue mask")

    axes[0, 2].imshow(
        color.label2rgb(labels, image=rgb, alpha=0.35)
    )
    axes[0, 2].set_title("Nuclear segmentation")

    order_map = axes[1, 0].imshow(
        field["order"],
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Local nematic order S")
    figure.colorbar(order_map, ax=axes[1, 0], fraction=0.046)

    density_map = axes[1, 1].imshow(field["density"])
    axes[1, 1].set_title("Orientational density")
    figure.colorbar(
        density_map,
        ax=axes[1, 1],
        fraction=0.046,
    )

    axes[1, 2].imshow(rgb)
    if not defects.empty:
        positive = defects.loc[defects["charge"] == 0.5]
        negative = defects.loc[defects["charge"] == -0.5]
        plus_one = defects.loc[defects["charge"] == 1.0]
        minus_one = defects.loc[defects["charge"] == -1.0]
        axes[1, 2].scatter(
            positive["x_px"], positive["y_px"],
            marker="+", s=140, linewidths=2, color="#0072B2",
        )
        axes[1, 2].scatter(
            negative["x_px"], negative["y_px"],
            marker="x", s=120, linewidths=2, color="#D55E00",
        )
        axes[1, 2].scatter(
            plus_one["x_px"], plus_one["y_px"],
            marker="*", s=260, facecolors="none", edgecolors="#009E73",
        )
        axes[1, 2].scatter(
            minus_one["x_px"], minus_one["y_px"],
            marker="s", s=180, facecolors="none", edgecolors="#CC79A7",
        )
    axes[1, 2].set_title("Persistent candidate defects")

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def draw_dense_director(
    background: np.ndarray,
    field: dict,
    tissue_mask: np.ndarray,
    output_path,
    grid_step_px: int = 12,
    min_order: float = 0.03,
    vector_length_frac: float = 0.55,
    color: str = "yellow",
    linewidth: float = 0.7,
    dpi: int = 95,
    title: str = "",
) -> None:
    """OrientationJ-style dense director overlay, confined to the mask.

    Unlike ``save_overlay``, which marks defects on a coarse field, this draws a
    line at every grid node inside the tissue - the look of an OrientationJ
    vector field - with uniform-length segments so orientation, not magnitude,
    is what the eye reads. Nodes outside the mask or below ``min_order`` are left
    blank, so no vectors appear where there is no tissue.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    theta = field["theta"]
    order = field["order"]
    height, width = tissue_mask.shape

    figure, ax = plt.subplots(figsize=(width / 180, height / 180))
    if background.ndim == 2:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(background)

    length = grid_step_px * vector_length_frac
    segments = []
    for y in range(grid_step_px, height - grid_step_px, grid_step_px):
        for x in range(grid_step_px, width - grid_step_px, grid_step_px):
            if not tissue_mask[y, x] or order[y, x] < min_order:
                continue
            angle = theta[y, x]
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            segments.append([(x - dx, y - dy), (x + dx, y + dy)])

    ax.add_collection(LineCollection(segments, colors=color,
                                     linewidths=linewidth, alpha=0.9))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return len(segments)
