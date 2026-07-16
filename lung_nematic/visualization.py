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

        if not positive.empty:
            axis.scatter(
                positive["x_px"],
                positive["y_px"],
                marker="+",
                s=180,
                linewidths=2.5,
                label="Candidate +1/2",
            )

        if not negative.empty:
            axis.scatter(
                negative["x_px"],
                negative["y_px"],
                marker="x",
                s=150,
                linewidths=2.5,
                label="Candidate -1/2",
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
        axes[1, 2].scatter(
            positive["x_px"],
            positive["y_px"],
            marker="+",
            s=140,
            linewidths=2,
        )
        axes[1, 2].scatter(
            negative["x_px"],
            negative["y_px"],
            marker="x",
            s=120,
            linewidths=2,
        )
    axes[1, 2].set_title("Persistent candidate defects")

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
