from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def compute_nematic_field(
    nuclei: pd.DataFrame,
    image_shape: tuple[int, int],
    sigma_px: float,
) -> dict[str, np.ndarray]:
    """
    Estimate a smooth 2D nematic tensor field from nuclear orientations.
    """
    height, width = image_shape
    density_impulse = np.zeros((height, width), dtype=np.float32)
    qxx_impulse = np.zeros_like(density_impulse)
    qxy_impulse = np.zeros_like(density_impulse)

    if nuclei.empty:
        return {
            "density": density_impulse,
            "order": density_impulse.copy(),
            "theta": density_impulse.copy(),
        }

    xs = np.clip(
        np.rint(nuclei["x_px"]).astype(int),
        0,
        width - 1,
    )
    ys = np.clip(
        np.rint(nuclei["y_px"]).astype(int),
        0,
        height - 1,
    )

    for x, y, theta, weight in zip(
        xs,
        ys,
        nuclei["theta_rad"].to_numpy(),
        nuclei["anisotropy_weight"].to_numpy(),
    ):
        density_impulse[y, x] += weight
        qxx_impulse[y, x] += weight * np.cos(2 * theta)
        qxy_impulse[y, x] += weight * np.sin(2 * theta)

    density = gaussian_filter(
        density_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )
    qxx = gaussian_filter(
        qxx_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )
    qxy = gaussian_filter(
        qxy_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )

    order = np.sqrt(qxx**2 + qxy**2) / (density + 1e-12)
    theta = (0.5 * np.arctan2(qxy, qxx)) % np.pi

    return {
        "density": density,
        "order": np.clip(order, 0, 1),
        "theta": theta,
    }


def compute_global_order(nuclei: pd.DataFrame) -> float:
    if nuclei.empty:
        return float("nan")

    weights = nuclei["anisotropy_weight"].to_numpy()
    angles = nuclei["theta_rad"].to_numpy()

    if weights.sum() <= 0:
        return float("nan")

    value = np.sum(
        weights * np.exp(1j * 2 * angles)
    ) / weights.sum()
    return float(abs(value))


def get_density_threshold(
    density: np.ndarray,
    tissue_mask: np.ndarray,
    quantile: float,
) -> float:
    values = density[(density > 0) & tissue_mask]
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, quantile))
