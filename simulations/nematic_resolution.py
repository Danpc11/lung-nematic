"""
Deciding when a measured nematic order is real.

A director field built from discrete objects carries counting noise. For N
randomly oriented rods the resultant of the doubled angles has magnitude of
order 1/sqrt(N) purely by chance, so a small smoothing window reports order
where there is none. The usual response - "use at least 30 samples per window" -
is a serviceable rule of thumb but it is not a test, and it hides the fact that
the achievable window size is fixed by the objects themselves.

That constraint is worth stating plainly. A window of radius R holds
``pi * R^2 * phi / A_cell`` objects at packing fraction ``phi``, and packing
cannot exceed about 1. So

    R_min = sqrt(N_min * A_cell / (pi * phi))

is a floor set by cell size, not by tissue density: for a 50 x 11 um fibroblast
at full packing and N_min = 30, R_min is about 64 um whatever the sample. There
is no denser region to retreat to. Either the window is widened past R_min - at
the cost of being unable to resolve defect pairs closer than roughly 2R - or the
measurement is abandoned.

What this module adds is the test the rule of thumb stands in for. For each
window the local sample count is measured, the null distribution of |S| at that
count is evaluated, and order is reported only where it exceeds the null
quantile *for its own N*. Sparse windows are then held to a stricter standard
than dense ones, automatically, instead of every window being judged against a
single global threshold.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter


@lru_cache(maxsize=512)
def null_order_quantile(
    n_samples: int,
    quantile: float = 0.95,
    n_trials: int = 20000,
    seed: int = 0,
) -> float:
    """Order parameter reached by chance alone, for ``n_samples`` random rods.

    Monte Carlo rather than analytic, because the Rayleigh approximation is only
    good for large N and the interesting windows are small.
    """
    if n_samples < 1:
        return 1.0
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, np.pi, size=(n_trials, n_samples))
    cos_sum = np.cos(2 * angles).sum(axis=1)
    sin_sum = np.sin(2 * angles).sum(axis=1)
    magnitudes = np.hypot(cos_sum, sin_sum) / n_samples
    return float(np.quantile(magnitudes, quantile))


def analytic_null_quantile(n_samples: int, quantile: float = 0.95) -> float:
    """Rayleigh approximation, for orientation and for checking the sampler."""
    if n_samples < 1:
        return 1.0
    return float(np.sqrt(-np.log(1.0 - quantile) / n_samples))


def minimum_window_radius_um(
    cell_area_um2: float,
    min_samples: int = 30,
    packing_fraction: float = 1.0,
) -> float:
    """Smallest window that can hold ``min_samples`` objects at this packing."""
    packing = max(float(packing_fraction), 1e-6)
    return float(np.sqrt(min_samples * cell_area_um2 / (np.pi * packing)))


def adaptive_window_radius_um(
    density_per_um2: float,
    min_samples: int = 30,
    floor_um: float = 5.0,
) -> float:
    """Window radius that captures ``min_samples`` at the measured density."""
    density = max(float(density_per_um2), 1e-12)
    return float(max(np.sqrt(min_samples / (np.pi * density)), floor_um))


def director_field(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    shape_um: tuple[float, float],
    grid_step_um: float,
    sigma_um: float,
) -> dict[str, np.ndarray]:
    """Coarse-grained director, order, and the *count* behind each estimate."""
    width_um, height_um = shape_um
    nx = max(int(np.ceil(width_um / grid_step_um)), 1)
    ny = max(int(np.ceil(height_um / grid_step_um)), 1)

    qxx = np.zeros((ny, nx))
    qxy = np.zeros((ny, nx))
    counts = np.zeros((ny, nx))
    if x.size:
        ix = np.clip((x / grid_step_um).astype(int), 0, nx - 1)
        iy = np.clip((y / grid_step_um).astype(int), 0, ny - 1)
        np.add.at(qxx, (iy, ix), np.cos(2 * theta))
        np.add.at(qxy, (iy, ix), np.sin(2 * theta))
        np.add.at(counts, (iy, ix), 1.0)

    sigma = sigma_um / grid_step_um
    qxx_s = gaussian_filter(qxx, sigma)
    qxy_s = gaussian_filter(qxy, sigma)
    counts_s = gaussian_filter(counts, sigma)

    # Effective number of objects contributing to each smoothed estimate. A
    # Gaussian of width sigma integrates to 2*pi*sigma^2 in area, so the count
    # per unit area must be scaled back up to a count.
    effective_n = counts_s * 2.0 * np.pi * sigma**2

    with np.errstate(invalid="ignore", divide="ignore"):
        order = np.sqrt(qxx_s**2 + qxy_s**2) / np.maximum(counts_s, 1e-12)

    return {
        "theta": np.mod(0.5 * np.arctan2(qxy_s, qxx_s), np.pi),
        "order": np.clip(np.nan_to_num(order), 0.0, 1.0),
        "effective_n": effective_n,
        "density": counts_s / grid_step_um**2,
        "sigma_um": sigma_um,
        "grid_step_um": grid_step_um,
    }


def significant_order_mask(
    field: dict[str, np.ndarray],
    quantile: float = 0.95,
    max_lookup_n: int = 400,
) -> np.ndarray:
    """Where does the measured order exceed the null *for its own sample count*?

    Each window is compared against the distribution its own N implies, so a
    sparse window must show much stronger apparent order than a dense one to
    count. This is the test the "30 samples" rule approximates.
    """
    effective_n = field["effective_n"]
    order = field["order"]

    counts = np.clip(np.rint(effective_n).astype(int), 0, max_lookup_n)
    thresholds = np.ones_like(order)
    for value in np.unique(counts):
        if value < 2:
            continue
        thresholds[counts == value] = null_order_quantile(int(value), quantile)
    return (order > thresholds) & (counts >= 2)


def resolved_defects(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    shape_um: tuple[float, float],
    cell_area_um2: float,
    grid_step_um: float = 6.0,
    min_samples: int = 30,
    quantile: float = 0.95,
    sigma_um: float | None = None,
    plaquette_step: int = 2,
    min_separation_um: float | None = None,
) -> dict:
    """Detect +/-1/2 defects only where the local order beats its own null.

    When ``sigma_um`` is None the window is chosen from the measured density so
    that a typical window holds ``min_samples`` objects. The returned
    diagnostics state the resulting resolution limit, because widening the
    window to defeat noise necessarily blurs defect pairs closer than about
    twice its radius - the two cannot both be had.
    """
    width_um, height_um = shape_um
    n_cells = int(x.size)
    mean_density = n_cells / max(width_um * height_um, 1e-9)

    if sigma_um is None:
        sigma_um = adaptive_window_radius_um(mean_density, min_samples)
    if min_separation_um is None:
        min_separation_um = 2.0 * sigma_um

    field = director_field(x, y, theta, shape_um, grid_step_um, sigma_um)
    significant = significant_order_mask(field, quantile)

    theta_grid = field["theta"]
    ny, nx = theta_grid.shape
    rows = np.arange(0, ny - plaquette_step, plaquette_step)
    cols = np.arange(0, nx - plaquette_step, plaquette_step)

    detections: dict[str, np.ndarray] = {"plus": np.zeros((0, 2)),
                                         "minus": np.zeros((0, 2))}
    if rows.size >= 2 and cols.size >= 2:
        corners = [
            theta_grid[np.ix_(rows, cols)],
            theta_grid[np.ix_(rows, cols + plaquette_step)],
            theta_grid[np.ix_(rows + plaquette_step, cols + plaquette_step)],
            theta_grid[np.ix_(rows + plaquette_step, cols)],
        ]
        phases = [2 * corner for corner in corners]
        winding = np.zeros_like(phases[0])
        for index in range(4):
            delta = phases[(index + 1) % 4] - phases[index]
            winding += np.arctan2(np.sin(delta), np.cos(delta))
        charge = winding / (4 * np.pi)

        # every corner of the plaquette must sit in a window that beat its null
        valid = (
            significant[np.ix_(rows, cols)]
            & significant[np.ix_(rows, cols + plaquette_step)]
            & significant[np.ix_(rows + plaquette_step, cols + plaquette_step)]
            & significant[np.ix_(rows + plaquette_step, cols)]
        )

        for key, target in (("plus", 0.5), ("minus", -0.5)):
            hit = valid & (np.abs(charge - target) < 0.2)
            row_index, col_index = np.nonzero(hit)
            points = np.column_stack([
                (cols[col_index] + plaquette_step / 2) * grid_step_um,
                (rows[row_index] + plaquette_step / 2) * grid_step_um,
            ])
            kept: list[np.ndarray] = []
            for point in points:
                if all(np.hypot(*(point - other)) >= min_separation_um
                       for other in kept):
                    kept.append(point)
            detections[key] = np.array(kept) if kept else np.zeros((0, 2))

    typical_n = float(np.median(field["effective_n"][field["density"] > 0])) \
        if (field["density"] > 0).any() else 0.0
    r_min_full_packing = minimum_window_radius_um(cell_area_um2, min_samples, 1.0)

    return {
        "plus": detections["plus"],
        "minus": detections["minus"],
        "field": field,
        "significant": significant,
        "diagnostics": {
            "n_cells": n_cells,
            "sigma_um": float(sigma_um),
            "mean_density_per_um2": float(mean_density),
            "typical_samples_per_window": typical_n,
            "null_quantile_at_typical_n": null_order_quantile(
                max(int(round(typical_n)), 1), quantile
            ),
            "median_order_where_significant": float(
                np.median(field["order"][significant])
            ) if significant.any() else float("nan"),
            "significant_area_fraction": float(significant.mean()),
            "resolution_limit_um": float(min_separation_um),
            "r_min_at_full_packing_um": r_min_full_packing,
            "window_is_adequate": bool(sigma_um >= r_min_full_packing * 0.95),
        },
    }
