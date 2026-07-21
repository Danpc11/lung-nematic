"""
Boundary conditions and defect detection in the periodic focus model.

The focus model moves cells on a periodic domain (``np.mod`` wrap, minimum-image
interactions), so every coarse-grained field must be smoothed with
``mode="wrap"``. The default reflect mode double-counts edge cells and
manufactures order in the corners: with fully random orientations and a nominal
5 % threshold, the corner false-positive rate runs to tens of percent under
reflect and drops to the nominal level under wrap.

These tests pin that down at two levels: the raw ``gaussian_filter`` behaviour
that motivated the fix, and the assembled director field the model actually
produces. The planted ±1/2 tests confirm the field is still able to represent a
real defect once the boundary is correct.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from simulations.fibrofocus import FocusConfig, FocusSimulation


# --------------------------------------------------------------- the mechanism
def _corner_false_positive_rate(mode: str, n_trials: int = 120,
                                nx: int = 60, ny: int = 60,
                                n_cells: int = 600, sigma: float = 3.0) -> float:
    """FP rate in the corner against a 5 % threshold from the centre.

    Fully random orientations have zero true order, so any apparent order is a
    boundary artifact of the smoothing mode.
    """
    def order_map(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, nx, n_cells)
        y = rng.uniform(0, ny, n_cells)
        theta = rng.uniform(0, np.pi, n_cells)
        qxx = np.zeros((ny, nx))
        qxy = np.zeros((ny, nx))
        counts = np.zeros((ny, nx))
        ix = np.clip(x.astype(int), 0, nx - 1)
        iy = np.clip(y.astype(int), 0, ny - 1)
        np.add.at(qxx, (iy, ix), np.cos(2 * theta))
        np.add.at(qxy, (iy, ix), np.sin(2 * theta))
        np.add.at(counts, (iy, ix), 1.0)
        qxx = gaussian_filter(qxx, sigma, mode=mode)
        qxy = gaussian_filter(qxy, sigma, mode=mode)
        density = gaussian_filter(counts, sigma, mode=mode)
        with np.errstate(invalid="ignore", divide="ignore"):
            order = np.sqrt(qxx**2 + qxy**2) / np.maximum(density, 1e-9)
        return np.clip(np.nan_to_num(order), 0, 1)

    centre_vals, corner_vals = [], []
    for trial in range(n_trials):
        order = order_map(trial)
        centre_vals.append(order[ny // 2 - 2:ny // 2 + 2, nx // 2 - 2:nx // 2 + 2].mean())
        corner_vals.append(order[:3, :3].mean())
    threshold = np.quantile(centre_vals, 0.95)
    return float((np.array(corner_vals) > threshold).mean())


def test_reflect_inflates_corners_but_wrap_does_not():
    """The bug and its fix, quantified: reflect fails, wrap is at nominal."""
    reflect_fp = _corner_false_positive_rate("reflect")
    wrap_fp = _corner_false_positive_rate("wrap")
    assert reflect_fp > 0.15, (
        f"reflect should inflate corner false positives; got {reflect_fp:.2f}"
    )
    assert wrap_fp < 0.10, (
        f"wrap should keep corner false positives near nominal; got {wrap_fp:.2f}"
    )


# -------------------------------------------------- the assembled model field
def test_focus_field_has_no_corner_bias_with_random_orientations():
    """The model's own director field must not be brighter in the corners."""
    config = replace(FocusConfig(), total_time_h=2.0)
    sim = FocusSimulation(config)
    sim.theta = sim.rng.uniform(0, np.pi, sim.n_cells)

    field = sim.director_field(sigma_um=25.0)
    order = field["order"]
    ny, nx = order.shape

    corner = np.mean([
        order[:3, :3].mean(), order[:3, -3:].mean(),
        order[-3:, :3].mean(), order[-3:, -3:].mean(),
    ])
    centre = order[ny // 2 - 2:ny // 2 + 2, nx // 2 - 2:nx // 2 + 2].mean()
    ratio = corner / max(centre, 1e-9)
    assert 0.7 < ratio < 1.4, (
        f"corner order should match the centre on a periodic field; ratio {ratio:.2f}"
    )


def test_density_field_wraps():
    """A cell at the edge must contribute to the density on the opposite edge."""
    config = replace(FocusConfig(), total_time_h=2.0)
    sim = FocusSimulation(config)
    # place every cell in a thin strip at the left edge
    sim.x = sim.rng.uniform(0, 2.0, sim.n_cells)
    sim.y = sim.rng.uniform(0, config.height_um, sim.n_cells)

    field = sim.director_field(sigma_um=15.0)
    density = field["density"]
    # with wrap, smoothing a left-edge strip leaks onto the right edge
    left = density[:, :3].sum()
    right = density[:, -3:].sum()
    assert right > 0.05 * left, (
        "density from an edge strip should wrap onto the opposite edge"
    )


# ------------------------------------------------------ planted ±1/2 defects
def _plant_defect(sim: FocusSimulation, x0: float, y0: float, charge: float):
    """Set every cell's orientation to the far field of a defect at (x0, y0)."""
    dx = sim.x - x0
    dy = sim.y - y0
    sim.theta = np.mod(charge * np.arctan2(dy, dx), np.pi)


@pytest.mark.parametrize("charge", [0.5, -0.5])
def test_planted_half_defect_is_representable(charge):
    """A planted ±1/2 must produce a low-order core in an otherwise ordered field.

    This is not a detector test; it checks the corrected field can still carry a
    real defect - order dips at the singularity and recovers around it - rather
    than being flattened by the boundary handling.
    """
    config = replace(FocusConfig(), total_time_h=2.0)
    sim = FocusSimulation(config)
    x0, y0 = config.width_um / 2, config.height_um / 2
    _plant_defect(sim, x0, y0, charge)

    field = sim.director_field(sigma_um=20.0)
    order = field["order"]
    ny, nx = order.shape
    ix = int(x0 / config.grid_step_um)
    iy = int(y0 / config.grid_step_um)

    core = order[iy - 1:iy + 2, ix - 1:ix + 2].mean()
    surround = order[iy - 6:iy + 7, ix - 6:ix + 7].mean()
    assert core < surround, (
        f"a planted {charge:+.1f} defect should have a disordered core; "
        f"core {core:.2f} vs surround {surround:.2f}"
    )
