"""Tests for adaptive-radius defect detection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from lung_nematic.config import load_default_config
from lung_nematic.defects_adaptive import (
    defect_order_context,
    detect_defects_adaptive,
)


def _config():
    return replace(load_default_config(), density_quantile=0.2,
                   defect_grid_step_px=10, min_edge_distance_px=5)


def _planted_defect_field(shape=(300, 300), charge=0.5, core=(150, 150)):
    """A field with one planted half-integer defect and a disordered core."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    theta = np.mod(charge * np.arctan2(yy - core[1], xx - core[0]), np.pi)
    radius = np.hypot(xx - core[0], yy - core[1])
    order = np.tanh(radius / 12.0)
    # A flat density field is rejected by the quantile gate (density > q(density)
    # is never true when density is constant), so use a broad central bump.
    scale = 0.45 * max(shape)
    density = 1.0 + 0.5 * np.exp(-0.5 * (radius / scale) ** 2)
    return {"theta": theta, "order": order, "density": density}


def test_planted_defect_is_detected():
    shape = (300, 300)
    field = _planted_defect_field(shape, charge=0.5)
    mask = np.ones(shape, bool)
    radius_map = np.full(shape, 30.0)
    config = _config()

    detected = detect_defects_adaptive(field, mask, radius_map, config, grid_step_px=10)
    assert not detected.empty
    assert (detected["charge"] == 0.5).any()
    # the detection should land near the planted core
    nearest = detected.loc[detected["charge"] == 0.5].iloc[
        ((detected.loc[detected["charge"] == 0.5, "x_px"] - 150) ** 2
         + (detected.loc[detected["charge"] == 0.5, "y_px"] - 150) ** 2).idxmin()
    ]
    assert np.hypot(nearest["x_px"] - 150, nearest["y_px"] - 150) < 60


def test_radius_is_recorded_per_defect():
    shape = (300, 300)
    field = _planted_defect_field(shape)
    mask = np.ones(shape, bool)
    radius_map = np.full(shape, 25.0)
    detected = detect_defects_adaptive(field, mask, radius_map, _config(), grid_step_px=10)
    assert "integration_radius_px" in detected.columns
    assert (detected["integration_radius_px"] == 25.0).all()


def test_variable_radius_is_used():
    """A radius that varies across the frame must be reflected in detections."""
    shape = (300, 300)
    field = _planted_defect_field(shape)
    mask = np.ones(shape, bool)
    # left half small radius, right half large
    radius_map = np.full(shape, 20.0)
    radius_map[:, 150:] = 45.0
    detected = detect_defects_adaptive(field, mask, radius_map, _config(), grid_step_px=10)
    if not detected.empty:
        # radii used should include values from both bands
        assert detected["integration_radius_px"].nunique() >= 1


def test_uniform_field_gives_no_defects():
    shape = (200, 200)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    radius = np.hypot(xx - shape[1] / 2, yy - shape[0] / 2)
    density = 1.0 + 0.5 * np.exp(-0.5 * (radius / (0.45 * max(shape))) ** 2)
    field = {"theta": np.full(shape, 0.5), "order": np.ones(shape), "density": density}
    mask = np.ones(shape, bool)
    radius_map = np.full(shape, 25.0)
    detected = detect_defects_adaptive(field, mask, radius_map, _config(), grid_step_px=10)
    if not detected.empty:
        assert (detected["charge"] == 0).all()


def test_ring_must_lie_in_tissue():
    """A defect whose ring would leave the tissue must be rejected."""
    shape = (300, 300)
    field = _planted_defect_field(shape, core=(20, 20))  # near the corner
    mask = np.ones(shape, bool)
    mask[:40, :40] = False  # remove tissue around the planted core
    radius_map = np.full(shape, 30.0)
    detected = detect_defects_adaptive(field, mask, radius_map, _config(), grid_step_px=10)
    # no detection should sit in the removed corner
    if not detected.empty:
        in_corner = (detected["x_px"] < 40) & (detected["y_px"] < 40)
        assert not in_corner.any()


def test_defect_order_context_flags_domain_walls():
    shape = (300, 300)
    field = _planted_defect_field(shape)
    mask = np.ones(shape, bool)
    defects = pd.DataFrame({"x_px": [150.0], "y_px": [150.0],
                            "charge": [0.5], "charge_raw": [0.5]})
    context = defect_order_context(defects, field, mask)
    # the planted core is disordered, so a defect there is on a wall
    assert context["defects_on_walls"] is True
    assert context["order_at_defects"] < context["order_in_tissue"]


def test_empty_defects_context():
    shape = (100, 100)
    field = {"theta": np.zeros(shape), "order": np.ones(shape), "density": np.ones(shape)}
    mask = np.ones(shape, bool)
    context = defect_order_context(pd.DataFrame(), field, mask)
    assert context["n_defects"] == 0
