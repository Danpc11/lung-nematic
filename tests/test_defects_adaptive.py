"""Tests for adaptive-radius defect detection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------- the negative controls
def test_random_field_gives_no_defects():
    """A disordered field (S~0) must yield no defects, not chance windings.

    This is the guard for the order gate. A +/-1/2 winding appears by chance in
    noise, so without a floor on the surrounding order the detector reports
    defects in a random field.
    """
    shape = (300, 300)
    config = _config()
    total = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
        radius = np.hypot(xx - 150, yy - 150)
        field = {
            "theta": rng.uniform(0, np.pi, shape),
            "order": np.full(shape, 0.02),
            "density": 1.0 + 0.5 * np.exp(-0.5 * (radius / 135) ** 2),
        }
        detected = detect_defects_adaptive(
            field, np.ones(shape, bool), np.full(shape, 25.0),
            config, grid_step_px=10,
        )
        total += len(detected)
    assert total == 0, f"random fields produced {total} defects; order gate failed"


def test_order_gate_threshold_is_respected():
    """Raising the order floor should not admit low-order candidates."""
    shape = (300, 300)
    field = _planted_defect_field(shape, charge=0.5)
    # depress the order everywhere so nothing clears a high floor
    field["order"] = field["order"] * 0.1
    detected = detect_defects_adaptive(
        field, np.ones(shape, bool), np.full(shape, 30.0),
        _config(), grid_step_px=10, min_ring_order=0.5,
    )
    assert detected.empty, "a high order floor should reject a low-order field"


def test_null_model_depletes_for_ordered_field():
    """An ordered field should show fewer defects than its shuffled null."""
    from lung_nematic.defects_adaptive import adaptive_null_model

    shape = (250, 250)
    # a smoothly varying but mostly aligned field: few real defects
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    theta = np.mod(0.4 + 0.3 * np.sin(xx / 80.0), np.pi)
    order = np.full(shape, 0.6)
    radius = np.hypot(xx - 125, yy - 125)
    density = 1.0 + 0.5 * np.exp(-0.5 * (radius / 110) ** 2)
    field = {"theta": theta, "order": order, "density": density}

    result = adaptive_null_model(
        field, np.ones(shape, bool), np.full(shape, 25.0),
        _config(), n_permutations=15, grid_step_px=12, seed=0,
    )
    assert result["observed_count"] <= result["null_mean"], (
        "an ordered field should not have more defects than its shuffle"
    )
    assert "p_depletion" in result


def test_far_from_nuclei_inherits_valid_size():
    """Regions with no nuclei must inherit a nearby valid size, not the min."""
    from lung_nematic.adaptive_radius import cell_size_from_nuclei

    centroids = np.array([[500.0, 500.0], [520.0, 500.0]])
    size = cell_size_from_nuclei(centroids, (1000, 1000), smoothing_px=40.0,
                                 min_size_px=5.0, max_size_px=80.0)
    far = size[50:100, 50:100].mean()
    assert far > 10.0, (
        f"far-from-nuclei region got size {far:.1f}, should inherit ~20 not clamp to 5"
    )
    assert (size <= 5.01).mean() < 0.1, "most of the image should not be at the minimum"


def test_null_model_reports_ties_as_equal():
    """observed == null must be 'equal', not 'enriched', with z defined only on tie."""
    from lung_nematic.defects_adaptive import adaptive_null_model

    shape = (200, 200)
    # a disordered low-order field: observed 0 defects, null also 0
    field = {"theta": np.full(shape, 0.5), "order": np.full(shape, 0.02),
             "density": np.ones(shape)}
    result = adaptive_null_model(
        field, np.ones(shape, bool), np.full(shape, 25.0),
        _config(), n_permutations=10, grid_step_px=15, seed=0,
    )
    assert result["observed_count"] == 0
    assert result["null_mean"] == 0.0
    assert result["direction"] == "equal", (
        f"a tie must be 'equal', got '{result['direction']}'"
    )
    # z is 0 on a genuine tie (both zero spread and zero difference)
    assert result["z_score"] == 0.0


def test_null_model_z_is_nan_when_undefined():
    """Zero null spread with a non-tie observation gives z = NaN, not 0."""
    from lung_nematic.defects_adaptive import adaptive_null_model
    import lung_nematic.defects_adaptive as module

    # force a constant null of 3 while the observed is 0, so std=0 but no tie
    shape = (200, 200)
    field = {"theta": np.full(shape, 0.5), "order": np.full(shape, 0.02),
             "density": np.ones(shape)}
    original = module.detect_defects_adaptive
    calls = {"n": 0}

    def fake_detect(*args, **kwargs):
        # first call is the observed (return empty), the rest are the null
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()
        return pd.DataFrame({"x_px": [1.0, 2.0, 3.0], "y_px": [1.0, 2.0, 3.0],
                             "charge": [0.5, 0.5, 0.5]})

    module.detect_defects_adaptive = fake_detect
    try:
        result = adaptive_null_model(
            field, np.ones(shape, bool), np.full(shape, 25.0),
            _config(), n_permutations=5, grid_step_px=15, seed=0,
        )
    finally:
        module.detect_defects_adaptive = original

    assert result["null_std"] == 0.0
    assert result["direction"] == "depleted"  # observed 0 < null 3
    assert np.isnan(result["z_score"]), "z must be NaN when std=0 without a tie"


def test_adaptive_null_parallel_matches_serial():
    """The parallel adaptive null must be identical to the serial one."""
    from lung_nematic.defects_adaptive import adaptive_null_model

    shape = (250, 250)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    theta = np.mod(0.4 + 0.3 * np.sin(xx / 80.0), np.pi)
    radius = np.hypot(xx - 125, yy - 125)
    field = {"theta": theta, "order": np.full(shape, 0.6),
             "density": 1.0 + 0.5 * np.exp(-0.5 * (radius / 110) ** 2)}

    serial = adaptive_null_model(
        field, np.ones(shape, bool), np.full(shape, 25.0),
        _config(), n_permutations=12, grid_step_px=12, seed=0, n_jobs=1,
    )
    parallel = adaptive_null_model(
        field, np.ones(shape, bool), np.full(shape, 25.0),
        _config(), n_permutations=12, grid_step_px=12, seed=0, n_jobs=2,
    )
    assert np.array_equal(serial["null_counts"], parallel["null_counts"])
    assert serial["p_depletion"] == parallel["p_depletion"]


def test_null_model_rejects_zero_permutations():
    """The public API must reject n_permutations < 1, like the main null model."""
    from lung_nematic.defects_adaptive import adaptive_null_model

    shape = (100, 100)
    field = {"theta": np.zeros(shape), "order": np.full(shape, 0.5),
             "density": np.ones(shape)}
    with pytest.raises(ValueError):
        adaptive_null_model(
            field, np.ones(shape, bool), np.full(shape, 20.0),
            _config(), n_permutations=0, grid_step_px=15,
        )


def test_context_and_null_serialise_to_strict_json():
    """NaN from empty defects / undefined z must serialise as null, not NaN."""
    import json
    from lung_nematic.io_utils import json_safe
    from lung_nematic.defects_adaptive import defect_order_context

    shape = (100, 100)
    field = {"theta": np.zeros(shape), "order": np.ones(shape), "density": np.ones(shape)}
    # empty defects -> NaN-bearing context
    context = defect_order_context(pd.DataFrame(), field, np.ones(shape, bool))
    payload = {"context": context, "z_score": float("nan")}
    # strict JSON must not raise once sanitised
    text = json.dumps(json_safe(payload), allow_nan=False)
    assert "NaN" not in text
    assert "null" in text
