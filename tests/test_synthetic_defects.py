"""
Detector validation on synthetic fields with physical defect cores.

These go beyond "something with the right charge was detected". Each test
checks *where* the detection landed, and the fields carry a real ``tanh`` core
so the order parameter actually falls at the singularity — without which the
colocalization machinery cannot be exercised at all.

The centrepiece is ``test_two_half_defects_resolution_limit``, which measures
the separation at which the integer ring detector stops confusing two ``+1/2``
cores for a genuine ``+1``. The README states that ambiguity; this pins it to a
number.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lung_nematic.config import load_default_config
from lung_nematic.defects import (
    detect_defects_single_scale,
    detect_integer_defects_single_scale,
)
from lung_nematic.synthetic import (
    director_field,
    integer_charge_sweep,
    localization_error,
    tissue_mask,
    two_half_defect_field,
    two_half_resolution_sweep,
    uniform_field,
)

SHAPE = (340, 340)
CENTRE = (SHAPE[1] / 2.0, SHAPE[0] / 2.0)


@pytest.fixture
def config():
    """Fine grid and a generous ring, so localization is actually testable."""
    return replace(
        load_default_config(),
        defect_grid_step_px=8,
        density_quantile=0.30,
        min_edge_distance_px=20,
        integer_defect_loop_radius_px=30,
        integer_defect_loop_points=12,
        integer_min_separation_px=60.0,
    )


@pytest.fixture
def mask():
    return tissue_mask(SHAPE)


# --------------------------------------------------------------- core profile
def test_core_loses_nematic_order():
    """S must vanish at the singularity and recover over the core size."""
    core = 8.0
    field = director_field(SHAPE, [(*CENTRE, 0.5)], core_size_px=core)
    row = int(CENTRE[1])
    column = int(CENTRE[0])

    at_core = field["order"][row, column]
    at_one_core = field["order"][row, column + int(core)]
    far = field["order"][row, column + int(8 * core)]

    assert at_core < 0.05, "order should collapse at the singularity"
    assert 0.5 < at_one_core < 0.95, "order should be partly recovered at r = xi"
    assert far > 0.99, "order should saturate far from the core"
    assert at_core < at_one_core < far, "S(r) must increase monotonically"


def test_core_size_sets_the_recovery_length():
    narrow = director_field(SHAPE, [(*CENTRE, 0.5)], core_size_px=4.0)
    wide = director_field(SHAPE, [(*CENTRE, 0.5)], core_size_px=32.0)
    row, column = int(CENTRE[1]), int(CENTRE[0]) + 10
    assert narrow["order"][row, column] > wide["order"][row, column]


# ------------------------------------------------------------- half-integer
@pytest.mark.parametrize("charge", [0.5, -0.5])
def test_half_integer_defect_is_localized(config, mask, charge):
    """Detection is not enough: it must land on the core."""
    expected = (CENTRE[0] + 5.0, CENTRE[1] + 5.0)
    field = director_field(SHAPE, [(*expected, charge)], core_size_px=8.0)
    detected = detect_defects_single_scale(field, mask, config)

    assert not detected.empty
    assert (detected["charge"] == charge).any()

    error = localization_error(detected, expected, charge=charge)
    tolerance = 1.5 * config.defect_grid_step_px
    assert error <= tolerance, (
        f"nearest {charge:+.1f} detection is {error:.1f} px from the core, "
        f"tolerance {tolerance:.1f} px"
    )


def test_uniform_field_has_no_defects(config, mask):
    """A defect-free field must stay defect-free."""
    field = uniform_field(SHAPE, angle=0.4)
    detected = detect_defects_single_scale(field, mask, config)
    if not detected.empty:
        assert (detected["charge"] == 0).all()


# ------------------------------------------------------------------ integer
@pytest.mark.parametrize("charge", [1.0, -1.0])
def test_integer_defect_is_localized_and_counted_once(config, mask, charge):
    """One defect must give one detection, not one per enclosing ring."""
    field = director_field(SHAPE, [(*CENTRE, charge)], core_size_px=8.0)
    detected = detect_integer_defects_single_scale(field, mask, config)

    matching = detected.loc[detected["charge"] == charge]
    assert len(matching) == 1, (
        f"expected a single {charge:+.0f} defect, got {len(matching)}; "
        "ring detections are probably not being clustered"
    )
    assert matching["n_ring_detections"].iloc[0] > 1, (
        "the clustering should be collapsing several ring detections"
    )
    error = localization_error(detected, CENTRE, charge=charge)
    assert error <= config.defect_grid_step_px


def test_integer_charge_is_recovered_across_core_sizes(config):
    """Raw enclosed charge should stay near the ideal for a clean field."""
    sweep = integer_charge_sweep(config, shape=SHAPE)
    assert (sweep["n_detected"] == 1).all()
    assert (sweep["charge_raw_abs_error"] < 0.15).all(), (
        "raw charge drifts from +1 even without noise; a tolerance parameter "
        "would need calibrating against this"
    )


# ------------------------------------------- the ambiguity the README states
def test_two_half_defects_resolution_limit(config):
    """Where does the ring stop confusing two +1/2 for one +1?

    Below roughly ``d = 1.6 R`` the pair sits inside the ring and is reported
    as an integer defect; above it, only the two half-integer cores survive.
    The half-integer layer resolves both at every separation tested, so a +1
    should be treated as suspect whenever a +1/2 pair is present nearby.
    """
    sweep = two_half_resolution_sweep(config, shape=SHAPE)

    assert not sweep.empty
    assert sweep["resolved_two_halves"].all(), (
        "the half-integer layer should resolve both cores at every separation"
    )

    close = sweep.loc[sweep["separation_over_radius"] <= 1.0]
    far = sweep.loc[sweep["separation_over_radius"] >= 2.0]

    assert (close["n_plus_one"] >= 1).all(), (
        "a pair well inside the ring must register as enclosed charge +1"
    )
    assert (far["n_plus_one"] == 0).all(), (
        "a widely separated pair must not be reported as an integer defect"
    )

    # the transition must be somewhere in between, and it must be a transition
    transition = sweep.loc[sweep["n_plus_one"] == 0, "separation_over_radius"]
    assert not transition.empty
    threshold = transition.min()
    assert 1.0 < threshold < 2.5, f"unexpected threshold at d/R = {threshold:.2f}"


def test_two_half_field_encloses_unit_charge(config, mask):
    """Sanity: the constructed pair really does carry total charge +1."""
    field = two_half_defect_field(SHAPE, separation_px=10.0, centre=CENTRE)
    detected = detect_defects_single_scale(field, mask, config)
    half_total = detected.loc[
        np.isclose(detected["charge"].abs(), 0.5), "charge"
    ].sum()
    assert half_total == pytest.approx(1.0, abs=0.5)


# --------------------------------------------------------------- aggregation
def test_group_summary_includes_integer_defects():
    """+/-1 counts must survive into the group-level summary."""
    from lung_nematic.batch import summarize_by_group

    frame = pd.DataFrame(
        {
            "group": ["case", "case", "control"],
            "global_nematic_order_S": [0.3, 0.4, 0.2],
            "local_S_median": [0.3, 0.4, 0.2],
            "n_defects_total": [10, 12, 4],
            "n_plus_half": [5, 6, 2],
            "n_minus_half": [5, 6, 2],
            "n_plus_one": [2, 1, 0],
            "n_minus_one": [1, 0, 0],
            "net_topological_charge": [0.0, 0.0, 0.0],
            "defect_density_mm2": [100.0, 120.0, 40.0],
            "mean_defect_confidence": [0.8, 0.9, 0.7],
        }
    )
    summary = summarize_by_group(frame)
    aggregated = {column for column, _ in summary.columns}
    assert "n_plus_one" in aggregated
    assert "n_minus_one" in aggregated
    assert summary.loc["case", ("n_plus_one", "mean")] == pytest.approx(1.5)
