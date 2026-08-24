"""Tests for the field-based global order parameter and its finite-size null.

The load-bearing test is `test_global_order_differs_between_field_types`. The
bug it guards against produced byte-identical `global_nematic_order_S` across
nuclear, collagen and fused runs of the same image, because the value was read
from the nuclei table and never saw the field. Nothing errored and every
downstream statistic looked plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lung_nematic.nematic import (
    compute_global_order,
    compute_global_order_from_field,
    compute_nematic_field,
    expected_order_under_randomness,
    global_order_null,
)

SHAPE = (300, 300)
SIGMA = 15.0


def _nuclei(n=2000, spread=None, seed=0, center=0.7):
    rng = np.random.default_rng(seed)
    height, width = SHAPE
    angles = (
        rng.uniform(0.0, np.pi, n)
        if spread is None
        else rng.normal(center, spread, n) % np.pi
    )
    return pd.DataFrame(
        {
            "x_px": rng.uniform(20, width - 20, n),
            "y_px": rng.uniform(20, height - 20, n),
            "theta_rad": angles,
            "anisotropy_weight": rng.uniform(0.3, 1.0, n),
        }
    )


# ------------------------------------------------------- field-based order
def test_field_order_reproduces_the_nuclei_value_on_the_nuclear_field():
    """Old and new nuclear numbers must stay comparable.

    Gaussian smoothing is linear and conserves mass, so summing the smoothed
    nematic vectors recovers the sum of the impulses up to boundary losses.
    """
    nuclei = _nuclei(spread=0.4)
    field = compute_nematic_field(nuclei, SHAPE, SIGMA)
    mask = np.ones(SHAPE, dtype=bool)
    assert compute_global_order_from_field(field, mask) == pytest.approx(
        compute_global_order(nuclei), rel=0.10
    )


def test_aligned_field_scores_far_above_random_field():
    mask = np.ones(SHAPE, dtype=bool)
    aligned = compute_global_order_from_field(
        compute_nematic_field(_nuclei(spread=0.12), SHAPE, SIGMA), mask
    )
    random = compute_global_order_from_field(
        compute_nematic_field(_nuclei(spread=None), SHAPE, SIGMA), mask
    )
    assert aligned > 0.8
    assert random < 0.1


def test_global_order_differs_between_field_types():
    """Two orientation sources over one image must not give one number.

    Byte-identical values across field types was the signature of the bug:
    the order was read from the nuclei table regardless of which field the run
    was analysing.
    """
    nuclei = _nuclei(spread=0.3, seed=1)
    nuclear = compute_nematic_field(nuclei, SHAPE, SIGMA)

    # A second, differently oriented source standing in for collagen.
    other = nuclei.copy()
    other["theta_rad"] = (nuclei["theta_rad"] + np.pi / 2) % np.pi
    other.loc[other.index[:800], "theta_rad"] = np.random.default_rng(
        2
    ).uniform(0.0, np.pi, 800)
    collagen = compute_nematic_field(other, SHAPE, SIGMA)

    mask = np.ones(SHAPE, dtype=bool)
    assert compute_global_order_from_field(
        nuclear, mask
    ) != pytest.approx(compute_global_order_from_field(collagen, mask))


def test_tissue_mask_restricts_the_average():
    nuclei = _nuclei(spread=0.2)
    field = compute_nematic_field(nuclei, SHAPE, SIGMA)
    full = np.ones(SHAPE, dtype=bool)
    half = np.zeros(SHAPE, dtype=bool)
    half[:150] = True
    assert compute_global_order_from_field(field, full) != pytest.approx(
        compute_global_order_from_field(field, half)
    )


def test_order_is_nan_on_empty_tissue():
    field = compute_nematic_field(_nuclei(), SHAPE, SIGMA)
    assert np.isnan(
        compute_global_order_from_field(field, np.zeros(SHAPE, dtype=bool))
    )


def test_order_is_bounded():
    nuclei = _nuclei(spread=0.01)
    field = compute_nematic_field(nuclei, SHAPE, SIGMA)
    value = compute_global_order_from_field(field, np.ones(SHAPE, dtype=bool))
    assert 0.0 <= value <= 1.0


# ------------------------------------------------------ finite-size null
def test_null_is_calibrated_under_random_orientations():
    """The property that killed the block-permutation null.

    Blocks over the smoothed field stayed correlated through the kernel and
    rejected ~45% of purely random fields at alpha = 0.05. Permuting at the
    source has no block-size parameter and calibrates.
    """
    p_values = []
    for seed in range(120):
        p_values.append(
            global_order_null(
                _nuclei(spread=None, seed=500 + seed),
                n_permutations=99,
                seed=seed,
            )["global_order_p"]
        )
    p_values = np.array(p_values)
    assert 0.40 < p_values.mean() < 0.60
    assert (p_values < 0.05).mean() < 0.12


def test_null_detects_real_alignment():
    result = global_order_null(_nuclei(spread=0.5, seed=7), 199, seed=1)
    assert result["global_order_excess"] > 5.0
    assert result["global_order_p"] <= 0.01


def test_permuted_floor_matches_the_analytic_rayleigh_value():
    nuclei = _nuclei(spread=None, seed=3)
    result = global_order_null(nuclei, 399, seed=0)
    analytic = expected_order_under_randomness(
        nuclei["anisotropy_weight"].to_numpy()
    )
    assert result["global_order_null_mean"] == pytest.approx(
        analytic, rel=0.10
    )


def test_fewer_nuclei_raise_the_floor():
    """The confound itself: S grows as ~1/sqrt(N) with no change in biology."""
    floors = [
        global_order_null(_nuclei(n=n, spread=None, seed=11), 99, seed=0)[
            "global_order_null_mean"
        ]
        for n in (500, 2000, 8000)
    ]
    assert floors[0] > floors[1] > floors[2]
    # Excess is what removes the confound: same alignment, same excess.
    excesses = [
        global_order_null(_nuclei(n=n, spread=0.5, seed=12), 199, seed=0)[
            "global_order_excess"
        ]
        for n in (2000, 8000)
    ]
    assert excesses[1] / excesses[0] > 1.4


def test_null_is_reproducible_and_seed_dependent():
    nuclei = _nuclei(spread=0.5, seed=4)
    first = global_order_null(nuclei, 99, seed=0)
    assert first == global_order_null(nuclei, 99, seed=0)
    assert (
        first["global_order_null_mean"]
        != global_order_null(nuclei, 99, seed=1)["global_order_null_mean"]
    )


def test_p_value_uses_the_add_one_correction():
    """No permutation test may report p = 0."""
    result = global_order_null(_nuclei(spread=0.02, seed=5), 99, seed=0)
    assert result["global_order_p"] == pytest.approx(1 / 100)


def test_null_handles_empty_input():
    empty = pd.DataFrame(
        {"theta_rad": [], "anisotropy_weight": []}, dtype=float
    )
    result = global_order_null(empty, 99, seed=0)
    assert np.isnan(result["global_order_observed"])
    assert np.isnan(result["global_order_excess"])
