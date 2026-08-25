"""Tests for focus-domain nematic architecture.

Each quantity here has an analytically known value on a synthetic field, so the
tests pin numbers rather than merely exercising code paths. A director analysis
that is subtly wrong - a mishandled modulo-pi branch, a tangent computed as a
normal - still returns plausible angles and fractions, so "it ran" proves
nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lung_nematic.focus import (
    analyze_focus,
    boundary_anchoring,
    charge_consistency,
    domain_shape,
    enclosed_charge_from_boundary,
    enclosed_charge_from_defects,
    euler_characteristic,
    radial_order_profile,
    splay_bend_decomposition,
)

SIZE = 300
CENTRE = 150.0
YY, XX = np.mgrid[0:SIZE, 0:SIZE]
DISC = ((XX - CENTRE) ** 2 + (YY - CENTRE) ** 2) < 100**2


def field(theta, order=0.8):
    return {
        "theta": theta,
        "order": np.full_like(theta, order),
        "density": np.ones_like(theta),
    }


def defect_field(charge, x=CENTRE, y=CENTRE):
    return charge * np.arctan2(YY - y, XX - x)


# ------------------------------------------------------ Euler characteristic
def test_disc_and_annulus_have_different_euler_characteristics():
    """The Poincare-Hopf target is chi, not the constant +1.

    A focus sectioned obliquely can appear as an annulus, where the expected
    enclosed charge is 0 rather than +1. Computing chi keeps that sectioning
    artefact from being read as a physical anomaly.
    """
    annulus = DISC & ~(((XX - CENTRE) ** 2 + (YY - CENTRE) ** 2) < 30**2)
    assert euler_characteristic(DISC) == 1
    assert euler_characteristic(annulus) == 0


def test_two_separate_discs_give_chi_two():
    a = ((XX - 80) ** 2 + (YY - 80) ** 2) < 40**2
    b = ((XX - 220) ** 2 + (YY - 220) ** 2) < 40**2
    assert euler_characteristic(a | b) == 2


def test_empty_mask_has_zero_euler_characteristic():
    assert euler_characteristic(np.zeros((10, 10), dtype=bool)) == 0


# --------------------------------------------------------- boundary winding
@pytest.mark.parametrize("charge", [0.5, -0.5, 1.0, -1.0])
def test_boundary_winding_recovers_an_imposed_charge(charge):
    result = enclosed_charge_from_boundary(field(defect_field(charge)), DISC)
    assert result["enclosed_charge"] == pytest.approx(charge, abs=0.02)


def test_two_half_defects_sum_to_one():
    """The Poincare-Hopf prediction for a focus: one +1 or two +1/2."""
    theta = defect_field(0.5, x=110.0) + defect_field(0.5, x=190.0)
    result = enclosed_charge_from_boundary(field(theta), DISC)
    assert result["enclosed_charge"] == pytest.approx(1.0, abs=0.02)


def test_a_defect_outside_the_domain_is_not_counted():
    theta = defect_field(1.0, x=20.0, y=20.0)
    result = enclosed_charge_from_boundary(field(theta), DISC)
    assert result["enclosed_charge"] == pytest.approx(0.0, abs=0.05)


def test_uniform_field_encloses_no_charge():
    result = enclosed_charge_from_boundary(
        field(np.full((SIZE, SIZE), 0.7)), DISC
    )
    assert result["enclosed_charge"] == pytest.approx(0.0, abs=0.02)


def test_annulus_sums_outer_and_inner_boundaries():
    annulus = DISC & ~(((XX - CENTRE) ** 2 + (YY - CENTRE) ** 2) < 30**2)
    result = enclosed_charge_from_boundary(field(defect_field(1.0)), annulus)
    assert result["enclosed_charge"] == pytest.approx(0.0, abs=0.02)
    assert result["euler_characteristic"] == 0


def test_disconnected_components_sum_all_boundaries():
    a = ((XX - 80) ** 2 + (YY - 80) ** 2) < 40**2
    b = ((XX - 220) ** 2 + (YY - 220) ** 2) < 40**2
    theta = defect_field(1.0, x=80.0, y=80.0) + defect_field(
        1.0, x=220.0, y=220.0
    )
    result = enclosed_charge_from_boundary(field(theta), a | b)
    assert result["enclosed_charge"] == pytest.approx(2.0, abs=0.03)
    assert result["euler_characteristic"] == 2


# ------------------------------------------------------------- consistency
def test_the_two_charge_routes_agree_when_detection_is_correct():
    theta = defect_field(0.5, x=110.0) + defect_field(0.5, x=190.0)
    defects = pd.DataFrame({"x_px": [110.0, 190.0], "y_px": [CENTRE] * 2,
                            "charge": [0.5, 0.5]})
    result = charge_consistency(field(theta), defects, DISC)
    assert result["detector_consistent"]
    assert result["poincare_hopf_satisfied"]
    assert result["n_defects_inside"] == 2


def test_a_missed_defect_breaks_detector_consistency_only():
    """Separating the two failures is the point of having both routes.

    The boundary still measures +1, so the physics holds; only the detector is
    wrong. Reporting one flag would confuse a tuning problem with a result.
    """
    theta = defect_field(0.5, x=110.0) + defect_field(0.5, x=190.0)
    only_one = pd.DataFrame({"x_px": [110.0], "y_px": [CENTRE],
                             "charge": [0.5]})
    result = charge_consistency(field(theta), only_one, DISC)
    assert not result["detector_consistent"]
    assert result["poincare_hopf_satisfied"]


def test_a_field_violating_poincare_hopf_is_flagged():
    """Charge 0 in a simply connected domain is not a droplet with anchoring."""
    result = charge_consistency(
        field(np.full((SIZE, SIZE), 0.7)), pd.DataFrame(
            columns=["x_px", "y_px", "charge"], dtype=float), DISC
    )
    assert result["detector_consistent"]
    assert not result["poincare_hopf_satisfied"]


def test_defects_outside_the_mask_are_excluded():
    defects = pd.DataFrame({"x_px": [CENTRE, 5.0], "y_px": [CENTRE, 5.0],
                            "charge": [0.5, 0.5]})
    result = enclosed_charge_from_defects(defects, DISC)
    assert result["n_defects"] == 1
    assert result["enclosed_charge"] == pytest.approx(0.5)


# ---------------------------------------------------------------- anchoring
def test_tangential_field_reads_as_tangential():
    """Myofibroblasts parallel to the focus margin: the expected case."""
    theta = np.arctan2(YY - CENTRE, XX - CENTRE) + np.pi / 2
    result = boundary_anchoring(field(theta), DISC)
    assert result["median_deg"] < 15.0
    assert result["fraction_tangential"] > 0.95


def test_radial_field_reads_as_homeotropic():
    result = boundary_anchoring(field(np.arctan2(YY - CENTRE, XX - CENTRE)),
                                DISC)
    assert result["median_deg"] > 75.0
    assert result["fraction_homeotropic"] > 0.95


def test_anchoring_angles_are_folded_to_a_quarter_turn():
    """The director has no head or tail; angles above 90 degrees are a bug."""
    theta = np.arctan2(YY - CENTRE, XX - CENTRE) + np.pi / 2
    result = boundary_anchoring(field(theta), DISC)
    assert result["angles_deg"].min() >= 0.0
    assert result["angles_deg"].max() <= 90.0 + 1e-9


# ------------------------------------------------------------- splay / bend
def test_radial_field_is_pure_splay():
    result = splay_bend_decomposition(
        field(np.arctan2(YY - CENTRE, XX - CENTRE)), DISC
    )
    assert result["bend_fraction"] < 0.05


def test_circular_field_is_pure_bend():
    theta = np.arctan2(YY - CENTRE, XX - CENTRE) + np.pi / 2
    assert splay_bend_decomposition(field(theta), DISC)["bend_fraction"] > 0.95


def test_uniform_field_has_undefined_bend_fraction():
    """No deformation means the ratio has no value - NaN, not zero."""
    result = splay_bend_decomposition(field(np.full((SIZE, SIZE), 0.7)), DISC)
    assert np.isnan(result["bend_fraction"])
    assert result["mean_splay2"] == pytest.approx(0.0, abs=1e-12)


def test_gradients_survive_the_modulo_pi_branch():
    """A director wrapping through pi must not register as deformation.

    theta and theta + pi are the same director. Differencing the raw angle
    array turns the representation's wrap into a spurious gradient spike, which
    would inflate both splay and bend wherever the branch happens to fall.
    """
    theta = np.where(XX < CENTRE, 0.05, 0.05 + np.pi)
    result = splay_bend_decomposition(field(theta), DISC)
    assert result["mean_splay2"] < 1e-6
    assert result["mean_bend2"] < 1e-6


# ------------------------------------------------------------ radial profile
def test_radial_profile_spans_margin_to_core():
    theta = np.arctan2(YY - CENTRE, XX - CENTRE)
    order = np.full((SIZE, SIZE), 0.3)
    distance = np.hypot(XX - CENTRE, YY - CENTRE)
    order[distance < 50] = 0.9      # ordered core, disordered rim

    profile = radial_order_profile(
        {"theta": theta, "order": order, "density": np.ones_like(order)}, DISC
    )
    assert len(profile) >= 5
    assert profile.iloc[0]["median_order"] < profile.iloc[-1]["median_order"]
    assert profile.depth_mid.is_monotonic_increasing


def test_radial_profile_is_empty_for_an_empty_mask():
    theta = np.zeros((SIZE, SIZE))
    assert radial_order_profile(field(theta),
                                np.zeros((SIZE, SIZE), bool)).empty


# ----------------------------------------------------------------- geometry
def test_shape_separates_a_round_cut_from_an_oblique_one():
    """Only a near-tangential cut gives a genuine 2D slice."""
    elongated = (((XX - CENTRE) / 100.0) ** 2
                 + ((YY - CENTRE) / 25.0) ** 2) < 1.0
    round_cut = domain_shape(DISC)
    oblique = domain_shape(elongated)
    assert round_cut["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
    assert oblique["aspect_ratio"] > 3.0
    assert round_cut["circularity"] > oblique["circularity"]


def test_shape_converts_area_when_calibrated():
    result = domain_shape(DISC, microns_per_pixel=0.114679)
    assert result["area_mm2"] == pytest.approx(
        result["area_px"] * 0.114679**2 / 1e6
    )


# -------------------------------------------------------------- integration
def test_analyze_focus_returns_a_flat_summary():
    theta = defect_field(0.5, x=110.0) + defect_field(0.5, x=190.0)
    defects = pd.DataFrame({"x_px": [110.0, 190.0], "y_px": [CENTRE] * 2,
                            "charge": [0.5, 0.5]})
    summary = analyze_focus(field(theta), defects, DISC, 0.114679)

    assert summary["charge_boundary"] == pytest.approx(1.0, abs=0.02)
    assert summary["poincare_hopf_satisfied"]
    assert 0.0 <= summary["anchoring_median_deg"] <= 90.0
    assert summary["shape_area_px"] > 0
    assert all(not isinstance(v, np.ndarray) for v in summary.values())
