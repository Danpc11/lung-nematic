"""Tests for gel-to-histology cross-mapping.

The failure this module exists to prevent is a confident number produced from an
invalid comparison: two defect densities measured at different physical scales,
or a stiffness inferred by extrapolating a three-point fit far outside its
range. Most of the tests below assert that such cases are refused or flagged
rather than silently returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lung_nematic.crossmap import (
    calibration_curve,
    charge_balance,
    dimensionless_density,
    infer_stiffness,
    sigmas_for_microns,
    steady_state_window,
)

HISTOLOGY_MPP = 0.114679
GEL_MPP = 1 / 1.42


# ------------------------------------------------------------ scale matching
def test_the_same_physical_scale_gives_different_pixel_sigmas():
    """The core incomparability: one pixel sigma is two physical scales."""
    histology = sigmas_for_microns((8.0,), HISTOLOGY_MPP)[0]
    gel = sigmas_for_microns((8.0,), GEL_MPP)[0]
    assert histology == pytest.approx(69.8, abs=0.5)
    assert gel == pytest.approx(11.4, abs=0.2)
    assert histology / gel == pytest.approx(GEL_MPP / HISTOLOGY_MPP, rel=1e-6)


def test_packaged_defaults_are_not_scale_matched():
    """Documents the mismatch that motivated this module.

    sigma = 70 px is 8.0 um in histology; sigma = 40 px is 28.2 um on a gel.
    Density falls roughly as 1/sigma^2, so this alone is about a 12x offset.
    """
    histology_um = 70 * HISTOLOGY_MPP
    gel_um = 40 * GEL_MPP
    assert histology_um == pytest.approx(8.0, abs=0.1)
    assert gel_um == pytest.approx(28.2, abs=0.2)
    assert (gel_um / histology_um) ** 2 > 10


def test_sub_pixel_sigma_is_refused():
    with pytest.raises(ValueError, match="below one pixel"):
        sigmas_for_microns((0.3,), GEL_MPP)


def test_non_positive_calibration_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        sigmas_for_microns((8.0,), 0.0)


# --------------------------------------------------------- dimensionless rho
def test_dimensionless_density_is_scale_free():
    """Two systems with the same texture must agree after normalisation.

    Halving the correlation length at four times the density is the same
    texture viewed at a different magnification; rho*xi^2 must not notice.
    """
    a = dimensionless_density([400.0], [50.0])
    b = dimensionless_density([1600.0], [25.0])
    assert a[0] == pytest.approx(b[0])


def test_unusable_correlation_length_yields_nan():
    """A negative xi means the fit failed; propagating it would be a lie.

    On a nearly perfectly ordered field the autocorrelation never decays and the
    fit extrapolates negative - observed in practice as -213 px. That is
    meaningless, not merely large.
    """
    assert np.isnan(dimensionless_density([150.0], [-213.0])[0])
    assert np.isnan(dimensionless_density([150.0], [0.0])[0])
    assert np.isnan(dimensionless_density([150.0], [np.nan])[0])


def test_dimensionless_density_uses_consistent_units():
    """rho in per mm^2 and xi in um: 1000 um = 1 mm must cancel exactly."""
    assert dimensionless_density([1.0], [1000.0])[0] == pytest.approx(1.0)


# ---------------------------------------------------------------- calibration
def test_calibration_recovers_an_imposed_exponent():
    stiffness = np.array([5.0, 10.0, 23.0])
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": stiffness,
                      "rho_xi2": 0.8 * stiffness**-0.6})
    )
    assert curve["log_slope"] == pytest.approx(-0.6, abs=1e-6)
    assert curve["n_points"] == 3


def test_calibration_recovers_a_positive_exponent_too():
    """The sign is the scientific result, so both directions must work."""
    stiffness = np.array([5.0, 10.0, 23.0])
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": stiffness,
                      "rho_xi2": 0.05 * stiffness**0.9})
    )
    assert curve["log_slope"] == pytest.approx(0.9, abs=1e-6)


def test_calibration_refuses_a_single_point():
    with pytest.raises(ValueError, match="at least two"):
        calibration_curve(
            pd.DataFrame({"stiffness_kPa": [10.0], "rho_xi2": [0.4]})
        )


def test_calibration_requires_its_columns():
    with pytest.raises(ValueError, match="missing columns"):
        calibration_curve(pd.DataFrame({"stiffness_kPa": [5.0, 10.0]}))


def test_calibration_drops_non_positive_responses():
    """A zero density cannot enter a log fit; it must be dropped, not crash."""
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": [5.0, 10.0, 23.0],
                      "rho_xi2": [0.4, 0.0, 0.15]})
    )
    assert curve["n_points"] == 2


# ------------------------------------------------------------------ inversion
def test_inversion_round_trips_within_the_calibrated_range():
    stiffness = np.array([5.0, 10.0, 23.0])
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": stiffness,
                      "rho_xi2": 0.8 * stiffness**-0.6})
    )
    result = infer_stiffness([0.8 * 12.0**-0.6], curve)
    assert result.inferred_stiffness_kPa.iloc[0] == pytest.approx(12.0, rel=1e-6)
    assert bool(result.within_calibrated_range.iloc[0])


def test_extrapolation_is_flagged_not_hidden():
    """A three-point fit on 5-23 kPa says nothing about 200 kPa."""
    stiffness = np.array([5.0, 10.0, 23.0])
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": stiffness,
                      "rho_xi2": 0.8 * stiffness**-0.6})
    )
    result = infer_stiffness([0.8 * 200.0**-0.6, 0.8 * 0.5**-0.6], curve)
    assert not result.within_calibrated_range.any()
    assert np.isfinite(result.inferred_stiffness_kPa).all()


def test_inversion_refuses_a_flat_calibration():
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": [5.0, 23.0], "rho_xi2": [0.4, 0.4]})
    )
    with pytest.raises(ValueError, match="too flat to invert"):
        infer_stiffness([0.4], curve)


def test_inversion_returns_nan_for_impossible_responses():
    curve = calibration_curve(
        pd.DataFrame({"stiffness_kPa": [5.0, 23.0], "rho_xi2": [0.4, 0.2]})
    )
    assert np.isnan(infer_stiffness([-1.0, 0.0], curve)
                    .inferred_stiffness_kPa).all()


# --------------------------------------------------------- steady state gate
def test_plateau_is_recognised_as_steady():
    kinetics = pd.DataFrame({"frame": range(20), "n_defects": [40] * 20,
                             "births": [5] * 20, "deaths": [5] * 20})
    assert steady_state_window(kinetics)["reached_steady_state"]


def test_coarsening_series_is_refused():
    """No plateau means no temporal anchor for comparing to fixed tissue."""
    kinetics = pd.DataFrame({"frame": range(20),
                             "n_defects": list(range(60, 40, -1)),
                             "births": [1] * 20, "deaths": [6] * 20})
    result = steady_state_window(kinetics)
    assert not result["reached_steady_state"]
    assert "imbalance" in result["reason"]


def test_short_series_is_refused_rather_than_guessed():
    kinetics = pd.DataFrame({"frame": range(4), "n_defects": [40] * 4,
                             "births": [5] * 4, "deaths": [5] * 4})
    assert not steady_state_window(kinetics)["reached_steady_state"]
    assert steady_state_window(pd.DataFrame(
        columns=["frame", "n_defects", "births", "deaths"]
    ))["reached_steady_state"] is False


# ------------------------------------------------------ sectioning diagnostic
def test_balanced_charges_indicate_a_clean_two_dimensional_slice():
    defects = pd.DataFrame({"charge": [0.5] * 10 + [-0.5] * 10})
    result = charge_balance(defects)
    assert result["ratio"] == pytest.approx(1.0)
    assert result["imbalance"] == pytest.approx(0.0)


def test_charge_imbalance_flags_a_three_dimensional_cut():
    """Pair creation balances the classes in 2D; a 3D cut need not."""
    result = charge_balance(pd.DataFrame({"charge": [0.5] * 15 + [-0.5] * 5}))
    assert result["imbalance"] == pytest.approx(0.5)


def test_charge_balance_handles_no_defects():
    result = charge_balance(pd.DataFrame({"charge": []}, dtype=float))
    assert result["n_plus"] == 0
    assert np.isnan(result["ratio"])
