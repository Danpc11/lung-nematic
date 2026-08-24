"""Tests for the reparameterized model and its identifiability analysis.

The load-bearing tests are the ones that assert *failure*: that a parameter the
reparameterization was meant to expose as unidentifiable really does show up in
the null space. A test suite that only checks the happy path would pass on a
model that quietly reintroduced the scale degeneracy.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lungtwin.design import DEFAULT_NOISE_SD, VisitSchedule, routine_followup
from lungtwin.identifiability import (
    analyze,
    monte_carlo_check,
    profile_likelihood,
    sensitivity_matrix,
    visits_required,
)
from lungtwin.model import Channel, Parameters, ReserveMode, observe, simulate
from lungtwin.report import render

CORE = ["fvc0", "dlco0", "r_i", "kappa", "beta"]


# --------------------------------------------------------------- the model
@pytest.mark.parametrize("switch", [None, 0.0, 0.5, 1.0])
def test_burden_matches_closed_form(switch):
    """B(t) = r*t - beta*max(0, t - t_switch) when the reserve is decoupled."""
    times = np.array([0.0, 0.25, 0.5, 1.0, 1.5])
    params = Parameters()
    burden = simulate(times, params, switch, ReserveMode.NONE)["burden"]

    start = np.inf if switch is None else switch
    expected = params.r_i * times
    if np.isfinite(start):
        expected = expected - params.beta * np.maximum(0.0, times - start)
    assert np.allclose(burden, expected, atol=1e-9)


def test_treatment_start_is_integrated_not_sampled():
    """The pre-treatment window must survive, however the visits fall.

    A discretisation that evaluates treatment only at visit times deletes the
    interval before the first post-switch visit. With a switch at 3 months and
    visits at 0/6/12 months, that would make the patient look treated from
    baseline - erasing the only window in which beta separates from r_i.
    """
    times = np.array([0.0, 0.5, 1.0])
    params = Parameters()
    switched = simulate(times, params, 0.25, ReserveMode.NONE)["burden"]
    from_baseline = simulate(times, params, 0.0, ReserveMode.NONE)["burden"]
    assert switched[-1] > from_baseline[-1] + 0.5


def test_baseline_conventions_are_fixed():
    """B(0)=0 and R(0)=1, so fvc0/dlco0 are literally the baseline values."""
    times = np.array([0.0, 1.0])
    params = Parameters()
    states = simulate(times, params, None, ReserveMode.FEEDBACK)
    assert states["burden"][0] == 0.0
    assert states["reserve"][0] == 1.0
    seen = observe(times, params, None, ReserveMode.NONE)
    assert seen["fvc_pct"][0] == pytest.approx(params.fvc0)
    assert seen["dlco_pct"][0] == pytest.approx(params.dlco0)


def test_spo2_requires_a_reserve_state():
    with pytest.raises(ValueError, match="ReserveMode.NONE"):
        observe(np.array([0.0, 1.0]), Parameters(), None, ReserveMode.NONE,
                (Channel.SPO2,))


# ------------------------------------------------------------- structural
def test_dangling_reserve_is_rejected_up_front():
    """alpha under ReserveMode.NONE is the original spec's dead parameter."""
    schedule = routine_followup(4, treatment_start=0.5)
    with pytest.raises(ValueError, match="identically zero"):
        analyze(Parameters(), CORE + ["alpha"], schedule, ReserveMode.NONE)


def test_dangling_reserve_shows_zero_sensitivity_when_forced():
    """Bypassing the guard must still expose the null direction, not hide it."""
    schedule = routine_followup(4, treatment_start=0.5)
    params = Parameters()
    matrix = sensitivity_matrix(
        params, CORE + ["alpha"], schedule, ReserveMode.OBSERVED
    )
    # Under OBSERVED the SpO2 channel is absent from this schedule, so alpha
    # still reaches nothing: its column must be exactly zero.
    assert np.allclose(matrix[:, -1], 0.0)


def test_beta_unidentifiable_without_a_treatment_change():
    params, schedule = Parameters(), routine_followup(4, treatment_start=None)
    report = analyze(params, CORE, schedule, ReserveMode.NONE)
    assert report.rank == 4
    assert list(report.null_directions[0]) == ["beta"]


def test_only_the_difference_is_identifiable_under_constant_treatment():
    params = Parameters()
    report = analyze(params, CORE, routine_followup(4, treatment_start=0.0),
                     ReserveMode.NONE)
    assert report.rank == 4
    direction = report.null_directions[0]
    assert set(direction) == {"r_i", "beta"}
    # Equal and same-signed loading: the null direction is r_i - beta held fixed.
    assert direction["r_i"] == pytest.approx(direction["beta"], abs=1e-3)


def test_treatment_switch_restores_full_rank():
    report = analyze(Parameters(), CORE, routine_followup(4, treatment_start=0.5),
                     ReserveMode.NONE)
    assert report.is_structurally_identifiable
    assert report.rank == 5


def test_missing_dlco_channel_kills_kappa_and_dlco0():
    schedule = routine_followup(4, treatment_start=0.5, channels=(Channel.FVC,))
    report = analyze(Parameters(), CORE, schedule, ReserveMode.NONE)
    assert report.rank == 3
    dead = set().union(*(set(d) for d in report.null_directions))
    assert dead == {"kappa", "dlco0"}


def test_feedback_mode_makes_alpha_reachable():
    """With gamma > 0 the reserve acts on the burden rate, so alpha is live."""
    params = Parameters(gamma=0.8)
    schedule = routine_followup(6, treatment_start=0.5)
    matrix = sensitivity_matrix(params, ["r_i", "alpha"], schedule,
                                ReserveMode.FEEDBACK)
    assert np.abs(matrix[:, 1]).max() > 0.0


def test_feedback_is_invisible_at_short_horizons():
    """Honest limitation: the nonlinearity is undetectable over 18 months.

    The mechanistic layer only earns its place if accelerating decline is
    distinguishable from linear decline. Over routine follow-up the difference
    is far below the measurement noise, so this must not be sold as a testable
    claim at that horizon.
    """
    times = np.array([0.0, 0.5, 1.0, 1.5])
    linear = observe(times, Parameters(gamma=0.0), None,
                     ReserveMode.FEEDBACK)["fvc_pct"]
    accelerating = observe(times, Parameters(gamma=1.0), None,
                           ReserveMode.FEEDBACK)["fvc_pct"]
    gap = np.abs(linear - accelerating).max()
    assert gap < DEFAULT_NOISE_SD["fvc_pct"] / 3.0


# --------------------------------------------------------------- practical
def test_crlb_matches_monte_carlo_for_well_behaved_parameters():
    """The Fisher bound must be validated, not trusted."""
    params = Parameters()
    schedule = routine_followup(8, treatment_start=0.5)
    result = monte_carlo_check(params, CORE, schedule, ReserveMode.NONE,
                               n_replicates=200, seed=1)
    for name in ("fvc0", "dlco0", "r_i", "beta"):
        ratio = result["robust_sd"][name] / result["crlb"][name]
        assert 0.7 < ratio < 1.4, f"{name}: {ratio}"


def test_kappa_has_a_global_failure_the_local_bound_cannot_see():
    """kappa is only identifiable while burden actually accrues.

    kappa multiplies B, so in replicates whose fitted trajectory is nearly flat
    it is unconstrained and runs to arbitrary values at no cost in likelihood.
    The CRLB linearises at the truth and reports a reasonable number; the
    classical empirical standard deviation is orders of magnitude larger while
    the robust one still agrees with the bound. That gap is the diagnostic.
    """
    result = monte_carlo_check(Parameters(), CORE, routine_followup(8,
                               treatment_start=0.5), ReserveMode.NONE,
                               n_replicates=200, seed=1)
    assert result["empirical_sd"]["kappa"] > 20 * result["crlb"]["kappa"]
    assert result["robust_sd"]["kappa"] < 3.0 * result["crlb"]["kappa"]
    assert 0.0 < result["collapse_fraction"] < 0.15


def test_precision_improves_monotonically_with_visits():
    params = Parameters()
    errors = []
    for n_visits in (4, 6, 8, 12):
        report = analyze(params, CORE,
                         routine_followup(n_visits, treatment_start=0.5),
                         ReserveMode.NONE)
        errors.append(report.standard_errors["r_i"])
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))


def test_four_visits_cannot_pin_the_personal_decline_rate():
    """The headline practical result, asserted so a regression would show."""
    report = analyze(Parameters(), CORE, routine_followup(4, treatment_start=0.5),
                     ReserveMode.NONE)
    assert report.is_structurally_identifiable
    # SE exceeds the true value: cannot separate a progressor from a stable
    # patient on four visits when beta must also be estimated.
    assert report.standard_errors["r_i"] > Parameters().r_i


def test_free_beta_makes_the_decline_rate_unreachable_at_any_horizon():
    """No follow-up length rescues r_i while beta is also being estimated.

    beta is informed only by the pre-treatment window, whose length is fixed by
    when therapy started. Visits after the switch add information about
    r_i - beta but almost none about their separation, so their correlation
    tends to 1 and SE(r_i) plateaus. Following the patient longer does not help.
    """
    found = visits_required(Parameters(), CORE, {"r_i": 1.0},
                            treatment_start=0.5, max_visits=30)
    assert found["r_i"] is None

    long_run = analyze(Parameters(), CORE,
                       routine_followup(30, treatment_start=0.5),
                       ReserveMode.NONE)
    assert long_run.standard_errors["r_i"] > 5.0
    assert abs(long_run.correlations[2, 4]) > 0.99


def test_fixing_beta_by_prior_makes_the_design_feasible():
    """The practical payoff of pinning beta from the antifibrotic trials."""
    without_beta = ["fvc0", "dlco0", "r_i", "kappa"]
    found = visits_required(Parameters(), without_beta, {"r_i": 1.0},
                            treatment_start=0.5, max_visits=30)
    assert found["r_i"] is not None
    assert 4 < found["r_i"] <= 12


def test_pretreatment_window_is_the_binding_constraint():
    """With total follow-up held fixed, a longer untreated window is what helps."""
    params = Parameters()
    errors = [
        analyze(params, CORE, routine_followup(8, treatment_start=start),
                ReserveMode.NONE).standard_errors["r_i"]
        for start in (0.25, 0.5, 1.0, 1.5, 2.0)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    assert errors[0] / errors[-1] > 5.0


# -------------------------------------------------------------- profiling
def test_noise_free_profile_is_flat_along_a_null_direction():
    """beta with no treatment change: every value fits the data equally well."""
    params = Parameters()
    schedule = routine_followup(4, treatment_start=None)
    grid = np.linspace(0.0, 5.0, 9)
    profile = profile_likelihood(params, CORE, schedule, "beta", grid,
                                 ReserveMode.NONE)
    assert np.ptp(profile) < 1e-6


def test_noise_free_profile_is_sharp_for_an_identifiable_parameter():
    params = Parameters()
    schedule = routine_followup(8, treatment_start=0.5)
    grid = np.linspace(params.r_i - 2.0, params.r_i + 2.0, 9)
    profile = profile_likelihood(params, CORE, schedule, "r_i", grid,
                                 ReserveMode.NONE)

    # Minimum sits at the truth and the fit there is exact (noise-free data).
    assert profile[np.argmin(profile)] == pytest.approx(0.0, abs=1e-6)
    assert grid[int(np.argmin(profile))] == pytest.approx(params.r_i, abs=0.6)

    # Curvature must reproduce the Fisher bound: for Gaussian noise the profile
    # rises as 0.5*(delta/SE)^2. This cross-validates the two machineries
    # against each other - a profile that disagreed with the CRLB would mean one
    # of them is wrong.
    report = analyze(params, CORE, schedule, ReserveMode.NONE)
    expected = 0.5 * ((grid - params.r_i) / report.standard_errors["r_i"]) ** 2
    assert np.allclose(profile, expected, atol=5e-3)

    # And the honest reading of that curvature: the profile is nearly flat over
    # a range wider than the parameter itself.
    assert profile.max() < 0.5


# ---------------------------------------------------------------- plumbing
def test_missing_dlco_reduces_precision_without_breaking_rank():
    params = Parameters()
    full = routine_followup(8, treatment_start=0.5)
    sparse = routine_followup(8, treatment_start=0.5, dlco_missing_rate=0.6,
                              seed=3)
    assert sparse.n_observations < full.n_observations
    assert analyze(params, CORE, sparse, ReserveMode.NONE).standard_errors[
        "kappa"
    ] > analyze(params, CORE, full, ReserveMode.NONE).standard_errors["kappa"]


def test_schedule_rejects_a_non_zero_baseline():
    with pytest.raises(ValueError, match="start at 0"):
        VisitSchedule(times=np.array([0.5, 1.0]))


def test_report_names_the_null_direction():
    report = analyze(Parameters(), CORE, routine_followup(4), ReserveMode.NONE)
    text = render(report)
    assert "NOT identifiable" in text
    assert "beta" in text
    assert "informative prior" in text


def test_report_flags_a_useless_precision_bound():
    report = analyze(Parameters(), CORE, routine_followup(4, treatment_start=0.5),
                     ReserveMode.NONE)
    text = render(report)
    assert "Cramer-Rao" in text
    assert "SE(r_i)" in text
