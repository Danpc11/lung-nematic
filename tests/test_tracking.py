"""Tests for time-lapse defect tracking.

The tests that matter are the ones asserting the tracker recovers a *known*
kinematics from synthetic frames. A tracker that silently mislinks produces a
plausible speed table from nonsense, so correctness has to be pinned against
ground truth rather than against "it ran".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lung_nematic.tracking import (
    calibrate,
    defect_kinetics,
    estimate_drift,
    motility_by_charge,
    subtract_drift,
    track_defects,
    track_summary,
)


def _series(n_frames=25, plus_velocity=3.0, jitter=0.3, drift=(0.0, 0.0),
            n_plus=6, n_minus=6, seed=0):
    """+1/2 defects propel ballistically, -1/2 defects only diffuse.

    Each +1/2 gets its OWN propulsion direction. A self-propelled defect moves
    along its own polar axis, and those axes are not aligned with each other, so
    a series where every +1/2 travels the same way is a collective flow rather
    than self-propulsion - and it breaks median-based drift estimation, since
    half the field then shares a common velocity that is not drift.
    """
    rng = np.random.default_rng(seed)
    plus = rng.uniform(20, 280, (n_plus, 2))
    minus = rng.uniform(20, 280, (n_minus, 2))
    angles = rng.uniform(0, 2 * np.pi, n_plus)
    heading = np.column_stack([np.cos(angles), np.sin(angles)])

    frames = []
    for t in range(n_frames):
        shift = np.array([drift[0] * t, drift[1] * t])
        p = plus + plus_velocity * t * heading + shift
        p = p + rng.normal(0, jitter, plus.shape)
        m = minus + shift + rng.normal(0, jitter, minus.shape)
        frames.append(
            pd.DataFrame(
                {
                    "x_px": np.r_[p[:, 0], m[:, 0]],
                    "y_px": np.r_[p[:, 1], m[:, 1]],
                    "charge": [0.5] * n_plus + [-0.5] * n_minus,
                }
            )
        )
    return frames


# ------------------------------------------------------------------ linking
def test_recovers_known_track_count():
    tracks = track_defects(_series(), max_displacement_px=10)
    assert tracks.track_id.nunique() == 12
    assert len(tracks) == 12 * 25


def test_recovers_the_imposed_velocity():
    tracks = track_defects(_series(plus_velocity=3.0), max_displacement_px=10)
    speeds = motility_by_charge(tracks).set_index("charge")
    assert speeds.loc[0.5, "mean_speed_px_per_frame"] == pytest.approx(
        3.0, abs=0.2
    )


def test_positive_defects_outrun_negative_ones():
    """The activity signature: +1/2 self-propels, -1/2 does not."""
    tracks = track_defects(_series(), max_displacement_px=10)
    speeds = motility_by_charge(tracks).set_index("charge")
    assert (
        speeds.loc[0.5, "mean_speed_px_per_frame"]
        > 4 * speeds.loc[-0.5, "mean_speed_px_per_frame"]
    )


def test_straightness_separates_ballistic_from_diffusive():
    summary = track_summary(track_defects(_series(), max_displacement_px=10))
    ballistic = summary[summary.charge > 0].straightness
    diffusive = summary[summary.charge < 0].straightness
    assert ballistic.min() > 0.9
    assert diffusive.max() < 0.3


def test_charge_is_conserved_within_a_track():
    """A defect that changes charge is two events, not one moving defect."""
    tracks = track_defects(_series(), max_displacement_px=10)
    for _, group in tracks.groupby("track_id"):
        assert group.charge.nunique() == 1


def test_charge_classes_never_cross_link():
    """A +1/2 must not adopt a nearby -1/2's identity even when co-located."""
    frames = [
        pd.DataFrame({"x_px": [10.0, 10.5], "y_px": [10.0, 10.0],
                      "charge": [0.5, -0.5]}),
        pd.DataFrame({"x_px": [10.5, 10.0], "y_px": [10.0, 10.0],
                      "charge": [-0.5, 0.5]}),
    ]
    tracks = track_defects(frames, max_displacement_px=5)
    assert tracks.track_id.nunique() == 2
    for _, group in tracks.groupby("track_id"):
        assert group.charge.nunique() == 1


def test_gate_prevents_linking_across_the_field():
    """Without a gate the solver links an annihilated defect to a stranger.

    The Hungarian algorithm returns a globally optimal matching whether or not
    any pair is physically plausible, so the displacement gate is what stops a
    death-plus-birth from being reported as one very fast defect.
    """
    frames = [
        pd.DataFrame({"x_px": [10.0], "y_px": [10.0], "charge": [0.5]}),
        pd.DataFrame({"x_px": [900.0], "y_px": [900.0], "charge": [0.5]}),
    ]
    linked = track_defects(frames, max_displacement_px=20)
    assert linked.track_id.nunique() == 2, "gate failed to break the link"

    permissive = track_defects(frames, max_displacement_px=2000)
    assert permissive.track_id.nunique() == 1


def test_rejects_a_non_positive_gate():
    with pytest.raises(ValueError, match="must be positive"):
        track_defects(_series(), max_displacement_px=0)


def test_rejects_frames_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        track_defects([pd.DataFrame({"x_px": [1.0], "y_px": [1.0]})],
                      max_displacement_px=5)


def test_handles_empty_frames_without_crashing():
    frames = [
        pd.DataFrame({"x_px": [10.0], "y_px": [10.0], "charge": [0.5]}),
        pd.DataFrame(columns=["x_px", "y_px", "charge"], dtype=float),
        pd.DataFrame({"x_px": [12.0], "y_px": [10.0], "charge": [0.5]}),
    ]
    tracks = track_defects(frames, max_displacement_px=5)
    # No gap closing, so the defect either side of the empty frame is two
    # tracks. Documented behaviour, asserted so a future gap-closer is a
    # deliberate change rather than an accident.
    assert tracks.track_id.nunique() == 2


# -------------------------------------------------------------------- drift
def _contrast(tracks):
    speeds = motility_by_charge(tracks).set_index("charge")
    return (speeds.loc[0.5, "mean_speed_px_per_frame"]
            - speeds.loc[-0.5, "mean_speed_px_per_frame"])


def test_drift_compresses_the_contrast_rather_than_offsetting_it():
    """Drift is a common velocity VECTOR, and speed is a magnitude.

    It is tempting to assume drift cancels out of speed(+1/2) - speed(-1/2)
    because it affects both classes. It does not: propulsion directions are
    independent, so |v_prop + v_drift| != |v_prop| + |v_drift|, and the contrast
    shrinks. Asserting the compression keeps anyone from reading an
    unregistered contrast as an activity estimate.
    """
    clean = track_defects(_series(drift=(0.0, 0.0), n_plus=10, n_minus=10),
                          max_displacement_px=15)
    drifted = track_defects(_series(drift=(0.0, 2.0), n_plus=10, n_minus=10),
                            max_displacement_px=15)
    assert _contrast(drifted) < 0.75 * _contrast(clean)


def test_removing_drift_restores_the_contrast():
    drifted = track_defects(_series(drift=(0.0, 2.0), n_plus=10, n_minus=10),
                            max_displacement_px=15)
    clean = track_defects(_series(drift=(0.0, 0.0), n_plus=10, n_minus=10),
                          max_displacement_px=15)
    corrected = subtract_drift(drifted, estimate_drift(drifted))
    assert _contrast(corrected) == pytest.approx(_contrast(clean), rel=0.25)


def test_drift_is_estimated_and_removable():
    tracks = track_defects(
        _series(drift=(1.5, -0.8), n_plus=14, n_minus=14),
        max_displacement_px=15,
    )
    drift = estimate_drift(tracks)
    assert drift.drift_x_px.median() == pytest.approx(1.5, abs=0.6)
    assert drift.drift_y_px.median() == pytest.approx(-0.8, abs=0.6)

    corrected = motility_by_charge(
        subtract_drift(tracks, drift)
    ).set_index("charge")
    assert corrected.loc[-0.5, "mean_speed_px_per_frame"] < 1.0


# ----------------------------------------------------------------- kinetics
def test_kinetics_counts_births_and_deaths():
    frames = [
        pd.DataFrame({"x_px": [10.0, 50.0], "y_px": [10.0, 50.0],
                      "charge": [0.5, -0.5]}),
        pd.DataFrame({"x_px": [11.0], "y_px": [10.0], "charge": [0.5]}),
        pd.DataFrame({"x_px": [12.0, 300.0], "y_px": [10.0, 300.0],
                      "charge": [0.5, 0.5]}),
    ]
    table = defect_kinetics(track_defects(frames, max_displacement_px=5))
    assert table.n_defects.tolist() == [2, 1, 2]
    assert table.loc[1, "deaths"] == 1
    assert table.loc[2, "births"] == 1


def test_kinetics_reports_density_when_area_is_given():
    tracks = track_defects(_series(n_frames=3), max_displacement_px=10)
    table = defect_kinetics(tracks, {0: 0.5, 1: 0.5, 2: 0.5})
    assert table.defect_density_mm2.iloc[0] == pytest.approx(12 / 0.5)


# -------------------------------------------------------------- calibration
def test_calibration_converts_pixels_and_frames():
    tracks = track_defects(_series(plus_velocity=3.0), max_displacement_px=10)
    speeds = calibrate(motility_by_charge(tracks), microns_per_pixel=0.5,
                       seconds_per_frame=600.0)
    row = speeds.set_index("charge").loc[0.5]
    assert row["mean_speed_um_per_s"] == pytest.approx(
        row["mean_speed_px_per_frame"] * 0.5 / 600.0
    )


def test_calibration_adds_a_time_column():
    tracks = track_defects(_series(n_frames=4), max_displacement_px=10)
    table = calibrate(defect_kinetics(tracks), 0.5, 600.0)
    assert table.time_s.tolist() == [0.0, 600.0, 1200.0, 1800.0]


@pytest.mark.parametrize("mpp,spf", [(0, 600.0), (0.5, 0), (-1, 600.0)])
def test_calibration_rejects_non_positive_scales(mpp, spf):
    with pytest.raises(ValueError, match="must both be positive"):
        calibrate(pd.DataFrame({"frame": [0]}), mpp, spf)
