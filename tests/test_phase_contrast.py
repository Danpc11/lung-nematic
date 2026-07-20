"""Tests for the phase-contrast NHLF analysis module."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lung_nematic.config import load_default_config
from lung_nematic.phase_contrast import (
    StiffnessOrderCalibration,
    analyze_phase_contrast,
    build_calibration,
    cell_texture_mask,
    orientation_correlation_length,
    phase_contrast_field,
)


def _striped_cells(shape=(400, 400), spacing=12, angle=0.0, noise=0.0, seed=0):
    """Synthetic aligned-cell texture: parallel bright ridges at a given angle."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    coordinate = xx * np.cos(angle) + yy * np.sin(angle)
    image = 0.5 + 0.5 * np.sin(2 * np.pi * coordinate / spacing)
    if noise:
        image = image + rng.normal(0, noise, shape)
    return np.stack([image] * 3, axis=-1)


@pytest.fixture
def config():
    return replace(
        load_default_config(),
        sigmas_px=(18.0, 25.0, 32.0),
        density_quantile=0.30,
        min_scales_for_persistence=2,
        defect_grid_step_px=12,
        min_edge_distance_px=20,
    )


def test_aligned_texture_is_highly_ordered(config):
    image = _striped_cells(angle=0.4, noise=0.02)
    result = analyze_phase_contrast(image, config)
    assert result["local_S_median"] > 0.6, "aligned stripes should be well-ordered"


def test_recovered_orientation_matches_input():
    for angle in (0.0, np.pi / 4, np.pi / 3):
        image = _striped_cells(angle=angle)
        field = phase_contrast_field(image, sigma_px=20.0)
        # director is perpendicular to the intensity gradient of the stripes
        measured = np.median(field["theta"][field["order"] > 0.5])
        expected = (angle + np.pi / 2) % np.pi
        difference = min(abs(measured - expected), np.pi - abs(measured - expected))
        assert difference < 0.2, f"angle {angle}: got {measured}, expected {expected}"


def test_coverage_mask_finds_textured_region():
    image = np.full((300, 300, 3), 0.5)
    image[100:200, 100:200] = _striped_cells((100, 100), noise=0.1)
    mask = cell_texture_mask(image, quantile=0.5)
    inside = mask[120:180, 120:180].mean()
    outside = mask[:80, :80].mean()
    assert inside > outside, "textured region should score higher coverage"


def test_correlation_length_rises_with_order():
    aligned = phase_contrast_field(_striped_cells(spacing=12, noise=0.01), 20.0)
    noisy = phase_contrast_field(_striped_cells(spacing=12, noise=0.4), 20.0)
    length_aligned = orientation_correlation_length(aligned)
    length_noisy = orientation_correlation_length(noisy)
    # a cleaner field stays correlated at least as far (both may hit the ceiling,
    # reported as a negative sentinel, so compare magnitudes)
    assert abs(length_aligned) >= abs(length_noisy) * 0.8


# ------------------------------------------------------ stiffness calibration
def test_calibration_is_monotonic_and_invertible():
    calibration = build_calibration(
        stiffness_kPa=[1.0, 5.0, 23.0, 50.0],
        order_values=[0.30, 0.45, 0.62, 0.70],
    )
    assert calibration.r_squared > 0.9
    # forward prediction increases with stiffness
    assert calibration.predict_order(50.0) > calibration.predict_order(1.0)


def test_inverse_returns_a_range_not_a_point():
    calibration = build_calibration(
        stiffness_kPa=[5.0, 23.0],
        order_values=[0.45, 0.62],
        order_std=[0.05, 0.05],
    )
    estimate = calibration.estimate_stiffness(0.53, order_uncertainty=0.05)
    assert estimate["low_kPa"] < estimate["stiffness_kPa"] < estimate["high_kPa"], (
        "an inferred stiffness must carry an interval, never be a bare point"
    )


def test_inverse_flags_extrapolation():
    calibration = build_calibration(
        stiffness_kPa=[5.0, 23.0],
        order_values=[0.45, 0.62],
    )
    inside = calibration.estimate_stiffness(0.53)
    outside = calibration.estimate_stiffness(0.20)
    assert inside["in_calibrated_range"]
    assert not outside["in_calibrated_range"], (
        "an order below the calibrated span must be flagged as extrapolation"
    )


def test_calibration_needs_two_distinct_stiffnesses():
    with pytest.raises(ValueError):
        build_calibration([5.0, 5.0], [0.4, 0.4])
