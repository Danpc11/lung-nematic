"""Tests for the adaptive integration radius."""

from __future__ import annotations

import numpy as np
import pytest

from lung_nematic.adaptive_radius import (
    adaptive_radius_map,
    cell_size_from_coherence,
    cell_size_from_nuclei,
    summarise_radius_map,
)


def test_dense_nuclei_give_smaller_radius_than_sparse():
    """Tight nuclei (epithelium) must get a smaller radius than sparse (stroma)."""
    shape = (400, 400)
    # left half: dense grid of nuclei; right half: sparse
    dense = np.array([(x, y) for x in range(20, 180, 8) for y in range(20, 380, 8)])
    sparse = np.array([(x, y) for x in range(220, 380, 40) for y in range(20, 380, 40)])
    centroids = np.vstack([dense, sparse]).astype(float)

    size = cell_size_from_nuclei(centroids, shape, smoothing_px=40.0)
    radius = adaptive_radius_map(size, cells_per_radius=4.5)

    left = radius[:, 60:120].mean()
    right = radius[:, 280:340].mean()
    assert left < right, (
        f"dense region should get a smaller radius; left {left:.1f} right {right:.1f}"
    )


def test_radius_encloses_requested_cells():
    """Radius should be about cells_per_radius local cell lengths."""
    shape = (200, 200)
    centroids = np.array([(x, y) for x in range(20, 180, 20)
                          for y in range(20, 180, 20)], dtype=float)
    size = cell_size_from_nuclei(centroids, shape, smoothing_px=30.0,
                                 min_size_px=5.0, max_size_px=80.0)
    radius = adaptive_radius_map(size, cells_per_radius=4.5,
                                 min_radius_px=6.0, max_radius_px=200.0)
    typical_size = np.median(size)
    typical_radius = np.median(radius)
    assert typical_radius == pytest.approx(4.5 * typical_size, rel=0.3)


def test_radius_is_clipped_to_band():
    shape = (100, 100)
    centroids = np.array([(50.0, 50.0), (52.0, 52.0)])
    size = cell_size_from_nuclei(centroids, shape)
    radius = adaptive_radius_map(size, min_radius_px=10.0, max_radius_px=40.0)
    assert radius.min() >= 10.0
    assert radius.max() <= 40.0


def test_coherence_size_is_bounded():
    theta = np.random.default_rng(0).uniform(0, np.pi, (100, 100))
    field = {"theta": theta, "order": np.ones((100, 100))}
    size = cell_size_from_coherence(field, min_size_px=5.0, max_size_px=50.0)
    assert size.min() >= 5.0
    assert size.max() <= 50.0


def test_summary_reports_spread():
    radius = np.linspace(10, 50, 100).reshape(10, 10)
    summary = summarise_radius_map(radius)
    assert summary["radius_min_px"] == pytest.approx(10.0)
    assert summary["radius_max_px"] == pytest.approx(50.0)
    assert summary["radius_iqr_px"] > 0


def test_no_nuclei_falls_back_gracefully():
    """With too few nuclei the map is the mid-band, not a crash or all-NaN."""
    size = cell_size_from_nuclei(np.zeros((0, 2)), (100, 100),
                                 min_size_px=8.0, max_size_px=60.0)
    assert np.isfinite(size).all()
    assert 8.0 <= np.median(size) <= 60.0
