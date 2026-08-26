"""Collective-flow tests use known translations and analytic directions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from lung_nematic.flow import collective_flow, flow_summary


def test_collective_flow_recovers_known_image_translation():
    rng = np.random.default_rng(4)
    texture = gaussian_filter(rng.normal(size=(96, 128)), 2.0)
    texture = (255 * (texture - texture.min())
               / (texture.max() - texture.min())).astype(np.uint8)
    first = np.repeat(texture[..., None], 3, axis=2)
    second_gray = shift(texture.astype(float), (2, 3), order=1, mode="nearest")
    second = np.repeat(second_gray[..., None], 3, axis=2).astype(np.uint8)
    result = collective_flow(
        first, second, downsample=2, smoothing_px=2,
        director_sigma_px=8, subtract_median_translation=False,
    )
    mask = result["mask"]
    assert np.median(result["u_px_per_frame"][mask]) == pytest.approx(3, abs=0.8)
    assert np.median(result["v_px_per_frame"][mask]) == pytest.approx(2, abs=0.8)


def test_flow_summary_reports_analytic_parallel_alignment():
    result = {
        "mask": np.ones((3, 3), dtype=bool),
        "speed_px_per_frame": np.full((3, 3), 2.0),
        "flow_director_alignment": np.ones((3, 3)),
        "drift_u_px_per_frame": 0.0,
        "drift_v_px_per_frame": 0.0,
    }
    summary = flow_summary(result, microns_per_pixel=0.5,
                           seconds_per_frame=60)
    assert summary["mean_speed_um_per_min"] == 1.0
    assert summary["mean_flow_director_alignment"] == 1.0
    assert summary["fraction_parallel"] == 1.0
    assert summary["fraction_perpendicular"] == 0.0
