"""Tests for line integral convolution of a nematic director field.

A LIC that mishandles the modulo-pi branch still produces a plausible-looking
picture, and one that aliases against the pixel grid produces a beautiful one
whose orientation is systematically wrong at oblique angles. Neither failure is
visible by eye, so the tests measure the orientation recovered *from the render*
and compare it against the field that went in.
"""

from __future__ import annotations

import numpy as np
import pytest

from lung_nematic.lic import (
    lic_rgb,
    line_integral_convolution,
    recovered_orientation,
)

SIZE = 300
YY, XX = np.mgrid[0:SIZE, 0:SIZE]
INNER = np.zeros((SIZE, SIZE), dtype=bool)
INNER[50:-50, 50:-50] = True


def orientation_error_deg(theta, **kwargs):
    texture = line_integral_convolution(theta, **kwargs)
    recovered = recovered_orientation(texture, sigma=3.0)
    difference = (recovered - theta + np.pi / 2) % np.pi - np.pi / 2
    return float(np.degrees(np.abs(difference[INNER])).mean())


@pytest.mark.parametrize("angle", [0.0, 0.4, np.pi / 4, 1.1, np.pi / 2, 1.9])
def test_streaks_follow_the_field_at_every_angle(angle):
    """No axial bias.

    With white noise the render was off by 1.9 degrees at 0 and 90 but 13 to 17
    at oblique angles - the signature of aliasing against the bilinear sampler.
    Band-limiting the seed noise brought every angle under 4.
    """
    assert orientation_error_deg(np.full((SIZE, SIZE), angle)) < 5.0


def test_white_noise_would_reintroduce_the_axial_bias():
    """Pins why noise_sigma is not cosmetic."""
    rng = np.random.default_rng(0)
    white = rng.random((SIZE, SIZE))
    oblique = np.full((SIZE, SIZE), 0.4)
    biased = orientation_error_deg(oblique, noise=white)
    clean = orientation_error_deg(oblique)
    assert biased > 2 * clean


def test_representation_wrap_does_not_change_the_render():
    """theta and theta + pi are the same director.

    Stepping along (cos theta, sin theta) naively reverses wherever the array
    wraps, folding the streamline back on itself exactly where the texture
    matters most.
    """
    plain = np.full((SIZE, SIZE), 0.4)
    wrapped = np.where(XX < SIZE // 2, 0.4, 0.4 + np.pi)
    assert orientation_error_deg(wrapped) == pytest.approx(
        orientation_error_deg(plain), abs=1.5
    )


@pytest.mark.parametrize("charge", [0.5, -0.5, 1.0, -1.0])
def test_defect_fields_render_faithfully(charge):
    theta = charge * np.arctan2(YY - SIZE / 2, XX - SIZE / 2)
    assert orientation_error_deg(theta) < 6.0


def test_output_is_bounded_and_masked():
    mask = ((XX - 150) ** 2 + (YY - 150) ** 2) < 100**2
    texture = line_integral_convolution(np.full((SIZE, SIZE), 0.4), mask)
    inside = texture[mask]
    assert np.isfinite(inside).all()
    assert inside.min() >= 0.0 and inside.max() <= 1.0
    # Outside is NaN, not zero, so a caller can composite without a black rim
    # appearing where there is simply no data.
    assert np.isnan(texture[~mask]).all()


def test_rgb_render_has_the_expected_shape_and_background():
    mask = ((XX - 150) ** 2 + (YY - 150) ** 2) < 80**2
    rgb = lic_rgb(np.full((SIZE, SIZE), 0.4), None, mask,
                  background=(18, 18, 22))
    assert rgb.shape == (SIZE, SIZE, 3)
    assert rgb.dtype == np.uint8
    assert (rgb[~mask] == np.array([18, 18, 22])).all()


def test_order_modulation_dims_disordered_regions():
    """Disordered regions must not read as confidently as ordered ones."""
    theta = np.full((SIZE, SIZE), 0.4)
    order = np.where(XX < SIZE // 2, 0.1, 1.0)
    rgb = lic_rgb(theta, order, np.ones((SIZE, SIZE), dtype=bool))
    assert rgb[:, : SIZE // 2].mean() < rgb[:, SIZE // 2:].mean()


def test_measurement_tool_is_accurate_on_synthetic_stripes():
    """Guards the test instrument itself.

    If recovered_orientation were biased, every test above would be measuring
    the measuring stick. On pure stripes it is accurate to under 2 degrees.
    """
    for angle in (0.0, 0.4, np.pi / 4, np.pi / 2, 1.9):
        stripes = np.cos(
            0.8 * (XX * np.cos(angle + np.pi / 2) + YY * np.sin(angle + np.pi / 2))
        )
        recovered = recovered_orientation(stripes, sigma=3.0)
        difference = (recovered - angle + np.pi / 2) % np.pi - np.pi / 2
        assert np.degrees(np.abs(difference[INNER])).mean() < 2.0
