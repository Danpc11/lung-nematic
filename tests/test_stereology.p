"""Tests for 3D-to-2D defect stereology.

Every case here has an analytically known answer. A stereology routine that is
subtly wrong still returns plausible densities - that is exactly its danger,
since the whole point is to correct a bias nobody can see by eye - so the tests
pin numbers against volumes whose line orientation was imposed, not inspected.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulations.stereology import (
    ISOTROPIC_FACTOR,
    anisotropy_factor,
    count_section_defects,
    director_volume_from_points,
    estimate_line_density,
    section_plane,
    section_sweep,
)

SIZE = 40
EXTENT = (300.0, 300.0, 300.0)
N_RODS = 45000


def _volume(kind: str, seed: int = 0):
    """Build a director volume whose disclination geometry is imposed."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, EXTENT[0], (N_RODS, 3))

    if kind == "lines_along_z":
        # Half-integer disclinations running parallel to z: the director turns
        # in the xy plane and does not depend on z at all.
        centres_x, centres_y = rng.uniform(60, 240, (2, 6))
        angle = np.zeros(N_RODS)
        for index in range(6):
            charge = 0.5 if index % 2 == 0 else -0.5
            angle = angle + charge * np.arctan2(
                positions[:, 1] - centres_y[index],
                positions[:, 0] - centres_x[index],
            )
        directors = np.stack(
            [np.cos(angle), np.sin(angle), np.zeros(N_RODS)], axis=1
        )
    elif kind == "isotropic":
        directors = rng.normal(size=(N_RODS, 3))
    elif kind == "uniform":
        directors = np.tile(np.array([1.0, 0.0, 0.0]), (N_RODS, 1))
    else:  # pragma: no cover - guard against a typo in a test
        raise ValueError(kind)

    return director_volume_from_points(
        positions, directors, (SIZE, SIZE, SIZE), EXTENT, sigma_voxels=1.2
    )


# ------------------------------------------------------------- construction
def test_volume_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        director_volume_from_points(
            np.zeros((5, 3)), np.zeros((4, 3)), (8, 8, 8), EXTENT
        )


def test_antiparallel_rods_do_not_cancel():
    """Directors are head-tail symmetric; averaging them as vectors is wrong.

    A voxel holding equal numbers of +x and -x rods is perfectly ordered along
    x. Vector averaging would report no orientation at all.
    """
    positions = np.full((200, 3), 150.0)
    directors = np.tile(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), (100, 1))
    volume = director_volume_from_points(
        positions, directors, (8, 8, 8), EXTENT, sigma_voxels=0.0
    )
    filled = volume["density"] > 0
    values = np.linalg.eigvalsh(volume["tensor"][filled])
    # Leading eigenvalue of <n n> is 1 for perfect alignment.
    assert values[..., -1].max() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------- isotropy
def test_isotropic_lines_give_angle_independent_density():
    """The condition under which N_A = L_V / 2 may be used at all."""
    sweep = section_sweep(_volume("isotropic"), EXTENT,
                          polar_angles_deg=(0, 30, 60, 90), n_azimuths=3,
                          n_samples=110)
    assert anisotropy_factor(sweep)["ratio"] < 1.35


def test_lines_parallel_to_a_section_are_not_cut():
    """The failure a single histological density cannot reveal.

    Disclinations running along z are cut by a plane normal to z and missed
    entirely by a plane containing z. Same tissue, same defects, and a count
    that goes from tens per mm^2 to zero depending only on the cutting angle.
    """
    sweep = section_sweep(_volume("lines_along_z"), EXTENT,
                          polar_angles_deg=(0, 30, 60, 90), n_azimuths=3,
                          n_samples=110)
    by_angle = sweep.groupby("polar_deg").areal_density_mm2.mean()
    assert by_angle.loc[0.0] > 10.0
    assert by_angle.loc[90.0] == pytest.approx(0.0, abs=1e-9)


def test_uniform_director_has_no_defects_at_any_angle():
    sweep = section_sweep(_volume("uniform"), EXTENT,
                          polar_angles_deg=(0, 45, 90), n_azimuths=2,
                          n_samples=90)
    assert sweep.n_defects.sum() == 0


# ------------------------------------------------------------- the filters
def test_density_gate_is_absolute_not_a_quantile():
    """A relative gate discards a fixed share of every section.

    The first version of this module used a 0.4 quantile. On a uniformly filled
    volume that removed 58% of the plane - and every defect with it - reporting
    zero where four were present. An absolute threshold relative to the mean
    keeps a full section full.
    """
    volume = _volume("lines_along_z")
    section = section_plane(volume, 0.0, 0.0, n_samples=110)
    counts = count_section_defects(section, min_density_ratio=0.3)
    assert counts["valid_fraction"] > 0.9
    assert counts["n_total"] > 0


def test_out_of_plane_regions_are_excluded():
    """Where the director is nearly normal to the cut, its apparent in-plane
    angle is noise, and counting it would manufacture obliquity-dependent
    defects - the very artefact this module measures."""
    volume = _volume("lines_along_z")
    section = section_plane(volume, 90.0, 0.0, n_samples=110)
    permissive = count_section_defects(section, min_in_plane=0.0)
    strict = count_section_defects(section, min_in_plane=0.5)
    assert strict["valid_fraction"] <= permissive["valid_fraction"]


def test_section_stays_inside_the_volume_at_every_angle():
    """Sampling past the edge would read zero-padding as tissue."""
    volume = _volume("isotropic")
    for polar in (0, 45, 90):
        section = section_plane(volume, polar, 37.0, n_samples=110)
        assert np.isfinite(section["theta"]).all()
        assert section["density"].min() > 0


# ------------------------------------------------------------- line density
def test_line_density_uses_the_isotropic_factor():
    sweep = section_sweep(_volume("isotropic"), EXTENT,
                          polar_angles_deg=(0, 45, 90), n_azimuths=2,
                          n_samples=100)
    estimate = estimate_line_density(sweep)
    assert estimate["line_density_mm_per_mm3"] == pytest.approx(
        estimate["mean_areal_density_mm2"] / ISOTROPIC_FACTOR
    )
    assert estimate["isotropy_assumed"] is True


def test_solid_angle_weighting_downweights_the_poles():
    """Equal steps in polar angle do not sample the sphere uniformly.

    A plain mean over the swept angles over-weights the poles, where the
    solid-angle element vanishes. Weighting by sin(theta) is what makes the
    estimate an average over the sphere rather than over the list of angles.
    """
    import pandas as pd

    sweep = pd.DataFrame({
        "polar_deg": [0.0, 90.0],
        "areal_density_mm2": [1000.0, 10.0],
    })
    estimate = estimate_line_density(sweep)
    # sin(0) = 0, so the pole carries no weight and the equator dominates.
    assert estimate["mean_areal_density_mm2"] == pytest.approx(10.0)


def test_anisotropy_factor_handles_an_all_zero_sweep():
    import pandas as pd

    sweep = pd.DataFrame({"polar_deg": [0.0, 90.0],
                          "areal_density_mm2": [0.0, 0.0]})
    assert np.isnan(anisotropy_factor(sweep)["ratio"])
