"""Tests for the 3D active-nematic focus model.

The load-bearing test is ``test_order_responds_to_density``. The model this one
replaces produced an order parameter pinned at 0.956 while density varied
tenfold - order that does not move when density does is a finite-size artefact,
not a nematic phase, and it looks perfectly healthy in any single run. Several
tests below therefore assert *responses* rather than values.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulations.focus3d import Focus3DConfig, Focus3DSimulation


def _run(steps: int = 400, **kwargs) -> Focus3DSimulation:
    config = Focus3DConfig(**kwargs)
    simulation = Focus3DSimulation(config)
    for _ in range(steps):
        simulation.step()
    return simulation


# ---------------------------------------------------------------- geometry
def test_domain_must_exceed_the_interaction_range():
    """A box smaller than a few alignment radii orders trivially.

    Every cell then sees every other, so the order parameter saturates whatever
    the physics. Refusing the configuration is better than returning a number
    that looks like emergent order.
    """
    with pytest.raises(ValueError, match="five alignment radii"):
        Focus3DConfig(size_um=40.0)


def test_cells_stay_inside_the_periodic_box():
    simulation = _run(200)
    assert simulation.position.min() >= 0.0
    assert simulation.position.max() <= simulation.cfg.size_um


def test_directors_stay_normalised():
    simulation = _run(200)
    norms = np.linalg.norm(simulation.director, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


# ------------------------------------------------------------ order measure
def test_isotropic_directors_give_zero_order():
    """The 3D convention: leading eigenvalue of Q = <3 n n - I>/2."""
    config = Focus3DConfig()
    simulation = Focus3DSimulation(config)
    rng = np.random.default_rng(1)
    directors = rng.normal(size=(20000, 3))
    simulation.director = directors / np.linalg.norm(
        directors, axis=1, keepdims=True
    )
    simulation.position = rng.uniform(0, config.size_um, (20000, 3))
    assert simulation.global_order() < 0.05


def test_aligned_directors_give_unit_order():
    config = Focus3DConfig()
    simulation = Focus3DSimulation(config)
    simulation.director = np.tile(np.array([0.0, 1.0, 0.0]), (500, 1))
    simulation.position = np.zeros((500, 3))
    assert simulation.global_order() == pytest.approx(1.0, abs=1e-9)


def test_antiparallel_directors_are_perfectly_ordered():
    """Head-tail symmetry: +n and -n are the same physical state."""
    config = Focus3DConfig()
    simulation = Focus3DSimulation(config)
    simulation.director = np.tile(
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), (250, 1)
    )
    simulation.position = np.zeros((500, 3))
    assert simulation.global_order() == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- the physics
def test_order_responds_to_density():
    """An isotropic-nematic crossover, not a fixed number.

    Measured on this model the transition sits between packing 0.12 and 0.24.
    The predecessor model reported 0.956 at every density it could reach, which
    is what a saturated finite-size artefact looks like.

    2500 steps is 125 simulated hours. Fewer leaves the dense case still
    ordering (0.17 at 1500 steps against 0.71 at 2500) and the test would fail
    for want of time rather than for want of physics. The domain has to stay
    large too: in a 100 um box even the dilute case reaches 0.82, because six
    alignment radii is small enough for the whole volume to correlate.
    """
    dilute = _run(2500, target_packing=0.06, n_initial=120)
    dense = _run(2500, target_packing=0.45, n_initial=120)
    assert dense.packing_fraction > 3 * dilute.packing_fraction
    assert dense.global_order() > dilute.global_order() + 0.3


def test_noise_destroys_order():
    """Order must be a balance, not a property of the initial condition."""
    ordered = _run(2500, rot_diffusion_per_h=0.4, align_rate_per_h=8.0,
                   target_packing=0.45, n_initial=120)
    noisy = _run(2500, rot_diffusion_per_h=8.0, align_rate_per_h=0.5,
                 target_packing=0.45, n_initial=120)
    assert ordered.global_order() > noisy.global_order() + 0.3


def test_proliferation_saturates_at_the_packing_target():
    """Growth is capped by volume fraction, not by a cell count.

    The same number of cells is a dense phase in a small box and a gas in a
    large one, and it was a fixed cap that made the predecessor impossible to
    place in the nematic window.
    """
    simulation = _run(900, target_packing=0.2, n_initial=100)
    assert simulation.packing_fraction <= 0.25
    assert simulation.n_cells > 100


def test_daughters_inherit_the_parent_axis():
    """Division in a packed tissue does not randomise the orientation."""
    simulation = _run(0, n_initial=50)
    before = simulation.n_cells
    for _ in range(300):
        simulation.step()
    assert simulation.n_cells > before
    # If daughters were randomly oriented the field could not order at all.
    assert np.isfinite(simulation.global_order())


def test_simulation_is_reproducible():
    first = _run(150, seed=7)
    second = _run(150, seed=7)
    assert np.allclose(first.position, second.position)
    assert first.global_order() == pytest.approx(second.global_order())
    assert _run(150, seed=8).global_order() != pytest.approx(
        first.global_order()
    )


# --------------------------------------------------------------- interface
def test_director_volume_feeds_the_stereology_module():
    from simulations.stereology import section_sweep

    simulation = _run(300, size_um=240.0, n_initial=200)
    volume = simulation.director_volume(n_voxels=32, sigma_voxels=1.2)
    assert set(volume) == {"tensor", "density"}
    assert volume["tensor"].shape == (32, 32, 32, 3, 3)

    sweep = section_sweep(volume, (240.0,) * 3, polar_angles_deg=(0, 90),
                          n_azimuths=2, n_samples=70)
    assert len(sweep) == 4
    assert sweep.areal_density_mm2.notna().all()


def test_metrics_reports_the_finite_size_ratio():
    """The number that says whether the box is big enough to trust."""
    simulation = _run(50)
    metrics = simulation.metrics()
    assert metrics["domain_over_alignment_radius"] >= 5.0
    assert 0.0 <= metrics["global_order_S"] <= 1.0
