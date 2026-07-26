from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulations.alveolar3d import (
    AT1,
    AT2,
    COLLAPSED,
    HUMAN_CHRONIC_RATE_SCALE,
    Alveolar3DConfig,
    Alveolar3DRenderConfig,
    Alveolar3DSimulation,
    human_chronic_3d_config,
    run_and_record_3d,
)


def _small_config() -> Alveolar3DConfig:
    return replace(
        Alveolar3DConfig(),
        epithelial_at2_per_alveolus=4,
        epithelial_at1_per_alveolus=2,
        n_resident_fibroblasts=14,
        max_mesenchymal_cells=40,
        total_time_h=2.0,
        dt_h=2.0,
        seed=21,
    )


def test_geometry_and_agents_have_real_xyz_coordinates():
    simulation = Alveolar3DSimulation(_small_config())

    assert simulation.centres.shape == (7, 3)
    assert np.linalg.matrix_rank(simulation.centres[1:]) == 3
    assert simulation.epithelial_xyz().shape[1] == 3
    assert simulation.fibroblast_xyz.shape[1] == 3
    assert np.std(simulation.epithelial_xyz()[:, 2]) > 0
    assert np.std(simulation.fibroblast_xyz[:, 2]) > 0


def test_initial_epithelial_number_ratio_matches_configuration():
    simulation = Alveolar3DSimulation(_small_config())
    n_at2 = int(np.sum(simulation.epithelial_state == AT2))
    n_at1 = int(np.sum(simulation.epithelial_state == AT1))

    assert n_at2 / n_at1 == pytest.approx(2.0)


def test_default_epithelial_morphometry_matches_human_targets():
    config = Alveolar3DConfig()
    simulation = Alveolar3DSimulation(config)
    n_at2 = int(np.sum(simulation.epithelial_state == AT2))
    n_at1 = int(np.sum(simulation.epithelial_state == AT1))
    represented_at2_area = n_at2 * config.at2_apical_surface_um2
    represented_at1_area = n_at1 * config.at1_apical_surface_um2
    at2_surface_fraction = represented_at2_area / (
        represented_at1_area + represented_at2_area
    )

    assert n_at2 / n_at1 == pytest.approx(67.0 / 40.0, rel=0.01)
    assert at2_surface_fraction == pytest.approx(0.0565, rel=0.02)
    assert config.at1_apical_surface_um2 / config.at2_apical_surface_um2 > 27


def test_breathing_expands_open_alveoli_but_not_collapsed_alveoli():
    simulation = Alveolar3DSimulation(_small_config())
    simulation.alveolar_state[0] = COLLAPSED
    simulation.radius_um[0] *= simulation.cfg.collapsed_radius_fraction
    simulation._update_tidal_strain()

    scale = simulation.respiratory_scale(1.0)

    assert scale[0] == pytest.approx(1.0)
    assert np.all(scale[1:] > 1.0)


def test_open_fibroblasts_remain_in_thin_interstitial_shell():
    simulation = Alveolar3DSimulation(_small_config())
    simulation.step()
    centre = simulation.centres[simulation.fibroblast_owner]
    radius = np.linalg.norm(simulation.fibroblast_xyz - centre, axis=1)
    expected = simulation.open_radius_um[simulation.fibroblast_owner]
    half = 0.5 * simulation.cfg.interstitial_thickness_um

    assert np.all(radius >= expected - half - 1e-8)
    assert np.all(radius <= expected + half + 1e-8)


def test_kinetic_scale_slows_biology_without_changing_displayed_clock():
    config = replace(_small_config(), kinetic_rate_scale=0.35)
    simulation = Alveolar3DSimulation(config)
    simulation.step()

    assert simulation.time_h == pytest.approx(config.dt_h)
    assert simulation.biological_time_days == pytest.approx(
        config.dt_h / 24.0 * 0.35
    )


def test_human_chronic_preset_uses_a_two_year_clinical_clock():
    config = human_chronic_3d_config(seed=7)

    assert config.total_time_h / 24.0 == pytest.approx(2.0 * 365.25)
    assert config.kinetic_rate_scale == pytest.approx(
        HUMAN_CHRONIC_RATE_SCALE
    )
    assert config.dt_h == pytest.approx(12.0)
    assert config.temporal_calibration == "human_chronic_disease_scale"


def test_human_chronic_duration_can_change_without_changing_rates():
    one_year = human_chronic_3d_config(years=1.0)
    three_years = human_chronic_3d_config(years=3.0)

    assert one_year.total_time_h / 24.0 == pytest.approx(365.25)
    assert three_years.total_time_h / 24.0 == pytest.approx(3.0 * 365.25)
    assert one_year.kinetic_rate_scale == three_years.kinetic_rate_scale


def test_short_3d_run_writes_frames_and_timeseries(tmp_path):
    outputs = run_and_record_3d(
        _small_config(),
        tmp_path,
        state_every_h=2.0,
        breathing_subframes=2,
        fps=2,
        render_config=Alveolar3DRenderConfig(
            figure_width=4.5,
            figure_height=4.0,
            dpi=35,
            breathing_frames_per_cycle=2,
            sphere_latitudes=7,
            sphere_longitudes=10,
            matrix_points_per_alveolus=5,
        ),
        make_gif=False,
        make_mp4=False,
    )

    assert outputs["n_frames"] == 4
    assert (tmp_path / "timeseries_3d.tsv").is_file()
    assert len(list((tmp_path / "frames_3d").glob("frame_*.png"))) == 4


def test_accelerated_scenario_forms_central_3d_focus():
    simulation = Alveolar3DSimulation().run()
    cells = np.bincount(
        simulation.fibroblast_owner,
        minlength=simulation.n_alveoli,
    )
    myofibroblasts = np.bincount(
        simulation.fibroblast_owner,
        weights=simulation.myofibroblast.astype(float),
        minlength=simulation.n_alveoli,
    )

    assert simulation.alveolar_state[simulation.injury_alveolus] == COLLAPSED
    assert cells[0] > 2 * np.mean(cells[1:])
    assert myofibroblasts[0] > np.sum(myofibroblasts[1:])


def test_invalid_number_of_alveoli_is_rejected():
    with pytest.raises(ValueError):
        replace(Alveolar3DConfig(), n_alveoli=8).validate()


def test_invalid_kinetic_scale_is_rejected():
    with pytest.raises(ValueError):
        replace(Alveolar3DConfig(), kinetic_rate_scale=0.0).validate()


def test_presentation_mode_renders_a_png(tmp_path):
    """The clean illustration style must produce a frame, like the default."""
    from simulations.alveolar3d.render import draw_3d_frame

    simulation = Alveolar3DSimulation(_small_config())
    simulation.run()
    output = tmp_path / "presentation_frame.png"
    record = draw_3d_frame(
        simulation,
        output,
        render_config=Alveolar3DRenderConfig(
            presentation=True,
            figure_width=6.0,
            figure_height=4.0,
            dpi=40,
            sphere_latitudes=7,
            sphere_longitudes=10,
            matrix_points_per_alveolus=5,
        ),
        frame_index=0,
    )
    assert output.is_file()
    assert output.stat().st_size > 0
    # the render still returns the metrics record, presentation or not
    assert "time_d" in record


def test_presentation_legend_reflects_state_not_time():
    """KRT8+ and collagen rows appear only when the model reports them."""
    import matplotlib

    from simulations.alveolar3d.render import _presentation_legend
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # healthy state: no transitional cells, no myofibroblasts, baseline matrix
    healthy = {
        "frac_KRT8": 0.0, "frac_aberrant": 0.0, "n_myofibroblast": 0,
        "n_collapsed": 0, "n_indurated": 0, "max_stiffness_kpa": 2.0,
        "time_d": 0.0,
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _presentation_legend(ax, healthy)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert not any("KRT8" in label for label in labels)
    assert not any("Collagen" in label for label in labels)
    assert not any("Myofibroblast" in label for label in labels)

    # fibrotic state: transitional cells and stiff matrix present
    fibrotic = {
        "frac_KRT8": 0.05, "frac_aberrant": 0.02, "n_myofibroblast": 14,
        "n_collapsed": 1, "n_indurated": 0, "max_stiffness_kpa": 8.0,
        "time_d": 365.0,
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _presentation_legend(ax, fibrotic)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert any("KRT8" in label for label in labels)
    assert any("Collagen" in label for label in labels)
    assert any("Myofibroblast" in label for label in labels)


def test_stage_caption_is_driven_by_state():
    """A zero-injury state must not be captioned as injured."""
    from simulations.alveolar3d.render import _stage_caption

    healthy = {
        "time_d": 0.0, "frac_KRT8": 0.0, "frac_aberrant": 0.0,
        "n_myofibroblast": 0, "n_collapsed": 0, "n_indurated": 0,
        "max_stiffness_kpa": 2.0,
    }
    assert "healthy" in _stage_caption(healthy)

    # even at a late timestamp, no injury means no injury caption
    late_but_healthy = dict(healthy, time_d=200.0)
    caption = _stage_caption(late_but_healthy)
    assert "injury" not in caption and "focus" not in caption
