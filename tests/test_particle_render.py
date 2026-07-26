from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulations.alveolar import AlveolarConfig, CoupledSimulation
from simulations.alveolar.model import AT1, AT2
from simulations.alveolar.particle_render import (
    ParticleRenderConfig,
    accelerated_particle_demo_config,
    build_particle_snapshot,
    draw_particle_frame,
    run_and_record_particles,
)


def _small_config() -> AlveolarConfig:
    return replace(
        AlveolarConfig(),
        width_um=460.0,
        height_um=460.0,
        alveolar_diameter_um=160.0,
        segment_length_um=18.0,
        n_resident_fibroblasts=18,
        total_time_h=1.0,
        dt_h=1.0,
        rate_scale=1.0,
        seed=7,
    )


def test_snapshot_uses_human_at2_to_at1_cell_number_ratio():
    coupled = CoupledSimulation(_small_config())
    snapshot = build_particle_snapshot(coupled)

    surface_at2 = np.mean(coupled.epithelium.state == AT2)
    n_at2 = int(np.sum(snapshot.epithelial_state == AT2))
    n_at1 = int(np.sum(snapshot.epithelial_state == AT1))

    assert surface_at2 == pytest.approx(
        coupled.cfg.healthy_at2_surface_fraction,
        abs=0.015,
    )
    assert n_at2 / n_at1 == pytest.approx(
        coupled.cfg.healthy_at2_to_at1_number_ratio,
        rel=0.25,
    )
    assert snapshot.n_mesenchymal == coupled.mesenchyme.n_cells
    assert snapshot.epithelial_xyz.shape[1] == 3
    assert snapshot.mesenchymal_xyz.shape[1] == 3
    assert np.isfinite(snapshot.epithelial_xyz).all()
    assert np.isfinite(snapshot.mesenchymal_xyz).all()


def test_breathing_moves_cells_in_depth_without_changing_xy():
    coupled = CoupledSimulation(_small_config())
    render = ParticleRenderConfig(breathing_frames_per_cycle=6)
    expiration = build_particle_snapshot(coupled, render, frame_index=0)
    inspiration = build_particle_snapshot(coupled, render, frame_index=3)

    assert np.array_equal(
        expiration.epithelial_xyz[:, :2],
        inspiration.epithelial_xyz[:, :2],
    )
    assert inspiration.epithelial_xyz[:, 2].mean() > (
        expiration.epithelial_xyz[:, 2].mean()
    )


def test_demo_uses_many_compact_alveoli_and_thin_interstitium():
    config = accelerated_particle_demo_config()
    coupled = CoupledSimulation(config)

    assert coupled.epithelium.geometry.n_alveoli >= 15
    assert config.septal_thickness_um <= 5.0
    assert coupled.mesenchyme.permitted_mask().mean() < 0.10


def test_particle_frame_is_written(tmp_path):
    coupled = CoupledSimulation(_small_config())
    output = tmp_path / "frame.png"
    metrics = draw_particle_frame(
        coupled,
        output,
        ParticleRenderConfig(dpi=55, figure_width=5.0, figure_height=4.0),
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert metrics["n_epithelial_particles"] > 0
    assert metrics["n_mesenchymal_particles"] == coupled.mesenchyme.n_cells


def test_short_run_writes_frames_and_timeseries(tmp_path):
    outputs = run_and_record_particles(
        _small_config(),
        tmp_path,
        frame_every_h=1.0,
        fps=2,
        render_config=ParticleRenderConfig(
            dpi=45,
            figure_width=4.5,
            figure_height=3.5,
            show_colourbar=False,
        ),
        make_gif=False,
        make_mp4=False,
    )

    assert outputs["n_frames"] == 2
    assert (tmp_path / "particle_timeseries.tsv").is_file()
    assert len(list((tmp_path / "particle_frames").glob("*.png"))) == 2


def test_orbit_frames_can_be_combined_into_gif(tmp_path):
    outputs = run_and_record_particles(
        _small_config(),
        tmp_path,
        frame_every_h=1.0,
        fps=2,
        render_config=ParticleRenderConfig(
            dpi=40,
            figure_width=4.0,
            figure_height=3.0,
            orbit_deg_per_frame=15.0,
            show_colourbar=False,
        ),
        make_gif=True,
        make_mp4=False,
    )

    gif = tmp_path / "alveolar_particles.gif"
    assert outputs["gif"] == str(gif)
    assert gif.is_file()
    assert gif.stat().st_size > 1_000


def test_breathing_subframes_expand_each_biological_snapshot(tmp_path):
    outputs = run_and_record_particles(
        _small_config(),
        tmp_path,
        frame_every_h=1.0,
        fps=3,
        breathing_subframes=3,
        render_config=ParticleRenderConfig(
            dpi=35,
            figure_width=4.0,
            figure_height=3.0,
            breathing_frames_per_cycle=3,
            show_colourbar=False,
        ),
        make_gif=False,
        make_mp4=False,
    )

    assert outputs["n_frames"] == 6


@pytest.mark.parametrize(
    ("frame_every_h", "fps"),
    [(0.0, 4), (-1.0, 4), (1.0, 0), (1.0, -1)],
)
def test_invalid_recording_parameters_raise(tmp_path, frame_every_h, fps):
    with pytest.raises(ValueError):
        run_and_record_particles(
            _small_config(),
            tmp_path,
            frame_every_h=frame_every_h,
            fps=fps,
            make_gif=False,
            make_mp4=False,
        )


def test_invalid_breathing_subframes_raise(tmp_path):
    with pytest.raises(ValueError):
        run_and_record_particles(
            _small_config(),
            tmp_path,
            breathing_subframes=0,
            make_gif=False,
            make_mp4=False,
        )
