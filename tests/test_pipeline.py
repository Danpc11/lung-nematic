from __future__ import annotations
from dataclasses import replace

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from lung_nematic.collagen_field import (
    compute_collagen_field,
    detect_multiscale_collagen_defects,
)
from lung_nematic.colocalization import eligible_plaquette_centers
from lung_nematic.config import AnalysisConfig, load_default_config
from lung_nematic.defects import detect_defects_single_scale
from lung_nematic.fused_field import compute_fused_field
from lung_nematic.null_model import run_null_model


def _mask(shape):
    mask = np.ones(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask


def _defect_field(shape, sign):
    """Analytic +/-1/2 defect director field, singularity inside a plaquette."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    # Offset by 5 px so the singularity sits inside a plaquette, not on a node.
    angle = np.arctan2(yy - (height / 2 + 5), xx - (width / 2 + 5))
    theta = (sign * 0.5 * angle) % np.pi
    rng = np.random.default_rng(0)
    density = 1.0 + 0.01 * rng.random(shape)
    return {"density": density, "order": np.ones(shape), "theta": theta}


def _detect_config():
    return AnalysisConfig(
        sigmas_px=(1.0,),
        defect_grid_step_px=10,
        density_quantile=0.0,
        min_edge_distance_px=5,
        defect_cluster_radius_px=20.0,
        min_scales_for_persistence=1,
    )


def _synthetic_he(path, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    img = np.full((200, 200, 3), (232, 184, 210), dtype=np.uint8)
    for _ in range(40):
        cy, cx = rng.integers(20, 180, size=2)
        yy, xx = np.mgrid[0:200, 0:200]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= rng.integers(9, 25)
        img[blob] = (110, 45, 125)
    Image.fromarray(img).save(path)


def test_default_config_loads():
    config = load_default_config()
    assert isinstance(config, AnalysisConfig)
    config.validate()


def test_plus_half_defect():
    config = _detect_config()
    field = _defect_field((200, 200), sign=+1)
    detected = detect_defects_single_scale(field, _mask((200, 200)), config)
    assert (detected["charge"] == 0.5).any()


def test_minus_half_defect():
    config = _detect_config()
    field = _defect_field((200, 200), sign=-1)
    detected = detect_defects_single_scale(field, _mask((200, 200)), config)
    assert (detected["charge"] == -0.5).any()


def test_uniform_field_has_no_defects():
    config = _detect_config()
    rng = np.random.default_rng(0)
    field = {
        "density": 1.0 + 0.01 * rng.random((200, 200)),
        "order": np.ones((200, 200)),
        "theta": np.full((200, 200), 0.7),
    }
    detected = detect_defects_single_scale(field, _mask((200, 200)), config)
    assert detected.empty or (detected["charge"].abs() == 0.5).sum() == 0


def test_collagen_uniform_field():
    # Parallel stripes -> single coherent orientation -> no defects.
    yy, xx = np.mgrid[0:200, 0:200]
    eosin = 0.5 + 0.5 * np.sin(2 * np.pi * yy / 10.0)
    mask = _mask((200, 200))
    field = compute_collagen_field(eosin, sigma_px=8.0)
    coherent = field["order"] > 0.5
    assert np.nanstd(field["theta"][coherent]) < 0.2
    defects, _, _ = detect_multiscale_collagen_defects(
        eosin, mask, _detect_config()
    )
    assert defects.empty


def test_fused_field():
    shape = (50, 50)
    nuclear = {
        "density": np.ones(shape),
        "order": np.ones(shape),
        "theta": np.full(shape, 0.2),
    }
    collagen = {
        "density": np.zeros(shape),  # no collagen -> fused follows nuclei
        "order": np.zeros(shape),
        "theta": np.full(shape, 1.4),
    }
    fused = compute_fused_field(nuclear, collagen)
    assert set(fused) == {"density", "order", "theta"}
    assert np.all(fused["theta"] >= 0) and np.all(fused["theta"] < np.pi)
    assert np.allclose(fused["theta"], 0.2, atol=1e-3)


def test_null_model_reproducibility():
    rng = np.random.default_rng(1)
    nuclei = pd.DataFrame({
        "x_px": rng.integers(5, 195, 300).astype(float),
        "y_px": rng.integers(5, 195, 300).astype(float),
        "theta_rad": rng.uniform(0, np.pi, 300),
        "anisotropy_weight": rng.uniform(0.2, 1.0, 300),
    })
    mask = _mask((200, 200))
    config = _detect_config()
    a = run_null_model(nuclei, mask, config, n_permutations=6, downsample=2, seed=7)
    b = run_null_model(nuclei, mask, config, n_permutations=6, downsample=2, seed=7)
    assert np.array_equal(a["null_totals"], b["null_totals"])


def test_null_model_parallel_matches_serial():
    """The parallel path must be bit-for-bit identical to the serial one.

    A permutation null whose p-value shifts with ``n_jobs`` is a silent bug.
    Determinism comes from one fixed seed per permutation, so this compares
    n_jobs=1 against n_jobs=2 and asserts identical null distributions.
    """
    rng = np.random.default_rng(1)
    nuclei = pd.DataFrame({
        "x_px": rng.integers(5, 195, 300).astype(float),
        "y_px": rng.integers(5, 195, 300).astype(float),
        "theta_rad": rng.uniform(0, np.pi, 300),
        "anisotropy_weight": rng.uniform(0.2, 1.0, 300),
    })
    mask = _mask((200, 200))
    config = _detect_config()

    serial = run_null_model(nuclei, mask, config, n_permutations=12,
                            downsample=2, seed=7, n_jobs=1)
    parallel = run_null_model(nuclei, mask, config, n_permutations=12,
                              downsample=2, seed=7, n_jobs=2)
    assert np.array_equal(serial["null_totals"], parallel["null_totals"])
    assert serial["p_two_sided"] == parallel["p_two_sided"]
    assert parallel["null_workers_used"] >= 1


def test_collagen_null_model_parallel_matches_serial():
    """The collagen route shares the executor, so it must be deterministic too."""
    from lung_nematic.null_model import run_collagen_null_model

    rng = np.random.default_rng(2)
    eosin = rng.uniform(0.1, 1.0, (120, 120))
    mask = _mask((120, 120))
    config = _detect_config()

    serial = run_collagen_null_model(eosin, mask, config, n_permutations=12,
                                     downsample=2, seed=5, n_jobs=1)
    parallel = run_collagen_null_model(eosin, mask, config, n_permutations=12,
                                       downsample=2, seed=5, n_jobs=2)
    assert np.array_equal(serial["null_totals"], parallel["null_totals"])


@pytest.mark.parametrize("bad", [0, -2, -5])
def test_null_model_rejects_invalid_n_jobs(bad):
    rng = np.random.default_rng(1)
    nuclei = pd.DataFrame({
        "x_px": rng.integers(5, 195, 50).astype(float),
        "y_px": rng.integers(5, 195, 50).astype(float),
        "theta_rad": rng.uniform(0, np.pi, 50),
        "anisotropy_weight": rng.uniform(0.2, 1.0, 50),
    })
    mask = _mask((200, 200))
    config = _detect_config()
    with pytest.raises(ValueError):
        run_null_model(nuclei, mask, config, n_permutations=4,
                       downsample=2, seed=7, n_jobs=bad)


def test_colocalization_same_valid_gate():
    # Left half has zero density -> ineligible; every eligible center is right.
    shape = (200, 200)
    rng = np.random.default_rng(0)
    density = 1.0 + 0.01 * rng.random(shape)
    density[:, : shape[1] // 2] = 0.0  # left half ineligible
    field = {"density": density, "order": np.ones(shape),
             "theta": np.zeros(shape)}
    config = _detect_config()
    centers = eligible_plaquette_centers(field, _mask(shape), config)
    assert len(centers) > 0
    assert np.all(centers[:, 0] > shape[1] // 2 - config.defect_grid_step_px)


def test_cli_nuclear_and_strict_json(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _synthetic_he(images / "sampleA.png")
    output = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "lung_nematic", "--input", str(images),
         "--output", str(output), "--field", "nuclear"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "summary_metrics.csv").exists()
    summaries = list(output.glob("*/*_summary.json"))
    assert summaries
    text = summaries[0].read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # standard JSON parses


def _integer_field(shape, sign):
    """Analytic +/-1 defect: aster (theta=phi) or saddle (theta=-phi)."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    phi = np.arctan2(yy - (height / 2 + 5), xx - (width / 2 + 5))
    theta = (sign * phi) % np.pi
    rng = np.random.default_rng(0)
    density = 1.0 + 0.01 * rng.random(shape)
    return {"density": density, "order": np.ones(shape), "theta": theta}


def _integer_config():
    return replace(
        _detect_config(),
        detect_integer_defects=True,
        integer_defect_loop_radius_px=20,
        integer_defect_loop_points=8,
    )


def test_plus_one_defect():
    from lung_nematic.defects import detect_integer_defects_single_scale

    field = _integer_field((200, 200), sign=+1)
    detected = detect_integer_defects_single_scale(
        field, _mask((200, 200)), _integer_config()
    )
    assert (detected["charge"] == 1.0).any()


def test_minus_one_defect():
    from lung_nematic.defects import detect_integer_defects_single_scale

    field = _integer_field((200, 200), sign=-1)
    detected = detect_integer_defects_single_scale(
        field, _mask((200, 200)), _integer_config()
    )
    assert (detected["charge"] == -1.0).any()


def test_integer_layer_is_opt_in():
    # With the aster field but the default config (layer off), the half-integer
    # detector must not emit +/-1 charges.
    from lung_nematic.defects import single_scale_detections

    field = _integer_field((200, 200), sign=+1)
    off = single_scale_detections(field, _mask((200, 200)), _detect_config())
    if not off.empty:
        assert (off["charge"].abs() == 1.0).sum() == 0


def test_spiral_angle_aster_and_vortex():
    from lung_nematic.defect_maps import estimate_spiral_angle

    height = width = 120
    yy, xx = np.mgrid[0:height, 0:width]
    phi = np.arctan2(yy - 60, xx - 60)
    aster = phi % np.pi
    vortex = (phi + np.pi / 2) % np.pi
    assert abs(estimate_spiral_angle(aster, 60, 60)) < 0.15
    assert abs(abs(estimate_spiral_angle(vortex, 60, 60)) - np.pi / 2) < 0.15


def test_render_defect_map_writes_file(tmp_path):
    import pandas as pd
    from lung_nematic.defect_maps import render_defect_map

    height = width = 120
    yy, xx = np.mgrid[0:height, 0:width]
    theta = (np.arctan2(yy - 60, xx - 60)) % np.pi
    field = {"density": np.ones((height, width)),
             "order": np.ones((height, width)), "theta": theta}
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    defect = pd.Series({"defect_id": 1, "x_px": 60.0, "y_px": 60.0, "charge": 1.0})
    out = tmp_path / "map.png"
    meta = render_defect_map(rgb, field, defect, out, window_px=100)
    assert out.exists()
    assert meta["spiral_class"] == "aster"
