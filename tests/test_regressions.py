"""Regression tests for packaging, metadata, coverage, and batch cleanup."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from lung_nematic.batch import analyze_folder
from lung_nematic.config import load_default_config
from lung_nematic.io_utils import load_metadata, resolve_metadata
from lung_nematic.phase_contrast import analyze_phase_contrast


def _phase_image(shape=(160, 160)) -> np.ndarray:
    image = np.full((*shape, 3), 0.5, dtype=float)
    yy, xx = np.mgrid[0:80, 0:80]
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * xx / 10.0)
    image[40:120, 40:120] = np.stack([stripes] * 3, axis=-1)
    return image


def test_phase_coverage_uses_texture_mask_not_filled_envelope():
    config = replace(
        load_default_config(),
        sigmas_px=(12.0,),
        min_scales_for_persistence=1,
        min_edge_distance_px=5,
    )
    result = analyze_phase_contrast(_phase_image(), config, coverage_quantile=0.5)

    assert result["coverage_fraction"] == result["coverage_mask"].mean()
    assert result["coverage_fraction"] <= result["mask"].mean()


def test_metadata_falls_back_to_filename_when_relative_path_is_blank(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "filename": ["sample.png"],
            "relative_path": [pd.NA],
            "image_id": ["S01"],
            "group": ["fibrosis"],
            "microns_per_pixel": [0.45],
        }
    ).to_csv(csv_path, index=False)

    metadata = load_metadata(csv_path)
    resolved = resolve_metadata(
        tmp_path / "images" / "sample.png",
        metadata,
        default_microns_per_pixel=1.0,
        root=tmp_path / "images",
    )

    assert resolved["image_id"] == "S01"
    assert resolved["group"] == "fibrosis"
    assert resolved["microns_per_pixel"] == 0.45


def test_clean_batch_removes_stale_processing_errors(tmp_path, monkeypatch):
    images = tmp_path / "images"
    output = tmp_path / "output"
    images.mkdir()
    output.mkdir()
    (images / "sample.png").write_bytes(b"placeholder")
    stale = output / "processing_errors.csv"
    stale.write_text("error\nold failure\n", encoding="utf-8")

    monkeypatch.setattr(
        "lung_nematic.batch.analyze_image",
        lambda *args, **kwargs: {"image_id": "sample", "group": "test"},
    )
    analyze_folder(images, output, load_default_config())

    assert not stale.exists()
