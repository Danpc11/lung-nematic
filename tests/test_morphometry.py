"""Tests for cell and nucleus morphometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk

from lung_nematic.morphometry import (
    measure_objects,
    morphometry_summary,
    segment_cells_from_nuclei,
    write_tsv,
)


def _labelled_disks(shape=(200, 200), centres=((50, 50), (50, 150), (150, 100)),
                    radius=12):
    labels = np.zeros(shape, dtype=int)
    for index, (cy, cx) in enumerate(centres, start=1):
        rr, cc = disk((cy, cx), radius, shape=shape)
        labels[rr, cc] = index
    return labels


def test_measure_objects_reports_geometry():
    labels = _labelled_disks()
    objects = measure_objects(labels, "nucleus")
    assert len(objects) == 3
    assert set(["area_px2", "major_axis_px", "equivalent_diameter_px"]).issubset(objects.columns)
    # a disk of radius 12 has area ~ pi*144
    assert objects["area_px2"].median() == pytest.approx(np.pi * 144, rel=0.15)


def test_microns_columns_appear_with_scale():
    labels = _labelled_disks()
    with_scale = measure_objects(labels, "nucleus", microns_per_pixel=0.5)
    without = measure_objects(labels, "nucleus")
    assert "equivalent_diameter_um" in with_scale.columns
    assert "equivalent_diameter_um" not in without.columns
    # 0.5 um/px halves the diameter number
    assert with_scale["equivalent_diameter_um"].median() == pytest.approx(
        without["equivalent_diameter_px"].median() * 0.5
    )


def test_cell_territories_are_larger_than_nuclei():
    """Watershed territories must enclose and exceed their seed nuclei."""
    shape = (200, 200)
    nuclei = _labelled_disks(shape, radius=8)
    mask = np.ones(shape, bool)
    cells = segment_cells_from_nuclei(nuclei, mask)
    assert cells.max() == nuclei.max()  # one territory per nucleus
    nucleus_area = (nuclei > 0).sum()
    cell_area = (cells > 0).sum()
    assert cell_area > nucleus_area, "territories should expand beyond nuclei"


def test_cell_territories_respect_mask():
    shape = (200, 200)
    nuclei = _labelled_disks(shape, radius=8)
    mask = np.zeros(shape, bool)
    mask[20:180, 20:180] = True
    cells = segment_cells_from_nuclei(nuclei, mask)
    # no territory pixel outside the mask
    assert not (cells > 0)[~mask].any()


def test_summary_on_empty_is_safe():
    empty = pd.DataFrame(columns=["kind", "equivalent_diameter_px", "area_px2",
                                  "aspect_ratio"])
    summary = morphometry_summary(empty)
    assert summary["n"] == 0


def test_write_tsv_is_tab_separated(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_tsv(frame, tmp_path / "out.tsv")
    text = path.read_text()
    assert "\t" in text
    assert "," not in text.split("\n")[0]  # header has no commas
    reloaded = pd.read_csv(path, sep="\t")
    assert list(reloaded.columns) == ["a", "b"]
