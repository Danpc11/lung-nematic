"""Regression tests: the output tree must not contaminate the input tree, and
image_id values must be unique in the namespace that actually reaches disk.

Both failure modes are silent. Re-analysed overlays produce plausible metrics
rather than errors, and colliding identifiers overwrite results in place, so
neither shows up without an explicit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lung_nematic.batch import analyze_folder
from lung_nematic.config import load_default_config
from lung_nematic.io_utils import (
    derived_seed,
    discover_images,
    safe_identifier,
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


# ------------------------------------------------------- output re-ingestion
def test_discovery_excludes_the_output_subtree(tmp_path):
    images = tmp_path / "data"
    (images / "control").mkdir(parents=True)
    _synthetic_he(images / "control" / "a.png")

    results = images / "results" / "a"
    results.mkdir(parents=True)
    _synthetic_he(results / "a_nuclear_overlay.png", seed=1)

    found = discover_images(images, exclude_dirs=[images / "results"])
    assert [path.name for path in found] == ["a.png"]


def test_output_dir_equal_to_input_dir_is_rejected(tmp_path):
    images = tmp_path / "data"
    images.mkdir()
    _synthetic_he(images / "a.png")

    with pytest.raises(ValueError, match="must not be the same directory"):
        analyze_folder(images, images, load_default_config())


def test_second_run_over_nested_output_is_idempotent(tmp_path):
    """The canonical failure: --output inside --input, run twice."""
    images = tmp_path / "data"
    (images / "control").mkdir(parents=True)
    _synthetic_he(images / "control" / "a.png")
    output = images / "results"

    config = load_default_config()
    first, first_errors = analyze_folder(images, output, config)
    second, second_errors = analyze_folder(images, output, config)

    assert first_errors.empty and second_errors.empty
    assert len(first) == 1, "one input image"
    assert len(second) == 1, "overlays from run 1 were re-analysed as tissue"
    pd.testing.assert_frame_equal(first, second)


# ------------------------------------------------------ identifier namespace
@pytest.mark.parametrize("value", ["", ".", "..", "   ", "case.", "case "])
def test_reserved_or_trailing_identifiers_are_rejected(value):
    """'..' would resolve to the parent of output_root, not a child of it."""
    if value in {"case.", "case "}:
        # These are normalised, not rejected, but must not keep the trailing
        # character that NTFS silently strips.
        assert safe_identifier(value) == "case"
        return
    with pytest.raises(ValueError):
        safe_identifier(value)


def test_distinct_ids_that_normalize_alike_are_rejected(tmp_path):
    images = tmp_path / "data"
    images.mkdir()
    _synthetic_he(images / "one.png")
    _synthetic_he(images / "two.png", seed=2)

    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "filename": ["one.png", "two.png"],
            "image_id": ["case/01", "case_01"],
            "group": ["control", "control"],
        }
    ).to_csv(metadata, index=False)

    with pytest.raises(ValueError, match="after normalization"):
        analyze_folder(
            images,
            tmp_path / "out",
            load_default_config(),
            metadata_csv=metadata,
        )


def test_reserved_identifier_is_caught_before_any_image_is_processed(tmp_path):
    images = tmp_path / "data"
    images.mkdir()
    _synthetic_he(images / "one.png")

    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"filename": ["one.png"], "image_id": [".."], "group": ["control"]}
    ).to_csv(metadata, index=False)

    output = tmp_path / "out"
    with pytest.raises(ValueError):
        analyze_folder(
            images, output, load_default_config(), metadata_csv=metadata
        )
    # Nothing may have been written outside the output root.
    assert not (tmp_path / "..").resolve().joinpath("_nuclei.tsv").exists()
    assert list(output.glob("*")) == []


# ------------------------------------------------------------ derived seeds
def test_derived_seed_is_deterministic_and_decorrelated():
    assert derived_seed(42, "imgA", "nuclear") == derived_seed(
        42, "imgA", "nuclear"
    )
    assert derived_seed(42, "imgA", "nuclear") != derived_seed(
        42, "imgB", "nuclear"
    )
    assert derived_seed(42, "imgA", "nuclear") != derived_seed(
        42, "imgA", "collagen"
    )
    assert derived_seed(1, "imgA", "nuclear") != derived_seed(
        2, "imgA", "nuclear"
    )


# ------------------------------------------------ per-field batch reporting
def test_two_fields_accumulate_instead_of_overwriting(tmp_path):
    from dataclasses import replace

    images = tmp_path / "data"
    images.mkdir()
    _synthetic_he(images / "a.png")
    output = tmp_path / "out"

    config = load_default_config()
    analyze_folder(images, output, replace(config, field_type="nuclear"))
    analyze_folder(images, output, replace(config, field_type="collagen"))

    assert (output / "summary_metrics_nuclear.csv").exists()
    assert (output / "summary_metrics_collagen.csv").exists()
    combined = pd.read_csv(output / "summary_metrics.csv")
    assert set(combined["field_type"]) == {"nuclear", "collagen"}
    assert len(combined) == 2
