"""Regression tests for field-specific quality-control flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lung_nematic.metrics import summarize_image


def _summary(field_type: str) -> dict:
    shape = (8, 8)
    empty_nuclei = pd.DataFrame(
        columns=["theta_rad", "anisotropy_weight"]
    )
    field = {
        "density": np.ones(shape),
        "order": np.ones(shape),
        "theta": np.zeros(shape),
    }
    return summarize_image(
        metadata={
            "filename": "sample.jpg",
            "image_id": "sample",
            "group": "test",
        },
        image_shape=shape,
        tissue_mask=np.ones(shape, dtype=bool),
        nuclei=empty_nuclei,
        oriented_nuclei=empty_nuclei,
        field=field,
        defects=pd.DataFrame(),
        density_quantile=0.0,
        representative_sigma_px=1.0,
        min_oriented_nuclei=200,
        field_type=field_type,
    )


def test_low_orientation_count_only_applies_to_nuclear_fields():
    nuclear = _summary("nuclear")
    collagen = _summary("collagen")
    fused = _summary("fused")

    assert nuclear["low_nuclear_orientation_count"] is True
    assert nuclear["low_orientation_count"] is True
    for non_nuclear in (collagen, fused):
        assert non_nuclear["low_nuclear_orientation_count"] is None
        assert non_nuclear["low_orientation_count"] is None
