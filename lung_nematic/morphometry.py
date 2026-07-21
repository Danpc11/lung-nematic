"""
Cell and nucleus morphometry.

Two size questions come up repeatedly: how big are the nuclei, and how big are
the whole cells. The nucleus question is answered by ``segment_nuclei`` already;
this module adds the cell question and packages both as size distributions with
per-object measurements, written as TSV.

Cell bodies are rarely separable in H&E - cytoplasm is faint and cells touch -
so whole-cell boundaries are estimated by watershed expansion from the nuclei
into the tissue, which gives a defensible per-cell territory (a Voronoi-like
partition constrained to stay inside the tissue mask) rather than a true
membrane segmentation. In phase contrast the cell bodies are visible directly,
so there the coverage texture is segmented instead. Each path is labelled in the
output so the two are never silently compared.

Sizes are reported in pixels and, when a scale is supplied, in microns. The
median cell size feeds the adaptive integration radius; the full distribution is
the morphometric read-out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, segmentation


def segment_cells_from_nuclei(
    nucleus_labels: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    """Estimate whole-cell territories by watershed expansion from nuclei.

    Each nucleus seeds one cell; the tissue is partitioned so every tissue pixel
    is assigned to its nearest nucleus, constrained to the mask. This is a
    territory estimate, not a membrane segmentation, and should be described as
    such.
    """
    if nucleus_labels.max() == 0:
        return np.zeros_like(nucleus_labels)
    # distance from each pixel to the nearest nucleus, watershed on its negative
    distance = ndimage.distance_transform_edt(nucleus_labels == 0)
    cells = segmentation.watershed(distance, nucleus_labels, mask=tissue_mask)
    return cells


def measure_objects(
    labels: np.ndarray,
    kind: str,
    microns_per_pixel: float | None = None,
) -> pd.DataFrame:
    """One row per labelled object with geometry, optionally in microns."""
    rows = []
    for region in measure.regionprops(labels):
        minor = max(float(region.axis_minor_length), 1e-6)
        major = float(region.axis_major_length)
        row = {
            "kind": kind,
            "label": int(region.label),
            "x_px": float(region.centroid[1]),
            "y_px": float(region.centroid[0]),
            "area_px2": float(region.area),
            "major_axis_px": major,
            "minor_axis_px": minor,
            "aspect_ratio": major / minor,
            "equivalent_diameter_px": float(region.equivalent_diameter_area),
            "orientation_rad": float(region.orientation),
        }
        if microns_per_pixel is not None:
            scale = microns_per_pixel
            row.update({
                "area_um2": row["area_px2"] * scale**2,
                "major_axis_um": major * scale,
                "minor_axis_um": minor * scale,
                "equivalent_diameter_um": row["equivalent_diameter_px"] * scale,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def morphometry_summary(objects: pd.DataFrame) -> dict:
    """Distribution summary for one object set (nuclei or cells)."""
    if objects.empty:
        return {"n": 0}
    diameter = objects["equivalent_diameter_px"]
    summary = {
        "kind": objects["kind"].iloc[0],
        "n": int(len(objects)),
        "diameter_px_median": float(diameter.median()),
        "diameter_px_iqr": float(diameter.quantile(0.75) - diameter.quantile(0.25)),
        "area_px2_median": float(objects["area_px2"].median()),
        "aspect_ratio_median": float(objects["aspect_ratio"].median()),
    }
    if "equivalent_diameter_um" in objects:
        summary["diameter_um_median"] = float(objects["equivalent_diameter_um"].median())
    return summary


def quantify_morphology(
    tissue_mask: np.ndarray,
    hed: np.ndarray,
    config,
    microns_per_pixel: float | None = None,
    include_cells: bool = True,
) -> dict:
    """Full morphometry for a histology image: nuclei and estimated cells.

    Returns per-object tables and their summaries. The cell table is the
    watershed territory estimate, flagged ``kind="cell_territory"`` so it is not
    mistaken for a membrane segmentation.
    """
    from .segmentation import segment_nuclei

    nucleus_labels, nuclei = segment_nuclei(tissue_mask, hed, config)
    nucleus_objects = measure_objects(nucleus_labels, "nucleus", microns_per_pixel)

    result = {
        "nuclei": nucleus_objects,
        "nuclei_summary": morphometry_summary(nucleus_objects),
    }

    if include_cells:
        cell_labels = segment_cells_from_nuclei(nucleus_labels, tissue_mask)
        cell_objects = measure_objects(cell_labels, "cell_territory", microns_per_pixel)
        result["cells"] = cell_objects
        result["cells_summary"] = morphometry_summary(cell_objects)

    return result


def quantify_phase_contrast_morphology(
    image: np.ndarray,
    microns_per_pixel: float | None = None,
    coverage_quantile: float = 0.35,
) -> dict:
    """Morphometry for a phase-contrast frame, where cell bodies are visible.

    Cells are segmented from the coverage texture directly (no nuclear stain to
    seed from), so the objects here are ``kind="cell"`` - actual cell bodies,
    distinct from the histology territory estimate.
    """
    from .phase_contrast import cell_texture_mask

    coverage = cell_texture_mask(image, quantile=coverage_quantile)
    # separate touching cells by watershed on the coverage distance transform
    distance = ndimage.distance_transform_edt(coverage)
    from skimage.feature import peak_local_max
    coords = peak_local_max(distance, min_distance=10, labels=coverage)
    seeds = np.zeros(coverage.shape, dtype=int)
    for index, (y, x) in enumerate(coords, start=1):
        seeds[y, x] = index
    cell_labels = segmentation.watershed(-distance, seeds, mask=coverage)
    cell_objects = measure_objects(cell_labels, "cell", microns_per_pixel)

    return {
        "cells": cell_objects,
        "cells_summary": morphometry_summary(cell_objects),
        "coverage_fraction": float(coverage.mean()),
    }


def write_tsv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a table as TSV. TSV, not CSV, is the project's tabular format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return path
