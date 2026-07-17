from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage import filters, measure, morphology, segmentation

from ._compat import remove_small_holes, remove_small_objects
from skimage.feature import peak_local_max

from .config import AnalysisConfig


def _safe_threshold(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.inf
    if np.allclose(finite.min(), finite.max()):
        return float(finite.min())
    return max(
        float(filters.threshold_otsu(finite)),
        float(np.percentile(finite, 55)),
    )


def segment_nuclei(
    tissue_mask: np.ndarray,
    hed: np.ndarray,
    config: AnalysisConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Segment nuclei from the hematoxylin channel using watershed.

    Returns
    -------
    labels:
        Integer image containing accepted nuclear objects.
    nuclei:
        One row per accepted nucleus with geometry and orientation.
    """
    hematoxylin = hed[:, :, 0]
    values = hematoxylin[tissue_mask]

    if values.size < 100:
        return (
            np.zeros(tissue_mask.shape, dtype=np.int32),
            pd.DataFrame(),
        )

    threshold = _safe_threshold(values)
    nuclear_mask = (hematoxylin > threshold) & tissue_mask
    nuclear_mask = remove_small_objects(
        nuclear_mask,
        max(5, int(config.min_nucleus_area_px * 0.5)),
    )
    nuclear_mask = morphology.opening(
        nuclear_mask,
        morphology.disk(1),
    )
    nuclear_mask = remove_small_holes(
        nuclear_mask,
        20,
    )

    distance = distance_transform_edt(nuclear_mask)
    peaks = peak_local_max(
        distance,
        min_distance=4,
        threshold_abs=1.5,
        labels=nuclear_mask,
        exclude_border=False,
    )

    markers = np.zeros_like(distance, dtype=np.int32)
    if len(peaks) == 0:
        return markers, pd.DataFrame()

    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    raw_labels = segmentation.watershed(
        -distance,
        markers,
        mask=nuclear_mask,
    )

    accepted_labels = np.zeros_like(raw_labels, dtype=np.int32)
    rows: list[dict] = []
    accepted_id = 1

    for region in measure.regionprops(raw_labels):
        minor_axis = max(float(region.axis_minor_length), 1e-6)
        major_axis = float(region.axis_major_length)
        aspect_ratio = major_axis / minor_axis

        plausible = (
            config.min_nucleus_area_px
            <= region.area
            <= config.max_nucleus_area_px
            and config.min_major_axis_px
            <= major_axis
            <= config.max_major_axis_px
            and minor_axis >= config.min_minor_axis_px
            and aspect_ratio <= config.max_aspect_ratio
        )
        if not plausible:
            continue

        # Convert skimage's row-based orientation into an undirected
        # angle measured from the x-axis.
        theta = (np.pi / 2 - region.orientation) % np.pi
        anisotropy_weight = max(
            0.0,
            (aspect_ratio - 1.0) / (aspect_ratio + 1.0),
        )

        accepted_labels[raw_labels == region.label] = accepted_id
        rows.append(
            {
                "nucleus_id": accepted_id,
                "x_px": float(region.centroid[1]),
                "y_px": float(region.centroid[0]),
                "area_px": float(region.area),
                "major_axis_px": major_axis,
                "minor_axis_px": minor_axis,
                "aspect_ratio": aspect_ratio,
                "eccentricity": float(region.eccentricity),
                "theta_rad": float(theta),
                "theta_deg": float(np.degrees(theta)),
                "anisotropy_weight": anisotropy_weight,
            }
        )
        accepted_id += 1

    return accepted_labels, pd.DataFrame(rows)


def select_oriented_nuclei(
    nuclei: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if nuclei.empty:
        return nuclei.copy()

    return nuclei.loc[
        nuclei["aspect_ratio"]
        >= config.min_aspect_ratio_for_orientation
    ].copy()
