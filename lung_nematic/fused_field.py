"""
Fused nuclear + collagen director field.

The nuclear field carries orientation where cells are dense; the collagen
(structure-tensor) field carries it where fibers dominate and nuclei are
sparse. Neither alone covers fibrotic tissue well. This module combines them in
Q-tensor space, weighting each source by its local confidence, so the fused
director follows nuclei in cellular regions and collagen in fibrous regions.

Both inputs are the ``{"density", "order", "theta"}`` dicts already produced by
``compute_nematic_field`` (nuclear) and ``compute_collagen_field`` (collagen) at
the same scale. The output is the same kind of dict, so it plugs into
``detect_defects_single_scale`` and ``cluster_multiscale_defects`` unchanged.

Confidence weights:
    nuclear  c_n = normalised nuclear orientational density,
    collagen c_c = coherence * normalised eosin density.
Each source contributes its weighted Q-tensor ``c * order * (cos2t, sin2t)``;
the fused angle is the argument of the summed Q, and the fused order is its
magnitude divided by the total confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .collagen_field import compute_collagen_field
from .defects import cluster_multiscale_defects, detect_defects_single_scale
from .nematic import compute_nematic_field


def _normalise(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale to [0, 1] by the 99th percentile (robust to outliers)."""
    reference = np.percentile(array, 99)
    if reference <= eps:
        return np.zeros_like(array)
    return np.clip(array / reference, 0.0, 1.0)


def compute_fused_field(
    nuclear_field: dict[str, np.ndarray],
    collagen_field: dict[str, np.ndarray],
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Combine a nuclear and a collagen field into one director field.

    The fused nematic tensor is a presence-weighted average of the two source
    tensors:

        Q_fused = (w_n Q_n + w_c Q_c) / (w_n + w_c),

    where each source tensor already carries its own order,
    ``Q_i = order_i (cos 2t_i, sin 2t_i)``, and the weights are *presence* only:
    ``w_n`` = normalised nuclear density, ``w_c`` = normalised eosin density.
    Coherence therefore enters exactly once (through ``order_c`` inside
    ``Q_c``); the weights measure how much of each material is present, not how
    aligned it is. The fused angle is the argument of ``Q_fused`` and the fused
    order its magnitude.
    """
    w_nuclear = _normalise(nuclear_field["density"])
    w_collagen = _normalise(collagen_field["density"])

    qxx = (
        w_nuclear * nuclear_field["order"] * np.cos(2 * nuclear_field["theta"])
        + w_collagen
        * collagen_field["order"]
        * np.cos(2 * collagen_field["theta"])
    )
    qxy = (
        w_nuclear * nuclear_field["order"] * np.sin(2 * nuclear_field["theta"])
        + w_collagen
        * collagen_field["order"]
        * np.sin(2 * collagen_field["theta"])
    )

    total_weight = w_nuclear + w_collagen
    theta = (0.5 * np.arctan2(qxy, qxx)) % np.pi
    order = np.clip(np.sqrt(qxx**2 + qxy**2) / (total_weight + eps), 0, 1)

    return {
        "density": total_weight,
        "order": order,
        "theta": theta,
    }


def detect_multiscale_fused_defects(
    oriented_nuclei: pd.DataFrame,
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    inner_scale_px: float = 1.5,
) -> tuple[pd.DataFrame, dict[float, dict[str, np.ndarray]], pd.DataFrame]:
    """Multi-scale candidate defect detection on the fused field."""
    fields: dict[float, dict[str, np.ndarray]] = {}
    all_detections: list[pd.DataFrame] = []

    for sigma in config.sigmas_px:
        nuclear_field = compute_nematic_field(
            oriented_nuclei, tissue_mask.shape, sigma
        )
        collagen_field = compute_collagen_field(
            eosin,
            sigma,
            inner_scale_px,
            tissue_mask=tissue_mask,
            mask_normalized=config.mask_normalized_smoothing,
        )
        fused = compute_fused_field(nuclear_field, collagen_field)
        fields[float(sigma)] = fused

        detections = detect_defects_single_scale(fused, tissue_mask, config)
        if not detections.empty:
            detections["sigma_px"] = float(sigma)
            all_detections.append(detections)

    if all_detections:
        raw_detections = pd.concat(all_detections, ignore_index=True)
    else:
        raw_detections = pd.DataFrame()

    defects = cluster_multiscale_defects(
        raw_detections, len(config.sigmas_px), config
    )
    return defects, fields, raw_detections
