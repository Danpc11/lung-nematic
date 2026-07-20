"""
Turning a defect candidate into a feature vector.

The detector already attaches numbers to every candidate it proposes - charge,
how many scales it persisted across, local order, distance to the tissue edge.
This module gathers those, adds a handful of measurements of the director field
*around* the candidate, and returns one row per candidate. That table is what a
classifier is trained on, and what the trained classifier scores at inference.

The design choice that matters: several features are things the eye cannot read
off the overlay - multiscale persistence, the local coherence gradient, the
winding residual. Including them is deliberate. If hand-labels could be
reproduced from the drawn director alone, the classifier would just be encoding
one person's reading of one picture. The features the eye cannot see are what
let it discover an objective rule - "the ones you keep almost always persist
across three scales" - that then stands in for the eye.

Nothing here is model-specific. The same extractor serves histology (collagen
or nuclear fields) and the phase-contrast gels, so a classifier trained on one
can at least be *tried* on the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# The columns the classifier sees. Kept explicit so a saved model can check it
# is being given the same features it was trained on.
FEATURE_COLUMNS = [
    "charge",
    "charge_raw_abs_error",
    "scale_fraction",
    "scales_detected",
    "mean_local_order",
    "order_at_core",
    "order_annulus_mean",
    "order_core_annulus_ratio",
    "coherence_std_local",
    "density_local",
    "edge_distance_px",
    "winding_residual",
    "nearest_neighbour_px",
    "nearest_opposite_px",
]


def _sample(array: np.ndarray, x: float, y: float) -> float:
    ix = int(np.clip(round(x), 0, array.shape[1] - 1))
    iy = int(np.clip(round(y), 0, array.shape[0] - 1))
    return float(array[iy, ix])


def _disc(array: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    """Values of ``array`` inside a disc, for local statistics."""
    iy, ix = np.mgrid[
        max(0, int(y - radius)):min(array.shape[0], int(y + radius) + 1),
        max(0, int(x - radius)):min(array.shape[1], int(x + radius) + 1),
    ]
    inside = (ix - x) ** 2 + (iy - y) ** 2 <= radius**2
    return array[iy[inside], ix[inside]]


def _annulus(array: np.ndarray, x: float, y: float,
             inner: float, outer: float) -> np.ndarray:
    iy, ix = np.mgrid[
        max(0, int(y - outer)):min(array.shape[0], int(y + outer) + 1),
        max(0, int(x - outer)):min(array.shape[1], int(x + outer) + 1),
    ]
    distance_sq = (ix - x) ** 2 + (iy - y) ** 2
    ring = (distance_sq >= inner**2) & (distance_sq <= outer**2)
    return array[iy[ring], ix[ring]]


def extract_features(
    candidates: pd.DataFrame,
    field: dict[str, np.ndarray],
    core_radius_px: float = 10.0,
    annulus_scale: float = 3.0,
) -> pd.DataFrame:
    """One feature row per candidate defect.

    ``candidates`` is the detector output (needs at least ``x_px``, ``y_px``,
    ``charge``); ``field`` is the director field the candidates were found in,
    with ``theta``, ``order`` and ``density``. Extra detector columns
    (``scale_fraction``, ``mean_local_order`` ...) are used when present and
    filled with sensible defaults when not, so the extractor works on raw
    single-scale detections and on fully persistent defects alike.
    """
    if candidates.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    order = field["order"]
    density = field.get("density", np.ones_like(order))
    theta = field["theta"]

    coords = candidates[["x_px", "y_px"]].to_numpy(dtype=float)
    charges = candidates["charge"].to_numpy(dtype=float)

    rows = []
    for index, (x, y) in enumerate(coords):
        charge = charges[index]
        core_vals = _disc(order, x, y, core_radius_px)
        annulus_vals = _annulus(order, x, y,
                                core_radius_px, core_radius_px * annulus_scale)
        coherence_local = _disc(order, x, y, core_radius_px * annulus_scale)

        order_core = float(core_vals.mean()) if core_vals.size else 0.0
        order_annulus = float(annulus_vals.mean()) if annulus_vals.size else 0.0

        # winding residual: a genuine +/-1/2 has charge_raw near +/-0.5; a noise
        # plaquette has a raw value that rounds there but sits far from it
        charge_raw = float(candidates.iloc[index].get("charge_raw", charge))
        residual = abs(charge_raw - charge)

        # distance to the nearest other candidate, and nearest of opposite sign,
        # because a real +1/2 usually has a -1/2 partner not far away
        others = np.delete(coords, index, axis=0)
        other_charges = np.delete(charges, index)
        if others.size:
            distances = np.hypot(others[:, 0] - x, others[:, 1] - y)
            nearest = float(distances.min())
            opposite = other_charges * charge < 0
            nearest_opp = float(distances[opposite].min()) if opposite.any() else np.nan
        else:
            nearest, nearest_opp = np.nan, np.nan

        rows.append({
            "candidate_index": index,
            "x_px": float(x),
            "y_px": float(y),
            "charge": charge,
            "charge_raw_abs_error": residual,
            "scale_fraction": float(candidates.iloc[index].get("scale_fraction", 1.0)),
            "scales_detected": float(candidates.iloc[index].get("scales_detected", 1)),
            "mean_local_order": float(candidates.iloc[index].get(
                "mean_local_order", order_core)),
            "order_at_core": order_core,
            "order_annulus_mean": order_annulus,
            "order_core_annulus_ratio": order_core / (order_annulus + 1e-6),
            "coherence_std_local": float(coherence_local.std())
            if coherence_local.size else 0.0,
            "density_local": _sample(density, x, y),
            "edge_distance_px": float(candidates.iloc[index].get(
                "mean_edge_distance_px",
                candidates.iloc[index].get("edge_distance_min_px", np.nan))),
            "winding_residual": residual,
            "nearest_neighbour_px": nearest,
            "nearest_opposite_px": nearest_opp,
        })

    frame = pd.DataFrame(rows)
    # a stable id so labels can be joined back even after reordering
    frame["candidate_id"] = [
        f"{int(round(r.x_px))}_{int(round(r.y_px))}_{r.charge:+.1f}"
        for r in frame.itertuples()
    ]
    return frame


def feature_matrix(features: pd.DataFrame) -> np.ndarray:
    """The numeric matrix a model consumes, with NaNs made explicit.

    Missing neighbour distances (a lone candidate) are encoded as a large
    sentinel rather than dropped, because "no partner nearby" is itself
    informative - it usually means artefact.
    """
    matrix = np.array(features.reindex(columns=FEATURE_COLUMNS).to_numpy(dtype=float), copy=True)
    # neighbour columns: NaN -> large distance
    for name in ("nearest_neighbour_px", "nearest_opposite_px"):
        column = FEATURE_COLUMNS.index(name)
        matrix[np.isnan(matrix[:, column]), column] = 1e4
    # any remaining NaN -> column median
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if np.isnan(values).any():
            with np.errstate(all="ignore"):
                median = np.nanmedian(values) if not np.isnan(values).all() else np.nan
            values[np.isnan(values)] = 0.0 if np.isnan(median) else median
    return matrix
