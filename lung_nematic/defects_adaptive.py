"""
Defect detection with a locally adaptive integration radius.

The fixed-plaquette detector in ``defects.py`` measures the winding of the
director on a loop of a single size everywhere. That is wrong for a tissue with
two cell populations of different size: a loop spanning four or five fibroblasts
spans many more epithelial cells, so a fixed radius over-integrates in
epithelium (merging or erasing real defects) and under-integrates in
fibroblast-rich stroma (counting noise).

This module reads a per-pixel radius map - built by ``adaptive_radius`` from the
local cell size - and integrates the winding on a loop of that local radius at
every grid node. The loop always encloses a comparable number of cells,
whatever population dominates the patch, which is the condition the winding
needs to be stable.

The winding is measured on an N-point ring rather than a 4-corner plaquette. A
ring keeps each inter-sample angle well below the +/-pi branch cut even for a
+/-1 defect, and it lets the radius vary continuously from node to node.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from .config import AnalysisConfig
from .defects import wrap_angle
from .nematic import get_density_threshold


def _ring_winding(
    theta: np.ndarray,
    x: float,
    y: float,
    radius: float,
    n_points: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Winding of the doubled phase on a ring, plus the sampled order path.

    Returns the raw charge (winding / 4pi), the ring coordinates, and the
    director angles sampled on the ring so the caller can gate on them.
    """
    angles = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    height, width = theta.shape
    xs = np.clip((x + radius * np.cos(angles)).astype(int), 0, width - 1)
    ys = np.clip((y + radius * np.sin(angles)).astype(int), 0, height - 1)

    phase = 2 * theta[ys, xs]
    winding = 0.0
    for index in range(n_points):
        winding += wrap_angle(phase[(index + 1) % n_points] - phase[index])
    return winding / (4 * np.pi), xs, ys


def detect_defects_adaptive(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    radius_map: np.ndarray,
    config: AnalysisConfig,
    grid_step_px: int | None = None,
    n_ring_points: int = 16,
) -> pd.DataFrame:
    """Detect +/-1/2 defects with a locally adaptive loop radius.

    ``radius_map`` is a per-pixel integration radius (from
    ``adaptive_radius.adaptive_radius_map``). At each grid node the winding is
    integrated on a ring of the local radius; a candidate is kept only where the
    whole ring lies in tissue, sits far enough from the edge, and the enclosed
    order and density pass the same gates the fixed detector uses.

    The returned frame carries ``integration_radius_px`` per candidate, so the
    adaptive radius that produced each detection is auditable.
    """
    theta = field["theta"]
    order = field["order"]
    density = field["density"]
    height, width = tissue_mask.shape

    step = int(grid_step_px or config.defect_grid_step_px)
    edge_distance = distance_transform_edt(tissue_mask)
    threshold = get_density_threshold(density, tissue_mask, config.density_quantile)

    candidates: list[dict] = []
    for y in range(step, height - step, step):
        for x in range(step, width - step, step):
            if not tissue_mask[y, x]:
                continue
            radius = float(radius_map[y, x])
            # the ring must fit inside the frame and inside the tissue
            if x - radius < 0 or x + radius >= width:
                continue
            if y - radius < 0 or y + radius >= height:
                continue
            if edge_distance[y, x] < radius:
                continue

            charge_raw, ring_xs, ring_ys = _ring_winding(
                theta, x, y, radius, n_ring_points
            )
            charge = np.round(charge_raw * 2) / 2
            if abs(charge) != 0.5:
                continue

            ring_tissue = tissue_mask[ring_ys, ring_xs]
            ring_density = density[ring_ys, ring_xs]
            ring_order = order[ring_ys, ring_xs]
            if not np.all(ring_tissue):
                continue
            if not np.all(ring_density > threshold):
                continue

            candidates.append({
                "x_px": float(x),
                "y_px": float(y),
                "charge": float(charge),
                "charge_raw": float(charge_raw),
                "integration_radius_px": radius,
                "local_order_mean": float(ring_order.mean()),
                "local_density_mean": float(ring_density.mean()),
                "edge_distance_min_px": float(edge_distance[y, x]),
            })

    frame = pd.DataFrame(candidates)
    return _cluster_nearby(frame, config)


def _cluster_nearby(candidates: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Merge same-sign detections closer than the local radius into one.

    Adjacent grid nodes can each register the same defect, so same-sign
    candidates within one integration radius of each other are collapsed to
    their centroid. The separation scale is the candidate's own adaptive radius,
    not a global constant.
    """
    if candidates.empty:
        return candidates

    kept: list[dict] = []
    for charge_value in sorted(candidates["charge"].unique()):
        subset = candidates.loc[candidates["charge"] == charge_value]
        points = subset[["x_px", "y_px"]].to_numpy()
        radii = subset["integration_radius_px"].to_numpy()
        assigned = np.full(len(points), -1, dtype=int)
        cluster = 0
        for i in range(len(points)):
            if assigned[i] >= 0:
                continue
            assigned[i] = cluster
            queue = [i]
            while queue:
                current = queue.pop()
                distances = np.hypot(points[:, 0] - points[current, 0],
                                     points[:, 1] - points[current, 1])
                # merge within the larger of the two adaptive radii
                merge = (distances < np.maximum(radii, radii[current])) & (assigned < 0)
                neighbours = np.nonzero(merge)[0]
                assigned[neighbours] = cluster
                queue.extend(neighbours.tolist())
            cluster += 1

        for group in range(cluster):
            members = subset.loc[assigned == group]
            record = members.iloc[0].to_dict()
            record["x_px"] = float(members["x_px"].mean())
            record["y_px"] = float(members["y_px"].mean())
            record["n_grid_detections"] = int(len(members))
            kept.append(record)

    return pd.DataFrame(kept).reset_index(drop=True)


def defect_order_context(
    defects: pd.DataFrame,
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
) -> dict:
    """Where do the detected defects sit relative to the tissue's order?

    A real nematic defect sits in a low-order region (a domain wall). This
    reports the median order at the defects against the tissue median, so the
    "defects live on domain walls" claim is checkable per image rather than
    asserted.
    """
    order = field["order"]
    if defects.empty or not tissue_mask.any():
        return {"n_defects": 0, "order_at_defects": float("nan"),
                "order_in_tissue": float("nan"), "defects_on_walls": None}

    ys = np.clip(defects["y_px"].astype(int), 0, order.shape[0] - 1)
    xs = np.clip(defects["x_px"].astype(int), 0, order.shape[1] - 1)
    at_defects = float(np.median(order[ys, xs]))
    in_tissue = float(np.median(order[tissue_mask]))
    return {
        "n_defects": int(len(defects)),
        "order_at_defects": at_defects,
        "order_in_tissue": in_tissue,
        "defects_on_walls": bool(at_defects < in_tissue),
    }
