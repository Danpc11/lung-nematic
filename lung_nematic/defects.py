from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN

from .config import AnalysisConfig
from .nematic import compute_nematic_field, get_density_threshold


DEFECT_COLUMNS = [
    "defect_id",
    "x_px",
    "y_px",
    "charge",
    "scales_detected",
    "scale_fraction",
    "mean_local_order",
    "mean_edge_distance_px",
    "confidence",
    "sigmas_px",
]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def detect_defects_single_scale(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame:
    density = field["density"]
    theta = field["theta"]
    order = field["order"]
    height, width = tissue_mask.shape

    xs = np.arange(0, width, config.defect_grid_step_px)
    ys = np.arange(0, height, config.defect_grid_step_px)

    phase = 2 * theta[np.ix_(ys, xs)]
    sampled_density = density[np.ix_(ys, xs)]
    sampled_order = order[np.ix_(ys, xs)]
    sampled_tissue = tissue_mask[np.ix_(ys, xs)]

    edge_distance = distance_transform_edt(tissue_mask)
    sampled_edge = edge_distance[np.ix_(ys, xs)]

    threshold = get_density_threshold(
        density,
        tissue_mask,
        config.density_quantile,
    )

    candidates: list[dict] = []

    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            corner_phase = np.array(
                [
                    phase[row, col],
                    phase[row, col + 1],
                    phase[row + 1, col + 1],
                    phase[row + 1, col],
                ]
            )

            winding = sum(
                wrap_angle(
                    corner_phase[(index + 1) % 4]
                    - corner_phase[index]
                )
                for index in range(4)
            )

            # phase = 2*theta, therefore q = winding/(4*pi)
            charge_raw = winding / (4 * np.pi)
            charge = np.round(charge_raw * 2) / 2

            corner_density = np.array(
                [
                    sampled_density[row, col],
                    sampled_density[row, col + 1],
                    sampled_density[row + 1, col + 1],
                    sampled_density[row + 1, col],
                ]
            )
            corner_order = np.array(
                [
                    sampled_order[row, col],
                    sampled_order[row, col + 1],
                    sampled_order[row + 1, col + 1],
                    sampled_order[row + 1, col],
                ]
            )
            corner_tissue = np.array(
                [
                    sampled_tissue[row, col],
                    sampled_tissue[row, col + 1],
                    sampled_tissue[row + 1, col + 1],
                    sampled_tissue[row + 1, col],
                ]
            )
            corner_edge = np.array(
                [
                    sampled_edge[row, col],
                    sampled_edge[row, col + 1],
                    sampled_edge[row + 1, col + 1],
                    sampled_edge[row + 1, col],
                ]
            )

            valid = (
                np.all(corner_tissue)
                and np.all(corner_density > threshold)
                and np.all(
                    corner_edge >= config.min_edge_distance_px
                )
            )

            if valid and abs(charge) == 0.5:
                candidates.append(
                    {
                        "x_px": float(
                            xs[col] + config.defect_grid_step_px / 2
                        ),
                        "y_px": float(
                            ys[row] + config.defect_grid_step_px / 2
                        ),
                        "charge": float(charge),
                        "charge_raw": float(charge_raw),
                        "local_order_mean": float(
                            corner_order.mean()
                        ),
                        "local_density_mean": float(
                            corner_density.mean()
                        ),
                        "edge_distance_min_px": float(
                            corner_edge.min()
                        ),
                    }
                )

    return pd.DataFrame(candidates)


def detect_integer_defects_single_scale(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Detect integer (+/-1) defects by winding on an N-point ring.

    A +/-1 defect (aster, vortex or saddle) makes the director wind by 2*pi
    around an enclosing loop, i.e. the doubled phase 2*theta winds by 4*pi. On a
    4-corner plaquette each edge would carry ~pi of that winding, right at the
    +/-pi branch cut where the sign is ambiguous. Sampling the director on a
    ring of ``integer_defect_loop_points`` points (>= 6) keeps each per-edge
    step well below pi, so the full winding is recovered. Candidates are kept
    only when ``round(charge) == +/-1``; half-integer defects are left to
    ``detect_defects_single_scale``.

    Interpretation caveat: the ring measures the *total enclosed* winding, so a
    +1 can be a genuine aster/vortex or two unresolved +1/2 cores inside the
    ring. Read this layer alongside the +/-1/2 layer.
    """
    density = field["density"]
    theta = field["theta"]
    order = field["order"]
    height, width = tissue_mask.shape

    step = config.defect_grid_step_px
    radius = int(config.integer_defect_loop_radius_px)
    n_points = int(config.integer_defect_loop_points)

    threshold = get_density_threshold(
        density, tissue_mask, config.density_quantile
    )
    edge_distance = distance_transform_edt(tissue_mask)

    angles = np.arange(n_points) * (2 * np.pi / n_points)
    ring_dx = np.rint(radius * np.cos(angles)).astype(int)
    ring_dy = np.rint(radius * np.sin(angles)).astype(int)

    xs = np.arange(radius, width - radius, step)
    ys = np.arange(radius, height - radius, step)

    candidates: list[dict] = []
    for cy in ys:
        ry = cy + ring_dy
        for cx in xs:
            rx = cx + ring_dx

            ring_tissue = tissue_mask[ry, rx]
            ring_density = density[ry, rx]
            ring_edge = edge_distance[ry, rx]

            valid = (
                np.all(ring_tissue)
                and np.all(ring_density > threshold)
                and np.all(ring_edge >= config.min_edge_distance_px)
            )
            if not valid:
                continue

            phase = 2 * theta[ry, rx]
            # Closed-loop winding: phase[k+1]-phase[k], last edge wraps to first.
            steps = wrap_angle(np.diff(phase, append=phase[:1]))
            winding = float(steps.sum())

            charge_raw = winding / (4 * np.pi)
            charge = np.round(charge_raw * 2) / 2

            if abs(charge) == 1.0:
                candidates.append(
                    {
                        "x_px": float(cx),
                        "y_px": float(cy),
                        "charge": float(charge),
                        "charge_raw": float(charge_raw),
                        "local_order_mean": float(order[ry, rx].mean()),
                        "local_density_mean": float(ring_density.mean()),
                        "edge_distance_min_px": float(ring_edge.min()),
                    }
                )

    return _thin_integer_candidates(pd.DataFrame(candidates), config)


def _thin_integer_candidates(
    candidates: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Collapse redundant ring detections of the same integer defect.

    Unlike the half-integer plaquette test, which fires only on the plaquette
    containing the singularity, *every* ring that encloses an integer defect
    registers it. Detections therefore fill a disc of the ring radius around
    the true core, and a single +1 produces two dozen candidates at typical
    settings - a count inflated by a factor set by grid step and ring radius
    rather than by the tissue.

    Candidates of the same sign closer than ``integer_min_separation_px`` are
    agglomerated and replaced by their centroid, which is where the enclosed
    defect actually sits. The separation should be about twice the ring radius,
    since that is the diameter of the disc a single defect illuminates.
    """
    if candidates.empty:
        return candidates

    separation = float(config.integer_min_separation_px)
    if separation <= 0:
        return candidates

    clustered: list[dict] = []
    for charge_value in sorted(candidates["charge"].unique()):
        subset = candidates.loc[candidates["charge"] == charge_value]
        points = subset[["x_px", "y_px"]].to_numpy(dtype=float)
        assigned = np.full(len(points), -1, dtype=int)
        n_clusters = 0

        for index in range(len(points)):
            if assigned[index] >= 0:
                continue
            assigned[index] = n_clusters
            queue = [index]
            while queue:                       # single-link agglomeration
                current = queue.pop()
                distances = np.hypot(
                    points[:, 0] - points[current, 0],
                    points[:, 1] - points[current, 1],
                )
                neighbours = np.nonzero((distances < separation) & (assigned < 0))[0]
                assigned[neighbours] = n_clusters
                queue.extend(neighbours.tolist())
            n_clusters += 1

        for cluster in range(n_clusters):
            members = subset.loc[assigned == cluster]
            record = members.iloc[0].to_dict()
            record["x_px"] = float(members["x_px"].mean())
            record["y_px"] = float(members["y_px"].mean())
            record["charge_raw"] = float(members["charge_raw"].mean())
            record["n_ring_detections"] = int(len(members))
            clustered.append(record)

    if not clustered:
        return candidates.iloc[0:0]
    return pd.DataFrame(clustered).reset_index(drop=True)


def single_scale_detections(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Half-integer detections, plus the +/-1 ring layer when enabled.

    This is the single entry point the nuclear, collagen and fused multi-scale
    detectors share, so the integer layer is available identically for all
    three fields.
    """
    half = detect_defects_single_scale(field, tissue_mask, config)
    if not config.detect_integer_defects:
        return half
    integer = detect_integer_defects_single_scale(field, tissue_mask, config)
    frames = [frame for frame in (half, integer) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cluster_multiscale_defects(
    detections: pd.DataFrame,
    number_of_scales: int,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=DEFECT_COLUMNS)

    rows: list[dict] = []
    defect_id = 1

    for charge in sorted(detections["charge"].unique()):
        subset = detections.loc[
            detections["charge"] == charge
        ].copy()
        if subset.empty:
            continue

        labels = DBSCAN(
            eps=config.defect_cluster_radius_px,
            min_samples=config.min_scales_for_persistence,
        ).fit(subset[["x_px", "y_px"]]).labels_
        subset["cluster"] = labels

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue

            cluster = subset.loc[subset["cluster"] == cluster_id]
            scales_detected = int(cluster["sigma_px"].nunique())
            if scales_detected < config.min_scales_for_persistence:
                continue

            scale_fraction = scales_detected / number_of_scales
            mean_order = float(cluster["local_order_mean"].mean())
            mean_edge = float(
                cluster["edge_distance_min_px"].mean()
            )

            # Conservative 0-1 score: persistence dominates, while
            # local order adds supporting evidence.
            confidence = float(
                np.clip(
                    0.75 * scale_fraction
                    + 0.25 * mean_order,
                    0,
                    1,
                )
            )

            rows.append(
                {
                    "defect_id": defect_id,
                    "x_px": float(cluster["x_px"].mean()),
                    "y_px": float(cluster["y_px"].mean()),
                    "charge": float(charge),
                    "scales_detected": scales_detected,
                    "scale_fraction": scale_fraction,
                    "mean_local_order": mean_order,
                    "mean_edge_distance_px": mean_edge,
                    "confidence": confidence,
                    "sigmas_px": ",".join(
                        f"{value:g}"
                        for value in sorted(
                            cluster["sigma_px"].unique()
                        )
                    ),
                }
            )
            defect_id += 1

    return pd.DataFrame(rows, columns=DEFECT_COLUMNS)


def detect_multiscale_defects(
    oriented_nuclei: pd.DataFrame,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> tuple[
    pd.DataFrame,
    dict[float, dict[str, np.ndarray]],
    pd.DataFrame,
]:
    fields: dict[float, dict[str, np.ndarray]] = {}
    all_detections: list[pd.DataFrame] = []

    for sigma in config.sigmas_px:
        field = compute_nematic_field(
            oriented_nuclei,
            tissue_mask.shape,
            sigma,
        )
        fields[float(sigma)] = field

        detections = single_scale_detections(
            field,
            tissue_mask,
            config,
        )
        if not detections.empty:
            detections["sigma_px"] = float(sigma)
            all_detections.append(detections)

    if all_detections:
        raw_detections = pd.concat(
            all_detections,
            ignore_index=True,
        )
    else:
        raw_detections = pd.DataFrame()

    defects = cluster_multiscale_defects(
        raw_detections,
        len(config.sigmas_px),
        config,
    )
    return defects, fields, raw_detections
