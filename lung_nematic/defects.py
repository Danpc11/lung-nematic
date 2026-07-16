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


def cluster_multiscale_defects(
    detections: pd.DataFrame,
    number_of_scales: int,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=DEFECT_COLUMNS)

    rows: list[dict] = []
    defect_id = 1

    for charge in (0.5, -0.5):
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

        detections = detect_defects_single_scale(
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
