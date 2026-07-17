"""
Defect - local-order colocalization test.

A candidate defect is more meaningful if it sits inside an organised structure
(a fibroblastic focus, an aligned collagen bundle) than in featureless
parenchyma. Two complementary measures are reported, which resolve the apparent
tension between "the core of a defect is disordered" and "a defect can be
embedded in order":

- ``S_core``: local order sampled *at* the defect. A genuine ``+/-1/2`` defect
  is a director singularity, so this is expected to be *low* (a sanity check).
- ``S_annulus``: local order in a ring around the defect. If defects sit inside
  organised regions this is expected to be *higher* than at random tissue.

Both are compared against a bootstrap null built from random *eligible plaquette
centres* -- locations that pass the exact same four-vertex gate the detector
applies (all four grid corners inside tissue, above the density threshold and
far enough from the edge). This fixes the earlier mismatch where controls were
filtered on the central pixel only while defects required all four corners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from .config import AnalysisConfig
from .nematic import get_density_threshold


def eligible_plaquette_centers(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Centres of grid plaquettes whose four corners pass the detector gate.

    Returns an ``(n, 2)`` array of ``(x_px, y_px)`` centres, identical in
    geometry to where ``detect_defects_single_scale`` is allowed to place a
    defect.
    """
    density = field["density"]
    height, width = tissue_mask.shape
    step = config.defect_grid_step_px

    xs = np.arange(0, width, step)
    ys = np.arange(0, height, step)
    threshold = get_density_threshold(
        density, tissue_mask, config.density_quantile
    )
    edge = distance_transform_edt(tissue_mask)

    dens = density[np.ix_(ys, xs)]
    tiss = tissue_mask[np.ix_(ys, xs)]
    edge_s = edge[np.ix_(ys, xs)]

    ok = tiss & (dens > threshold) & (edge_s >= config.min_edge_distance_px)
    # A plaquette is valid when all four of its corners are ok.
    valid_plaquette = (
        ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    )
    rows, cols = np.nonzero(valid_plaquette)
    centers_x = xs[cols] + step / 2.0
    centers_y = ys[rows] + step / 2.0
    return np.column_stack([centers_x, centers_y]).astype(float)


def _sample_point(score: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = score.shape
    xi = np.clip(np.rint(x).astype(int), 0, width - 1)
    yi = np.clip(np.rint(y).astype(int), 0, height - 1)
    return score[yi, xi]


def _annulus_offsets(inner_px: float, outer_px: float) -> tuple[np.ndarray, np.ndarray]:
    radius = int(np.ceil(outer_px))
    dy, dx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance_sq = dx**2 + dy**2
    ring = (distance_sq >= inner_px**2) & (distance_sq <= outer_px**2)
    return dy[ring].ravel(), dx[ring].ravel()


def _sample_annulus(
    score: np.ndarray,
    tissue_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    offsets: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    height, width = score.shape
    dy, dx = offsets
    means = np.empty(x.size, dtype=float)
    for index in range(x.size):
        yy = np.clip(np.rint(y[index]).astype(int) + dy, 0, height - 1)
        xx = np.clip(np.rint(x[index]).astype(int) + dx, 0, width - 1)
        inside = tissue_mask[yy, xx]
        means[index] = score[yy, xx][inside].mean() if inside.any() else np.nan
    return means


def _summary(observed: float, null: np.ndarray, n_bootstrap: int) -> dict:
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return {
            "observed": float(observed),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_higher": float("nan"),
            "p_lower": float("nan"),
            "p_two_sided": float("nan"),
            "direction": "undefined",
        }
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if null.size > 1 else 0.0
    z = (observed - null_mean) / null_std if null_std > 0 else float("nan")
    p_higher = (1 + int(np.sum(null >= observed))) / (null.size + 1)
    p_lower = (1 + int(np.sum(null <= observed))) / (null.size + 1)
    return {
        "observed": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z,
        "p_higher": p_higher,
        "p_lower": p_lower,
        "p_two_sided": min(1.0, 2 * min(p_higher, p_lower)),
        "direction": (
            "higher_than_random"
            if observed > null_mean
            else "lower_than_random"
            if observed < null_mean
            else "equal"
        ),
    }


def run_colocalization(
    defects: pd.DataFrame,
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    representative_sigma_px: float,
    score: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap test of local order at defects (core and annulus) vs random.

    Controls are drawn from eligible plaquette centres (same four-vertex gate as
    detection). The annulus radii are
    ``[inner_frac, outer_frac] * representative_sigma_px``.

    Returns a flat dict with ``core_*`` and ``annulus_*`` statistics plus the
    raw bootstrap arrays under ``core_null`` and ``annulus_null``.
    """
    score_map = field["order"] if score is None else score

    centers = eligible_plaquette_centers(field, tissue_mask, config)
    n_defects = int(len(defects))

    base = {
        "n_defects": n_defects,
        "n_eligible_centers": int(len(centers)),
        "n_bootstrap": n_bootstrap,
        "annulus_inner_px": config.colocalization_annulus_inner_frac
        * representative_sigma_px,
        "annulus_outer_px": config.colocalization_annulus_outer_frac
        * representative_sigma_px,
    }

    if defects.empty or len(centers) == 0:
        empty = {"observed": np.nan, "null_mean": np.nan, "null_std": np.nan,
                 "z_score": np.nan, "p_higher": np.nan, "p_lower": np.nan,
                 "p_two_sided": np.nan, "direction": "undefined"}
        result = dict(base)
        for prefix in ("core", "annulus"):
            for key, value in empty.items():
                result[f"{prefix}_{key}"] = value
        result["core_null"] = np.zeros(0)
        result["annulus_null"] = np.zeros(0)
        return result

    inner_px = base["annulus_inner_px"]
    outer_px = base["annulus_outer_px"]
    offsets = _annulus_offsets(inner_px, outer_px)

    defect_x = defects["x_px"].to_numpy()
    defect_y = defects["y_px"].to_numpy()

    observed_core = float(np.mean(_sample_point(score_map, defect_x, defect_y)))
    observed_annulus = float(
        np.nanmean(
            _sample_annulus(score_map, tissue_mask, defect_x, defect_y, offsets)
        )
    )

    rng = np.random.default_rng(seed)
    core_null = np.empty(n_bootstrap, dtype=float)
    annulus_null = np.empty(n_bootstrap, dtype=float)
    center_x = centers[:, 0]
    center_y = centers[:, 1]
    for index in range(n_bootstrap):
        picks = rng.integers(0, len(centers), size=n_defects)
        px = center_x[picks]
        py = center_y[picks]
        core_null[index] = np.mean(_sample_point(score_map, px, py))
        annulus_null[index] = np.nanmean(
            _sample_annulus(score_map, tissue_mask, px, py, offsets)
        )

    result = dict(base)
    for prefix, observed, null in (
        ("core", observed_core, core_null),
        ("annulus", observed_annulus, annulus_null),
    ):
        summary = _summary(observed, null, n_bootstrap)
        for key, value in summary.items():
            result[f"{prefix}_{key}"] = value
        result[f"{prefix}_null"] = null
    return result
