"""
Permutation null model for candidate topological defects.

The observed ``+/-1/2`` defect count is only interpretable against a baseline:
how many defects would the *same* nuclei produce if their orientations carried
no spatial coherence? This module answers that by holding nuclear positions,
anisotropy weights, the tissue mask, the density field, the density threshold,
the edge filter and the multi-scale persistence rule fixed, and shuffling only
the orientation angles. Everything except spatial orientational order is
preserved, so the test isolates that order.

Because a spatially ordered director field has *fewer* singularities than a
random one, tissue with real orientational order is expected to be *depleted*
in defects relative to the null. The surviving defects are the structurally
meaningful ones. The direction is reported from the data, not assumed.

Detection reuses ``detect_defects_single_scale`` and
``cluster_multiscale_defects`` unchanged, so the null is identical to the main
pipeline by construction. Both the observed statistic and every permutation are
evaluated through the same (optionally down-sampled) path, which keeps the
comparison internally valid. A down-sample factor of 2 reproduces the full-
resolution defect count exactly while running several times faster; larger
factors trade fidelity for speed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import AnalysisConfig
from .defects import cluster_multiscale_defects, detect_defects_single_scale


def _prepare(
    oriented_nuclei: pd.DataFrame,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    downsample: int,
) -> dict:
    """Precompute everything that is constant across permutations."""
    factor = max(1, int(downsample))
    height, width = tissue_mask.shape
    down_height = (height + factor - 1) // factor
    down_width = (width + factor - 1) // factor

    xs = np.clip(
        np.rint(oriented_nuclei["x_px"].to_numpy() / factor).astype(int),
        0,
        down_width - 1,
    )
    ys = np.clip(
        np.rint(oriented_nuclei["y_px"].to_numpy() / factor).astype(int),
        0,
        down_height - 1,
    )
    weights = oriented_nuclei["anisotropy_weight"].to_numpy()

    down_mask = tissue_mask[::factor, ::factor][:down_height, :down_width]

    scaled_config = replace(
        config,
        defect_grid_step_px=max(
            1, round(config.defect_grid_step_px / factor)
        ),
        min_edge_distance_px=config.min_edge_distance_px / factor,
        defect_cluster_radius_px=config.defect_cluster_radius_px / factor,
    )

    # Density impulse and per-scale density fields are permutation-invariant
    # (weights and positions never move), so cache them once.
    density_impulse = np.zeros((down_height, down_width), dtype=np.float32)
    np.add.at(density_impulse, (ys, xs), weights)
    density = {
        float(sigma): gaussian_filter(
            density_impulse, float(sigma) / factor, mode="constant"
        )
        for sigma in config.sigmas_px
    }

    return {
        "factor": factor,
        "shape": (down_height, down_width),
        "xs": xs,
        "ys": ys,
        "weights": weights,
        "mask": down_mask,
        "config": scaled_config,
        "density": density,
        "sigmas": tuple(float(s) for s in config.sigmas_px),
    }


def _count_defects(theta: np.ndarray, prepared: dict) -> tuple[int, int, int]:
    """Return (total, n_plus_half, n_minus_half) for one orientation set."""
    shape = prepared["shape"]
    xs, ys, weights = prepared["xs"], prepared["ys"], prepared["weights"]
    factor = prepared["factor"]

    qxx_impulse = np.zeros(shape, dtype=np.float32)
    qxy_impulse = np.zeros(shape, dtype=np.float32)
    np.add.at(qxx_impulse, (ys, xs), weights * np.cos(2 * theta))
    np.add.at(qxy_impulse, (ys, xs), weights * np.sin(2 * theta))

    detections: list[pd.DataFrame] = []
    for sigma in prepared["sigmas"]:
        scaled_sigma = sigma / factor
        density = prepared["density"][sigma]
        qxx = gaussian_filter(qxx_impulse, scaled_sigma, mode="constant")
        qxy = gaussian_filter(qxy_impulse, scaled_sigma, mode="constant")
        order = np.clip(
            np.sqrt(qxx**2 + qxy**2) / (density + 1e-12), 0, 1
        )
        angle = (0.5 * np.arctan2(qxy, qxx)) % np.pi
        field = {"density": density, "order": order, "theta": angle}

        detected = detect_defects_single_scale(
            field, prepared["mask"], prepared["config"]
        )
        if not detected.empty:
            detected["sigma_px"] = sigma
            detections.append(detected)

    if detections:
        raw = pd.concat(detections, ignore_index=True)
    else:
        raw = pd.DataFrame()

    defects = cluster_multiscale_defects(
        raw, len(prepared["sigmas"]), prepared["config"]
    )
    if defects.empty:
        return 0, 0, 0
    return (
        len(defects),
        int((defects["charge"] == 0.5).sum()),
        int((defects["charge"] == -0.5).sum()),
    )


def run_null_model(
    oriented_nuclei: pd.DataFrame,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    n_permutations: int = 199,
    downsample: int = 2,
    mode: str = "shuffle",
    seed: int = 0,
) -> dict:
    """
    Permutation test for the ``+/-1/2`` defect count of one image.

    Parameters
    ----------
    oriented_nuclei, tissue_mask, config:
        The same objects the main pipeline builds for the image.
    n_permutations:
        Number of orientation shuffles. The empirical p-value resolution is
        ``1 / (n_permutations + 1)``.
    downsample:
        Spatial down-sampling factor for speed. ``2`` reproduces the full-
        resolution defect count; larger values are faster but approximate.
    mode:
        ``"shuffle"`` permutes the observed angles among positions (preserves
        the angle marginal, destroys spatial coherence). ``"uniform"`` draws
        fresh angles on ``[0, pi)`` (also destroys the marginal).
    seed:
        Seed for the permutation RNG (reproducibility).

    Returns
    -------
    dict
        Observed counts, null summary statistics, effect direction, empirical
        p-values, and the raw per-permutation counts under ``"null_totals"``.
    """
    if mode not in {"shuffle", "uniform"}:
        raise ValueError("mode must be 'shuffle' or 'uniform'.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    keys = [
        "n_oriented_nuclei",
        "n_permutations",
        "null_mode",
        "null_downsample",
        "null_seed",
        "observed_total",
        "observed_plus_half",
        "observed_minus_half",
        "null_mean",
        "null_std",
        "null_median",
        "null_q2_5",
        "null_q97_5",
        "z_score",
        "log2_enrichment",
        "p_depletion",
        "p_enrichment",
        "p_two_sided",
        "direction",
    ]

    if oriented_nuclei.empty:
        result = {key: np.nan for key in keys}
        result.update(
            n_oriented_nuclei=0,
            n_permutations=n_permutations,
            null_mode=mode,
            null_downsample=int(max(1, downsample)),
            null_seed=seed,
            observed_total=0,
            observed_plus_half=0,
            observed_minus_half=0,
            direction="undefined",
            null_totals=np.zeros(0, dtype=int),
        )
        return result

    prepared = _prepare(oriented_nuclei, tissue_mask, config, downsample)
    theta_observed = oriented_nuclei["theta_rad"].to_numpy()

    observed_total, observed_plus, observed_minus = _count_defects(
        theta_observed, prepared
    )

    rng = np.random.default_rng(seed)
    null_totals = np.empty(n_permutations, dtype=int)
    for index in range(n_permutations):
        if mode == "shuffle":
            theta = rng.permutation(theta_observed)
        else:
            theta = rng.uniform(0.0, np.pi, size=theta_observed.shape)
        null_totals[index] = _count_defects(theta, prepared)[0]

    null_mean = float(null_totals.mean())
    null_std = float(null_totals.std(ddof=1)) if n_permutations > 1 else 0.0
    z_score = (
        (observed_total - null_mean) / null_std
        if null_std > 0
        else float("nan")
    )
    log2_enrichment = (
        float(np.log2(observed_total / null_mean))
        if observed_total > 0 and null_mean > 0
        else float("nan")
    )

    # Empirical p-values with +1 smoothing so they are never exactly zero.
    p_depletion = (1 + int(np.sum(null_totals <= observed_total))) / (
        n_permutations + 1
    )
    p_enrichment = (1 + int(np.sum(null_totals >= observed_total))) / (
        n_permutations + 1
    )
    p_two_sided = min(1.0, 2 * min(p_depletion, p_enrichment))

    if observed_total < null_mean:
        direction = "depleted"
    elif observed_total > null_mean:
        direction = "enriched"
    else:
        direction = "equal"

    return {
        "n_oriented_nuclei": int(len(oriented_nuclei)),
        "n_permutations": n_permutations,
        "null_mode": mode,
        "null_downsample": prepared["factor"],
        "null_seed": seed,
        "observed_total": observed_total,
        "observed_plus_half": observed_plus,
        "observed_minus_half": observed_minus,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_median": float(np.median(null_totals)),
        "null_q2_5": float(np.percentile(null_totals, 2.5)),
        "null_q97_5": float(np.percentile(null_totals, 97.5)),
        "z_score": z_score,
        "log2_enrichment": log2_enrichment,
        "p_depletion": p_depletion,
        "p_enrichment": p_enrichment,
        "p_two_sided": p_two_sided,
        "direction": direction,
        "null_totals": null_totals,
    }


def save_null_histogram(result: dict, output_path: str | Path, title: str = "") -> None:
    """Plot the null distribution with the observed count marked."""
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    null_totals = np.asarray(result["null_totals"])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if null_totals.size:
        bins = range(int(null_totals.min()), int(null_totals.max()) + 2)
        axis.hist(null_totals, bins=bins, align="left", alpha=0.8)
    axis.axvline(
        result["observed_total"],
        color="crimson",
        linewidth=2.5,
        label=f"observed = {result['observed_total']}",
    )
    axis.axvline(
        result["null_mean"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"null mean = {result['null_mean']:.1f}",
    )
    axis.set_xlabel("candidate +/-1/2 defects per image")
    axis.set_ylabel("permutations")
    subtitle = (
        f"{result['direction']} | p(two-sided) = {result['p_two_sided']:.3f} "
        f"| z = {result['z_score']:.2f} | N = {result['n_permutations']}"
    )
    axis.set_title(f"{title}\n{subtitle}" if title else subtitle)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _prepare_collagen(
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    downsample: int,
    inner_scale_px: float,
) -> dict:
    """Precompute the permutation-invariant parts of the collagen null."""
    factor = max(1, int(downsample))
    down_eosin = eosin[::factor, ::factor].astype(np.float32)
    down_mask = tissue_mask[::factor, ::factor][
        : down_eosin.shape[0], : down_eosin.shape[1]
    ]

    gx = gaussian_filter(down_eosin, inner_scale_px, order=[0, 1]) * down_mask
    gy = gaussian_filter(down_eosin, inner_scale_px, order=[1, 0]) * down_mask
    tys, txs = np.nonzero(down_mask)

    scaled_config = replace(
        config,
        defect_grid_step_px=max(
            1, round(config.defect_grid_step_px / factor)
        ),
        min_edge_distance_px=config.min_edge_distance_px / factor,
        defect_cluster_radius_px=config.defect_cluster_radius_px / factor,
    )
    density = {
        float(sigma): gaussian_filter(down_eosin, float(sigma) / factor)
        for sigma in config.sigmas_px
    }
    return {
        "factor": factor,
        "mask": down_mask,
        "gx": gx,
        "gy": gy,
        "tys": tys,
        "txs": txs,
        "config": scaled_config,
        "density": density,
        "sigmas": tuple(float(s) for s in config.sigmas_px),
    }


def _count_collagen_defects(
    gx: np.ndarray, gy: np.ndarray, prepared: dict
) -> tuple[int, int, int]:
    """Detect collagen defects for a given gradient arrangement."""
    factor = prepared["factor"]
    sxx, syy, sxy = gx * gx, gy * gy, gx * gy

    detections: list[pd.DataFrame] = []
    for sigma in prepared["sigmas"]:
        scaled_sigma = sigma / factor
        jxx = gaussian_filter(sxx, scaled_sigma)
        jyy = gaussian_filter(syy, scaled_sigma)
        jxy = gaussian_filter(sxy, scaled_sigma)
        coherence = np.clip(
            np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-12), 0, 1
        )
        theta = ((0.5 * np.arctan2(2 * jxy, jxx - jyy)) + np.pi / 2) % np.pi
        field = {
            "density": prepared["density"][sigma],
            "order": coherence,
            "theta": theta,
        }
        detected = detect_defects_single_scale(
            field, prepared["mask"], prepared["config"]
        )
        if not detected.empty:
            detected["sigma_px"] = sigma
            detections.append(detected)

    if detections:
        raw = pd.concat(detections, ignore_index=True)
    else:
        raw = pd.DataFrame()
    defects = cluster_multiscale_defects(
        raw, len(prepared["sigmas"]), prepared["config"]
    )
    if defects.empty:
        return 0, 0, 0
    return (
        len(defects),
        int((defects["charge"] == 0.5).sum()),
        int((defects["charge"] == -0.5).sum()),
    )


def run_collagen_null_model(
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    n_permutations: int = 199,
    downsample: int = 3,
    inner_scale_px: float = 1.5,
    seed: int = 0,
) -> dict:
    """
    Permutation null for the collagen (structure-tensor) defect count.

    The collagen field is dense, so orientations cannot be shuffled per nucleus.
    The analogue is to shuffle the per-pixel eosin gradient vectors among tissue
    positions and then coarse-grain: this destroys spatial fiber coherence while
    preserving the local orientation marginal and the collagen density field.
    Detection is identical to the collagen pipeline, and both the observed
    statistic and every permutation run through the same (down-sampled) path.

    Returns the same keys as ``run_null_model`` plus ``null_totals``.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    prepared = _prepare_collagen(
        eosin, tissue_mask, config, downsample, inner_scale_px
    )
    gx, gy = prepared["gx"], prepared["gy"]
    tys, txs = prepared["tys"], prepared["txs"]

    observed_total, observed_plus, observed_minus = _count_collagen_defects(
        gx, gy, prepared
    )

    values_x = gx[tys, txs]
    values_y = gy[tys, txs]
    rng = np.random.default_rng(seed)
    null_totals = np.empty(n_permutations, dtype=int)
    for index in range(n_permutations):
        order = rng.permutation(values_x.size)
        gx_perm = gx.copy()
        gy_perm = gy.copy()
        gx_perm[tys, txs] = values_x[order]
        gy_perm[tys, txs] = values_y[order]
        null_totals[index] = _count_collagen_defects(gx_perm, gy_perm, prepared)[0]

    null_mean = float(null_totals.mean())
    null_std = float(null_totals.std(ddof=1)) if n_permutations > 1 else 0.0
    z_score = (
        (observed_total - null_mean) / null_std
        if null_std > 0
        else float("nan")
    )
    log2_enrichment = (
        float(np.log2(observed_total / null_mean))
        if observed_total > 0 and null_mean > 0
        else float("nan")
    )
    p_depletion = (1 + int(np.sum(null_totals <= observed_total))) / (
        n_permutations + 1
    )
    p_enrichment = (1 + int(np.sum(null_totals >= observed_total))) / (
        n_permutations + 1
    )
    p_two_sided = min(1.0, 2 * min(p_depletion, p_enrichment))
    if observed_total < null_mean:
        direction = "depleted"
    elif observed_total > null_mean:
        direction = "enriched"
    else:
        direction = "equal"

    return {
        "source": "collagen",
        "n_permutations": n_permutations,
        "null_mode": "gradient_shuffle",
        "null_downsample": prepared["factor"],
        "null_seed": seed,
        "observed_total": observed_total,
        "observed_plus_half": observed_plus,
        "observed_minus_half": observed_minus,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_median": float(np.median(null_totals)),
        "null_q2_5": float(np.percentile(null_totals, 2.5)),
        "null_q97_5": float(np.percentile(null_totals, 97.5)),
        "z_score": z_score,
        "log2_enrichment": log2_enrichment,
        "p_depletion": p_depletion,
        "p_enrichment": p_enrichment,
        "p_two_sided": p_two_sided,
        "direction": direction,
        "null_totals": null_totals,
    }
