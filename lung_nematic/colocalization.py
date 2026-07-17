"""
Defect - local-order colocalization test.

A candidate defect is more interesting if it sits inside an organised structure
(a fibroblastic focus, an aligned collagen bundle) than in featureless
parenchyma. This test asks whether the detected defects fall in regions of
higher local order than random tissue locations would.

For a given field it takes the local order ``S`` at each defect, averages it,
and compares that average against a bootstrap null built by scoring the same
number of random valid locations many times. "Valid" means the same gate the
detector uses: inside tissue, above the density threshold, and away from the
edge. The score field defaults to the field's local order but any per-pixel map
can be supplied (e.g. collagen density, or a focus score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from .config import AnalysisConfig
from .nematic import get_density_threshold


def _valid_mask(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Locations the detector would accept: tissue, dense, away from edge."""
    threshold = get_density_threshold(
        field["density"], tissue_mask, config.density_quantile
    )
    edge_distance = distance_transform_edt(tissue_mask)
    return (
        tissue_mask
        & (field["density"] > threshold)
        & (edge_distance >= config.min_edge_distance_px)
    )


def run_colocalization(
    defects: pd.DataFrame,
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    score: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Bootstrap test: is local order higher at defects than at random tissue?

    Parameters
    ----------
    defects:
        Candidate defect table (needs ``x_px``, ``y_px``).
    field:
        The field the defects were detected on (``density``/``order``/``theta``).
    score:
        Optional per-pixel score map. Defaults to ``field["order"]``.
    n_bootstrap:
        Number of random location sets drawn for the null.

    Returns
    -------
    dict
        Observed mean score at defects, bootstrap null summary, z-score,
        empirical p-values and the effect direction.
    """
    score_map = field["order"] if score is None else score

    keys = [
        "n_defects",
        "n_bootstrap",
        "observed_mean_score",
        "null_mean",
        "null_std",
        "null_q2_5",
        "null_q97_5",
        "z_score",
        "p_higher",
        "p_lower",
        "p_two_sided",
        "direction",
    ]

    valid = _valid_mask(field, tissue_mask, config)
    valid_ys, valid_xs = np.nonzero(valid)

    if defects.empty or valid_ys.size == 0:
        result = {key: np.nan for key in keys}
        result.update(
            n_defects=int(len(defects)),
            n_bootstrap=n_bootstrap,
            direction="undefined",
            null_scores=np.zeros(0),
        )
        return result

    height, width = tissue_mask.shape
    dx = np.clip(np.rint(defects["x_px"].to_numpy()).astype(int), 0, width - 1)
    dy = np.clip(np.rint(defects["y_px"].to_numpy()).astype(int), 0, height - 1)
    observed_mean = float(np.mean(score_map[dy, dx]))
    n_defects = len(defects)

    valid_scores = score_map[valid_ys, valid_xs]
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        picks = rng.integers(0, valid_scores.size, size=n_defects)
        null_scores[index] = valid_scores[picks].mean()

    null_mean = float(null_scores.mean())
    null_std = float(null_scores.std(ddof=1)) if n_bootstrap > 1 else 0.0
    z_score = (
        (observed_mean - null_mean) / null_std
        if null_std > 0
        else float("nan")
    )
    p_higher = (1 + int(np.sum(null_scores >= observed_mean))) / (
        n_bootstrap + 1
    )
    p_lower = (1 + int(np.sum(null_scores <= observed_mean))) / (
        n_bootstrap + 1
    )
    p_two_sided = min(1.0, 2 * min(p_higher, p_lower))

    if observed_mean > null_mean:
        direction = "higher_order_at_defects"
    elif observed_mean < null_mean:
        direction = "lower_order_at_defects"
    else:
        direction = "equal"

    return {
        "n_defects": n_defects,
        "n_bootstrap": n_bootstrap,
        "observed_mean_score": observed_mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_q2_5": float(np.percentile(null_scores, 2.5)),
        "null_q97_5": float(np.percentile(null_scores, 97.5)),
        "z_score": z_score,
        "p_higher": p_higher,
        "p_lower": p_lower,
        "p_two_sided": p_two_sided,
        "direction": direction,
        "null_scores": null_scores,
    }
