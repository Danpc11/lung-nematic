from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def compute_nematic_field(
    nuclei: pd.DataFrame,
    image_shape: tuple[int, int],
    sigma_px: float,
) -> dict[str, np.ndarray]:
    """
    Estimate a smooth 2D nematic tensor field from nuclear orientations.
    """
    height, width = image_shape
    density_impulse = np.zeros((height, width), dtype=np.float32)
    qxx_impulse = np.zeros_like(density_impulse)
    qxy_impulse = np.zeros_like(density_impulse)

    if nuclei.empty:
        return {
            "density": density_impulse,
            "order": density_impulse.copy(),
            "theta": density_impulse.copy(),
        }

    xs = np.clip(
        np.rint(nuclei["x_px"]).astype(int),
        0,
        width - 1,
    )
    ys = np.clip(
        np.rint(nuclei["y_px"]).astype(int),
        0,
        height - 1,
    )

    for x, y, theta, weight in zip(
        xs,
        ys,
        nuclei["theta_rad"].to_numpy(),
        nuclei["anisotropy_weight"].to_numpy(),
    ):
        density_impulse[y, x] += weight
        qxx_impulse[y, x] += weight * np.cos(2 * theta)
        qxy_impulse[y, x] += weight * np.sin(2 * theta)

    density = gaussian_filter(
        density_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )
    qxx = gaussian_filter(
        qxx_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )
    qxy = gaussian_filter(
        qxy_impulse,
        sigma=float(sigma_px),
        mode="constant",
    )

    order = np.sqrt(qxx**2 + qxy**2) / (density + 1e-12)
    theta = (0.5 * np.arctan2(qxy, qxx)) % np.pi

    return {
        "density": density,
        "order": np.clip(order, 0, 1),
        "theta": theta,
    }


def compute_global_order(nuclei: pd.DataFrame) -> float:
    if nuclei.empty:
        return float("nan")

    weights = nuclei["anisotropy_weight"].to_numpy()
    angles = nuclei["theta_rad"].to_numpy()

    if weights.sum() <= 0:
        return float("nan")

    value = np.sum(
        weights * np.exp(1j * 2 * angles)
    ) / weights.sum()
    return float(abs(value))


def get_density_threshold(
    density: np.ndarray,
    tissue_mask: np.ndarray,
    quantile: float,
) -> float:
    values = density[(density > 0) & tissue_mask]
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, quantile))


def compute_global_order_from_field(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray | None = None,
) -> float:
    """Density-weighted global nematic order of an orientation field.

    ``compute_global_order`` reads nuclear orientations directly and therefore
    returns the same number whatever field is being analysed. That made
    ``global_nematic_order_S`` field-invariant: nuclear, collagen and fused runs
    over one image reported byte-identical values, so the column described
    nuclei even when the run was labelled collagen.

    This computes the order from the field itself:

        S = |sum_tissue rho * S_local * exp(2i*theta)| / sum_tissue rho

    Because Gaussian smoothing is linear and conserves mass, on the nuclear
    field this reproduces the nuclei-based value up to boundary losses, so old
    and new nuclear numbers stay comparable. On the collagen and fused fields it
    finally measures what its name claims.
    """
    density = np.asarray(field["density"], dtype=float)
    order = np.asarray(field["order"], dtype=float)
    theta = np.asarray(field["theta"], dtype=float)

    if tissue_mask is not None:
        selection = np.asarray(tissue_mask, dtype=bool)
        density = density[selection]
        order = order[selection]
        theta = theta[selection]

    total = float(density.sum())
    if total <= 0 or density.size == 0:
        return float("nan")

    resultant = np.sum(density * order * np.exp(2j * theta))
    return float(min(abs(resultant) / total, 1.0))


def global_order_null(
    oriented_nuclei: pd.DataFrame,
    n_permutations: int = 199,
    seed: int = 0,
) -> dict[str, float]:
    """Finite-size floor for the global order, by permutation of the source.

    ``S`` is biased upward at finite sample size: for orientations drawn at
    random it converges not to 0 but to roughly ``1/sqrt(N_eff)``.  Dividing by
    that floor is *not* a correction: ``S / floor`` grows as ``sqrt(N_eff)`` at
    fixed biological alignment.  Instead we debias the squared resultant.  If
    ``q = sum(w^2) / sum(w)^2``, then

        E[S^2] = q + (1 - q) * S_population^2

    and the method-of-moments estimate is

        S_debiased = sqrt(max(0, (S^2 - q) / (1 - q))).

    ``global_order_excess`` is retained as a compatibility alias for this
    corrected effect size; it no longer means the old, sample-size-dependent
    ratio. ``global_order_p`` remains a significance measure and will, as it
    should, gain power with sample size.

    The null reassigns every orientation uniformly on [0, pi) while holding the
    anisotropy weights fixed, so the weight distribution - and hence the
    effective sample size - is exactly preserved and only the alignment is
    destroyed.

    A block-permutation null over the smoothed field was tried first and
    rejected: blocks of any size either stay correlated through the Gaussian
    kernel (anticonservative - 45% of random fields rejected at alpha = 0.05
    with 2-sigma blocks) or become too few to have power. Permuting at the
    source has no such tuning parameter. Calibration on random orientations
    gives mean p = 0.52 and a 4.0% rejection rate at alpha = 0.05, and the
    simulated floor matches the analytic Rayleigh value
    ``sqrt(pi)/2 * sqrt(sum w^2) / sum w`` to three decimals.

    Note the scope: this is the null for the *nuclear* orientation source. For
    collagen and fused fields the source orientations are per-pixel structure
    tensor estimates that this function does not receive, so the pipeline
    reports the null columns as NaN there rather than substituting a null that
    does not apply.
    """
    if oriented_nuclei.empty:
        return _empty_order_null(n_permutations)

    weights = oriented_nuclei["anisotropy_weight"].to_numpy(dtype=float)
    angles = oriented_nuclei["theta_rad"].to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return _empty_order_null(n_permutations)

    observed = float(abs(np.sum(weights * np.exp(2j * angles))) / total)
    finite_sample_q = float(np.sum(weights**2) / total**2)
    if finite_sample_q < 1.0:
        debiased_sq = (observed**2 - finite_sample_q) / (1.0 - finite_sample_q)
        debiased = float(np.sqrt(np.clip(debiased_sq, 0.0, 1.0)))
    else:
        # One effective observation contains no information about population
        # alignment after correcting its unavoidable unit resultant.
        debiased = float("nan")

    rng = np.random.default_rng(int(seed))
    null_values = np.empty(int(n_permutations))
    for index in range(int(n_permutations)):
        shuffled = rng.uniform(0.0, np.pi, size=angles.size)
        null_values[index] = abs(
            np.sum(weights * np.exp(2j * shuffled))
        ) / total

    null_mean = float(null_values.mean())
    return {
        "global_order_observed": observed,
        "global_order_null_mean": null_mean,
        "global_order_debiased": debiased,
        "global_order_excess": debiased,
        "global_order_p": float(
            (1 + int(np.sum(null_values >= observed))) / (1 + n_permutations)
        ),
        "global_order_n_permutations": int(n_permutations),
    }


def expected_order_under_randomness(weights: np.ndarray) -> float:
    """Analytic Rayleigh floor for ``S``, for a quick check without permuting."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.pi) / 2 * np.sqrt((weights**2).sum()) / total)


def _empty_order_null(n_permutations: int) -> dict[str, float]:
    return {
        "global_order_observed": float("nan"),
        "global_order_null_mean": float("nan"),
        "global_order_debiased": float("nan"),
        "global_order_excess": float("nan"),
        "global_order_p": float("nan"),
        "global_order_n_permutations": int(n_permutations),
    }
