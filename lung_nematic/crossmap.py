"""Bridge the stiffness-controlled gel time lapses to fixed histology.

The problem this solves
-----------------------
Defect counts from the two systems are not comparable as computed. The packaged
histology configuration detects at sigma = 70 px, which at 0.1147 um/px is 8.0
um; a gel frame at sigma = 40 px and 0.7042 um/px is 28.2 um. Since defect
density falls roughly as ``1/sigma^2`` - a coarser kernel averages neighbouring
defects away - a 3.5x mismatch in detection scale produces about a 12x
difference in density on its own, which swamps any biological effect.

Matching sigma in micrometres fixes the arithmetic but not the ambiguity: nuclei
are packed in tissue and spread on a gel, so matching physical length and
matching cell diameters are different choices answering different questions.

The way out is to compare *dimensionless* quantities. The orientational
correlation length ``xi`` is measured rather than chosen, and ``rho * xi^2`` -
defects per correlation area - is dimensionless. In a texture whose defects sit
roughly ``xi`` apart it is of order one, and departures from that are the
informative part. Together with ``S``, which is already dimensionless, it gives
two observables that cross systems without negotiating a calibration.

What the time lapse licenses you to infer
-----------------------------------------
A fixed section is one frozen instant of unknown age, so on its own it cannot be
placed anywhere on a coarsening curve. The gel series supplies the missing axis:

*The sign of the stiffness response.* Naively, stiffer means more contractile
means more active means more defects. But the steady-state density of an active
nematic goes as activity over elasticity, and stiffening the substrate raises
both. Which wins is an empirical question the gel answers and histology cannot.

*An effective stiffness.* Once ``rho*(E)`` is measured across the gel series it
can be inverted, turning a dimensionless density measured on a slide into a
stiffness estimate. This is falsifiable against instrumented microindentation on
an adjacent section, which is the validation that makes it worth doing.

*A staging readout.* Gel trajectories give both the steady-state band and the
pre-steady-state transient. A histological region whose ``rho*`` sits above the
steady-state band of every stiffness has not equilibrated - evidence that the
lesion is still coarsening rather than mature.

*Defect motility from a static image.* Velocity is unmeasurable in fixed tissue.
But if the gel establishes a relation between ``rho*`` (static, measurable in
both) and ``+1/2`` speed (dynamic, gel only), then ``rho*`` in histology implies
a motility. Validate by holding one stiffness out of the fit.

What it does not license
------------------------
Absolute lesion age, absolute in vivo activity, or anything about the
three-dimensional structure. A histological section cuts a 3D disclination
network, where defects are lines rather than points, so its areal density
estimates a line density per volume - not the same quantity as the gel's areal
point density. ``charge_balance`` gives a diagnostic for how badly this bites.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .defects import detect_multiscale_defects
from .nematic import compute_global_order_from_field, compute_nematic_field
from .phase_contrast import analyze_phase_contrast, orientation_correlation_length
from .preprocessing import make_tissue_mask
from .segmentation import segment_nuclei, select_oriented_nuclei

# The multiscale detectors confirm a defect by requiring it to persist across
# scales, so handing them a single sigma disables that filter and they return
# nothing. Each sweep point therefore uses a narrow band around the nominal
# scale: wide enough for persistence to mean something, narrow enough that the
# point still refers to one physical scale.
SCALE_BAND_RATIO = 1.4


def _band(sigma_px: float, ratio: float = SCALE_BAND_RATIO) -> tuple[float, ...]:
    return (float(sigma_px), float(sigma_px) * float(ratio))


def sigmas_for_microns(
    sigmas_um, microns_per_pixel: float
) -> tuple[float, ...]:
    """Convert a physical smoothing scale to pixels for a given calibration.

    Sweeps must be specified in micrometres. Reusing a pixel sigma across
    systems silently changes the physical scale, which is the single largest
    source of incomparability between the two datasets.
    """
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be positive")
    values = tuple(float(s) / microns_per_pixel for s in sigmas_um)
    if any(value < 1.0 for value in values):
        raise ValueError(
            f"sigmas {sigmas_um} um are below one pixel at "
            f"{microns_per_pixel} um/px; the sweep would be meaningless"
        )
    return values


def dimensionless_density(
    defect_density_mm2, correlation_length_um
) -> np.ndarray:
    """``rho * xi^2``: defects per correlation area, dimensionless.

    Returns NaN where the correlation length is unusable. A negative or absurdly
    large ``xi`` means the exponential fit did not converge, usually on a nearly
    perfectly ordered field where the autocorrelation never decays. That value
    is meaningless rather than merely large, and propagating it as a number
    would produce a confident, wrong comparison.
    """
    density = np.asarray(defect_density_mm2, dtype=float)
    xi_um = np.asarray(correlation_length_um, dtype=float)
    xi_mm = xi_um / 1000.0
    valid = np.isfinite(xi_mm) & (xi_mm > 0) & np.isfinite(density)
    out = np.full(density.shape, np.nan)
    np.divide(density * xi_mm**2, 1.0, out=out, where=valid)
    return out


def histology_scale_sweep(
    image: np.ndarray,
    config: AnalysisConfig,
    sigmas_um,
    microns_per_pixel: float,
) -> pd.DataFrame:
    """Defect density, order and correlation length versus physical scale.

    Segmentation runs once; only the smoothing scale varies. That keeps the
    sweep a property of the field rather than of the nuclei detector, so a
    change in the curve means the texture changed and not the segmentation.

    Comparing whole ``rho(sigma)`` curves is more robust than comparing single
    numbers, because the scale at which the density collapses *is* the
    correlation length - so the curve carries its own calibration.
    """
    tissue_mask = make_tissue_mask(image, config)
    nuclei = segment_nuclei(image, tissue_mask, config)
    oriented = select_oriented_nuclei(nuclei, config)
    if oriented.empty:
        raise ValueError("no oriented nuclei; cannot sweep scale")

    shape = image.shape[:2]
    area_mm2 = (
        float(tissue_mask.sum()) * microns_per_pixel**2 / 1_000_000.0
    )

    rows = []
    for sigma_um in sigmas_um:
        sigma_px = float(sigma_um) / microns_per_pixel
        if sigma_px < 1.0:
            continue
        scaled = replace(config, sigmas_px=_band(sigma_px))
        field = compute_nematic_field(oriented, shape, sigma_px)
        defects = detect_multiscale_defects(oriented, shape, tissue_mask, scaled)
        xi_px = orientation_correlation_length(field, tissue_mask)
        rows.append(
            {
                "sigma_um": float(sigma_um),
                "sigma_px": sigma_px,
                "sigma_band_px": _band(sigma_px),
                "n_defects": len(defects),
                "defect_density_mm2": (
                    len(defects) / area_mm2 if area_mm2 > 0 else np.nan
                ),
                "global_S": compute_global_order_from_field(field, tissue_mask),
                "correlation_length_um": xi_px * microns_per_pixel,
                "tissue_area_mm2": area_mm2,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["rho_xi2"] = dimensionless_density(
            table.defect_density_mm2, table.correlation_length_um
        )
    return table


def gel_scale_sweep(
    image: np.ndarray,
    config: AnalysisConfig,
    sigmas_um,
    microns_per_pixel: float,
) -> pd.DataFrame:
    """The same sweep on a phase-contrast frame, for a matched comparison."""
    rows = []
    for sigma_um in sigmas_um:
        sigma_px = float(sigma_um) / microns_per_pixel
        if sigma_px < 1.0:
            continue
        scaled = replace(config, sigmas_px=_band(sigma_px))
        result = analyze_phase_contrast(image, scaled)
        covered_mm2 = (
            result["coverage_fraction"]
            * image.shape[0] * image.shape[1]
            * microns_per_pixel**2 / 1_000_000.0
        )
        rows.append(
            {
                "sigma_um": float(sigma_um),
                "sigma_px": sigma_px,
                "sigma_band_px": _band(sigma_px),
                "n_defects": int(result["n_defects_total"]),
                "defect_density_mm2": (
                    result["n_defects_total"] / covered_mm2
                    if covered_mm2 > 0 else np.nan
                ),
                "global_S": result["global_nematic_order_S"],
                "correlation_length_um": (
                    result["correlation_length_px"] * microns_per_pixel
                ),
                "tissue_area_mm2": covered_mm2,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["rho_xi2"] = dimensionless_density(
            table.defect_density_mm2, table.correlation_length_um
        )
    return table


def steady_state_window(
    kinetics: pd.DataFrame,
    *,
    tail_fraction: float = 0.4,
    balance_tolerance: float = 0.25,
) -> dict:
    """Decide whether a gel series reached a defect steady state, and where.

    This gate is what licenses comparing a fixed section to the gel at all. A
    histological lesion has persisted for months, so the quantity it can be
    compared against is a steady-state level - not an arbitrary point on a
    transient. If the recording never plateaus, the comparison has no temporal
    anchor and should not be made.

    Steadiness is judged on the births-to-deaths balance rather than on a fitted
    decay exponent, because over a short series a plateau and a slow power law
    look alike while their nucleation balance does not.
    """
    if kinetics.empty:
        return {"reached_steady_state": False, "reason": "no frames"}

    cut = int(len(kinetics) * (1.0 - tail_fraction))
    tail = kinetics.iloc[cut:]
    if len(tail) < 3:
        return {"reached_steady_state": False, "reason": "tail too short"}

    births = float(tail["births"].sum())
    deaths = float(tail["deaths"].sum())
    total = births + deaths
    imbalance = abs(births - deaths) / total if total > 0 else 0.0

    counts = tail["n_defects"].to_numpy(dtype=float)
    trend = np.polyfit(np.arange(counts.size), counts, 1)[0]
    relative_trend = (
        abs(trend) * counts.size / counts.mean() if counts.mean() > 0 else np.inf
    )

    steady = imbalance <= balance_tolerance and relative_trend <= 0.25
    return {
        "reached_steady_state": bool(steady),
        "reason": "" if steady else (
            f"birth/death imbalance {imbalance:.2f} "
            f"(tol {balance_tolerance}), relative trend {relative_trend:.2f}"
        ),
        "first_frame": int(tail["frame"].iloc[0]),
        "last_frame": int(tail["frame"].iloc[-1]),
        "mean_n_defects": float(counts.mean()),
        "sd_n_defects": float(counts.std(ddof=1)) if counts.size > 1 else np.nan,
        "births": births,
        "deaths": deaths,
        "imbalance": imbalance,
    }


def calibration_curve(points: pd.DataFrame, response: str = "rho_xi2") -> dict:
    """Fit ``log(response) = a + b * log(stiffness)`` over the gel series.

    A power law is used because the steady-state defect density of an active
    nematic goes as activity over elastic constant, and both are expected to
    follow the substrate stiffness as power laws over a limited range. The
    exponent's *sign* is the result worth reporting: negative means elasticity
    stiffens faster than activity, so denser tissue is more ordered - which is
    the direction the histology appears to show and which a fixed section alone
    could never establish.

    Three stiffness levels give one degree of freedom after fitting two
    parameters. The fit is therefore a description, not a test; ``n_points`` is
    returned so nobody reports an R^2 from it as evidence.
    """
    required = {"stiffness_kPa", response}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    usable = points.dropna(subset=["stiffness_kPa", response])
    usable = usable[(usable["stiffness_kPa"] > 0) & (usable[response] > 0)]
    if len(usable) < 2:
        raise ValueError(
            "need at least two positive (stiffness, response) points to fit"
        )

    x = np.log(usable["stiffness_kPa"].to_numpy(dtype=float))
    y = np.log(usable[response].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)

    predicted = intercept + slope * x
    residual = y - predicted
    ss_total = float(((y - y.mean()) ** 2).sum())
    return {
        "response": response,
        "log_slope": float(slope),
        "log_intercept": float(intercept),
        "n_points": len(usable),
        "residual_sd": float(residual.std(ddof=1)) if len(residual) > 2 else np.nan,
        "r_squared": (
            float(1 - (residual**2).sum() / ss_total) if ss_total > 0 else np.nan
        ),
        "stiffness_range_kPa": (
            float(usable["stiffness_kPa"].min()),
            float(usable["stiffness_kPa"].max()),
        ),
    }


def infer_stiffness(
    observed_response,
    curve: dict,
    *,
    extrapolation_factor: float = 1.5,
    min_log_variation: float = 0.1,
) -> pd.DataFrame:
    """Invert the calibration: a dimensionless density implies a stiffness.

    Values requiring extrapolation beyond ``extrapolation_factor`` times the
    calibrated range are returned but flagged. A power law fitted on 5-23 kPa
    says nothing about 200 kPa, and an unflagged number there would be read as a
    measurement.
    """
    slope = curve["log_slope"]
    low, high = curve["stiffness_range_kPa"]

    # Testing for an exactly zero slope is not enough - a flat calibration fits
    # to about 1e-17, not to 0. What matters is whether the response varies
    # enough across the calibrated range to be inverted at all: the total
    # variation in log units is |slope| * log(high/low), and when that is small
    # the inversion amplifies measurement noise without bound.
    log_range = np.log(high / low) if low > 0 and high > low else 0.0
    variation = abs(slope) * log_range
    if variation < min_log_variation:
        raise ValueError(
            f"calibration is too flat to invert: the response varies by only "
            f"{variation:.3g} in log units across {low:g}-{high:g} kPa "
            f"(need {min_log_variation}). A near-flat response means defect "
            f"density does not track stiffness in this range."
        )

    values = np.atleast_1d(np.asarray(observed_response, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        stiffness = np.exp(
            (np.log(values) - curve["log_intercept"]) / slope
        )
    stiffness = np.where(values > 0, stiffness, np.nan)

    within = (stiffness >= low / extrapolation_factor) & (
        stiffness <= high * extrapolation_factor
    )
    return pd.DataFrame(
        {
            "observed_response": values,
            "inferred_stiffness_kPa": stiffness,
            "within_calibrated_range": within & np.isfinite(stiffness),
        }
    )


def charge_balance(defects: pd.DataFrame) -> dict:
    """Ratio of ``+1/2`` to ``-1/2`` defects - a sectioning diagnostic.

    In a closed two-dimensional active nematic defects are created and destroyed
    in pairs, so the two classes balance. A section through a three-dimensional
    tissue does not enjoy that constraint: it cuts disclination *lines*, and the
    apparent charge depends on the cutting angle. A histological imbalance
    larger than anything the gels show at any stiffness is therefore evidence
    that the section is not a clean two-dimensional slice, and that its density
    is not directly comparable to a gel's.
    """
    if defects.empty:
        return {"n_plus": 0, "n_minus": 0, "ratio": np.nan, "imbalance": np.nan}
    charge = defects["charge"]
    plus = int((charge > 0).sum())
    minus = int((charge < 0).sum())
    total = plus + minus
    return {
        "n_plus": plus,
        "n_minus": minus,
        "ratio": (plus / minus) if minus > 0 else np.nan,
        "imbalance": (abs(plus - minus) / total) if total > 0 else np.nan,
    }
