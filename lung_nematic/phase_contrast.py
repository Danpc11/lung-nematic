"""
Phase-contrast analysis of NHLF on defined-stiffness substrates.

The histology pipeline reads orientation from nuclei or from the eosin channel.
Live fibroblasts on a polyacrylamide gel have neither: they are unstained
elongated cells in phase contrast. But the object the pipeline actually needs -
a nematic director field - is recoverable here from exactly the same structure
tensor the collagen route already uses, because an image of aligned elongated
cells has the same oriented-texture statistics as an image of aligned fibers.

So this module does not reinvent the detector. It builds a director field from
phase contrast, hands it to the existing ``detect_defects_single_scale`` and the
existing null model, and adds the one thing the gels make possible that fixed
histology does not: a calibration of nematic order against a *known* substrate
stiffness.

That calibration runs one way only. Order rises with stiffness in a system where
stiffness is set by the experimenter, so a monotonic order(E) curve can be
fitted. Reading it backwards - inferring a stiffness from an order measured in
tissue - is an estimate bounded by that curve's scatter, not a measurement, and
the code labels it as such wherever it is produced.

Nothing here needs microns unless defect *densities* are wanted. Order,
coherence and correlation length in cell-diameter units are dimensionless and
compare across the gels and the histology directly; anything in mm^-2 does not,
and requires the per-system scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (
    binary_closing, binary_fill_holes, gaussian_filter, uniform_filter,
)

from .collagen_field import compute_collagen_field
from .config import AnalysisConfig
from .defects import (
    cluster_multiscale_defects,
    single_scale_detections,
)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse an RGB phase-contrast frame to a single intensity channel."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.shape[-1] >= 3:
        return array[..., :3].mean(axis=-1)
    return array[..., 0]


def flatten_illumination(image: np.ndarray, sigma_px: float = 80.0) -> np.ndarray:
    """Remove slow shading so the structure tensor sees texture, not vignetting.

    Phase-contrast frames often carry a broad intensity gradient across the
    field. Dividing by a heavily blurred copy removes it without touching the
    cell-scale texture the orientation is read from.
    """
    gray = to_grayscale(image)
    background = gaussian_filter(gray, sigma_px)
    flattened = gray / np.maximum(background, 1e-6)
    return flattened - flattened.mean()


def cell_texture_mask(
    image: np.ndarray,
    sigma_px: float = 6.0,
    quantile: float = 0.35,
) -> np.ndarray:
    """Where the frame actually contains cells, from local texture energy.

    Cell-covered regions have high local variance in phase contrast; bare gel is
    smooth. A threshold on local standard deviation separates the two well
    enough for a coverage mask, which the density gate then uses.
    """
    gray = flatten_illumination(image)
    local_mean = uniform_filter(gray, int(sigma_px))
    local_sq = uniform_filter(gray * gray, int(sigma_px))
    variance = np.clip(local_sq - local_mean**2, 0.0, None)
    energy = np.sqrt(variance)
    threshold = np.quantile(energy, quantile)
    raw = energy > threshold
    # Close small gaps and fill holes so the mask is a coverage *envelope*, not
    # a stipple of individual cells. The detector measures distance to the mask
    # edge; on a stippled mask almost every point is near an edge and all
    # defects are rejected, whereas the real boundary is the edge of the
    # confluent sheet.
    return binary_closing(raw, iterations=max(1, int(sigma_px)))


def coverage_envelope(mask: np.ndarray, sigma_px: float = 6.0) -> np.ndarray:
    """Filled envelope of a coverage mask, for edge-distance purposes only.

    The unfilled coverage mask gates on density (is a cell here?); the filled
    envelope defines the boundary of the confluent sheet, so edge distance
    measures distance to the sheet edge rather than to gaps between cells. The
    two must be kept separate: filling the mask that feeds the density gate
    flattens it and rejects every defect.
    """
    return binary_fill_holes(binary_closing(mask, iterations=max(1, int(sigma_px))))


# --------------------------------------------------------------------------
# orientation field
# --------------------------------------------------------------------------
def phase_contrast_field(
    image: np.ndarray,
    sigma_px: float,
    inner_scale_px: float = 1.5,
    mask: np.ndarray | None = None,
    mask_normalized: bool = True,
    flatten_sigma_px: float = 80.0,
) -> dict[str, np.ndarray]:
    """Nematic director field of aligned cells, via the structure tensor.

    Returns ``{"density", "order", "theta"}`` in the same form the histology
    fields use, so every downstream detector accepts it unchanged. ``order`` is
    the structure-tensor coherence, which is the local nematic order parameter
    of the texture; ``density`` is the smoothed flattened intensity, used only
    by the density gate.
    """
    flattened = flatten_illumination(image, flatten_sigma_px)
    # Shift into a positive range so the density gate behaves as it does on the
    # eosin channel, which is non-negative.
    intensity = flattened - flattened.min()
    result = compute_collagen_field(
        intensity,
        sigma_px=sigma_px,
        inner_scale_px=inner_scale_px,
        tissue_mask=mask,
        mask_normalized=mask_normalized,
    )
    # The detector gates on `density > quantile(density)`. On the eosin channel
    # that separates tissue from lumen, but a confluent monolayer has nearly
    # uniform intensity, so an intensity-based gate rejects the whole frame -
    # the same failure the synthetic uniform-density fields exposed. Here the
    # variable that separates signal from background is cell *coverage*, so the
    # smoothed coverage mask is used as the gating density. Orientation and
    # order are untouched; only what the detector counts as "present" changes.
    if mask is not None:
        # Density gate reads cell coverage (unfilled), so bare-gel gaps score
        # low and covered regions score high - a real contrast the quantile
        # gate can bite on, unlike near-uniform intensity.
        result["density"] = gaussian_filter(mask.astype(np.float32), sigma_px * 0.5) + 1e-3
    return result


def analyze_phase_contrast(
    image: np.ndarray,
    config: AnalysisConfig,
    representative_sigma_px: float | None = None,
    coverage_quantile: float = 0.35,
) -> dict:
    """Full single-image analysis: field, order, defects, coverage.

    This is the phase-contrast counterpart of ``analyze_image`` for one gel
    frame. It deliberately returns the field and mask alongside the summary so a
    caller can draw the director (see ``lung_nematic.visualization``) without
    recomputing anything.
    """
    gray = to_grayscale(image)
    coverage = cell_texture_mask(image, quantile=coverage_quantile)
    mask = coverage_envelope(coverage)   # sheet boundary for edge distance

    sigmas = list(config.sigmas_px)
    if representative_sigma_px is None:
        representative_sigma_px = float(sigmas[len(sigmas) // 2])

    # Run detection at every configured scale and keep only defects that
    # persist across them, exactly as the collagen route does. Passing a single
    # scale while the config demands persistence across two silently rejects
    # everything, so persistence is handled here rather than left to a single
    # detect call.
    import pandas as _pd
    fields_by_scale: dict[float, dict] = {}
    per_scale: list = []
    for sigma in config.sigmas_px:
        scale_field = phase_contrast_field(image, float(sigma), mask=coverage)
        # gate on coverage, but let defects live anywhere inside the sheet
        scale_field = dict(scale_field)
        fields_by_scale[float(sigma)] = scale_field
        detections = single_scale_detections(scale_field, mask, config)
        if not detections.empty:
            detections["sigma_px"] = float(sigma)
            per_scale.append(detections)
    raw = _pd.concat(per_scale, ignore_index=True) if per_scale else _pd.DataFrame()
    defects = cluster_multiscale_defects(raw, len(config.sigmas_px), config)
    field = fields_by_scale[representative_sigma_px] if representative_sigma_px in fields_by_scale \
        else phase_contrast_field(image, representative_sigma_px, mask=mask)

    order_in_cells = field["order"][mask]
    global_S = float(np.sqrt(
        np.mean(np.cos(2 * field["theta"][mask])) ** 2
        + np.mean(np.sin(2 * field["theta"][mask])) ** 2
    )) if mask.any() else float("nan")

    n_plus = int((defects["charge"] == 0.5).sum()) if not defects.empty else 0
    n_minus = int((defects["charge"] == -0.5).sum()) if not defects.empty else 0

    return {
        "field": field,
        "mask": mask,
        "defects": defects,
        "global_nematic_order_S": global_S,
        "local_S_median": float(np.median(order_in_cells)) if order_in_cells.size else float("nan"),
        "local_S_mean": float(np.mean(order_in_cells)) if order_in_cells.size else float("nan"),
        "coverage_fraction": float(mask.mean()),
        "n_plus_half": n_plus,
        "n_minus_half": n_minus,
        "n_defects_total": n_plus + n_minus,
        "net_topological_charge": 0.5 * (n_plus - n_minus),
        "correlation_length_px": orientation_correlation_length(field, mask),
        "representative_sigma_px": representative_sigma_px,
    }


def orientation_correlation_length(
    field: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    max_lag_px: int = 600,
) -> float:
    """Distance over which the director stays aligned, from the nematic
    autocorrelation of the doubled angle.

    Reported in pixels; divide by cell length (gels) or alveolar diameter
    (histology) to get a dimensionless, cross-comparable number.
    """
    theta = field["theta"]
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    if mask is not None:
        c2 = np.where(mask, c2, 0.0)
        s2 = np.where(mask, s2, 0.0)

    # radial average of the orientation autocorrelation via the FFT
    def autocorr(component: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft2(component)) ** 2
        return np.fft.irfft2(spectrum, s=component.shape)

    correlation = autocorr(c2) + autocorr(s2)
    correlation = np.fft.fftshift(correlation)
    correlation /= correlation.max() + 1e-12

    centre = np.array(correlation.shape) // 2
    yy, xx = np.indices(correlation.shape)
    radius = np.hypot(yy - centre[0], xx - centre[1]).astype(int)
    radial = np.bincount(radius.ravel(), correlation.ravel()) / np.maximum(
        np.bincount(radius.ravel()), 1
    )
    radial = radial[: min(max_lag_px, radial.size)]

    below = np.nonzero(radial < 1.0 / np.e)[0]
    if below.size:
        return float(below[0])
    # never decorrelated within the window: report the window, flagged by sign
    return -float(radial.size)


# --------------------------------------------------------------------------
# stiffness <-> order calibration
# --------------------------------------------------------------------------
@dataclass
class StiffnessOrderCalibration:
    """A fitted, monotonic order(stiffness) relation from the gel series.

    The forward direction, order as a function of a known substrate stiffness,
    is a fit to controlled data. The inverse, ``estimate_stiffness``, is an
    interpolation *bounded by that fit* and returns a range, never a point,
    because tissue order carries scatter the gels never resolve.
    """

    stiffness_kPa: np.ndarray
    order: np.ndarray
    order_std: np.ndarray | None = None
    _slope: float = field(default=0.0, init=False)
    _intercept: float = field(default=0.0, init=False)
    _log_fit: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.stiffness_kPa = np.asarray(self.stiffness_kPa, dtype=float)
        self.order = np.asarray(self.order, dtype=float)
        # order tends to rise with log-stiffness over the physiological decade,
        # so fit S = a * log10(E) + b unless that is degenerate
        x = np.log10(self.stiffness_kPa) if self._log_fit else self.stiffness_kPa
        if np.ptp(x) > 0:
            self._slope, self._intercept = np.polyfit(x, self.order, 1)

    def predict_order(self, stiffness_kPa: float | np.ndarray) -> np.ndarray:
        x = np.log10(np.asarray(stiffness_kPa, dtype=float))
        return self._slope * x + self._intercept

    def estimate_stiffness(
        self,
        order_value: float,
        order_uncertainty: float | None = None,
    ) -> dict:
        """Invert the fit. Returns a central estimate AND a range.

        This is the operation that reads a stiffness off a histology order
        value. It is an estimate, not a measurement: the returned interval
        reflects the gel scatter and any supplied measurement uncertainty, and
        the caller is expected to report the interval, not the point.
        """
        if self._slope == 0:
            return {"stiffness_kPa": float("nan"), "low_kPa": float("nan"),
                    "high_kPa": float("nan"), "in_calibrated_range": False}

        def invert(order: float) -> float:
            return float(10 ** ((order - self._intercept) / self._slope))

        central = invert(order_value)
        spread = order_uncertainty
        if spread is None and self.order_std is not None:
            spread = float(np.mean(self.order_std))
        spread = spread or 0.0
        low = invert(order_value + spread)   # higher order -> stiffer, so signs
        high = invert(order_value - spread)  # flip depending on slope sign
        low, high = sorted((low, high))

        calibrated_min = float(self.stiffness_kPa.min())
        calibrated_max = float(self.stiffness_kPa.max())
        return {
            "stiffness_kPa": central,
            "low_kPa": low,
            "high_kPa": high,
            "in_calibrated_range": bool(calibrated_min <= central <= calibrated_max),
            "calibrated_range_kPa": (calibrated_min, calibrated_max),
        }

    @property
    def r_squared(self) -> float:
        predicted = self.predict_order(self.stiffness_kPa)
        ss_res = float(np.sum((self.order - predicted) ** 2))
        ss_tot = float(np.sum((self.order - self.order.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_calibration(
    stiffness_kPa: list[float],
    order_values: list[float],
    order_std: list[float] | None = None,
) -> StiffnessOrderCalibration:
    """Fit order(stiffness) from paired gel measurements.

    Pass one order value per gel stiffness (or per replicate, repeating the
    stiffness). At least two distinct stiffnesses are needed for a slope; three
    or more let ``r_squared`` mean something.
    """
    stiffness = np.asarray(stiffness_kPa, dtype=float)
    order = np.asarray(order_values, dtype=float)
    if stiffness.size < 2 or np.unique(stiffness).size < 2:
        raise ValueError("need at least two distinct stiffnesses to fit order(E).")
    std = np.asarray(order_std, dtype=float) if order_std is not None else None
    return StiffnessOrderCalibration(stiffness, order, std)


# --------------------------------------------------------------------------
# batch driver
# --------------------------------------------------------------------------
def analyze_gel_series(
    image_paths: dict,
    config: AnalysisConfig,
    output_dir=None,
    draw_fields: bool = True,
) -> "pd.DataFrame":
    """Analyze a stiffness series of phase-contrast frames into one table.

    ``image_paths`` maps a stiffness in kPa to an image path (or a list of
    replicate paths). Returns one row per frame with order, coherence,
    correlation length and defect counts, and - when at least two stiffnesses
    are present - attaches a fitted ``order(E)`` calibration as an attribute.

    When ``output_dir`` is given and ``draw_fields`` is True, a director overlay
    is written per frame using the histology visualiser, so the gels and the
    tissue are drawn the same way.
    """
    import pandas as pd
    from pathlib import Path

    from .io_utils import read_rgb
    from .visualization import save_overlay

    rows = []
    for stiffness_kPa, paths in image_paths.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for index, path in enumerate(paths):
            rgb = read_rgb(str(path))
            result = analyze_phase_contrast(rgb, config)
            row = {k: v for k, v in result.items()
                   if k not in ("field", "mask", "defects")}
            row["stiffness_kPa"] = float(stiffness_kPa)
            row["path"] = str(path)
            row["replicate"] = index
            rows.append(row)

            if output_dir is not None and draw_fields:
                out = Path(output_dir) / f"field_{stiffness_kPa}kPa_{index}.png"
                save_overlay(
                    rgb, result["mask"], result["field"], result["defects"],
                    out, config, title=f"NHLF, {stiffness_kPa} kPa",
                )

    frame = pd.DataFrame(rows)

    distinct = frame["stiffness_kPa"].unique()
    if distinct.size >= 2:
        grouped = frame.groupby("stiffness_kPa")["local_S_median"]
        calibration = build_calibration(
            list(grouped.mean().index),
            list(grouped.mean().values),
            list(grouped.std().fillna(0.0).values),
        )
        frame.attrs["calibration"] = calibration
        frame.attrs["calibration_r_squared"] = calibration.r_squared
    return frame
