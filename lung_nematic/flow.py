"""Collective texture flow and its alignment with a nematic director field.

Phase-contrast optical flow follows image texture, not labelled cell identities.
Accordingly every output is named *collective flow*. Calling it single-cell
velocity would require segmentation or particle tracking that these images do
not provide.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.registration import optical_flow_tvl1

from .phase_contrast import (
    cell_texture_mask,
    flatten_illumination,
    phase_contrast_field,
)


def collective_flow(
    first: np.ndarray,
    second: np.ndarray,
    *,
    downsample: int = 4,
    smoothing_px: float = 12.0,
    director_sigma_px: float = 18.0,
    subtract_median_translation: bool = True,
) -> dict[str, np.ndarray | float]:
    """Estimate dense flow in original-pixel units between two frames.

    TV-L1 is run on a reduced grid because full-resolution 2048x1536 fields are
    unnecessarily expensive for collective motion. Displacements are scaled
    back to original pixels. Median translation is returned separately and,
    by default, removed as a conservative stage-drift correction; this may also
    remove genuine whole-sheet translation, so both raw drift and corrected
    flow are exported for audit.
    """
    if isinstance(downsample, bool) or downsample < 1:
        raise ValueError("downsample must be a positive integer")
    if smoothing_px < 0 or director_sigma_px <= 0:
        raise ValueError("smoothing_px must be non-negative and sigma positive")

    scale = 1.0 / int(downsample)
    a = flatten_illumination(first)
    b = flatten_illumination(second)
    if downsample > 1:
        a = zoom(a, scale, order=1, prefilter=False)
        b = zoom(b, scale, order=1, prefilter=False)
    a = _robust_unit_interval(a)
    b = _robust_unit_interval(b)

    v_small, u_small = optical_flow_tvl1(a, b)
    u_raw = u_small.astype(np.float32) * downsample
    v_raw = v_small.astype(np.float32) * downsample

    # The field is estimated on the same grid as flow, avoiding interpolation
    # of a modulo-pi angle. Sigma is converted to that grid explicitly.
    first_small = _resize_rgb(first, a.shape)
    second_small = _resize_rgb(second, a.shape)
    mask_scale = max(2.0, 6.0 / downsample)
    coverage_first = cell_texture_mask(first_small, sigma_px=mask_scale)
    coverage_second = cell_texture_mask(second_small, sigma_px=mask_scale)
    field = phase_contrast_field(
        first_small,
        max(director_sigma_px / downsample, 1.0),
        mask=coverage_first,
    )
    mask = coverage_first & coverage_second & np.isfinite(a) & np.isfinite(b)
    drift_u = float(np.median(u_raw[mask]))
    drift_v = float(np.median(v_raw[mask]))
    u = u_raw - drift_u if subtract_median_translation else u_raw.copy()
    v = v_raw - drift_v if subtract_median_translation else v_raw.copy()
    sigma_small = smoothing_px / downsample
    if sigma_small > 0:
        u = gaussian_filter(u, sigma_small)
        v = gaussian_filter(v, sigma_small)

    speed = np.hypot(u, v)
    flow_theta = np.arctan2(v, u)
    alignment = np.cos(2 * (flow_theta - field["theta"]))
    alignment = np.where(speed > 1e-6, alignment, np.nan)
    return {
        "u_px_per_frame": u.astype(np.float32),
        "v_px_per_frame": v.astype(np.float32),
        "speed_px_per_frame": speed.astype(np.float32),
        "flow_theta_rad": flow_theta.astype(np.float32),
        "director_theta_rad": field["theta"].astype(np.float32),
        "local_order": field["order"].astype(np.float32),
        "flow_director_alignment": alignment.astype(np.float32),
        "mask": mask,
        "drift_u_px_per_frame": drift_u,
        "drift_v_px_per_frame": drift_v,
    }


def flow_summary(
    result: dict[str, np.ndarray | float],
    microns_per_pixel: float,
    seconds_per_frame: float,
) -> dict[str, float]:
    """Summarise one interval with calibrated speed and nematic alignment."""
    if microns_per_pixel <= 0 or seconds_per_frame <= 0:
        raise ValueError("spatial and temporal calibration must be positive")
    mask = np.asarray(result["mask"], dtype=bool)
    speed = np.asarray(result["speed_px_per_frame"], dtype=float)[mask]
    alignment = np.asarray(result["flow_director_alignment"], dtype=float)[mask]
    speed_um_min = speed * microns_per_pixel * 60.0 / seconds_per_frame
    finite_alignment = alignment[np.isfinite(alignment)]
    return {
        "mean_speed_um_per_min": float(np.mean(speed_um_min)),
        "median_speed_um_per_min": float(np.median(speed_um_min)),
        "p90_speed_um_per_min": float(np.percentile(speed_um_min, 90)),
        "mean_flow_director_alignment": (
            float(np.mean(finite_alignment)) if finite_alignment.size else np.nan
        ),
        "fraction_parallel": (
            float(np.mean(finite_alignment > 0.5))
            if finite_alignment.size else np.nan
        ),
        "fraction_perpendicular": (
            float(np.mean(finite_alignment < -0.5))
            if finite_alignment.size else np.nan
        ),
        "drift_u_px_per_frame": float(result["drift_u_px_per_frame"]),
        "drift_v_px_per_frame": float(result["drift_v_px_per_frame"]),
    }


def _robust_unit_interval(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image)
    factors = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    if array.ndim == 3:
        factors += (1.0,)
    return zoom(array, factors, order=1, prefilter=False)
