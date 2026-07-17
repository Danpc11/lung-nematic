"""
Collagen (fiber) orientation field via the structure tensor.

The nuclear pipeline builds the director field from nuclear long axes, so in
fibrotic regions dominated by acellular eosinophilic collagen -- where nuclei
are sparse but the fiber architecture is exactly what matters mechanically --
the field is under-sampled. This module estimates orientation directly from the
eosin (collagen) channel using the structure tensor, which yields a *dense*
per-pixel director and a coherence map that plays the role of local order.

The output is the same ``{"density", "order", "theta"}`` dict the nuclear
pipeline produces, so it plugs into ``detect_defects_single_scale`` and
``cluster_multiscale_defects`` unchanged. That lets you detect candidate
topological defects in the collagen architecture with the *identical* winding /
multi-scale-persistence machinery used for nuclei, and compare the two.

Convention (validated on synthetic stripes): ``theta`` is the fiber direction,
measured from the x-axis, in ``[0, pi)``. It is the structure-tensor gradient
direction rotated by 90 degrees, since fibers run perpendicular to the dominant
intensity gradient. ``order`` is the structure-tensor coherence in ``[0, 1]``.
``density`` is the smoothed eosin intensity (a proxy for collagen amount), used
by the density threshold so defects are only reported inside collagen-rich
tissue.

Get the eosin channel from ``make_tissue_mask``, which already returns the HED
stain-separated stack: ``eosin = hed[:, :, 1]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import AnalysisConfig
from .defects import cluster_multiscale_defects, single_scale_detections


def compute_collagen_field(
    eosin: np.ndarray,
    sigma_px: float,
    inner_scale_px: float = 1.5,
    tissue_mask: np.ndarray | None = None,
    mask_normalized: bool = False,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Estimate a nematic field from collagen fibers via the structure tensor.

    Parameters
    ----------
    eosin:
        The eosin channel (``hed[:, :, 1]`` from ``make_tissue_mask``).
    sigma_px:
        Integration scale of the structure tensor, i.e. the coarse-graining
        length of the orientation field. Use the same ``sigmas_px`` as the
        nuclear field so the two are comparable.
    inner_scale_px:
        Gradient (noise) scale. Small; suppresses pixel noise before the
        derivative without smearing fiber orientation.
    tissue_mask, mask_normalized:
        When ``mask_normalized`` is True and a mask is given, the structure
        tensor is integrated with mask-normalized convolution
        (``gauss(field * mask) / gauss(mask)``). This confines fiber-orientation
        integration to tissue so the field does not mix orientations across an
        alveolar lumen.

    Returns
    -------
    dict
        ``{"density", "order", "theta"}`` with the same meaning the nuclear
        pipeline uses, so it is accepted by ``detect_defects_single_scale``.
    """
    image = eosin.astype(np.float32)

    # Gaussian derivatives at the inner scale. Axis 0 is rows (y), axis 1 is
    # columns (x); order=[0, 1] is d/dx, order=[1, 0] is d/dy.
    gx = gaussian_filter(image, inner_scale_px, order=[0, 1])
    gy = gaussian_filter(image, inner_scale_px, order=[1, 0])

    sigma = float(sigma_px)
    if mask_normalized and tissue_mask is not None:
        support = tissue_mask.astype(np.float32)
        gx = gx * support
        gy = gy * support
        normaliser = np.maximum(gaussian_filter(support, sigma), eps)
        jxx = gaussian_filter(gx * gx, sigma) / normaliser
        jyy = gaussian_filter(gy * gy, sigma) / normaliser
        jxy = gaussian_filter(gx * gy, sigma) / normaliser
        density = gaussian_filter(image * support, sigma) / normaliser
    else:
        jxx = gaussian_filter(gx * gx, sigma)
        jyy = gaussian_filter(gy * gy, sigma)
        jxy = gaussian_filter(gx * gy, sigma)
        density = gaussian_filter(image, sigma)

    # Dominant gradient direction, then rotate 90 deg to get the fiber direction.
    theta_gradient = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    theta = (theta_gradient + np.pi / 2) % np.pi

    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + eps)

    return {
        "density": density,
        "order": np.clip(coherence, 0, 1),
        "theta": theta,
    }


def detect_multiscale_collagen_defects(
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
    inner_scale_px: float = 1.5,
) -> tuple[pd.DataFrame, dict[float, dict[str, np.ndarray]], pd.DataFrame]:
    """
    Multi-scale candidate defect detection on the collagen orientation field.

    Mirrors ``detect_multiscale_defects`` but sources the field from the
    structure tensor of the eosin channel instead of nuclear orientations.
    Detection and multi-scale persistence are the same functions, so the
    ``+/-1/2`` candidates are directly comparable to the nuclear ones.

    Returns
    -------
    defects:
        Persistent candidate defects (same schema as the nuclear pipeline).
    fields:
        The collagen field at each smoothing scale.
    raw_detections:
        Per-scale detections before persistence clustering.
    """
    fields: dict[float, dict[str, np.ndarray]] = {}
    all_detections: list[pd.DataFrame] = []

    for sigma in config.sigmas_px:
        field = compute_collagen_field(
            eosin,
            sigma,
            inner_scale_px,
            tissue_mask=tissue_mask,
            mask_normalized=config.mask_normalized_smoothing,
        )
        fields[float(sigma)] = field

        detections = single_scale_detections(field, tissue_mask, config)
        if not detections.empty:
            detections["sigma_px"] = float(sigma)
            all_detections.append(detections)

    if all_detections:
        raw_detections = pd.concat(all_detections, ignore_index=True)
    else:
        raw_detections = pd.DataFrame()

    defects = cluster_multiscale_defects(
        raw_detections, len(config.sigmas_px), config
    )
    return defects, fields, raw_detections


def compute_global_collagen_order(
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    sigma_px: float,
    inner_scale_px: float = 1.5,
) -> float:
    """Coherence-weighted global fiber order S in [0, 1] within tissue."""
    field = compute_collagen_field(eosin, sigma_px, inner_scale_px)
    weights = field["order"][tissue_mask]
    angles = field["theta"][tissue_mask]
    if weights.sum() <= 0:
        return float("nan")
    value = np.sum(weights * np.exp(2j * angles)) / weights.sum()
    return float(abs(value))
