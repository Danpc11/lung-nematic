from __future__ import annotations

import numpy as np
from skimage import color, filters, morphology

from ._compat import remove_small_holes, remove_small_objects


def _safe_otsu(values: np.ndarray, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    if np.allclose(finite.min(), finite.max()):
        return float(finite.min())
    return float(filters.threshold_otsu(finite))


def make_tissue_mask(
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tissue mask and return HED stain-separated channels.

    The mask combines saturation and eosin information so that white
    background is excluded before estimating the nematic field.
    """
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_float)
    hed = color.rgb2hed(rgb_float)

    saturation = hsv[:, :, 1]
    eosin = hed[:, :, 1]

    saturation_threshold = max(_safe_otsu(saturation, 0.03), 0.03)
    positive_eosin = eosin[eosin > 0]
    eosin_threshold = (
        float(np.percentile(positive_eosin, 25))
        if positive_eosin.size
        else 0.005
    )

    mask = (
        (saturation > saturation_threshold)
        | (eosin > eosin_threshold)
    )
    mask = morphology.closing(mask, morphology.disk(3))
    mask = remove_small_objects(mask, 150)
    mask = remove_small_holes(mask, 250)

    return mask.astype(bool), hed
