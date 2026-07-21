"""
Adaptive integration radius for topological defect detection.

The winding of the director is integrated around a loop, and the loop radius is
a compromise. Too small and the winding is dominated by sampling noise; too
large and neighbouring defects merge and real structure is averaged away. A
single fixed radius cannot serve a tissue that contains two cell populations of
very different size - spindle-shaped fibroblasts tens of microns long and
compact epithelial cells a few microns across. A radius spanning four or five
fibroblasts spans many more epithelial cells, so a fixed value over-integrates
in epithelium (erasing real defects) and under-integrates in fibroblast-rich
stroma (counting noise).

The fix is to size the loop from the *local* cell size, so it always encloses a
comparable number of cells - the four-to-five the winding needs to be stable -
whatever population dominates that patch. Local cell size is estimated two ways
here, and the caller can pick:

  * from nuclear spacing, the typical nearest-neighbour distance between
    detected nuclei in a neighbourhood (direct, needs a nucleus mask);
  * from the orientation coherence length, the distance over which the director
    stays aligned (indirect, works on the collagen field alone).

Both return a per-pixel radius map that the detector then reads instead of a
scalar.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def cell_size_from_nuclei(
    nucleus_centroids: np.ndarray,
    shape: tuple[int, int],
    smoothing_px: float = 60.0,
    min_size_px: float = 5.0,
    max_size_px: float = 80.0,
) -> np.ndarray:
    """Local typical cell size, from the spacing between nuclei.

    For each nucleus the nearest-neighbour distance is a proxy for local cell
    size; those distances are splatted onto the frame and smoothed into a dense
    map. Epithelial regions (many close nuclei) get a small size, fibroblast
    stroma (sparse nuclei) a large one.
    """
    height, width = shape
    size_map = np.full(shape, np.nan)
    if len(nucleus_centroids) >= 2:
        from scipy.spatial import cKDTree

        tree = cKDTree(nucleus_centroids)
        # second neighbour: the first is the point itself
        distances, _ = tree.query(nucleus_centroids, k=2)
        nn = distances[:, 1]
        ys = np.clip(nucleus_centroids[:, 1].astype(int), 0, height - 1)
        xs = np.clip(nucleus_centroids[:, 0].astype(int), 0, width - 1)
        accum = np.zeros(shape)
        count = np.zeros(shape)
        np.add.at(accum, (ys, xs), nn)
        np.add.at(count, (ys, xs), 1.0)
        accum = gaussian_filter(accum, smoothing_px)
        count = gaussian_filter(count, smoothing_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            size_map = accum / np.maximum(count, 1e-9)

    # fill any gaps with the global median and clip to a sane band
    if np.isnan(size_map).all():
        size_map = np.full(shape, (min_size_px + max_size_px) / 2)
    else:
        median = np.nanmedian(size_map)
        size_map = np.where(np.isnan(size_map), median, size_map)
    return np.clip(size_map, min_size_px, max_size_px)


def cell_size_from_coherence(
    field: dict[str, np.ndarray],
    tissue_mask: np.ndarray | None = None,
    patch_px: float = 40.0,
    min_size_px: float = 5.0,
    max_size_px: float = 80.0,
) -> np.ndarray:
    """Local cell size proxied by the orientation coherence length.

    Where the director is locally coherent over a long distance the structures
    are large (fibroblasts); where it decorrelates quickly they are small
    (epithelium). The coherence is the magnitude of the locally averaged nematic
    tensor, mapped monotonically onto the size band.
    """
    theta = field["theta"]
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    coherence = np.hypot(
        uniform_filter(c2, int(patch_px)),
        uniform_filter(s2, int(patch_px)),
    )
    # coherence in [0, 1] -> size in [min, max]; high coherence => large cells
    size_map = min_size_px + (max_size_px - min_size_px) * np.clip(coherence, 0, 1)
    if tissue_mask is not None:
        median = float(np.median(size_map[tissue_mask])) if tissue_mask.any() else \
            (min_size_px + max_size_px) / 2
        size_map = np.where(tissue_mask, size_map, median)
    return size_map


def adaptive_radius_map(
    cell_size_px: np.ndarray,
    cells_per_radius: float = 4.5,
    min_radius_px: float = 6.0,
    max_radius_px: float = 60.0,
) -> np.ndarray:
    """Integration radius that encloses ``cells_per_radius`` local cells.

    The winding needs four to five cells around the loop to be stable, so the
    radius is that many local cell lengths, clipped to keep the loop neither
    sub-pixel nor larger than a typical focus.
    """
    radius = cells_per_radius * cell_size_px
    return np.clip(radius, min_radius_px, max_radius_px)


def summarise_radius_map(radius_map: np.ndarray,
                         tissue_mask: np.ndarray | None = None) -> dict:
    """Report the spread of the adaptive radius, for diagnostics and captions."""
    values = radius_map[tissue_mask] if tissue_mask is not None else radius_map.ravel()
    return {
        "radius_min_px": float(np.min(values)),
        "radius_median_px": float(np.median(values)),
        "radius_max_px": float(np.max(values)),
        "radius_iqr_px": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }
