from __future__ import annotations

import numpy as np
import pandas as pd

from .nematic import (
    compute_global_order,
    compute_global_order_from_field,
    get_density_threshold,
)


def _null_field(stats: dict | None, key: str) -> float:
    """Null-model column, NaN when the null was not run.

    NaN rather than 0 so "the null was not computed" stays distinguishable from
    "the null was computed and came out at zero".
    """
    if stats is None:
        return float("nan")
    return stats.get(key, float("nan"))


def summarize_image(
    metadata: dict,
    image_shape: tuple[int, int],
    tissue_mask: np.ndarray,
    nuclei: pd.DataFrame,
    oriented_nuclei: pd.DataFrame,
    field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    density_quantile: float,
    representative_sigma_px: float,
    detect_integer_defects: bool = False,
    global_order_null_stats: dict | None = None,
    min_oriented_nuclei: int = 200,
) -> dict:
    height, width = image_shape
    density_cutoff = get_density_threshold(
        field["density"],
        tissue_mask,
        density_quantile,
    )
    valid_local = (
        tissue_mask
        & (field["density"] > density_cutoff)
    )
    local_values = field["order"][valid_local]

    microns_per_pixel = metadata.get("microns_per_pixel")
    tissue_area_px = int(tissue_mask.sum())

    charges = (
        defects["charge"] if not defects.empty else pd.Series(dtype=float)
    )
    n_half = int((charges.abs() == 0.5).sum())
    n_integer = int((charges.abs() == 1.0).sum())

    if microns_per_pixel is not None:
        tissue_area_mm2 = (
            tissue_area_px
            * microns_per_pixel**2
            / 1_000_000
        )
    else:
        tissue_area_mm2 = float("nan")

    def density(count: int) -> float:
        if not np.isfinite(tissue_area_mm2) or tissue_area_mm2 <= 0:
            return float("nan")
        return count / tissue_area_mm2

    # The half-integer (plaquette winding) and integer (N-point ring) layers are
    # two detectors with different spatial support and sensitivity. Reporting
    # their sum under one name made defect_density_mm2 mean different things
    # depending on whether detect_integer_defects was set, so runs with and
    # without the flag were not comparable under the same column. The primary
    # column is now the flag-independent half-integer density; the integer layer
    # and the pooled count are reported separately, and the integer density is
    # NaN (not zero) when the layer was never run, so "not measured" and
    # "measured, none found" stay distinguishable.
    defect_density_mm2 = density(n_half)
    defect_density_integer_mm2 = (
        density(n_integer) if detect_integer_defects else float("nan")
    )
    defect_density_all_mm2 = density(len(defects))

    def quantile_or_nan(q: float) -> float:
        if local_values.size == 0:
            return float("nan")
        return float(np.quantile(local_values, q))

    return {
        "filename": metadata["filename"],
        "image_id": metadata["image_id"],
        "group": metadata["group"],
        "width_px": width,
        "height_px": height,
        "microns_per_pixel": microns_per_pixel,
        "tissue_area_px": tissue_area_px,
        "tissue_area_mm2": tissue_area_mm2,
        "n_nuclei": len(nuclei),
        "n_oriented_nuclei": len(oriented_nuclei),
        # Computed from `field`, not from the nuclei table: the nuclei-based
        # value is identical across field types and so cannot describe a
        # collagen or fused run.
        "global_nematic_order_S": compute_global_order_from_field(
            field, tissue_mask
        ),
        "global_nematic_order_S_nuclei": compute_global_order(oriented_nuclei),
        # S is biased upward as ~1/sqrt(N_eff), so it is confounded with nuclei
        # count and tissue area. Prefer the excess over the null when comparing
        # groups that differ in either.
        "global_order_null_mean": _null_field(
            global_order_null_stats, "global_order_null_mean"
        ),
        "global_order_excess": _null_field(
            global_order_null_stats, "global_order_excess"
        ),
        "global_order_p": _null_field(
            global_order_null_stats, "global_order_p"
        ),
        # Below this count the order parameter is dominated by finite-size
        # noise; flagged rather than dropped so the caller decides.
        "low_orientation_count": bool(
            len(oriented_nuclei) < int(min_oriented_nuclei)
        ),
        "local_S_q25": quantile_or_nan(0.25),
        "local_S_median": quantile_or_nan(0.50),
        "local_S_q75": quantile_or_nan(0.75),
        "n_defects_total": len(defects),
        "n_half_total": n_half,
        "n_integer_total": n_integer,
        "n_plus_half": int(
            (defects["charge"] == 0.5).sum()
            if not defects.empty
            else 0
        ),
        "n_minus_half": int(
            (defects["charge"] == -0.5).sum()
            if not defects.empty
            else 0
        ),
        "n_plus_one": int(
            (defects["charge"] == 1.0).sum()
            if not defects.empty
            else 0
        ),
        "n_minus_one": int(
            (defects["charge"] == -1.0).sum()
            if not defects.empty
            else 0
        ),
        "net_topological_charge": float(
            defects["charge"].sum()
            if not defects.empty
            else 0.0
        ),
        "defect_density_mm2": defect_density_mm2,
        "defect_density_integer_mm2": defect_density_integer_mm2,
        "defect_density_all_mm2": defect_density_all_mm2,
        "detect_integer_defects": bool(detect_integer_defects),
        "mean_defect_confidence": float(
            defects["confidence"].mean()
            if not defects.empty
            else float("nan")
        ),
        "representative_sigma_px": representative_sigma_px,
    }
