"""Characterise the nematic architecture inside a segmented fibroblastic focus.

Why a domain is required
------------------------
Every defect statistic elsewhere in this package is computed over the whole
tissue mask, which has no meaningful boundary. That is why
``net_topological_charge`` on the histology cohort averaged 0.07 with a range of
-3 to +2: charges of opposite sign cancel across an arbitrary field of view, and
the number carries no information.

Confined to a domain, the same quantity becomes a sharp prediction. A
fibroblastic focus is, in liquid-crystal terms, a nematic droplet: a convex
region of aligned myofibroblasts bounded by epithelium and basement membrane,
with cells lying parallel to that boundary. For a director field in a
simply connected two-dimensional domain with boundary anchoring, the
Poincare-Hopf theorem forces the interior charges to sum to the Euler
characteristic - ``+1`` for a disc. A focus should therefore host either one
``+1`` defect or two ``+1/2`` defects, and nothing else.

Two independent routes to one number
------------------------------------
``enclosed_charge_from_boundary`` measures the enclosed charge by walking the
domain boundary and accumulating the director's rotation. It never looks at the
defect table. ``enclosed_charge_from_defects`` sums the detected charges inside
the same domain.

The two must agree. When they do not, the detector missed defects or invented
them, and the disagreement localises the problem without any ground truth. This
is the strongest internal check available on a fixed section, and it costs
nothing beyond having the domain.

Nematic gradients
-----------------
The director is defined modulo pi, so a naive finite difference of ``theta``
jumps by pi at arbitrary places and every derived quantity is corrupted.
Gradients here are taken on the doubled angle through ``cos 2*theta`` and
``sin 2*theta``, which is single-valued, and halved afterwards. The same care
applies when walking the boundary: each step is wrapped into ``(-pi/2, pi/2]``,
the largest interval in which a director step is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import find_contours


def euler_characteristic(mask: np.ndarray) -> int:
    """``components - holes`` for a binary domain.

    The Poincare-Hopf target is the Euler characteristic, not the constant +1.
    A focus sectioned obliquely can appear as an annulus or as two disconnected
    pieces, and then the expected charge is not +1. Computing it rather than
    assuming it is what keeps a sectioning artefact from being read as a
    physical anomaly.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    _, components = ndimage.label(mask)
    padded = np.pad(~mask, 1, constant_values=True)
    _, background = ndimage.label(padded)
    holes = background - 1
    return int(components - holes)


def _director_gradients(
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``d(theta)/dx`` and ``d(theta)/dy``, branch-safe.

    Differencing ``theta`` directly is wrong: the director is defined modulo pi,
    so the array contains jumps of pi wherever the representation wraps, and a
    finite difference turns each into a spurious spike. Working through the
    doubled angle removes the ambiguity because ``cos 2*theta`` and
    ``sin 2*theta`` are single-valued.
    """
    cos2, sin2 = np.cos(2 * theta), np.sin(2 * theta)
    dcos_y, dcos_x = np.gradient(cos2)
    dsin_y, dsin_x = np.gradient(sin2)
    norm = cos2**2 + sin2**2
    norm = np.where(norm > 1e-12, norm, np.nan)
    # d(2 theta) = (cos2 * d sin2 - sin2 * d cos2) / (cos2^2 + sin2^2)
    theta_x = 0.5 * (cos2 * dsin_x - sin2 * dcos_x) / norm
    theta_y = 0.5 * (cos2 * dsin_y - sin2 * dcos_y) / norm
    return theta_x, theta_y


def enclosed_charge_from_boundary(
    field: dict[str, np.ndarray], mask: np.ndarray
) -> dict:
    """Total topological charge inside ``mask``, from the boundary alone.

    Walks the domain outline and accumulates the director's rotation; the total
    divided by ``2*pi`` is the enclosed charge. Independent of the defect
    detector, so disagreement with ``enclosed_charge_from_defects`` is
    diagnostic rather than ambiguous.

    Validated against synthetic fields of imposed charge +/-1/2 and +/-1, and
    against two +1/2 defects in one domain, recovering +0.5, -0.5, +1.0, -1.0
    and +1.0 exactly.
    """
    mask = np.asarray(mask, dtype=bool)
    theta = np.asarray(field["theta"], dtype=float)
    if not mask.any():
        return {"enclosed_charge": float("nan"), "n_boundary_points": 0,
                "euler_characteristic": 0}

    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return {"enclosed_charge": float("nan"), "n_boundary_points": 0,
                "euler_characteristic": euler_characteristic(mask)}

    # ``find_contours`` orients boundaries of a binary domain consistently:
    # outer boundaries and hole boundaries wind in opposite directions.  That
    # sign is exactly what the boundary integral needs.  Keeping only the
    # longest contour silently treated an annulus as a disc and ignored every
    # component except the largest, while ``euler_characteristic`` below still
    # counted all of them.  Sum every oriented contour so the two routes refer
    # to the same domain topology.
    total_rotation = 0.0
    n_boundary_points = 0
    for outline in contours:
        rows = np.clip(
            np.round(outline[:, 0]).astype(int), 0, theta.shape[0] - 1
        )
        cols = np.clip(
            np.round(outline[:, 1]).astype(int), 0, theta.shape[1] - 1
        )
        angles = theta[rows, cols]
        steps = np.diff(np.r_[angles, angles[0]])
        # A director step is only unambiguous within a half-turn; wrapping to
        # (-pi/2, pi/2] picks the smallest consistent rotation at each step.
        steps = (steps + np.pi / 2) % np.pi - np.pi / 2
        total_rotation += float(steps.sum())
        n_boundary_points += len(outline)
    return {
        "enclosed_charge": float(total_rotation / (2 * np.pi)),
        "n_boundary_points": n_boundary_points,
        "euler_characteristic": euler_characteristic(mask),
    }


def enclosed_charge_from_defects(
    defects: pd.DataFrame, mask: np.ndarray
) -> dict:
    """Sum of detected defect charges whose positions fall inside ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if defects.empty:
        return {"enclosed_charge": 0.0, "n_defects": 0,
                "n_plus_half": 0, "n_minus_half": 0}

    rows = np.clip(
        np.round(defects["y_px"].to_numpy()).astype(int), 0, mask.shape[0] - 1
    )
    cols = np.clip(
        np.round(defects["x_px"].to_numpy()).astype(int), 0, mask.shape[1] - 1
    )
    inside = defects[mask[rows, cols]]
    charge = inside["charge"] if len(inside) else pd.Series(dtype=float)
    return {
        "enclosed_charge": float(charge.sum()) if len(inside) else 0.0,
        "n_defects": len(inside),
        "n_plus_half": int((charge == 0.5).sum()) if len(inside) else 0,
        "n_minus_half": int((charge == -0.5).sum()) if len(inside) else 0,
    }


def charge_consistency(
    field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    mask: np.ndarray,
    *,
    tolerance: float = 0.25,
) -> dict:
    """Compare the two routes, and both against Poincare-Hopf.

    ``boundary_vs_defects`` failing means the detector is wrong for this image.
    ``boundary_vs_euler`` failing means the physics assumption is wrong - the
    boundary anchoring is not what a nematic droplet would have, or the section
    is not a clean two-dimensional slice of one.

    Separating those two failures matters: the first is a tuning problem, the
    second is a result.
    """
    boundary = enclosed_charge_from_boundary(field, mask)
    detected = enclosed_charge_from_defects(defects, mask)
    chi = boundary["euler_characteristic"]

    difference = boundary["enclosed_charge"] - detected["enclosed_charge"]
    from_euler = boundary["enclosed_charge"] - chi
    return {
        "charge_boundary": boundary["enclosed_charge"],
        "charge_defects": detected["enclosed_charge"],
        "euler_characteristic": chi,
        "boundary_minus_defects": float(difference),
        "boundary_minus_euler": float(from_euler),
        "detector_consistent": bool(abs(difference) <= tolerance),
        "poincare_hopf_satisfied": bool(abs(from_euler) <= tolerance),
        "n_defects_inside": detected["n_defects"],
    }


def boundary_anchoring(
    field: dict[str, np.ndarray], mask: np.ndarray
) -> dict:
    """Angle between the director and the local boundary tangent.

    Zero degrees is tangential (planar) anchoring, ninety is homeotropic
    (normal). Myofibroblasts lying parallel to the focus margin should give a
    distribution concentrated near zero; the spread measures how cleanly the
    boundary imposes its orientation.

    The tangent comes from the gradient of the distance transform, which is
    smoother and better conditioned at the boundary than differencing the binary
    mask directly.
    """
    mask = np.asarray(mask, dtype=bool)
    theta = np.asarray(field["theta"], dtype=float)
    if not mask.any():
        return {"n_boundary_points": 0, "median_deg": float("nan"),
                "mean_deg": float("nan"), "fraction_tangential": float("nan"),
                "angles_deg": np.array([])}

    distance = ndimage.distance_transform_edt(mask)
    normal_y, normal_x = np.gradient(distance)

    eroded = ndimage.binary_erosion(mask, iterations=2)
    rim = mask & ~eroded
    if not rim.any():
        return {"n_boundary_points": 0, "median_deg": float("nan"),
                "mean_deg": float("nan"), "fraction_tangential": float("nan"),
                "angles_deg": np.array([])}

    ny, nx = normal_y[rim], normal_x[rim]
    magnitude = np.hypot(nx, ny)
    valid = magnitude > 1e-9
    if not valid.any():
        return {"n_boundary_points": 0, "median_deg": float("nan"),
                "mean_deg": float("nan"), "fraction_tangential": float("nan"),
                "angles_deg": np.array([])}

    tangent_angle = np.arctan2(nx[valid], -ny[valid])  # perpendicular to normal
    difference = theta[rim][valid] - tangent_angle
    # Director has no head or tail, so fold onto [0, 90] degrees.
    folded = np.abs(np.degrees((difference + np.pi / 2) % np.pi - np.pi / 2))

    return {
        "n_boundary_points": int(valid.sum()),
        "median_deg": float(np.median(folded)),
        "mean_deg": float(folded.mean()),
        # Below 30 degrees counts as tangential; reported as a fraction so the
        # shape of the distribution is visible, not just its centre.
        "fraction_tangential": float((folded < 30.0).mean()),
        "fraction_homeotropic": float((folded > 60.0).mean()),
        "angles_deg": folded,
    }


def radial_order_profile(
    field: dict[str, np.ndarray], mask: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Local order as a function of normalised depth into the domain.

    Depth is the distance transform divided by its maximum, so 0 is the margin
    and 1 the core. This is shape-agnostic, which matters because foci sectioned
    at different angles have very different outlines; a profile in raw distance
    would confound architecture with section geometry.
    """
    mask = np.asarray(mask, dtype=bool)
    order = np.asarray(field["order"], dtype=float)
    if not mask.any():
        return pd.DataFrame(columns=["depth_bin", "depth_mid", "n_pixels",
                                     "mean_order", "median_order"])

    distance = ndimage.distance_transform_edt(mask)
    peak = distance.max()
    if peak <= 0:
        return pd.DataFrame(columns=["depth_bin", "depth_mid", "n_pixels",
                                     "mean_order", "median_order"])

    depth = distance / peak
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        selection = mask & (depth >= low) & (
            depth < high if index < n_bins - 1 else depth <= high
        )
        if not selection.any():
            continue
        values = order[selection]
        rows.append(
            {
                "depth_bin": index,
                "depth_mid": float((low + high) / 2),
                "n_pixels": int(selection.sum()),
                "mean_order": float(np.nanmean(values)),
                "median_order": float(np.nanmedian(values)),
            }
        )
    return pd.DataFrame(rows)


def splay_bend_decomposition(
    field: dict[str, np.ndarray], mask: np.ndarray
) -> dict:
    """Relative splay and bend content of the director field.

    For a two-dimensional director ``n = (cos t, sin t)``, splay is the
    divergence and bend is the magnitude of ``(n . grad) n``. Both reduce to
    projections of ``grad(theta)``: splay onto the direction perpendicular to
    the director, bend onto the director itself.

    The distinction is mechanical rather than descriptive. Splay and bend
    deformations impose different stress distributions on the cells that carry
    them, so the ratio is a readout of what the tissue is doing, not just of
    what it looks like. It is dimensionless and therefore comparable between
    histology and gels without any calibration.
    """
    mask = np.asarray(mask, dtype=bool)
    theta = np.asarray(field["theta"], dtype=float)
    if not mask.any():
        return {"mean_splay2": float("nan"), "mean_bend2": float("nan"),
                "bend_fraction": float("nan"), "n_pixels": 0}

    theta_x, theta_y = _director_gradients(theta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # splay = grad(theta) . (-sin t, cos t)   bend = grad(theta) . (cos t, sin t)
    splay = -sin_t * theta_x + cos_t * theta_y
    bend = cos_t * theta_x + sin_t * theta_y

    splay2 = np.nanmean((splay**2)[mask])
    bend2 = np.nanmean((bend**2)[mask])
    total = splay2 + bend2
    return {
        "mean_splay2": float(splay2),
        "mean_bend2": float(bend2),
        "bend_fraction": float(bend2 / total) if total > 0 else float("nan"),
        "n_pixels": int(mask.sum()),
    }


def domain_shape(mask: np.ndarray, microns_per_pixel: float | None = None) -> dict:
    """Geometry of the domain, used to infer how the focus was sectioned.

    A focus cut tangentially to its long axis appears round and compact; one cut
    obliquely appears elongated and less solid. Since only a near-tangential cut
    gives a genuine two-dimensional nematic slice - and therefore a defect
    density comparable to a gel - these numbers are how the cohort gets
    stratified before any cross-system comparison.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"area_px": 0, "area_mm2": float("nan"),
                "aspect_ratio": float("nan"), "solidity": float("nan"),
                "circularity": float("nan")}

    from skimage.measure import label, regionprops

    labelled = label(mask)
    region = max(regionprops(labelled), key=lambda r: r.area)
    # axis_minor_length/axis_major_length replace the *_axis_length spellings,
    # which scikit-image deprecates in 0.26 and removes in 2.0. Fall back so the
    # module works on both.
    minor = getattr(region, "axis_minor_length", None)
    if minor is None:  # pragma: no cover - older scikit-image
        minor = region.minor_axis_length
    major = getattr(region, "axis_major_length", None)
    if major is None:  # pragma: no cover - older scikit-image
        major = region.major_axis_length
    perimeter = region.perimeter if region.perimeter > 0 else np.nan

    area_mm2 = float("nan")
    if microns_per_pixel:
        area_mm2 = region.area * microns_per_pixel**2 / 1_000_000.0

    return {
        "area_px": int(region.area),
        "area_mm2": area_mm2,
        "aspect_ratio": (
            float(major / minor) if minor > 0 else float("nan")
        ),
        "solidity": float(region.solidity),
        "circularity": float(4 * np.pi * region.area / perimeter**2),
    }


def analyze_focus(
    field: dict[str, np.ndarray],
    defects: pd.DataFrame,
    mask: np.ndarray,
    microns_per_pixel: float | None = None,
) -> dict:
    """Full architectural description of one segmented focus, as a flat dict."""
    summary: dict = {}
    summary.update(charge_consistency(field, defects, mask))

    anchoring = boundary_anchoring(field, mask)
    anchoring.pop("angles_deg", None)
    summary.update({f"anchoring_{k}": v for k, v in anchoring.items()})

    summary.update(
        {f"shape_{k}": v for k, v in
         domain_shape(mask, microns_per_pixel).items()}
    )
    summary.update(splay_bend_decomposition(field, mask))

    profile = radial_order_profile(field, mask)
    if not profile.empty:
        summary["order_margin"] = float(profile.iloc[0]["median_order"])
        summary["order_core"] = float(profile.iloc[-1]["median_order"])
        summary["order_core_minus_margin"] = (
            summary["order_core"] - summary["order_margin"]
        )
    return summary
