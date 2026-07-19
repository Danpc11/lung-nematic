"""
Synthetic director fields with physical defect cores.

Detector tests are only as good as the fields they run on. A field built with
``order = 1`` everywhere, including at the singularity, cannot exercise the
colocalization test at all: the core/annulus contrast that test is designed to
measure does not exist in it. These builders give every defect a real core, so
the order parameter falls to zero at the singularity and recovers over a
coherence length:

    S(r) = S0 * tanh(r / xi)

Several defects multiply their profiles, so order drops near each core. The
director superposes through the doubled phase, which is the physically correct
combination for a nematic:

    theta(x) = sum_i q_i * atan2(y - y_i, x - x_i)   (mod pi)

The module is part of the package rather than the test suite because the
two-defect resolution sweep it supports is a methodological result, not an
internal check: it measures when the integer ring detector reports a genuine
+1 and when it is reporting two unresolved +1/2 cores.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .defects import (
    detect_defects_single_scale,
    detect_integer_defects_single_scale,
)


def tissue_mask(shape: tuple[int, int], border_px: int = 1) -> np.ndarray:
    """All-tissue mask with a border excluded, matching detector expectations."""
    mask = np.ones(shape, dtype=bool)
    if border_px > 0:
        mask[:border_px, :] = mask[-border_px:, :] = False
        mask[:, :border_px] = mask[:, -border_px:] = False
    return mask


def director_field(
    shape: tuple[int, int],
    defects: list[tuple[float, float, float]],
    core_size_px: float = 8.0,
    order_amplitude: float = 1.0,
    spiral_angle: float = 0.0,
    density: float | np.ndarray | None = None,
    density_noise: float = 0.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a director field containing the given defects.

    ``defects`` is a list of ``(x, y, charge)``. Charges add through the
    doubled phase, and each contributes a ``tanh`` core to the order parameter,
    so a singularity is genuinely disordered at its centre.

    ``spiral_angle`` offsets the director globally, which for a single ``+1``
    turns an aster (0) into a vortex (pi/2).
    """
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(float)

    theta = np.full(shape, float(spiral_angle))
    order = np.full(shape, float(order_amplitude))

    for x0, y0, charge in defects:
        theta = theta + charge * np.arctan2(yy - y0, xx - x0)
        radius = np.hypot(xx - x0, yy - y0)
        order = order * np.tanh(radius / max(core_size_px, 1e-9))

    rng = np.random.default_rng(seed)
    if density is None:
        # A flat density field is rejected wholesale by the detector: the gate
        # is `density > quantile(density)`, which no pixel of a constant field
        # can satisfy. Real tissue is never flat, so the default here is a
        # broad central bump - dense in the middle of the frame, thinner at the
        # edges - which leaves a contiguous eligible region at any quantile.
        height_, width_ = shape
        yy_, xx_ = np.mgrid[0:height_, 0:width_].astype(float)
        radius_ = np.hypot(xx_ - width_ / 2.0, yy_ - height_ / 2.0)
        scale_ = 0.45 * max(height_, width_)
        density_field = 1.0 + 0.5 * np.exp(-0.5 * (radius_ / scale_) ** 2)
    elif np.isscalar(density):
        density_field = np.full(shape, float(density))
    else:
        density_field = np.asarray(density, dtype=float)
    if density_noise > 0:
        density_field = density_field * (
            1.0 + rng.normal(0.0, density_noise, shape)
        )
        np.clip(density_field, 0.0, None, out=density_field)

    return {
        "theta": np.mod(theta, np.pi),
        "order": np.clip(order, 0.0, 1.0),
        "density": density_field,
    }


def uniform_field(shape: tuple[int, int], angle: float = 0.4,
                  **kwargs) -> dict[str, np.ndarray]:
    """Defect-free field; any detection on it is a false positive."""
    return director_field(shape, defects=[], spiral_angle=angle, **kwargs)


def two_half_defect_field(
    shape: tuple[int, int],
    separation_px: float,
    centre: tuple[float, float] | None = None,
    core_size_px: float = 8.0,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Two ``+1/2`` cores separated along x, with total enclosed charge ``+1``.

    This is the configuration the integer ring detector cannot distinguish from
    a genuine ``+1`` once the separation falls inside the ring.
    """
    height, width = shape
    cx, cy = centre if centre is not None else (width / 2, height / 2)
    half = separation_px / 2.0
    return director_field(
        shape,
        defects=[(cx - half, cy, 0.5), (cx + half, cy, 0.5)],
        core_size_px=core_size_px,
        **kwargs,
    )


def localization_error(
    detected: pd.DataFrame,
    expected: tuple[float, float],
    charge: float | None = None,
) -> float:
    """Distance from the expected core to the nearest detection of that charge.

    Returns ``inf`` when nothing of the requested charge was detected, so a
    caller can assert on the number without special-casing emptiness.
    """
    if detected.empty:
        return float("inf")
    subset = detected
    if charge is not None:
        subset = detected.loc[np.isclose(detected["charge"], charge)]
    if subset.empty:
        return float("inf")
    distances = np.hypot(
        subset["x_px"].to_numpy() - expected[0],
        subset["y_px"].to_numpy() - expected[1],
    )
    return float(distances.min())


def two_half_resolution_sweep(
    config: AnalysisConfig,
    shape: tuple[int, int] = (320, 320),
    separations_px: np.ndarray | None = None,
    core_size_px: float = 8.0,
    match_tolerance_px: float | None = None,
) -> pd.DataFrame:
    """When does the ring detector see two ``+1/2``, and when a single ``+1``?

    Two ``+1/2`` cores are placed a distance ``d`` apart and both detector
    layers are run. The outcome is reported against ``d / R``, where ``R`` is
    ``integer_defect_loop_radius_px``: below 1 the pair sits inside the ring and
    is enclosed as total charge ``+1``; above it the ring should stop reporting
    an integer defect while the half-integer layer resolves both cores.

    The README states this ambiguity; this function measures where it starts.
    """
    radius = float(config.integer_defect_loop_radius_px)
    if separations_px is None:
        separations_px = np.round(
            np.linspace(0.25 * radius, 3.0 * radius, 12)
        )
    if match_tolerance_px is None:
        match_tolerance_px = 1.5 * config.defect_grid_step_px

    height, width = shape
    centre = (width / 2.0, height / 2.0)
    mask = tissue_mask(shape)
    rows: list[dict] = []

    for separation in np.asarray(separations_px, dtype=float):
        field = two_half_defect_field(
            shape, separation, centre=centre, core_size_px=core_size_px
        )
        half_layer = detect_defects_single_scale(field, mask, config)
        integer_layer = detect_integer_defects_single_scale(field, mask, config)

        left = (centre[0] - separation / 2.0, centre[1])
        right = (centre[0] + separation / 2.0, centre[1])
        error_left = localization_error(half_layer, left, charge=0.5)
        error_right = localization_error(half_layer, right, charge=0.5)
        resolved_two = (
            error_left <= match_tolerance_px
            and error_right <= match_tolerance_px
            and int((half_layer["charge"] == 0.5).sum()) >= 2
        )
        n_integer = (
            int((integer_layer["charge"] == 1.0).sum())
            if not integer_layer.empty
            else 0
        )

        if resolved_two and n_integer:
            verdict = "both"
        elif resolved_two:
            verdict = "two_halves"
        elif n_integer:
            verdict = "one_integer"
        else:
            verdict = "none"

        rows.append(
            {
                "separation_px": float(separation),
                "separation_over_radius": float(separation / radius),
                "n_plus_half": int((half_layer["charge"] == 0.5).sum()),
                "n_plus_one": n_integer,
                "localization_error_left_px": error_left,
                "localization_error_right_px": error_right,
                "resolved_two_halves": bool(resolved_two),
                "verdict": verdict,
            }
        )

    return pd.DataFrame(rows)


def integer_charge_sweep(
    config: AnalysisConfig,
    shape: tuple[int, int] = (320, 320),
    core_sizes_px: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
) -> pd.DataFrame:
    """Raw enclosed charge for a genuine ``+1``, as the core is widened.

    Useful for calibrating how close ``charge_raw`` stays to the ideal value
    before rounding, which is what an explicit tolerance would be set against.
    """
    height, width = shape
    centre = (width / 2.0, height / 2.0)
    mask = tissue_mask(shape)
    rows = []
    for core in core_sizes_px:
        field = director_field(
            shape, defects=[(centre[0], centre[1], 1.0)], core_size_px=core
        )
        detected = detect_integer_defects_single_scale(field, mask, config)
        near = detected.loc[
            np.hypot(detected["x_px"] - centre[0], detected["y_px"] - centre[1])
            <= config.integer_defect_loop_radius_px
        ] if not detected.empty else detected
        rows.append(
            {
                "core_size_px": float(core),
                "n_detected": int(len(near)),
                "charge_raw_mean": float(near["charge_raw"].mean())
                if not near.empty else float("nan"),
                "charge_raw_abs_error": float(
                    np.abs(near["charge_raw"] - 1.0).mean()
                ) if not near.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)
