"""Track topological defects across time-lapse frames.

Why this module exists
----------------------
Every other analysis in this package describes a single frame. A time lapse of
cells on a calibrated gel carries something a fixed image cannot: whether the
nematic continually creates and annihilates opposite-charge defect pairs.

Pair turnover is the primary activity observable for these data. Defects live
only three to four frames, leaving too few displacement steps for a robust
``+1/2`` versus ``-1/2`` speed contrast. Kinematic tables remain useful for
quality control and future higher-frequency acquisitions, but they must not be
treated as an independent activity measurement here.

A passive nematic coarsens: defect density decays as a power law and the
texture orders out. An active one does not - activity keeps nucleating pairs,
so the density saturates at a steady state set by the ratio of activity to
elastic constant. Whether ``defect_density(t)`` decays or plateaus, and where
the plateau sits, is a second, independent estimate of the same activity.

Pair enrichment over a spatial null is essential: simultaneous unpaired
detections may be detector flicker, while nearby opposite-charge births and
deaths are the topology expected from active-nematic turnover.

Stage drift
-----------
Uncorrected stage drift adds a common velocity *vector* to every defect, and
speed is the magnitude of that sum. Because self-propelled defects head in
independent directions, ``|v_propulsion + v_drift|`` is not
``|v_propulsion| + |v_drift|``, so drift does not merely offset both charge
classes - it compresses the contrast between them. Measured on synthetic
frames, a drift of 2 px/frame against a propulsion of 3 px/frame shrinks
``speed(+1/2) - speed(-1/2)`` from 2.51 to 1.48, a 41% loss.

So drift must be removed, not reasoned around: run ``estimate_drift`` and
``subtract_drift`` before reading any speed, and treat unregistered frames as
giving a lower bound on activity rather than an estimate of it.

Units
-----
Everything here works in pixels and frames. Calibration to micrometres and
seconds is applied at the end by ``calibrate``, so the tracking does not have
to be redone when the acquisition metadata is finally located.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = ("x_px", "y_px", "charge")


def track_defects(
    frames: list[pd.DataFrame],
    max_displacement_px: float,
    *,
    max_gap: int = 0,
    charge_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Link per-frame defect tables into tracks.

    Linking is solved separately within each charge class, which enforces
    charge conservation structurally rather than by post-hoc filtering: a
    ``+1/2`` can only ever be matched to another ``+1/2``. A defect that
    genuinely changes charge is not a moving defect, it is one defect
    annihilating and another appearing, and the track table should say so.

    ``max_displacement_px`` gates the assignment. The Hungarian solver returns a
    globally optimal matching even when every pair is implausibly far apart, so
    without a gate a defect that annihilated would be linked to an unrelated one
    across the field. Set it from the physics: a few times the expected
    per-frame displacement, and well below the typical defect separation.

    ``max_gap`` is the number of missing intermediate frames that a track may
    survive. The displacement gate grows linearly with elapsed frames, so a
    link spanning one missing frame is allowed up to
    ``2 * max_displacement_px``. The default of zero preserves consecutive-only
    linking.

    Returns one row per (track, frame) with ``track_id``, ``frame``, ``x_px``,
    ``y_px`` and ``charge``. Tracks of length 1 are kept - a defect that appears
    and annihilates immediately is data about the nucleation rate, not noise to
    discard.
    """
    if max_displacement_px <= 0:
        raise ValueError("max_displacement_px must be positive")
    if isinstance(max_gap, bool) or not isinstance(max_gap, (int, np.integer)):
        raise TypeError("max_gap must be an integer")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    for index, table in enumerate(frames):
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"frame {index} is missing columns: {missing}")

    rows: list[dict] = []
    next_track_id = 0
    # Last observation for every track that may still be linked. Keeping track
    # state rather than only the preceding frame lets a detection disappear
    # briefly without forcing a new identity when it returns.
    active: dict[int, dict[str, float | int]] = {}

    for frame_index, table in enumerate(frames):
        table = table.reset_index(drop=True)
        current_tracks: dict[int, int] = {}

        active = {
            track_id: state
            for track_id, state in active.items()
            if frame_index - int(state["frame"]) <= max_gap + 1
        }

        if len(table) and active:
            active_ids = list(active)
            previous = pd.DataFrame([active[track_id] for track_id in active_ids])
            for charge in _charge_classes(previous, table, charge_tolerance):
                previous_rows = _rows_with_charge(previous, charge,
                                                  charge_tolerance)
                current_rows = _rows_with_charge(table, charge,
                                                 charge_tolerance)
                if not len(previous_rows) or not len(current_rows):
                    continue
                elapsed = (
                    frame_index
                    - previous.loc[previous_rows, "frame"].to_numpy(dtype=int)
                )
                for previous_row, current_row in _assign_with_gates(
                    previous.loc[previous_rows], table.loc[current_rows],
                    max_displacement_px * elapsed,
                ):
                    source = previous_rows[previous_row]
                    target = current_rows[current_row]
                    current_tracks[target] = active_ids[source]

        for row_index in range(len(table)):
            if row_index not in current_tracks:
                current_tracks[row_index] = next_track_id
                next_track_id += 1
            rows.append(
                {
                    "track_id": current_tracks[row_index],
                    "frame": frame_index,
                    "x_px": float(table.loc[row_index, "x_px"]),
                    "y_px": float(table.loc[row_index, "y_px"]),
                    "charge": float(table.loc[row_index, "charge"]),
                }
            )
            active[current_tracks[row_index]] = {
                "x_px": float(table.loc[row_index, "x_px"]),
                "y_px": float(table.loc[row_index, "y_px"]),
                "charge": float(table.loc[row_index, "charge"]),
                "frame": frame_index,
            }

    return pd.DataFrame(
        rows,
        columns=["track_id", "frame", "x_px", "y_px", "charge"],
    )


def _charge_classes(a: pd.DataFrame, b: pd.DataFrame, tolerance: float):
    values = np.concatenate([a["charge"].to_numpy(), b["charge"].to_numpy()])
    unique = []
    for value in values:
        if not any(abs(value - seen) <= tolerance for seen in unique):
            unique.append(float(value))
    return unique


def _rows_with_charge(table: pd.DataFrame, charge: float, tolerance: float):
    mask = (table["charge"] - charge).abs() <= tolerance
    return list(np.flatnonzero(mask.to_numpy()))


def _assign(previous: pd.DataFrame, current: pd.DataFrame, gate: float):
    """Gated Hungarian assignment. Yields (previous_row, current_row) pairs."""
    before = previous[["x_px", "y_px"]].to_numpy(dtype=float)
    after = current[["x_px", "y_px"]].to_numpy(dtype=float)
    distance = np.linalg.norm(before[:, None, :] - after[None, :, :], axis=2)

    # Forbidden pairs get a cost far above any admissible one, so the solver
    # only picks them when it has no alternative - and those picks are then
    # dropped by the gate below.
    cost = np.where(distance <= gate, distance, 1e9)
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        if distance[row, column] <= gate:
            yield int(row), int(column)


def _assign_with_gates(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    gates: np.ndarray,
):
    """Assign with one displacement gate per previous observation."""
    before = previous[["x_px", "y_px"]].to_numpy(dtype=float)
    after = current[["x_px", "y_px"]].to_numpy(dtype=float)
    distance = np.linalg.norm(before[:, None, :] - after[None, :, :], axis=2)
    allowed = distance <= np.asarray(gates, dtype=float)[:, None]
    cost = np.where(allowed, distance, 1e9)
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        if allowed[row, column]:
            yield int(row), int(column)


def track_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-track kinematics, in pixels per frame."""
    if tracks.empty:
        return pd.DataFrame(
            columns=["track_id", "charge", "n_frames", "first_frame",
                     "last_frame", "mean_speed_px_per_frame",
                     "net_displacement_px", "straightness"]
        )

    records = []
    for track_id, group in tracks.sort_values("frame").groupby("track_id"):
        x = group["x_px"].to_numpy()
        y = group["y_px"].to_numpy()
        elapsed = np.diff(group["frame"].to_numpy(dtype=float))
        steps = np.hypot(np.diff(x), np.diff(y))
        speeds = steps / elapsed
        path = float(steps.sum())
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        records.append(
            {
                "track_id": int(track_id),
                "charge": float(group["charge"].iloc[0]),
                "n_frames": len(group),
                "first_frame": int(group["frame"].iloc[0]),
                "last_frame": int(group["frame"].iloc[-1]),
                "mean_speed_px_per_frame": (
                    float(speeds.mean()) if speeds.size else float("nan")
                ),
                "net_displacement_px": net,
                # 1 means ballistic, near 0 means the defect wandered without
                # going anywhere. Self-propelled defects should be closer to 1
                # than diffusive ones over the same number of frames.
                "straightness": (net / path) if path > 0 else float("nan"),
            }
        )
    return pd.DataFrame(records)


def estimate_drift(tracks: pd.DataFrame) -> pd.DataFrame:
    """Median per-frame displacement of all tracked defects.

    A rigid translation of the field - stage drift - moves every defect
    identically, so the median step is an estimate of it. This is only valid
    when defect motion is otherwise uncorrelated; if the whole sheet is flowing,
    the median contains real physics and subtracting it removes signal. Compare
    against the ``+1/2`` and ``-1/2`` medians separately before subtracting: if
    they differ, the common component is not pure drift.
    """
    # A step spanning a detection gap has no unique frame on which to apply
    # its translation. Use consecutive observations for drift estimation;
    # gap-normalised steps remain valid for motility below.
    steps = _step_table(tracks)
    steps = steps[steps["gap"] == 1]
    if steps.empty:
        return pd.DataFrame(columns=["frame", "drift_x_px", "drift_y_px",
                                     "n_steps"])
    grouped = steps.groupby("frame")
    return pd.DataFrame(
        {
            "frame": grouped.size().index,
            "drift_x_px": grouped["dx_px"].median().to_numpy(),
            "drift_y_px": grouped["dy_px"].median().to_numpy(),
            "n_steps": grouped.size().to_numpy(),
        }
    ).reset_index(drop=True)


def subtract_drift(tracks: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    """Remove a cumulative per-frame translation from every track.

    If no drift estimate exists for a frame, retain the latest cumulative
    offset. Resetting to zero at that frame creates an artificial jump in every
    corrected trajectory.
    """
    drift = drift.sort_values("frame")
    offsets = {0: (0.0, 0.0)}
    cumulative_x = cumulative_y = 0.0
    for _, row in drift.iterrows():
        cumulative_x += float(row["drift_x_px"])
        cumulative_y += float(row["drift_y_px"])
        offsets[int(row["frame"])] = (cumulative_x, cumulative_y)

    corrected = tracks.copy()
    known_frames = np.array(sorted(offsets), dtype=int)
    known_offsets = [offsets[int(frame)] for frame in known_frames]

    def offset_at(frame):
        index = int(np.searchsorted(known_frames, int(frame), side="right") - 1)
        return known_offsets[index] if index >= 0 else (0.0, 0.0)

    shift = corrected["frame"].map(offset_at)
    corrected["x_px"] = corrected["x_px"] - shift.map(lambda s: s[0])
    corrected["y_px"] = corrected["y_px"] - shift.map(lambda s: s[1])
    return corrected


def _step_table(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(columns=["track_id", "frame", "charge", "dx_px",
                                     "dy_px", "gap",
                                     "speed_px_per_frame"])
    ordered = tracks.sort_values(["track_id", "frame"])
    grouped = ordered.groupby("track_id")
    step = pd.DataFrame(
        {
            "track_id": ordered["track_id"],
            "frame": ordered["frame"],
            "charge": ordered["charge"],
            "dx_px": grouped["x_px"].diff(),
            "dy_px": grouped["y_px"].diff(),
            "gap": grouped["frame"].diff(),
        }
    ).dropna(subset=["dx_px"])
    step["dx_px"] = step["dx_px"] / step["gap"]
    step["dy_px"] = step["dy_px"] / step["gap"]
    step["speed_px_per_frame"] = np.hypot(step["dx_px"], step["dy_px"])
    return step


def motility_by_charge(tracks: pd.DataFrame) -> pd.DataFrame:
    """Descriptive speed statistics per charge class.

    In an active nematic the ``+1/2`` class is faster than the ``-1/2`` class,
    and the contrast between them can scale with activity. In the present data,
    however, 3--4-frame lifetimes make this contrast too noisy to use as the
    activity observable; use pair nucleation instead.

    Remove stage drift first. Drift adds a common velocity vector, and since
    speed is a magnitude and propulsion directions are independent, it inflates
    both classes *and* compresses the contrast - it is not a common offset that
    cancels. On unregistered frames this figure is a lower bound on activity.
    """
    steps = _step_table(tracks)
    if steps.empty:
        return pd.DataFrame(
            columns=["charge", "n_steps", "n_tracks",
                     "mean_speed_px_per_frame", "median_speed_px_per_frame",
                     "sem_speed_px_per_frame"]
        )
    grouped = steps.groupby("charge")["speed_px_per_frame"]
    summary = pd.DataFrame(
        {
            "charge": grouped.mean().index,
            "n_steps": grouped.size().to_numpy(),
            "mean_speed_px_per_frame": grouped.mean().to_numpy(),
            "median_speed_px_per_frame": grouped.median().to_numpy(),
            "sem_speed_px_per_frame": (
                grouped.std(ddof=1) / np.sqrt(grouped.size())
            ).to_numpy(),
        }
    ).reset_index(drop=True)
    counts = steps.groupby("charge")["track_id"].nunique()
    summary["n_tracks"] = summary["charge"].map(counts).to_numpy()
    return summary


def defect_kinetics(
    tracks: pd.DataFrame,
    tissue_area_mm2_by_frame: dict[int, float] | None = None,
    seconds_per_frame: float | None = None,
    pair_radius_px: float = 100.0,
) -> pd.DataFrame:
    """Defect counts, turnover, and opposite-charge nucleation per frame.

    ``births`` counts tracks starting at that frame and ``deaths`` counts tracks
    ending at the previous one. In a coarsening passive nematic deaths exceed
    births and the count decays; in an active steady state the two balance and
    the count plateaus. That balance is the cleanest way to tell the two regimes
    apart without fitting anything. Births in the first frame describe the
    initial population and are deliberately excluded from nucleation.

    A nucleation pair is a one-to-one opposite-charge match within
    ``pair_radius_px``. One-to-one assignment prevents a dense cluster from
    counting the same event more than once. Physical rates require both the
    covered area and ``seconds_per_frame``; otherwise the pair count is still
    returned without fabricating a time calibration.
    """
    if tracks.empty:
        return pd.DataFrame(
            columns=["frame", "n_defects", "n_plus_half", "n_minus_half",
                     "births", "deaths", "nucleation_pairs"]
        )

    frames = np.arange(int(tracks["frame"].min()),
                       int(tracks["frame"].max()) + 1)
    first = tracks.groupby("track_id")["frame"].min()
    last = tracks.groupby("track_id")["frame"].max()
    birth_rows = tracks.merge(first.rename("first_frame"), on="track_id")
    birth_rows = birth_rows[birth_rows["frame"] == birth_rows["first_frame"]]

    records = []
    for frame in frames:
        present = tracks[tracks["frame"] == frame]
        records.append(
            {
                "frame": int(frame),
                "n_defects": len(present),
                "n_plus_half": int((present["charge"] > 0).sum()),
                "n_minus_half": int((present["charge"] < 0).sum()),
                "births": int((first == frame).sum()) if frame > frames[0] else 0,
                "deaths": int((last == frame - 1).sum()) if frame > frames[0] else 0,
                "nucleation_pairs": (
                    _opposite_charge_pairs(
                        birth_rows[birth_rows["frame"] == frame], pair_radius_px
                    )
                    if frame > frames[0] else 0
                ),
            }
        )
    table = pd.DataFrame(records)

    if tissue_area_mm2_by_frame:
        area = table["frame"].map(tissue_area_mm2_by_frame)
        table["covered_area_mm2"] = area
        table["defect_density_mm2"] = np.where(
            area > 0, table["n_defects"] / area, np.nan
        )
        if seconds_per_frame is not None:
            if seconds_per_frame <= 0:
                raise ValueError("seconds_per_frame must be positive")
            table["pair_nucleation_rate_mm2_h"] = np.where(
                area > 0,
                table["nucleation_pairs"] / area * 3600.0 / seconds_per_frame,
                np.nan,
            )
    return table


def _opposite_charge_pairs(events: pd.DataFrame, radius_px: float) -> int:
    """Return a maximum one-to-one matching of opposite-charge events."""
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    plus = events.loc[events["charge"] > 0, ["x_px", "y_px"]].to_numpy(float)
    minus = events.loc[events["charge"] < 0, ["x_px", "y_px"]].to_numpy(float)
    if not len(plus) or not len(minus):
        return 0
    distance = np.linalg.norm(plus[:, None, :] - minus[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(np.where(distance <= radius_px,
                                                    distance, 1e12))
    return int(np.sum(distance[rows, columns] <= radius_px))


def pair_events(
    frames_or_tracks: list[pd.DataFrame] | pd.DataFrame,
    radius_px: float = 100.0,
    n_null: int = 25,
) -> dict[str, float | int]:
    """Measure same-frame opposite-charge pairing at births and deaths.

    The input may be a track table or per-frame tables carrying ``track_id``.
    Tracks are required because birth and death are identity events; inferring
    them from nearest neighbours would repeat the circular tracking diagnostic
    this statistic is intended to replace.

    The null randomises the negative event positions uniformly within the
    observed field rectangle, retaining each frame's event counts. DataFrames
    may provide exact ``width_px`` and ``height_px`` in ``attrs``; otherwise the
    observed coordinate extent is used. A fixed seed makes CLI diagnostics and
    tests reproducible.
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    tracks = _coerce_tracks(frames_or_tracks)
    if tracks.empty:
        return _empty_pair_events()

    first = tracks.groupby("track_id")["frame"].min()
    last = tracks.groupby("track_id")["frame"].max()
    marked = tracks.merge(first.rename("first_frame"), on="track_id")
    marked = marked.merge(last.rename("last_frame"), on="track_id")
    first_frame, last_frame = int(tracks.frame.min()), int(tracks.frame.max())
    births = marked[(marked.frame == marked.first_frame)
                    & (marked.frame > first_frame)]
    deaths = marked[(marked.frame == marked.last_frame)
                    & (marked.frame < last_frame)]

    width = float(tracks.attrs.get("width_px", tracks.x_px.max() + 1))
    height = float(tracks.attrs.get("height_px", tracks.y_px.max() + 1))
    rng = np.random.default_rng(0)
    result: dict[str, float | int] = {}
    for name, events in (("birth", births), ("death", deaths)):
        paired, total = _paired_event_count(events, radius_px)
        null_fractions = []
        # The null separates from the observed value by more than an order of
        # magnitude, so its mean needs very few replicates to be precise enough
        # to act on. Copying the frame 200 times cost 78 s on a 2000-row series
        # and this diagnostic runs on every CLI invocation; one copy reused
        # across replicates gives the same answer in a fraction of the time.
        randomised = events.copy()
        negative = (randomised.charge < 0).to_numpy()
        n_negative = int(negative.sum())
        for _ in range(n_null):
            if n_negative:
                randomised.loc[negative, "x_px"] = rng.uniform(0, width, n_negative)
                randomised.loc[negative, "y_px"] = rng.uniform(0, height, n_negative)
            null_paired, _ = _paired_event_count(randomised, radius_px)
            null_fractions.append(null_paired / total if total else np.nan)
        result[f"n_{name}_events"] = total
        result[f"n_paired_{name}_events"] = paired
        result[f"paired_{name}_fraction"] = paired / total if total else np.nan
        result[f"null_paired_{name}_fraction"] = (
            float(np.nanmean(null_fractions)) if total else np.nan
        )
    return result


def _paired_event_count(events: pd.DataFrame, radius_px: float) -> tuple[int, int]:
    """Count event members belonging to a one-to-one opposite-charge pair."""
    pairs = sum(
        _opposite_charge_pairs(group, radius_px)
        for _, group in events.groupby("frame")
    )
    return 2 * pairs, len(events)


def _coerce_tracks(value: list[pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        tracks = value.copy()
        attrs = value.attrs.copy()
    else:
        rows = []
        attrs = {}
        for frame, table in enumerate(value):
            if "track_id" not in table.columns:
                raise ValueError("per-frame tables must contain track_id")
            rows.append(table.assign(frame=frame))
            attrs.update(table.attrs)
        tracks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    required = {"track_id", "frame", "x_px", "y_px", "charge"}
    missing = sorted(required - set(tracks.columns))
    if missing and not tracks.empty:
        raise ValueError(f"tracks are missing columns: {missing}")
    tracks.attrs.update(attrs)
    return tracks


def _empty_pair_events() -> dict[str, float | int]:
    return {
        "n_birth_events": 0, "n_paired_birth_events": 0,
        "paired_birth_fraction": np.nan,
        "null_paired_birth_fraction": np.nan,
        "n_death_events": 0, "n_paired_death_events": 0,
        "paired_death_fraction": np.nan,
        "null_paired_death_fraction": np.nan,
    }


def calibrate(
    table: pd.DataFrame,
    microns_per_pixel: float,
    seconds_per_frame: float,
) -> pd.DataFrame:
    """Convert pixel/frame columns to micrometre/second columns.

    Applied last so that tracking never has to be repeated when the acquisition
    metadata turns up. Nothing here can be inferred from the image files: the
    frame rate stored in an exported video is a playback rate, not the
    acquisition interval, and using it silently rescales every velocity.
    """
    if microns_per_pixel <= 0 or seconds_per_frame <= 0:
        raise ValueError(
            "microns_per_pixel and seconds_per_frame must both be positive"
        )
    out = table.copy()
    for column in out.columns:
        if column.endswith("_px_per_frame"):
            base = column[: -len("_px_per_frame")]
            out[f"{base}_um_per_s"] = (
                out[column] * microns_per_pixel / seconds_per_frame
            )
        elif column.endswith("_px"):
            out[f"{column[:-3]}_um"] = out[column] * microns_per_pixel
    if "frame" in out.columns:
        out["time_s"] = out["frame"] * seconds_per_frame
    return out


def detector_stability(
    frames: list[pd.DataFrame],
    match_radius_px: float,
    *,
    charge_tolerance: float = 1e-6,
) -> dict:
    """Measure detector flicker, independently of any tracking.

    Tracking quality confounds two different failures: defects that genuinely
    move too far to be linked, and detections that appear and vanish between
    frames. Both produce short tracks, and diagnosing them from the tracks is
    circular - the steps you measure are the steps the tracker chose to make.

    This looks only at the per-frame detection tables. For every detection it
    finds the nearest same-charge detection in the next frame, with no
    assignment and no gate. Two numbers come out:

    ``median_nn_px``
        How far defects actually move. Compare against ``null_median_nn_px``,
        the same statistic against a distant frame: if they are similar, there
        is no frame-to-frame continuity at all and the interval is too coarse.
    ``orphan_fraction``
        Detections with no same-charge neighbour within ``match_radius_px``.
        This is flicker. Above roughly 0.15 the tracker is fighting the
        detector rather than the physics, and the fix belongs in detection -
        more scales for the persistence filter - not in the linking gate.
    """
    nearest, null_nearest, per_frame_counts = [], [], []
    rng = np.random.default_rng(0)
    n_frames = len(frames)

    for index in range(n_frames - 1):
        current, following = frames[index], frames[index + 1]
        per_frame_counts.append(len(current))
        if current.empty or following.empty:
            continue

        # The null needs a frame far enough away to share no continuity. If
        # the series is too short to provide one, skip the null rather than
        # comparing the frame with itself - that returns a distance of zero,
        # which reads as a measurement rather than as a missing value.
        candidates = [
            f for f in range(n_frames)
            if abs(f - index) > min(10, max(n_frames // 4, 1))
        ]
        distant = frames[int(rng.choice(candidates))] if candidates else None

        for charge in sorted(set(current["charge"])):
            source = current.loc[
                (current["charge"] - charge).abs() <= charge_tolerance,
                ["x_px", "y_px"],
            ].to_numpy(dtype=float)
            if not len(source):
                continue
            comparisons = [(following, nearest)]
            if distant is not None:
                comparisons.append((distant, null_nearest))
            for other, sink in comparisons:
                target = other.loc[
                    (other["charge"] - charge).abs() <= charge_tolerance,
                    ["x_px", "y_px"],
                ].to_numpy(dtype=float)
                if not len(target):
                    continue
                distance = np.linalg.norm(
                    source[:, None, :] - target[None, :, :], axis=2
                )
                sink.extend(distance.min(axis=1))

    if not nearest:
        return {
            "n_comparisons": 0, "median_nn_px": float("nan"),
            "orphan_fraction": float("nan"), "null_median_nn_px": float("nan"),
            "count_fluctuation": float("nan"), "mean_count": float("nan"),
        }

    nearest = np.asarray(nearest)
    counts = np.asarray(per_frame_counts, dtype=float)
    return {
        "n_comparisons": int(nearest.size),
        "median_nn_px": float(np.median(nearest)),
        "p90_nn_px": float(np.percentile(nearest, 90)),
        "orphan_fraction": float(np.mean(nearest > match_radius_px)),
        "null_median_nn_px": (
            float(np.median(null_nearest)) if null_nearest else float("nan")
        ),
        # Frame-to-frame swing in the raw count, relative to its mean. Real
        # defect number changes slowly; a large swing is detection noise.
        "count_fluctuation": (
            float(np.median(np.abs(np.diff(counts))) / counts.mean())
            if counts.size > 1 and counts.mean() > 0 else float("nan")
        ),
        "mean_count": float(counts.mean()) if counts.size else float("nan"),
    }
