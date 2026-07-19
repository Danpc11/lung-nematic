"""
Which defects survive, and does the phenotype follow them?

Topological defects in an active nematic are normally transient: +1/2 and -1/2
partners nucleate in pairs, drift, and annihilate. The interesting claim in
fibrosis is that some of them stop doing that - that matrix laid down by the
cells immobilises the texture, and a defect that stops moving becomes a
persistent stress concentrator rather than a passing feature.

This module tests that directly. It links defects between snapshots into
trajectories, measures how long each one lives, and samples the local state -
myofibroblast fraction, stiffness, tidal strain, whether the site sits in a
collapsed alveolus - along every trajectory. That makes two questions
answerable:

  1. Which defects become stable, and do +1/2 and -1/2 differ?
  2. Do stable defects hold the myofibroblast phenotype around them, compared
     with transient defects and with matched random tissue?

The matching is nearest-neighbour within a maximum displacement, applied
separately per charge, so a +1/2 can never be mistaken for a -1/2. Defects that
cannot be matched are treated as having annihilated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DefectTrack:
    """One defect followed through time."""

    track_id: int
    charge: float
    birth_h: float
    positions: list[np.ndarray] = field(default_factory=list)
    times_h: list[float] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    alive: bool = True

    @property
    def lifetime_h(self) -> float:
        if len(self.times_h) < 2:
            return 0.0
        return float(self.times_h[-1] - self.times_h[0])

    @property
    def displacement_um(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(self.positions[-1] - self.positions[0]))

    @property
    def path_length_um(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        steps = np.diff(np.array(self.positions), axis=0)
        return float(np.linalg.norm(steps, axis=1).sum())

    def mean_sample(self, key: str) -> float:
        values = [s[key] for s in self.samples if key in s]
        return float(np.mean(values)) if values else float("nan")


class DefectTracker:
    """Links defects across snapshots and records the local state along each."""

    def __init__(self, max_displacement_um: float = 70.0):
        self.max_displacement_um = float(max_displacement_um)
        self.tracks: list[DefectTrack] = []
        self._active: dict[int, int] = {}      # track_id -> index in tracks
        self._next_id = 0

    def update(self, time_h: float, defects: dict, sampler=None) -> None:
        """Add one snapshot. ``sampler(point) -> dict`` reports the local state."""
        for charge, key in ((0.5, "plus"), (-0.5, "minus")):
            points = defects.get(key, np.zeros((0, 2)))
            self._match_charge(time_h, charge, np.asarray(points), sampler)

    def _match_charge(self, time_h: float, charge: float,
                      points: np.ndarray, sampler) -> None:
        live = [
            index for index, track in enumerate(self.tracks)
            if track.alive and track.charge == charge
        ]
        unmatched_tracks = set(live)
        used = set()

        # greedy nearest-neighbour matching, closest pairs first
        if live and len(points):
            pairs = []
            for index in live:
                last = self.tracks[index].positions[-1]
                for point_index, point in enumerate(points):
                    distance = float(np.linalg.norm(point - last))
                    if distance <= self.max_displacement_um:
                        pairs.append((distance, index, point_index))
            pairs.sort()
            for distance, index, point_index in pairs:
                if index not in unmatched_tracks or point_index in used:
                    continue
                track = self.tracks[index]
                track.positions.append(points[point_index])
                track.times_h.append(time_h)
                if sampler is not None:
                    track.samples.append(sampler(points[point_index]))
                unmatched_tracks.discard(index)
                used.add(point_index)

        # tracks with no partner in this frame have annihilated
        for index in unmatched_tracks:
            self.tracks[index].alive = False

        # leftover detections are new defects
        for point_index, point in enumerate(points):
            if point_index in used:
                continue
            track = DefectTrack(track_id=self._next_id, charge=charge,
                                birth_h=time_h)
            track.positions.append(point)
            track.times_h.append(time_h)
            if sampler is not None:
                track.samples.append(sampler(point))
            self.tracks.append(track)
            self._next_id += 1

    # ------------------------------------------------------------- summaries
    def summary(self, stable_threshold_h: float) -> dict:
        """Split tracks into stable and transient and compare their context."""
        result: dict = {}
        for charge, label in ((0.5, "plus"), (-0.5, "minus")):
            tracks = [t for t in self.tracks if t.charge == charge]
            if not tracks:
                continue
            lifetimes = np.array([t.lifetime_h for t in tracks])
            stable = [t for t in tracks if t.lifetime_h >= stable_threshold_h]
            transient = [t for t in tracks if t.lifetime_h < stable_threshold_h]
            result[label] = {
                "n_tracks": len(tracks),
                "n_stable": len(stable),
                "stable_fraction": len(stable) / len(tracks),
                "median_lifetime_h": float(np.median(lifetimes)),
                "max_lifetime_h": float(lifetimes.max()),
                "stable": _context(stable),
                "transient": _context(transient),
            }
        return result


def _context(tracks: list[DefectTrack]) -> dict:
    """Average local state over a group of tracks."""
    if not tracks:
        return {}
    keys = set()
    for track in tracks:
        for sample in track.samples:
            keys.update(sample.keys())
    out = {"n": len(tracks)}
    for key in sorted(keys):
        values = [track.mean_sample(key) for track in tracks]
        values = [v for v in values if v == v]
        if values:
            out[key] = float(np.mean(values))
    out["mean_displacement_um"] = float(
        np.mean([t.displacement_um for t in tracks])
    )
    out["mean_path_um"] = float(np.mean([t.path_length_um for t in tracks]))
    return out


def make_sampler(coupled, radius_um: float = 45.0):
    """Build a sampler reporting the local tissue state around a point."""
    mes = coupled.mesenchyme
    ep = coupled.epithelium

    def sampler(point: np.ndarray) -> dict:
        x, y = float(point[0]), float(point[1])
        ix = int(np.clip(x / mes.grid_step, 0, mes.nx - 1))
        iy = int(np.clip(y / mes.grid_step, 0, mes.ny - 1))

        near = np.zeros(0, dtype=bool)
        if mes.n_cells:
            near = (np.hypot(mes.x - x, mes.y - y) <= radius_um)
        myo_fraction = float(mes.myo[near].mean()) if near.any() else float("nan")

        label = mes.alveolus_label[iy, ix]
        collapsed = bool(label >= 0 and ep.alveolar_state[label] != 0)

        return {
            "myo_fraction": myo_fraction,
            "n_cells_near": int(near.sum()),
            "stiffness_kPa": float(mes.stiffness_kPa[iy, ix]),
            "strain_ratio": float(
                mes.strain[iy, ix] / max(mes.cfg.tidal_strain, 1e-9)
            ),
            "in_collapsed": float(collapsed),
        }

    return sampler


def random_control(coupled, n_points: int, radius_um: float = 45.0,
                   seed: int = 0) -> dict:
    """Same measurements at random permitted locations, as a null comparison."""
    mes = coupled.mesenchyme
    rng = np.random.default_rng(seed)
    mask = mes.permitted_mask()
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {}
    picks = rng.choice(xs.size, size=min(n_points, xs.size), replace=False)
    sampler = make_sampler(coupled, radius_um)
    samples = [
        sampler(np.array([(xs[i] + 0.5) * mes.grid_step,
                          (ys[i] + 0.5) * mes.grid_step]))
        for i in picks
    ]
    keys = samples[0].keys()
    out = {"n": len(samples)}
    for key in keys:
        values = [s[key] for s in samples if s[key] == s[key]]
        if values:
            out[key] = float(np.mean(values))
    return out
