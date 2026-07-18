"""
Alveolar epithelium: state machine, surfactant, collapse and collapse induration.

Each septal segment carries one epithelial state:

    empty (denuded)  ->  AT2  ->  KRT8+ transitional  ->  AT1        (repair)
                                        |
                                        v
                              aberrant basaloid  ->  EMT  ->  mesenchyme

The branch point is the whole disease. A KRT8+ segment either completes
differentiation to AT1, restoring the barrier, or stalls and becomes an
aberrant basaloid cell: senescent, profibrotic, EMT-prone. The literature
places this stall at the centre of IPF: transitional cells that fail to become
AT1 accumulate as KRT5-/KRT17+ aberrant basaloid cells, and profibrotic
mesenchyme feeds back to keep them in that state.

The primary lesion modelled here is exactly that failure: inside the injury
region the KRT8+ -> AT1 rate is suppressed. Nothing else is imposed.

Mechanics closes the loop. AT2 cells make surfactant; surfactant sets alveolar
surface tension; the Laplace pressure 2*gamma/r is opposed by tissue recoil, so
an alveolus whose AT2 population fails will derecruit - and small alveoli go
first. Collapsed alveoli suffer faster epithelial damage, and after long enough
in the collapsed state they become indurated (irreversible), which is the
reported route to permanent loss of alveoli in fibrosis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import AlveolarGeometry

# epithelial states
EMPTY, AT1, AT2, KRT8, ABERRANT = 0, 1, 2, 3, 4
STATE_NAMES = {EMPTY: "empty", AT1: "AT1", AT2: "AT2",
               KRT8: "KRT8+", ABERRANT: "aberrant"}

# alveolar states
OPEN, COLLAPSED, INDURATED = 0, 1, 2


@dataclass
class AlveolarConfig:
    """Parameters of the epithelial stage, in microns, hours and SI-ish units."""

    # ---- geometry ----
    width_um: float = 1400.0
    height_um: float = 1400.0
    alveolar_diameter_um: float = 200.0
    segment_length_um: float = 12.0
    geometry_jitter: float = 0.18

    # ---- time ----
    dt_h: float = 0.25
    total_time_h: float = 1440.0        # 60 days

    # ---- epithelial turnover (1/h) ----
    at1_damage_rate: float = 0.0004     # baseline loss of AT1 (slow turnover)
    at2_damage_rate: float = 0.0004
    at2_repopulation_rate: float = 0.05  # AT2 proliferating into denuded space
    at2_activation_rate: float = 0.06    # AT2 -> KRT8+, driven by denudation
    krt8_to_at1_rate: float = 0.030      # successful differentiation
    krt8_to_aberrant_rate: float = 0.0015  # baseline stalling
    aberrant_emt_rate: float = 0.004     # aberrant -> mesenchyme (leaves epithelium)
    # Clearance of aberrant cells by apoptosis and immune removal. Its failure
    # (apoptosis resistance) is a hallmark of the IPF myofibroblast/epithelial
    # axis, so this rate is one of the decisive knobs.
    aberrant_clearance_rate: float = 0.0025

    # ---- the primary lesion: AT2 that cannot finish differentiating ----
    injury_radius_um: float = 260.0
    repair_failure_factor: float = 0.10  # multiplies krt8_to_at1_rate inside
    # Dysfunctional AT2 keep attempting differentiation and keep failing, so
    # the lesion both drives cells into the transitional state and blocks the
    # exit from it. Driving alone is harmless; blocking alone is harmless.
    injury_activation_boost: float = 10.0
    injury_duration_h: float = 480.0     # 20 days, then the block is lifted

    # ---- profibrotic feedback (aberrant cells and mesenchyme) ----
    profibrotic_secretion: float = 1.0
    profibrotic_decay_per_h: float = 0.02
    profibrotic_range_um: float = 90.0
    repair_inhibition_strength: float = 3.0   # how strongly P blocks KRT8 -> AT1
    stall_promotion_strength: float = 4.0     # how strongly P promotes stalling

    # ---- surfactant and mechanics ----
    surfactant_production_per_h: float = 0.12
    surfactant_loss_per_h: float = 0.012
    surface_tension_min_mN_m: float = 5.0     # well-surfacted alveolus
    surface_tension_max_mN_m: float = 40.0    # surfactant-depleted
    tissue_recoil_Pa: float = 420.0           # opposes collapse
    reopening_hysteresis: float = 0.75        # must fall well below to reopen
    collapse_damage_factor: float = 6.0       # damage is faster in closed alveoli
    collapse_surfactant_penalty: float = 0.4  # closed alveoli make less surfactant
    induration_time_h: float = 240.0          # collapsed this long -> irreversible

    seed: int = 0

    def validate(self) -> None:
        if self.dt_h <= 0 or self.total_time_h <= 0:
            raise ValueError("dt_h and total_time_h must be positive.")
        if not 0 <= self.repair_failure_factor <= 1:
            raise ValueError("repair_failure_factor must lie in [0, 1].")
        if self.surface_tension_min_mN_m >= self.surface_tension_max_mN_m:
            raise ValueError("surface tension min must be below max.")
        if self.injury_activation_boost < 0:
            raise ValueError("injury_activation_boost must be non-negative.")
        if self.aberrant_clearance_rate < 0:
            raise ValueError("aberrant_clearance_rate must be non-negative.")
        if self.tissue_recoil_Pa <= 0:
            raise ValueError("tissue_recoil_Pa must be positive.")
        if not 0 < self.reopening_hysteresis <= 1:
            raise ValueError("reopening_hysteresis must lie in (0, 1].")
        for name in ("profibrotic_range_um", "induration_time_h",
                     "alveolar_diameter_um", "segment_length_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class AlveolarSimulation:
    """Epithelium on an alveolar tessellation, with surfactant-driven collapse."""

    def __init__(self, config: AlveolarConfig):
        config.validate()
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.geometry = AlveolarGeometry(
            width_um=config.width_um,
            height_um=config.height_um,
            alveolar_diameter_um=config.alveolar_diameter_um,
            jitter=config.geometry_jitter,
            segment_length_um=config.segment_length_um,
            seed=config.seed,
        )
        g = self.geometry
        n_seg = g.n_segments
        n_alv = g.n_alveoli
        if n_seg == 0 or n_alv == 0:
            raise ValueError("Geometry produced no alveoli; enlarge the domain.")

        # --- epithelium: healthy lung is mostly AT1 by area, AT2 at corners ---
        self.state = np.full(n_seg, AT1, dtype=np.int8)
        # AT2 sit at regular intervals along the septa rather than at random,
        # so no alveolus starts surfactant-poor by chance.
        stride = 8
        self.state[np.arange(n_seg) % stride == 0] = AT2

        # --- alveolar state ---
        self.alveolar_state = np.full(n_alv, OPEN, dtype=np.int8)
        self.surfactant = np.ones(n_alv)
        self.time_collapsed_h = np.zeros(n_alv)
        self.radius_um = g.alveolar_radii.copy()
        self.open_radius_um = g.alveolar_radii.copy()

        # --- profibrotic signal on a coarse grid ---
        self.grid_step_um = max(config.segment_length_um, 10.0)
        self.nx = int(np.ceil(config.width_um / self.grid_step_um))
        self.ny = int(np.ceil(config.height_um / self.grid_step_um))
        self.profibrotic = np.zeros((self.ny, self.nx))

        # --- the lesion: AT2 that cannot complete differentiation ---
        centre = np.array([config.width_um / 2, config.height_um / 2])
        self.injury_centre = centre
        self.segment_in_injury = (
            np.linalg.norm(g.segment_centre - centre, axis=1)
            <= config.injury_radius_um
        )

        self.mesenchymal_released = 0
        self.time_h = 0.0
        self.history: list[dict] = []

    # ------------------------------------------------------------- utilities
    def _grid_index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip((points[:, 0] / self.grid_step_um).astype(int), 0, self.nx - 1)
        iy = np.clip((points[:, 1] / self.grid_step_um).astype(int), 0, self.ny - 1)
        return iy, ix

    def _segment_profibrotic(self) -> np.ndarray:
        iy, ix = self._grid_index(self.geometry.segment_centre)
        return self.profibrotic[iy, ix]

    def _alveolar_counts(self, state: int) -> np.ndarray:
        """Number of segments in a given epithelial state, per alveolus."""
        g = self.geometry
        counts = np.zeros(g.n_alveoli)
        mask = self.state == state
        for column in (0, 1):
            owners = g.segment_alveoli[:, column]
            valid = mask & (owners >= 0)
            if valid.any():
                np.add.at(counts, owners[valid], 1.0)
        return counts

    def _segment_alveolar_property(self, values: np.ndarray) -> np.ndarray:
        """Average an per-alveolus quantity onto its segments."""
        g = self.geometry
        out = np.zeros(g.n_segments)
        weight = np.zeros(g.n_segments)
        for column in (0, 1):
            owners = g.segment_alveoli[:, column]
            valid = owners >= 0
            out[valid] += values[owners[valid]]
            weight[valid] += 1.0
        return out / np.maximum(weight, 1.0)

    # -------------------------------------------------------------- dynamics
    def _update_profibrotic(self) -> None:
        cfg = self.cfg
        g = self.geometry
        source = np.zeros((self.ny, self.nx))
        emitting = self.state == ABERRANT
        if emitting.any():
            iy, ix = self._grid_index(g.segment_centre[emitting])
            np.add.at(source, (iy, ix), cfg.profibrotic_secretion)
        sigma = cfg.profibrotic_range_um / self.grid_step_um
        self.profibrotic += cfg.dt_h * (
            gaussian_filter(source, sigma) - cfg.profibrotic_decay_per_h * self.profibrotic
        )
        np.clip(self.profibrotic, 0.0, None, out=self.profibrotic)

    def _update_epithelium(self) -> None:
        cfg = self.cfg
        g = self.geometry
        dt = cfg.dt_h
        n = g.n_segments

        signal = self._segment_profibrotic()
        collapsed = self.alveolar_state != OPEN
        seg_collapsed = self._segment_alveolar_property(collapsed.astype(float))
        damage_boost = 1.0 + (cfg.collapse_damage_factor - 1.0) * seg_collapsed

        draw = self.rng.random(n)
        new_state = self.state.copy()

        # --- AT1 and AT2 loss (accelerated in collapsed alveoli) ---
        is_at1 = self.state == AT1
        hazard = cfg.at1_damage_rate * damage_boost
        new_state[is_at1 & (draw < hazard * dt)] = EMPTY

        is_at2 = self.state == AT2
        hazard = cfg.at2_damage_rate * damage_boost
        new_state[is_at2 & (draw < hazard * dt)] = EMPTY

        # --- denuded segments repopulated by AT2 ---
        is_empty = self.state == EMPTY
        at2_fraction = self._segment_alveolar_property(
            self._alveolar_counts(AT2) / np.maximum(self._segments_per_alveolus(), 1)
        )
        hazard = cfg.at2_repopulation_rate * np.clip(at2_fraction * 4.0, 0.0, 1.0)
        new_state[is_empty & (draw < hazard * dt)] = AT2

        # --- AT2 activation into the transitional state, driven by denudation ---
        denuded = self._segment_alveolar_property(
            self._alveolar_counts(EMPTY) / np.maximum(self._segments_per_alveolus(), 1)
        )
        hazard = cfg.at2_activation_rate * np.clip(denuded * 3.0, 0.0, 1.0)
        if self.time_h < cfg.injury_duration_h:
            hazard = np.where(self.segment_in_injury,
                              cfg.at2_activation_rate * cfg.injury_activation_boost
                              * np.clip(0.1 + denuded * 3.0, 0.0, 1.0),
                              hazard)
        new_state[is_at2 & (draw < hazard * dt) & (new_state == AT2)] = KRT8

        # --- the branch point ---
        is_krt8 = self.state == KRT8
        repair = cfg.krt8_to_at1_rate / (1.0 + cfg.repair_inhibition_strength * signal)
        if self.time_h < cfg.injury_duration_h:
            repair = np.where(self.segment_in_injury,
                              repair * cfg.repair_failure_factor, repair)
        stall = cfg.krt8_to_aberrant_rate * (
            1.0 + cfg.stall_promotion_strength * signal
        )
        total = repair + stall
        fires = is_krt8 & (draw < total * dt)
        if fires.any():
            second = self.rng.random(n)
            to_at1 = fires & (second < repair / np.maximum(total, 1e-12))
            new_state[to_at1] = AT1
            new_state[fires & ~to_at1] = ABERRANT

        # --- aberrant cells: cleared, or lost to the mesenchyme via EMT ---
        is_aberrant = self.state == ABERRANT
        total_exit = cfg.aberrant_emt_rate + cfg.aberrant_clearance_rate
        exits = is_aberrant & (draw < total_exit * dt)
        if exits.any():
            second = self.rng.random(n)
            via_emt = exits & (second < cfg.aberrant_emt_rate / total_exit)
            self.mesenchymal_released += int(via_emt.sum())
            new_state[exits] = EMPTY

        self.state = new_state

    def _segments_per_alveolus(self) -> np.ndarray:
        if not hasattr(self, "_seg_per_alv"):
            g = self.geometry
            counts = np.zeros(g.n_alveoli)
            for column in (0, 1):
                owners = g.segment_alveoli[:, column]
                valid = owners >= 0
                np.add.at(counts, owners[valid], 1.0)
            self._seg_per_alv = counts
        return self._seg_per_alv

    def _update_surfactant_and_collapse(self) -> None:
        cfg = self.cfg
        dt = cfg.dt_h

        at2_counts = self._alveolar_counts(AT2)
        reference = np.maximum(self._segments_per_alveolus() * 0.12, 1.0)
        supply = np.clip(at2_counts / reference, 0.0, 1.5)
        closed = self.alveolar_state != OPEN
        supply = np.where(closed, supply * cfg.collapse_surfactant_penalty, supply)

        self.surfactant += dt * (
            cfg.surfactant_production_per_h * supply * (1.0 - self.surfactant)
            - cfg.surfactant_loss_per_h * self.surfactant
        )
        np.clip(self.surfactant, 0.0, 1.0, out=self.surfactant)

        # Laplace balance: collapse when surface tension beats tissue recoil.
        gamma = (
            cfg.surface_tension_max_mN_m
            - (cfg.surface_tension_max_mN_m - cfg.surface_tension_min_mN_m)
            * self.surfactant
        )
        # 2*gamma/r with gamma in mN/m and r in um gives kPa*1e-3 -> Pa via 1e3
        laplace_Pa = 2.0 * gamma / np.maximum(self.radius_um, 1e-6) * 1e3

        open_now = self.alveolar_state == OPEN
        collapsing = open_now & (laplace_Pa > cfg.tissue_recoil_Pa)
        self.alveolar_state[collapsing] = COLLAPSED
        self.radius_um[collapsing] = self.open_radius_um[collapsing] * 0.35

        collapsed_now = self.alveolar_state == COLLAPSED
        reopening = collapsed_now & (
            laplace_Pa < cfg.tissue_recoil_Pa * cfg.reopening_hysteresis
        )
        self.alveolar_state[reopening] = OPEN
        self.radius_um[reopening] = self.open_radius_um[reopening]
        self.time_collapsed_h[reopening] = 0.0

        still_collapsed = self.alveolar_state == COLLAPSED
        self.time_collapsed_h[still_collapsed] += dt
        indurating = still_collapsed & (
            self.time_collapsed_h >= cfg.induration_time_h
        )
        self.alveolar_state[indurating] = INDURATED

    def step(self) -> None:
        self._update_profibrotic()
        self._update_epithelium()
        self._update_surfactant_and_collapse()
        self.time_h += self.cfg.dt_h

    # -------------------------------------------------------------- readouts
    def metrics(self) -> dict:
        g = self.geometry
        n_seg = max(g.n_segments, 1)
        n_alv = max(g.n_alveoli, 1)
        return {
            "time_h": self.time_h,
            "time_d": self.time_h / 24.0,
            "frac_AT1": float((self.state == AT1).mean()),
            "frac_AT2": float((self.state == AT2).mean()),
            "frac_KRT8": float((self.state == KRT8).mean()),
            "frac_aberrant": float((self.state == ABERRANT).mean()),
            "frac_denuded": float((self.state == EMPTY).mean()),
            "frac_open": float((self.alveolar_state == OPEN).mean()),
            "frac_collapsed": float((self.alveolar_state == COLLAPSED).mean()),
            "frac_indurated": float((self.alveolar_state == INDURATED).mean()),
            "mean_surfactant": float(self.surfactant.mean()),
            "mesenchymal_released": int(self.mesenchymal_released),
            "mean_profibrotic": float(self.profibrotic.mean()),
            "n_alveoli": n_alv,
            "n_segments": n_seg,
        }

    def run(self, record_every_h: float = 12.0, callback=None) -> "AlveolarSimulation":
        cfg = self.cfg
        n_steps = int(round(cfg.total_time_h / cfg.dt_h))
        every = max(1, int(round(record_every_h / cfg.dt_h)))
        self.history.append(self.metrics())
        for index in range(1, n_steps + 1):
            self.step()
            if index % every == 0:
                self.history.append(self.metrics())
                if callback is not None:
                    callback(self)
        return self
