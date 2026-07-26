"""Small but genuinely three-dimensional alveolar fibrosis simulation.

Unlike :mod:`simulations.alveolar.particle_render`, this model does not invent a
display-only z coordinate.  Alveolar centres, epithelial cells, fibroblasts,
orientations, migration, neighbourhoods and respiratory deformation all live
in three spatial dimensions.

The model is deliberately small.  Seven near-touching spherical alveolar units
are sufficient to exercise interdependence, loss of ventilation and formation
of a fibroblastic focus without the memory cost of a full acinus.  It is a
mechanistic research prototype, not a calibrated clinical predictor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EMPTY, AT1, AT2, KRT8, ABERRANT = 0, 1, 2, 3, 4
OPEN, COLLAPSED, INDURATED = 0, 1, 2

EPITHELIAL_NAMES = {
    EMPTY: "empty",
    AT1: "AT1",
    AT2: "AT2",
    KRT8: "KRT8+",
    ABERRANT: "aberrant",
}
ALVEOLAR_NAMES = {
    OPEN: "open",
    COLLAPSED: "collapsed",
    INDURATED: "indurated",
}

# The visual model reaches its representative fibrotic endpoint in roughly
# twelve kinetic days.  The clinical-scale preset maps that trajectory onto
# two calendar years.  This is a disease-level calibration, not a claim that
# every cell transition has been measured with this duration in human lung.
HUMAN_CHRONIC_REFERENCE_DAYS = 2.0 * 365.25
HUMAN_CHRONIC_RATE_SCALE = 12.0 / HUMAN_CHRONIC_REFERENCE_DAYS


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Return unit vectors, preserving zero rows."""

    if vectors.size == 0:
        return vectors.copy()
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


def _fibonacci_sphere(count: int, phase: float = 0.0) -> np.ndarray:
    """Approximately uniform deterministic unit vectors on a sphere."""

    if count <= 0:
        return np.zeros((0, 3), dtype=float)
    index = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = golden_angle * index + phase
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


@dataclass(frozen=True)
class Alveolar3DConfig:
    """Numerical and biological parameters, in micrometres and hours."""

    n_alveoli: int = 7
    alveolar_diameter_um: float = 150.0
    interstitial_thickness_um: float = 5.0
    # Human morphometry gives about 67 AT2 and 40 AT1 cells for an average
    # 207,000 µm² alveolus.  The simulated 150 µm sphere is smaller, so 20 and
    # 12 preserve the 1.67 number ratio while matching its available surface.
    epithelial_at2_per_alveolus: int = 20
    epithelial_at1_per_alveolus: int = 12
    # Mean air-facing area per human cell (Crapo et al., 1982).
    at2_apical_surface_um2: float = 183.0
    at1_apical_surface_um2: float = 5_098.0
    n_resident_fibroblasts: int = 84
    max_mesenchymal_cells: int = 350

    dt_h: float = 2.0
    total_time_h: float = 24.0 * 12.0
    # Multiplies biological kinetics while the displayed clock remains in real
    # days. 0.4 means that 30 displayed days contain 12 accelerated-model days.
    kinetic_rate_scale: float = 1.0
    temporal_calibration: str = "accelerated_visual"
    seed: int = 12

    healthy_stiffness_kpa: float = 2.0
    activation_stiffness_kpa: float = 6.0
    max_stiffness_kpa: float = 40.0
    collagen_deposition_kpa_per_cell_day: float = 0.015
    collagen_degradation_per_day: float = 0.010

    healthy_tidal_strain: float = 0.05
    overstrain_threshold: float = 0.09
    breaths_per_min: float = 15.0

    injury_duration_days: float = 9.0
    at1_damage_per_day: float = 0.010
    at2_to_krt8_per_day: float = 0.080
    injury_activation_multiplier: float = 8.0
    krt8_to_at1_per_day: float = 0.25
    krt8_to_aberrant_per_day: float = 0.035
    injury_repair_fraction: float = 0.08
    aberrant_emt_per_day: float = 0.16
    aberrant_clearance_per_day: float = 0.025
    at2_repopulation_per_day: float = 0.08

    surfactant_recovery_per_day: float = 0.65
    surfactant_loss_per_day: float = 0.12
    collapse_surfactant_threshold: float = 0.56
    reopening_surfactant_threshold: float = 0.80
    collapse_stiffness_kpa: float = 10.0
    induration_delay_days: float = 10.0
    collapsed_radius_fraction: float = 0.35

    profibrotic_from_aberrant_per_day: float = 0.90
    profibrotic_from_myo_per_day: float = 0.018
    profibrotic_decay_per_day: float = 0.22
    profibrotic_diffusion_per_day: float = 0.18

    fibroblast_speed_um_per_day: float = 18.0
    fibroblast_length_um: float = 32.0
    fibroblast_width_um: float = 8.0
    alignment_radius_um: float = 38.0
    alignment_strength: float = 0.38
    steric_strength_um_per_day: float = 9.0
    activation_rate_per_day: float = 0.22
    strain_protection_strength: float = 5.5
    proliferation_per_day: float = 0.18
    fibroblast_death_per_day: float = 0.025
    myofibroblast_death_per_day: float = 0.007

    def validate(self) -> None:
        if not 1 <= self.n_alveoli <= 7:
            raise ValueError("n_alveoli must lie between 1 and 7.")
        if self.alveolar_diameter_um <= 0:
            raise ValueError("alveolar_diameter_um must be positive.")
        if not 0 < self.interstitial_thickness_um < self.alveolar_diameter_um:
            raise ValueError("interstitial_thickness_um is invalid.")
        if self.epithelial_at1_per_alveolus < 1:
            raise ValueError("epithelial_at1_per_alveolus must be positive.")
        if self.epithelial_at2_per_alveolus < 1:
            raise ValueError("epithelial_at2_per_alveolus must be positive.")
        if self.at1_apical_surface_um2 <= 0 or self.at2_apical_surface_um2 <= 0:
            raise ValueError("Epithelial apical surface areas must be positive.")
        if self.dt_h <= 0 or self.total_time_h <= 0:
            raise ValueError("dt_h and total_time_h must be positive.")
        if self.kinetic_rate_scale <= 0:
            raise ValueError("kinetic_rate_scale must be positive.")
        if not 0 < self.healthy_tidal_strain < 1:
            raise ValueError("healthy_tidal_strain must lie in (0, 1).")
        if not 0 < self.collapsed_radius_fraction < 1:
            raise ValueError("collapsed_radius_fraction must lie in (0, 1).")
        if self.max_mesenchymal_cells < self.n_resident_fibroblasts:
            raise ValueError(
                "max_mesenchymal_cells cannot be below the resident population."
            )

    @property
    def open_radius_um(self) -> float:
        return 0.5 * self.alveolar_diameter_um

    @property
    def centre_spacing_um(self) -> float:
        return self.alveolar_diameter_um + self.interstitial_thickness_um

    @property
    def healthy_at2_to_at1_number_ratio(self) -> float:
        return (
            self.epithelial_at2_per_alveolus
            / self.epithelial_at1_per_alveolus
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )


def compact_alveolar_centres(config: Alveolar3DConfig) -> np.ndarray:
    """Central alveolus plus six neighbours on the Cartesian axes."""

    spacing = config.centre_spacing_um
    directions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return spacing * directions[: config.n_alveoli]


class Alveolar3DSimulation:
    """Coupled epithelial, mesenchymal and matrix dynamics in real 3D."""

    def __init__(self, config: Alveolar3DConfig | None = None):
        self.cfg = config or Alveolar3DConfig()
        self.cfg.validate()
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.centres = compact_alveolar_centres(cfg)
        self.n_alveoli = len(self.centres)
        self.open_radius_um = np.full(self.n_alveoli, cfg.open_radius_um)
        self.radius_um = self.open_radius_um.copy()
        self.alveolar_state = np.full(self.n_alveoli, OPEN, dtype=np.int8)
        self.surfactant = np.ones(self.n_alveoli)
        self.stiffness_kpa = np.full(
            self.n_alveoli,
            cfg.healthy_stiffness_kpa,
        )
        self.profibrotic = np.zeros(self.n_alveoli)
        self.collapsed_time_days = np.zeros(self.n_alveoli)

        self.neighbour_matrix = self._build_neighbour_matrix()
        self.injury_alveolus = 0

        self.epithelial_owner: np.ndarray
        self.epithelial_normal: np.ndarray
        self.epithelial_state: np.ndarray
        self._seed_epithelium()

        self.fibroblast_xyz = np.zeros((0, 3), dtype=float)
        self.fibroblast_orientation = np.zeros((0, 3), dtype=float)
        self.myofibroblast = np.zeros(0, dtype=bool)
        self.fibroblast_owner = np.zeros(0, dtype=int)
        self._seed_fibroblasts()

        self.tidal_strain = np.full(
            self.n_alveoli,
            cfg.healthy_tidal_strain,
        )
        self.time_h = 0.0
        self.mesenchymal_released = 0

    # --------------------------------------------------------------- geometry
    def _build_neighbour_matrix(self) -> np.ndarray:
        delta = self.centres[:, None, :] - self.centres[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        return (
            (distance > 0)
            & (distance <= self.cfg.centre_spacing_um * 1.05)
        ).astype(float)

    def _seed_epithelium(self) -> None:
        cfg = self.cfg
        owners: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        states: list[np.ndarray] = []
        n_at2 = cfg.epithelial_at2_per_alveolus
        n_at1 = cfg.epithelial_at1_per_alveolus
        for owner in range(self.n_alveoli):
            count = n_at2 + n_at1
            owner_normals = _fibonacci_sphere(count, phase=owner * 0.47)
            # Interleave compact AT2 cells with the less numerous, broad AT1.
            order = np.arange(count)
            owner_state = np.full(count, AT2, dtype=np.int8)
            owner_state[order % 3 == 2] = AT1
            # Exact configured counts, including non-default experimental ratios.
            owner_state[:] = AT2
            at1_positions = np.linspace(0, count - 1, n_at1, dtype=int)
            owner_state[np.unique(at1_positions)] = AT1
            current_at1 = int(np.sum(owner_state == AT1))
            if current_at1 < n_at1:
                available = np.flatnonzero(owner_state == AT2)
                owner_state[available[: n_at1 - current_at1]] = AT1
            owners.append(np.full(count, owner, dtype=int))
            normals.append(owner_normals)
            states.append(owner_state)
        self.epithelial_owner = np.concatenate(owners)
        self.epithelial_normal = np.vstack(normals)
        self.epithelial_state = np.concatenate(states)

    def epithelial_xyz(self, inspiration_fraction: float = 0.0) -> np.ndarray:
        """Actual 3D positions of epithelial cell centres."""

        scale = self.respiratory_scale(inspiration_fraction)
        radius = self.radius_um * scale
        return (
            self.centres[self.epithelial_owner]
            + self.epithelial_normal
            * radius[self.epithelial_owner, None]
        )

    def respiratory_scale(self, inspiration_fraction: float) -> np.ndarray:
        """Per-alveolus radial scale for a respiratory phase in [0, 1]."""

        fraction = float(np.clip(inspiration_fraction, 0.0, 1.0))
        scale = 1.0 + self.tidal_strain * fraction
        scale[self.alveolar_state != OPEN] = 1.0
        return scale

    def displayed_fibroblast_xyz(
        self,
        inspiration_fraction: float = 0.0,
    ) -> np.ndarray:
        """Move interstitial agents with the wall during the rendered breath."""

        if self.fibroblast_xyz.size == 0:
            return self.fibroblast_xyz.copy()
        centre = self.centres[self.fibroblast_owner]
        scale = self.respiratory_scale(inspiration_fraction)[
            self.fibroblast_owner
        ]
        return centre + (self.fibroblast_xyz - centre) * scale[:, None]

    # --------------------------------------------------------------- seeding
    def _seed_fibroblasts(self) -> None:
        cfg = self.cfg
        count = cfg.n_resident_fibroblasts
        owner = np.arange(count, dtype=int) % self.n_alveoli
        normals = _fibonacci_sphere(count, phase=1.13)
        shell_offset = self.rng.uniform(
            -0.5 * cfg.interstitial_thickness_um,
            0.5 * cfg.interstitial_thickness_um,
            count,
        )
        radius = self.open_radius_um[owner] + shell_offset
        self.fibroblast_xyz = self.centres[owner] + normals * radius[:, None]
        random_orientation = self.rng.normal(size=(count, 3))
        # Project orientation into the local tangent plane.
        random_orientation -= (
            np.sum(random_orientation * normals, axis=1, keepdims=True)
            * normals
        )
        self.fibroblast_orientation = _normalise(random_orientation)
        self.myofibroblast = np.zeros(count, dtype=bool)
        self.fibroblast_owner = owner

    # --------------------------------------------------------------- readouts
    @property
    def n_mesenchymal(self) -> int:
        return len(self.fibroblast_xyz)

    @property
    def n_myofibroblast(self) -> int:
        return int(np.sum(self.myofibroblast))

    @property
    def biological_time_days(self) -> float:
        """Time experienced by rate-limited biology after kinetic scaling."""

        return (
            self.time_h
            / 24.0
            * self.cfg.kinetic_rate_scale
        )

    def _counts_by_alveolus(self, values: np.ndarray | None = None) -> np.ndarray:
        counts = np.zeros(self.n_alveoli, dtype=float)
        if values is None:
            values = np.ones(self.n_mesenchymal)
        if self.n_mesenchymal:
            np.add.at(counts, self.fibroblast_owner, values)
        return counts

    def epithelial_counts(self, state: int) -> np.ndarray:
        counts = np.zeros(self.n_alveoli, dtype=float)
        selected = self.epithelial_state == state
        np.add.at(counts, self.epithelial_owner[selected], 1.0)
        return counts

    # ------------------------------------------------------------- epithelium
    def _update_epithelium(self, dt_days: float) -> None:
        cfg = self.cfg
        state = self.epithelial_state
        owner = self.epithelial_owner
        draw = self.rng.random(len(state))
        new_state = state.copy()
        injury_active = self.biological_time_days < cfg.injury_duration_days
        injured = owner == self.injury_alveolus

        at1 = state == AT1
        damage = np.full(len(state), cfg.at1_damage_per_day)
        if injury_active:
            damage[injured] *= 5.0
        overstrain = np.clip(
            self.tidal_strain[owner] - cfg.overstrain_threshold,
            0.0,
            None,
        )
        damage += 0.5 * overstrain / max(cfg.overstrain_threshold, 1e-9)
        new_state[at1 & (draw < damage * dt_days)] = EMPTY

        at2 = state == AT2
        activation = np.full(len(state), cfg.at2_to_krt8_per_day)
        activation[~injured] *= 0.10
        if injury_active:
            activation[injured] *= cfg.injury_activation_multiplier
        new_state[
            at2
            & (draw < activation * dt_days)
            & (new_state == AT2)
        ] = KRT8

        krt8 = state == KRT8
        repair = np.full(len(state), cfg.krt8_to_at1_per_day)
        if injury_active:
            repair[injured] *= cfg.injury_repair_fraction
        repair /= 1.0 + 2.5 * self.profibrotic[owner]
        stall = cfg.krt8_to_aberrant_per_day * (
            1.0 + 5.0 * self.profibrotic[owner]
        )
        total = repair + stall
        firing = krt8 & (draw < total * dt_days)
        second = self.rng.random(len(state))
        to_at1 = firing & (second < repair / np.maximum(total, 1e-12))
        new_state[to_at1] = AT1
        new_state[firing & ~to_at1] = ABERRANT

        aberrant = state == ABERRANT
        total_exit = cfg.aberrant_emt_per_day + cfg.aberrant_clearance_per_day
        exiting = aberrant & (draw < total_exit * dt_days)
        second = self.rng.random(len(state))
        emt = exiting & (
            second < cfg.aberrant_emt_per_day / max(total_exit, 1e-12)
        )
        if np.any(emt):
            xyz = (
                self.centres[owner[emt]]
                + self.epithelial_normal[emt]
                * self.radius_um[owner[emt], None]
                * 0.92
            )
            self._append_mesenchymal(
                xyz,
                owner[emt],
                activated=np.ones(int(np.sum(emt)), dtype=bool),
            )
            self.mesenchymal_released += int(np.sum(emt))
        new_state[exiting] = EMPTY

        empty = state == EMPTY
        at2_fraction = self.epithelial_counts(AT2) / max(
            cfg.epithelial_at2_per_alveolus,
            1,
        )
        repopulation = (
            cfg.at2_repopulation_per_day
            * np.clip(at2_fraction[owner], 0.0, 1.0)
        )
        new_state[empty & (draw < repopulation * dt_days)] = AT2
        self.epithelial_state = new_state

    # -------------------------------------------------------------- alveoli
    def _update_signals_and_alveoli(self, dt_days: float) -> None:
        cfg = self.cfg
        at2 = self.epithelial_counts(AT2)
        aberrant = self.epithelial_counts(ABERRANT)
        myo = self._counts_by_alveolus(self.myofibroblast.astype(float))

        target_supply = np.clip(
            at2 / max(cfg.epithelial_at2_per_alveolus, 1),
            0.0,
            1.0,
        )
        self.surfactant += dt_days * (
            cfg.surfactant_recovery_per_day
            * target_supply
            * (1.0 - self.surfactant)
            - cfg.surfactant_loss_per_day * self.surfactant
        )
        np.clip(self.surfactant, 0.0, 1.0, out=self.surfactant)

        source = (
            cfg.profibrotic_from_aberrant_per_day * aberrant
            + cfg.profibrotic_from_myo_per_day * myo
        )
        degree = self.neighbour_matrix.sum(axis=1)
        diffusion = (
            self.neighbour_matrix @ self.profibrotic
            - degree * self.profibrotic
        )
        self.profibrotic += dt_days * (
            source
            - cfg.profibrotic_decay_per_day * self.profibrotic
            + cfg.profibrotic_diffusion_per_day * diffusion
        )
        np.clip(self.profibrotic, 0.0, 12.0, out=self.profibrotic)

        self.stiffness_kpa += dt_days * (
            cfg.collagen_deposition_kpa_per_cell_day * myo
            - cfg.collagen_degradation_per_day
            * (self.stiffness_kpa - cfg.healthy_stiffness_kpa)
        )
        np.clip(
            self.stiffness_kpa,
            cfg.healthy_stiffness_kpa,
            cfg.max_stiffness_kpa,
            out=self.stiffness_kpa,
        )

        open_now = self.alveolar_state == OPEN
        collapse = open_now & (
            (self.surfactant < cfg.collapse_surfactant_threshold)
            | (self.stiffness_kpa > cfg.collapse_stiffness_kpa)
        )
        self.alveolar_state[collapse] = COLLAPSED
        self.radius_um[collapse] = (
            self.open_radius_um[collapse] * cfg.collapsed_radius_fraction
        )

        collapsed = self.alveolar_state == COLLAPSED
        reopen = collapsed & (
            (self.surfactant > cfg.reopening_surfactant_threshold)
            & (self.stiffness_kpa < 0.8 * cfg.collapse_stiffness_kpa)
        )
        self.alveolar_state[reopen] = OPEN
        self.radius_um[reopen] = self.open_radius_um[reopen]
        self.collapsed_time_days[reopen] = 0.0

        collapsed = self.alveolar_state == COLLAPSED
        self.collapsed_time_days[collapsed] += dt_days
        indurate = collapsed & (
            self.collapsed_time_days >= cfg.induration_delay_days
        )
        self.alveolar_state[indurate] = INDURATED

    # ------------------------------------------------------------- breathing
    def _update_tidal_strain(self) -> None:
        cfg = self.cfg
        open_mask = self.alveolar_state == OPEN
        compliance = np.where(
            open_mask,
            1.0 / np.maximum(self.stiffness_kpa, 1e-9),
            0.0,
        )
        total = compliance.sum()
        if total <= 0:
            self.tidal_strain[:] = 0.0
            return
        self.tidal_strain = (
            cfg.healthy_tidal_strain
            * compliance
            * self.n_alveoli
            / total
        )
        np.clip(self.tidal_strain, 0.0, 0.18, out=self.tidal_strain)

    # ------------------------------------------------------------ mesenchyme
    def _append_mesenchymal(
        self,
        xyz: np.ndarray,
        owner: np.ndarray,
        *,
        activated: np.ndarray,
    ) -> None:
        available = self.cfg.max_mesenchymal_cells - self.n_mesenchymal
        if available <= 0 or len(xyz) == 0:
            return
        take = min(available, len(xyz))
        xyz = np.asarray(xyz[:take], dtype=float)
        owner = np.asarray(owner[:take], dtype=int)
        activated = np.asarray(activated[:take], dtype=bool)
        orientation = _normalise(self.rng.normal(size=(take, 3)))
        self.fibroblast_xyz = np.vstack([self.fibroblast_xyz, xyz])
        self.fibroblast_orientation = np.vstack(
            [self.fibroblast_orientation, orientation]
        )
        self.myofibroblast = np.concatenate(
            [self.myofibroblast, activated]
        )
        self.fibroblast_owner = np.concatenate(
            [self.fibroblast_owner, owner]
        )

    def _local_nematic_directors(self) -> np.ndarray:
        n = self.n_mesenchymal
        if n < 2:
            return self.fibroblast_orientation.copy()
        tree = cKDTree(self.fibroblast_xyz)
        neighbours = tree.query_ball_point(
            self.fibroblast_xyz,
            self.cfg.alignment_radius_um,
        )
        directors = self.fibroblast_orientation.copy()
        for index, nearby in enumerate(neighbours):
            if len(nearby) < 2:
                continue
            local = self.fibroblast_orientation[nearby]
            tensor = np.einsum("ni,nj->ij", local, local)
            _, vectors = np.linalg.eigh(tensor)
            director = vectors[:, -1]
            if np.dot(director, directors[index]) < 0:
                director = -director
            directors[index] = director
        return directors

    def _steric_displacement(self, dt_days: float) -> np.ndarray:
        n = self.n_mesenchymal
        displacement = np.zeros((n, 3), dtype=float)
        if n < 2:
            return displacement
        tree = cKDTree(self.fibroblast_xyz)
        pairs = tree.query_pairs(self.cfg.fibroblast_width_um)
        for left, right in pairs:
            delta = self.fibroblast_xyz[left] - self.fibroblast_xyz[right]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                direction = self.rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1e-12)
            else:
                direction = delta / distance
            overlap = self.cfg.fibroblast_width_um - distance
            push = (
                0.5
                * self.cfg.steric_strength_um_per_day
                * overlap
                / self.cfg.fibroblast_width_um
                * dt_days
            )
            displacement[left] += push * direction
            displacement[right] -= push * direction
        return displacement

    def _confine_mesenchyme(self) -> None:
        if self.n_mesenchymal == 0:
            return
        # Ownership follows the nearest alveolar centre in real 3D.
        delta = (
            self.fibroblast_xyz[:, None, :]
            - self.centres[None, :, :]
        )
        distance = np.linalg.norm(delta, axis=2)
        owner = np.argmin(distance, axis=1)
        self.fibroblast_owner = owner
        centre = self.centres[owner]
        radial = self.fibroblast_xyz - centre
        radius = np.linalg.norm(radial, axis=1)
        direction = radial / np.maximum(radius[:, None], 1e-12)

        open_mask = self.alveolar_state[owner] == OPEN
        half = 0.5 * self.cfg.interstitial_thickness_um
        shell_radius = np.clip(
            radius,
            self.open_radius_um[owner] - half,
            self.open_radius_um[owner] + half,
        )
        # A derecruited unit is no longer air space; the original alveolar
        # volume becomes available to form a focus.
        filled_radius = np.clip(
            radius,
            3.0,
            0.88 * self.open_radius_um[owner],
        )
        target_radius = np.where(open_mask, shell_radius, filled_radius)
        self.fibroblast_xyz = centre + direction * target_radius[:, None]

    def _update_mesenchyme(self, dt_days: float) -> None:
        cfg = self.cfg
        n = self.n_mesenchymal
        if n == 0:
            return

        owner = self.fibroblast_owner
        strain = self.tidal_strain[owner]
        stiffness_signal = 1.0 / (
            1.0
            + np.exp(
                -(
                    self.stiffness_kpa[owner]
                    - cfg.activation_stiffness_kpa
                )
            )
        )
        activation = (
            cfg.activation_rate_per_day
            * (
                0.15
                + stiffness_signal
                + self.profibrotic[owner]
                / (1.0 + self.profibrotic[owner])
            )
            / (
                1.0
                + cfg.strain_protection_strength
                * strain
                / max(cfg.healthy_tidal_strain, 1e-9)
            )
        )
        draw = self.rng.random(n)
        self.myofibroblast |= draw < activation * dt_days

        # Nematic alignment is computed from full xyz neighbourhoods.
        director = self._local_nematic_directors()
        alpha = np.clip(cfg.alignment_strength * dt_days, 0.0, 1.0)
        orientation = _normalise(
            (1.0 - alpha) * self.fibroblast_orientation
            + alpha * director
            + np.sqrt(dt_days) * 0.08 * self.rng.normal(size=(n, 3))
        )
        self.fibroblast_orientation = orientation

        centre = self.centres[owner]
        inward = _normalise(centre - self.fibroblast_xyz)
        collapsed = self.alveolar_state[owner] != OPEN
        chemotaxis = self.profibrotic[owner, None] * inward
        chemotaxis[~collapsed] *= 0.18
        speed = cfg.fibroblast_speed_um_per_day * np.where(
            self.myofibroblast,
            0.35,
            1.0,
        )
        displacement = (
            speed[:, None]
            * dt_days
            * _normalise(orientation + 0.9 * chemotaxis)
        )
        displacement += self._steric_displacement(dt_days)
        self.fibroblast_xyz += displacement
        self._confine_mesenchyme()

        # Proliferation is favoured inside a collapsed, profibrotic unit.
        owner = self.fibroblast_owner
        proliferative = (
            (self.alveolar_state[owner] != OPEN)
            & (
                self.rng.random(self.n_mesenchymal)
                < cfg.proliferation_per_day
                * (
                    0.2
                    + self.profibrotic[owner]
                    / (1.0 + self.profibrotic[owner])
                )
                * dt_days
            )
        )
        parent = np.flatnonzero(proliferative)
        if parent.size:
            jitter = self.rng.normal(
                scale=0.5 * cfg.fibroblast_width_um,
                size=(len(parent), 3),
            )
            self._append_mesenchymal(
                self.fibroblast_xyz[parent] + jitter,
                self.fibroblast_owner[parent],
                activated=self.myofibroblast[parent],
            )
            self._confine_mesenchyme()

        death_rate = np.where(
            self.myofibroblast,
            cfg.myofibroblast_death_per_day,
            cfg.fibroblast_death_per_day,
        )
        survive = self.rng.random(self.n_mesenchymal) >= death_rate * dt_days
        self.fibroblast_xyz = self.fibroblast_xyz[survive]
        self.fibroblast_orientation = self.fibroblast_orientation[survive]
        self.myofibroblast = self.myofibroblast[survive]
        self.fibroblast_owner = self.fibroblast_owner[survive]

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        dt_days = (
            self.cfg.dt_h
            / 24.0
            * self.cfg.kinetic_rate_scale
        )
        self._update_tidal_strain()
        self._update_epithelium(dt_days)
        self._update_signals_and_alveoli(dt_days)
        self._update_tidal_strain()
        self._update_mesenchyme(dt_days)
        self.time_h += self.cfg.dt_h

    def run(self) -> Alveolar3DSimulation:
        steps = round(self.cfg.total_time_h / self.cfg.dt_h)
        for _ in range(steps):
            self.step()
        return self

    def metrics(self) -> dict:
        epithelial_total = max(len(self.epithelial_state), 1)
        open_mask = self.alveolar_state == OPEN
        strain_open = self.tidal_strain[open_mask]
        return {
            "time_h": self.time_h,
            "time_d": self.time_h / 24.0,
            "biological_time_d": self.biological_time_days,
            "kinetic_rate_scale": self.cfg.kinetic_rate_scale,
            "temporal_calibration": self.cfg.temporal_calibration,
            "n_alveoli": self.n_alveoli,
            "n_open": int(np.sum(self.alveolar_state == OPEN)),
            "n_collapsed": int(np.sum(self.alveolar_state == COLLAPSED)),
            "n_indurated": int(np.sum(self.alveolar_state == INDURATED)),
            "frac_AT1": float(np.sum(self.epithelial_state == AT1) / epithelial_total),
            "frac_AT2": float(np.sum(self.epithelial_state == AT2) / epithelial_total),
            "frac_KRT8": float(np.sum(self.epithelial_state == KRT8) / epithelial_total),
            "frac_aberrant": float(
                np.sum(self.epithelial_state == ABERRANT) / epithelial_total
            ),
            "frac_denuded": float(
                np.sum(self.epithelial_state == EMPTY) / epithelial_total
            ),
            "n_mesenchymal": self.n_mesenchymal,
            "n_myofibroblast": self.n_myofibroblast,
            "mesenchymal_released": self.mesenchymal_released,
            "mean_stiffness_kpa": float(np.mean(self.stiffness_kpa)),
            "max_stiffness_kpa": float(np.max(self.stiffness_kpa)),
            "mean_surfactant": float(np.mean(self.surfactant)),
            "max_profibrotic": float(np.max(self.profibrotic)),
            "mean_tidal_strain_open": (
                float(np.mean(strain_open)) if strain_open.size else 0.0
            ),
        }


def accelerated_3d_demo_config(seed: int = 12) -> Alveolar3DConfig:
    """Configuration used by the visual demo; rates are intentionally fast."""

    return replace(Alveolar3DConfig(), seed=seed)


def human_chronic_3d_config(
    seed: int = 12,
    *,
    years: float = 2.0,
) -> Alveolar3DConfig:
    """Clinical-scale clock for a chronic lesion followed for ``years``.

    The relative mechanisms are unchanged from the demonstrator, but one
    twelve-day reference trajectory is distributed over two calendar years.
    Experimental epithelial transitions motivate the ordering of events;
    cohort and trial timescales constrain the displayed human disease clock.
    The result is suitable for sensitivity analysis, not patient prediction.
    """

    if years <= 0:
        raise ValueError("years must be positive.")
    return replace(
        Alveolar3DConfig(),
        seed=seed,
        dt_h=12.0,
        total_time_h=years * 365.25 * 24.0,
        kinetic_rate_scale=HUMAN_CHRONIC_RATE_SCALE,
        temporal_calibration="human_chronic_disease_scale",
    )
