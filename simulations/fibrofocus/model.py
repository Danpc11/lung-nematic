"""
Agent-based active-nematic model of fibroblastic focus formation in IPF.

The tissue is represented by elongated self-propelled rods (fibroblasts and
myofibroblasts) moving on a substrate whose local stiffness ``E`` evolves as
myofibroblasts deposit collagen and matrix turnover degrades it. An initial
alveolar type II (ATII) injury seeds the process by locally lowering the
activation threshold, mimicking TGF-beta release from damaged epithelium.

Nematic order and +/-1/2 topological defects are NOT imposed: they emerge once
the rod density exceeds the isotropic-nematic crossover, exactly as observed in
confluent fibroblast monolayers.

Parameter provenance
--------------------
Mechanical parameters are anchored to published measurements on spindle-shaped
cell monolayers (Blanch-Mercader et al., Phys. Rev. Lett. 126, 028101 (2021)):
collective speed ~21.4 um/h, cell number density ~8.2e-3 /um^2, flow-alignment
parameter nu ~ -1.1 (rod-like).

Stiffness thresholds follow the lung-fibrosis mechanobiology literature
(reviewed in Hinz, Proc. Am. Thorac. Soc. 9, 137 (2012)): healthy parenchyma is
soft (~0.2-2 kPa) and keeps fibroblasts quiescent; mechanical activation of
latent TGF-beta1 requires roughly >5 kPa; induction and maintenance of the
myofibroblast phenotype sits near ~16 kPa; established fibrotic tissue reaches
~20-100 kPa. Fibroblasts primed on stiff substrates retain the myofibroblast
phenotype after returning to soft substrates, which is represented here as
hysteresis (``memory_factor``).

Focus geometry targets come from 3D morphometry of IPF lungs (Jones et al.,
JCI Insight 1, e86375 (2016)): foci are discrete, non-interconnected structures
with volumes spanning ~1.3e4 to 9.9e7 um^3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


@dataclass
class FocusConfig:
    """All parameters that change the result, in physical units."""

    # ---- domain and time (um, hours) ----
    width_um: float = 600.0
    height_um: float = 600.0
    grid_step_um: float = 6.0
    dt_h: float = 0.05
    total_time_h: float = 336.0          # 14 days
    injury_duration_h: float = 96.0      # epithelial insult lasts 4 days

    # ---- rods (cells) ----
    n_initial: int = 220
    rod_length_um: float = 50.0
    rod_width_um: float = 11.0
    n_nodes: int = 3                     # nodes per rod for steric repulsion
    speed_um_per_h: float = 21.4         # Blanch-Mercader et al.
    speed_myo_factor: float = 0.15       # myofibroblasts are anchored/contractile
    rot_diffusion_per_h: float = 0.80    # orientational noise; sets defect density
    align_rate_per_h: float = 1.4        # nematic alignment    (~ K)
    repulsion_um_per_h: float = 60.0     # steric push
    # Durotaxis: fibroblasts drift up stiffness gradients. This is the feedback
    # that concentrates cells on the stiffening lesion and turns a diffuse
    # response into a compact focus.
    durotaxis_um2_per_kPa_h: float = 55.0

    # ---- proliferation ----
    prolif_rate_per_h: float = 0.022
    # Saturation is set by how much area the cells themselves occupy. A packing
    # fraction slightly above 1 allows the mild overlap seen in confluent
    # fibroblast layers. Setting carrying_density_per_um2 overrides this.
    max_packing_fraction: float = 1.05
    carrying_density_per_um2: float | None = None

    # ---- stiffness field (kPa) ----
    E_healthy_kPa: float = 2.0           # normal parenchyma
    E_max_kPa: float = 100.0             # saturation of fibrotic scar
    E_tgfb_kPa: float = 5.0              # mechanical TGF-beta activation
    E_act_kPa: float = 16.0              # myofibroblast phenotype threshold
    activation_width_kPa: float = 2.0    # sharpness of the switch
    memory_factor: float = 3.0           # hysteresis: deactivate at E_act/memory

    # Phenotype switching timescales. Activation takes ~half a day; reversion
    # is far slower, encoding the mechanical memory reported for lung
    # fibroblasts primed on stiff substrates (phenotype retained for ~2 weeks
    # after transfer to soft substrates).
    activation_rate_per_h: float = 0.08
    deactivation_rate_per_h: float = 0.004

    # ---- matrix turnover: the two knobs that set the point of no return ----
    deposition_rate_kPa_per_h: float = 0.10   # collagen output of myofibroblasts
    degradation_rate_per_h: float = 0.004     # MMP-mediated resolution

    # ---- injury (ATII lesion) ----
    injury_radius_um: float = 70.0
    injury_activation_drop_kPa: float = 12.0  # how much the insult lowers E_act
    # Damaged epithelium lays down a provisional (fibrin/fibronectin) matrix.
    # It is stiffer than healthy parenchyma and above the TGF-beta activation
    # threshold, but on its own below the myofibroblast threshold: the insult
    # primes the system without committing it.
    injury_provisional_E_kPa: float = 9.0
    injury_stiffening_rate_per_h: float = 0.15

    # ---- misc ----
    collagen_smoothing_um: float = 12.0
    seed: int = 0

    def validate(self) -> None:
        if self.dt_h <= 0 or self.total_time_h <= 0:
            raise ValueError("dt_h and total_time_h must be positive.")
        if self.rod_length_um <= self.rod_width_um:
            raise ValueError("rod_length_um must exceed rod_width_um.")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2.")
        if not 0 < self.E_healthy_kPa < self.E_max_kPa:
            raise ValueError("Require 0 < E_healthy_kPa < E_max_kPa.")
        if self.max_packing_fraction <= 0:
            raise ValueError("max_packing_fraction must be positive.")
        if (self.carrying_density_per_um2 is not None
                and self.carrying_density_per_um2 <= 0):
            raise ValueError("carrying_density_per_um2 must be positive or None.")
        if self.memory_factor < 1:
            raise ValueError("memory_factor must be >= 1 (1 = no memory).")
        for name in ("deposition_rate_kPa_per_h", "degradation_rate_per_h"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def cell_area_um2(self) -> float:
        """Projected area of one cell, modelled as an ellipse."""
        return float(np.pi * 0.25 * self.rod_length_um * self.rod_width_um)

    @property
    def carrying_density(self) -> float:
        """Saturation number density (cells / um^2).

        Derived from the cell's own footprint so that the packing fraction at
        saturation equals ``max_packing_fraction``. Using the C2C12 cell size
        this reproduces the ~8.2e-3 /um^2 reported for those monolayers; larger
        fibroblasts saturate at a correspondingly lower number density.
        """
        if self.carrying_density_per_um2 is not None:
            return float(self.carrying_density_per_um2)
        return float(self.max_packing_fraction / self.cell_area_um2)


class FocusSimulation:
    """Elongated active agents on an evolving stiffness field."""

    def __init__(self, config: FocusConfig):
        config.validate()
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # --- stiffness grid ---
        self.nx = int(round(config.width_um / config.grid_step_um))
        self.ny = int(round(config.height_um / config.grid_step_um))
        self.E = np.full((self.ny, self.nx), config.E_healthy_kPa, dtype=float)

        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        self.grid_x = (gx + 0.5) * config.grid_step_um
        self.grid_y = (gy + 0.5) * config.grid_step_um

        # injury mask centred in the domain (the ATII lesion)
        cx, cy = config.width_um / 2, config.height_um / 2
        r2 = (self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2
        self.injury_mask = r2 <= config.injury_radius_um ** 2
        self.injury_centre = (cx, cy)

        # --- rods ---
        n = config.n_initial
        self.x = self.rng.uniform(0, config.width_um, n)
        self.y = self.rng.uniform(0, config.height_um, n)
        self.theta = self.rng.uniform(0, np.pi, n)
        self.myo = np.zeros(n, dtype=bool)      # False = fibroblast

        self.time_h = 0.0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ utils
    @property
    def n_cells(self) -> int:
        return self.x.size

    def _wrap(self) -> None:
        self.x %= self.cfg.width_um
        self.y %= self.cfg.height_um

    def _node_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions of the nodes that discretise each rod (for steric forces)."""
        cfg = self.cfg
        offsets = np.linspace(-0.5, 0.5, cfg.n_nodes) * cfg.rod_length_um
        ux, uy = np.cos(self.theta), np.sin(self.theta)
        nx = self.x[:, None] + offsets[None, :] * ux[:, None]
        ny = self.y[:, None] + offsets[None, :] * uy[:, None]
        # keep nodes inside the periodic box (rod ends may stick out)
        nx = np.mod(nx, cfg.width_um)
        ny = np.mod(ny, cfg.height_um)
        owner = np.repeat(np.arange(self.n_cells), cfg.n_nodes)
        return nx.ravel(), ny.ravel(), owner

    def _sample_E(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ix = np.clip((x / cfg.grid_step_um).astype(int), 0, self.nx - 1)
        iy = np.clip((y / cfg.grid_step_um).astype(int), 0, self.ny - 1)
        return self.E[iy, ix]

    def _in_injury(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx, cy = self.injury_centre
        return (x - cx) ** 2 + (y - cy) ** 2 <= self.cfg.injury_radius_um ** 2

    # -------------------------------------------------------------- mechanics
    def _steric_and_alignment(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Steric displacement and nematic alignment torque from neighbours."""
        cfg = self.cfg
        nx_, ny_, owner = self._node_positions()
        points = np.column_stack([nx_, ny_])
        tree = cKDTree(points, boxsize=[cfg.width_um, cfg.height_um])

        contact = cfg.rod_width_um
        pairs = tree.query_pairs(contact, output_type="ndarray")

        fx = np.zeros(self.n_cells)
        fy = np.zeros(self.n_cells)
        qxx = np.zeros(self.n_cells)
        qxy = np.zeros(self.n_cells)

        if pairs.size:
            a, b = pairs[:, 0], pairs[:, 1]
            oa, ob = owner[a], owner[b]
            keep = oa != ob
            a, b, oa, ob = a[keep], b[keep], oa[keep], ob[keep]

            if a.size:
                dx = points[a, 0] - points[b, 0]
                dy = points[a, 1] - points[b, 1]
                # minimum-image convention (periodic domain)
                dx -= cfg.width_um * np.round(dx / cfg.width_um)
                dy -= cfg.height_um * np.round(dy / cfg.height_um)
                dist = np.sqrt(dx * dx + dy * dy) + 1e-9
                overlap = np.clip(1.0 - dist / contact, 0.0, 1.0)
                ux, uy = dx / dist, dy / dist
                np.add.at(fx, oa, overlap * ux)
                np.add.at(fx, ob, -overlap * ux)
                np.add.at(fy, oa, overlap * uy)
                np.add.at(fy, ob, -overlap * uy)

                # nematic alignment: accumulate neighbour Q components
                c2 = np.cos(2 * self.theta)
                s2 = np.sin(2 * self.theta)
                np.add.at(qxx, oa, c2[ob])
                np.add.at(qxy, oa, s2[ob])
                np.add.at(qxx, ob, c2[oa])
                np.add.at(qxy, ob, s2[oa])

        theta_local = 0.5 * np.arctan2(qxy, qxx)
        has_neighbour = (qxx**2 + qxy**2) > 1e-12
        return fx, fy, np.where(has_neighbour, theta_local, self.theta)

    def _local_density(self) -> np.ndarray:
        """Coarse-grained cell-area density per grid node (cells / um^2)."""
        cfg = self.cfg
        hist, _, _ = np.histogram2d(
            self.y, self.x,
            bins=[self.ny, self.nx],
            range=[[0, cfg.height_um], [0, cfg.width_um]],
        )
        cell_area = cfg.grid_step_um ** 2
        # Periodic domain: cells wrap, so the field must wrap too. The default
        # reflect mode double-counts edge cells and inflates apparent density
        # and order near the boundary.
        return gaussian_filter(hist / cell_area,
                               cfg.collagen_smoothing_um / cfg.grid_step_um,
                               mode="wrap")

    def _myo_density(self) -> np.ndarray:
        cfg = self.cfg
        if not self.myo.any():
            return np.zeros((self.ny, self.nx))
        hist, _, _ = np.histogram2d(
            self.y[self.myo], self.x[self.myo],
            bins=[self.ny, self.nx],
            range=[[0, cfg.height_um], [0, cfg.width_um]],
        )
        cell_area = cfg.grid_step_um ** 2
        return gaussian_filter(hist / cell_area,
                               cfg.collagen_smoothing_um / cfg.grid_step_um,
                               mode="wrap")

    # ------------------------------------------------------------ biology
    def _update_phenotype(self) -> None:
        """Stiffness-gated activation with mechanical memory (hysteresis)."""
        cfg = self.cfg
        E_local = self._sample_E(self.x, self.y)

        threshold = np.full(self.n_cells, cfg.E_act_kPa)
        if self.time_h < cfg.injury_duration_h:
            # Damaged ATII epithelium releases TGF-beta, lowering the bar
            # locally, but never below the mechanical TGF-beta threshold.
            inside = self._in_injury(self.x, self.y)
            threshold[inside] = np.maximum(
                cfg.E_tgfb_kPa,
                cfg.E_act_kPa - cfg.injury_activation_drop_kPa,
            )

        # probabilistic switch with a finite transition width
        on_hazard = cfg.activation_rate_per_h / (
            1.0 + np.exp(-(E_local - threshold) / cfg.activation_width_kPa)
        )
        draw = self.rng.random(self.n_cells)
        newly_on = (~self.myo) & (draw < on_hazard * cfg.dt_h)
        self.myo |= newly_on

        # Deactivation is much harder: mechanical memory keeps the phenotype
        # until stiffness falls well below the activation threshold, and even
        # then reversion is slow (deactivation_rate_per_h).
        E_deact = cfg.E_act_kPa / cfg.memory_factor
        off_hazard = cfg.deactivation_rate_per_h / (
            1.0 + np.exp((E_local - E_deact) / cfg.activation_width_kPa)
        )
        draw = self.rng.random(self.n_cells)
        self.myo &= ~(self.myo & (draw < off_hazard * cfg.dt_h))

    def _proliferate(self) -> None:
        cfg = self.cfg
        density = self._local_density()
        local = density[
            np.clip((self.y / cfg.grid_step_um).astype(int), 0, self.ny - 1),
            np.clip((self.x / cfg.grid_step_um).astype(int), 0, self.nx - 1),
        ]
        space = np.clip(1.0 - local / cfg.carrying_density, 0.0, 1.0)

        # profibrogenic stiffness range boosts proliferation
        E_local = self._sample_E(self.x, self.y)
        boost = 1.0 + 0.8 * np.clip(
            (E_local - cfg.E_healthy_kPa) / max(cfg.E_act_kPa, 1e-9), 0, 1.5
        )
        rate = cfg.prolif_rate_per_h * space * boost
        born = self.rng.random(self.n_cells) < rate * cfg.dt_h

        if born.any():
            k = int(born.sum())
            jitter = cfg.rod_width_um
            self.x = np.concatenate([self.x, self.x[born] + self.rng.normal(0, jitter, k)])
            self.y = np.concatenate([self.y, self.y[born] + self.rng.normal(0, jitter, k)])
            self.theta = np.concatenate(
                [self.theta, (self.theta[born] + self.rng.normal(0, 0.3, k)) % np.pi]
            )
            self.myo = np.concatenate([self.myo, self.myo[born]])
            self._wrap()

    def _update_stiffness(self) -> None:
        """dE/dt = deposition by myofibroblasts - matrix turnover."""
        cfg = self.cfg
        rho_myo = self._myo_density() / cfg.carrying_density
        deposition = (
            cfg.deposition_rate_kPa_per_h
            * np.clip(rho_myo, 0, 2.0)
            * (1.0 - self.E / cfg.E_max_kPa)
        )
        degradation = cfg.degradation_rate_per_h * (self.E - cfg.E_healthy_kPa)
        self.E += cfg.dt_h * (deposition - degradation)

        # While the epithelial insult is active, a provisional matrix forms in
        # the lesion. It primes the region above the TGF-beta threshold but,
        # by itself, relaxes away once the insult stops.
        if self.time_h < cfg.injury_duration_h:
            target = cfg.injury_provisional_E_kPa
            gap = target - self.E[self.injury_mask]
            self.E[self.injury_mask] += (
                cfg.dt_h * cfg.injury_stiffening_rate_per_h * np.maximum(gap, 0.0)
            )

        np.clip(self.E, cfg.E_healthy_kPa, cfg.E_max_kPa, out=self.E)

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        cfg = self.cfg
        fx, fy, theta_local = self._steric_and_alignment()

        speed = np.where(self.myo, cfg.speed_um_per_h * cfg.speed_myo_factor,
                         cfg.speed_um_per_h)

        # durotactic drift up the local stiffness gradient
        grad_y, grad_x = np.gradient(self.E, cfg.grid_step_um)
        ix = np.clip((self.x / cfg.grid_step_um).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / cfg.grid_step_um).astype(int), 0, self.ny - 1)
        duro_x = cfg.durotaxis_um2_per_kPa_h * grad_x[iy, ix]
        duro_y = cfg.durotaxis_um2_per_kPa_h * grad_y[iy, ix]

        self.x += cfg.dt_h * (
            speed * np.cos(self.theta) + cfg.repulsion_um_per_h * fx + duro_x
        )
        self.y += cfg.dt_h * (
            speed * np.sin(self.theta) + cfg.repulsion_um_per_h * fy + duro_y
        )
        self._wrap()

        # nematic torque toward the local director + rotational noise
        dtheta = np.arctan2(np.sin(2 * (theta_local - self.theta)),
                            np.cos(2 * (theta_local - self.theta))) / 2.0
        noise = self.rng.normal(0, np.sqrt(2 * cfg.rot_diffusion_per_h * cfg.dt_h),
                                self.n_cells)
        self.theta = (self.theta + cfg.dt_h * cfg.align_rate_per_h * dtheta + noise) % np.pi

        self._update_phenotype()
        self._proliferate()
        self._update_stiffness()
        self.time_h += cfg.dt_h

    # -------------------------------------------------------------- readouts
    def focus_metrics(self) -> dict:
        """Scalar readouts, including the focus-persistence indicators."""
        cfg = self.cfg
        inside = self.injury_mask
        return {
            "time_h": self.time_h,
            "time_d": self.time_h / 24.0,
            "n_cells": self.n_cells,
            "myo_fraction": float(self.myo.mean()) if self.n_cells else 0.0,
            "E_focus_kPa": float(self.E[inside].mean()),
            "E_max_kPa": float(self.E.max()),
            "fibrotic_area_frac": float((self.E > cfg.E_act_kPa).mean()),
            "global_order_S": self.global_order(),
            "density_per_um2": self.n_cells / (cfg.width_um * cfg.height_um),
            "area_fraction": (
                self.n_cells * cfg.cell_area_um2 / (cfg.width_um * cfg.height_um)
            ),
        }

    def global_order(self) -> float:
        if self.n_cells == 0:
            return 0.0
        return float(np.abs(np.mean(np.exp(2j * self.theta))))

    def director_field(self, sigma_um: float = 25.0) -> dict[str, np.ndarray]:
        """Coarse-grained director field, in the format lung_nematic expects."""
        cfg = self.cfg
        qxx = np.zeros((self.ny, self.nx))
        qxy = np.zeros((self.ny, self.nx))
        counts = np.zeros((self.ny, self.nx))

        ix = np.clip((self.x / cfg.grid_step_um).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / cfg.grid_step_um).astype(int), 0, self.ny - 1)
        np.add.at(qxx, (iy, ix), np.cos(2 * self.theta))
        np.add.at(qxy, (iy, ix), np.sin(2 * self.theta))
        np.add.at(counts, (iy, ix), 1.0)

        s = sigma_um / cfg.grid_step_um
        # Periodic domain -> wrap. See _cell_density for why reflect is wrong.
        qxx = gaussian_filter(qxx, s, mode="wrap")
        qxy = gaussian_filter(qxy, s, mode="wrap")
        density = gaussian_filter(counts, s, mode="wrap")

        with np.errstate(invalid="ignore", divide="ignore"):
            order = np.sqrt(qxx**2 + qxy**2) / np.maximum(density, 1e-9)
        theta = (0.5 * np.arctan2(qxy, qxx)) % np.pi

        return {
            "density": density,
            "order": np.clip(np.nan_to_num(order), 0, 1),
            "theta": theta,
        }

    def run(self, record_every_h: float = 4.0, callback=None) -> "FocusSimulation":
        cfg = self.cfg
        n_steps = int(round(cfg.total_time_h / cfg.dt_h))
        record_every = max(1, int(round(record_every_h / cfg.dt_h)))
        self.history.append(self.focus_metrics())
        for step_index in range(1, n_steps + 1):
            self.step()
            if step_index % record_every == 0:
                self.history.append(self.focus_metrics())
                if callback is not None:
                    callback(self)
        return self
