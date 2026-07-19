"""
Stage 3: the mesenchyme, confined by the alveolar architecture.

Fibroblasts do not live in air. In an intact alveolus they are restricted to the
interstitium - the thin space inside the septum, between the two epithelial
surfaces. That confinement is what keeps a healthy lung from filling in.

When an alveolus derecruits, its air space stops being air space. The collapsed
volume becomes available, cells released by EMT and recruited from neighbouring
septa migrate into it, and only there can their density climb high enough for
nematic order - and therefore for +/-1/2 topological defects - to exist at all.
The fibroblastic focus is what that filled-in, collapsed alveolus becomes.

So the geometry does the work: the epithelial stage decides *where* space opens
up, and the mesenchymal stage decides *what grows into it*.

Cells are elongated (ellipse footprint), align nematically, are pushed by steric
repulsion, drift up gradients of stiffness (durotaxis) and of the profibrotic
signal secreted by aberrant epithelium (chemotaxis). Myofibroblasts deposit
collagen, which stiffens the matrix and thickens the septum - the latter being
directly measurable in histology.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from .model import ABERRANT, OPEN, AlveolarSimulation


class MesenchymeLayer:
    """Elongated cells confined to interstitium plus collapsed air space."""

    def __init__(self, sim: AlveolarSimulation, config):
        from .model import apply_rate_scale

        self.sim = sim
        self.user_cfg = config
        self.cfg = apply_rate_scale(config)
        config = self.cfg
        self.rng = np.random.default_rng(config.seed + 101)

        g = sim.geometry
        step = config.mesenchyme_grid_step_um
        self.grid_step = step
        self.nx = int(np.ceil(config.width_um / step))
        self.ny = int(np.ceil(config.height_um / step))

        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        self.grid_x = (gx + 0.5) * step
        self.grid_y = (gy + 0.5) * step
        points = np.column_stack([self.grid_x.ravel(), self.grid_y.ravel()])

        # which alveolus contains each grid node (-1 = none)
        label = np.full(points.shape[0], -1, dtype=int)
        for alveolus in g.alveoli:
            inside = MplPath(alveolus.vertices).contains_points(points)
            label[inside & (label < 0)] = alveolus.index
        self.alveolus_label = label.reshape(self.ny, self.nx)

        # distance from every node to the nearest septum
        septal = np.ones((self.ny, self.nx), dtype=bool)
        for septum in g.septa:
            n_samples = max(2, int(septum.length_um / (step * 0.5)))
            ts = np.linspace(0.0, 1.0, n_samples)
            pts = septum.start[None, :] + ts[:, None] * (septum.end - septum.start)
            ix = np.clip((pts[:, 0] / step).astype(int), 0, self.nx - 1)
            iy = np.clip((pts[:, 1] / step).astype(int), 0, self.ny - 1)
            septal[iy, ix] = False
        self.distance_to_septum = distance_transform_edt(septal) * step

        # septal half-thickness grows where collagen accumulates
        self.septal_half_thickness = np.full(
            (self.ny, self.nx), 0.5 * config.septal_thickness_um
        )
        self.stiffness_kPa = np.full((self.ny, self.nx), config.E_healthy_kPa)

        self.strain = np.full((self.ny, self.nx), config.tidal_strain)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.theta = np.zeros(0)
        self.myo = np.zeros(0, dtype=bool)

        self.seed_interstitial_cells()

    # ---------------------------------------------------------------- strain
    def strain_field(self) -> np.ndarray:
        """Tidal strain amplitude, redistributed over what can still deform.

        The chest wall imposes a fixed tidal volume, so the total deformation
        is conserved. Each ventilated region takes a share proportional to its
        compliance (1/E), and derecruited alveoli take none at all. Therefore
        stiff and collapsed regions deform less *because* they are stiff, and
        whatever remains healthy must deform more to make up the difference.
        Losing alveoli concentrates strain on the survivors.
        """
        cfg = self.cfg
        ventilated = np.ones((self.ny, self.nx), dtype=bool)
        valid = self.alveolus_label >= 0
        states = self.sim.alveolar_state
        ventilated[valid] = states[self.alveolus_label[valid]] == OPEN

        compliance = np.where(ventilated, 1.0 / np.maximum(self.stiffness_kPa, 1e-6), 0.0)
        total = compliance.sum()
        if total <= 0:
            return np.zeros((self.ny, self.nx))
        # normalise so the summed volume change equals tidal strain x whole area
        return cfg.tidal_strain * compliance * compliance.size / total

    def sample_grid(self, field: np.ndarray, x: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
        ix = np.clip((x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((y / self.grid_step).astype(int), 0, self.ny - 1)
        return field[iy, ix]

    # ----------------------------------------------------------------- masks
    def permitted_mask(self) -> np.ndarray:
        """Where mesenchymal cells may be: interstitium, or collapsed air space."""
        interstitium = self.distance_to_septum <= self.septal_half_thickness
        collapsed = np.zeros_like(interstitium)
        states = self.sim.alveolar_state
        valid = self.alveolus_label >= 0
        collapsed[valid] = states[self.alveolus_label[valid]] != OPEN
        return interstitium | collapsed

    def _permitted_at(self, x: np.ndarray, y: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        ix = np.clip((x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((y / self.grid_step).astype(int), 0, self.ny - 1)
        return mask[iy, ix]

    # ------------------------------------------------------------- seeding
    def seed_interstitial_cells(self) -> None:
        """Resident fibroblasts, sparse, inside the septa."""
        cfg = self.cfg
        mask = self.permitted_mask()
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return
        take = min(cfg.n_resident_fibroblasts, xs.size)
        picks = self.rng.choice(xs.size, size=take, replace=False)
        self.x = (xs[picks] + 0.5) * self.grid_step
        self.y = (ys[picks] + 0.5) * self.grid_step
        self.theta = self.rng.uniform(0, np.pi, take)
        self.myo = np.zeros(take, dtype=bool)

    def add_from_emt(self, positions: np.ndarray) -> None:
        """Cells that left the epithelium by EMT enter the mesenchyme here."""
        if positions.size == 0:
            return
        n = positions.shape[0]
        self.x = np.concatenate([self.x, positions[:, 0]])
        self.y = np.concatenate([self.y, positions[:, 1]])
        self.theta = np.concatenate([self.theta, self.rng.uniform(0, np.pi, n)])
        # cells arriving via EMT are already activated
        self.myo = np.concatenate([self.myo, np.ones(n, dtype=bool)])

    # ------------------------------------------------------------- dynamics
    @property
    def n_cells(self) -> int:
        return int(self.x.size)

    def _node_positions(self):
        """Nodes along each cell's long axis: this is what gives it area."""
        cfg = self.cfg
        offsets = np.linspace(-0.5, 0.5, cfg.n_nodes) * cfg.cell_length_um
        ux, uy = np.cos(self.theta), np.sin(self.theta)
        nx = self.x[:, None] + offsets[None, :] * ux[:, None]
        ny = self.y[:, None] + offsets[None, :] * uy[:, None]
        owner = np.repeat(np.arange(self.n_cells), cfg.n_nodes)
        arm_x = (nx - self.x[:, None]).ravel()
        arm_y = (ny - self.y[:, None]).ravel()
        return nx.ravel(), ny.ravel(), owner, arm_x, arm_y

    def _steric_and_alignment(self):
        """Excluded-volume forces AND torques between elongated cells.

        Each cell is resolved into nodes along its long axis and nodes repel at
        the cell width. A force applied at a node that is offset from the cell
        centre exerts a torque, so two crossing cells rotate towards being
        parallel. Nematic alignment is therefore produced by the *shape*, not
        imposed by a rule - which is the whole reason for giving cells an area.
        """
        cfg = self.cfg
        n = self.n_cells
        fx = np.zeros(n); fy = np.zeros(n); torque = np.zeros(n)
        theta_local = self.theta.copy()
        if n < 2:
            return fx, fy, torque, theta_local

        node_x, node_y, owner, arm_x, arm_y = self._node_positions()
        points = np.column_stack([node_x, node_y])
        tree = cKDTree(points)
        pairs = tree.query_pairs(cfg.cell_width_um, output_type="ndarray")
        if pairs.size == 0:
            return fx, fy, torque, theta_local

        a, b = pairs[:, 0], pairs[:, 1]
        oa, ob = owner[a], owner[b]
        keep = oa != ob
        a, b, oa, ob = a[keep], b[keep], oa[keep], ob[keep]
        if a.size == 0:
            return fx, fy, torque, theta_local

        dx = points[a, 0] - points[b, 0]
        dy = points[a, 1] - points[b, 1]
        dist = np.hypot(dx, dy) + 1e-9
        overlap = np.clip(1.0 - dist / cfg.cell_width_um, 0.0, 1.0)
        ux, uy = dx / dist, dy / dist
        px, py = overlap * ux, overlap * uy

        np.add.at(fx, oa, px);  np.add.at(fx, ob, -px)
        np.add.at(fy, oa, py);  np.add.at(fy, ob, -py)
        # torque = arm x force, taken about each cell's own centre
        np.add.at(torque, oa, arm_x[a] * py - arm_y[a] * px)
        np.add.at(torque, ob, -(arm_x[b] * py - arm_y[b] * px))

        # local nematic director, for the weak co-alignment term
        qxx = np.zeros(n); qxy = np.zeros(n)
        c2, s2 = np.cos(2 * self.theta), np.sin(2 * self.theta)
        np.add.at(qxx, oa, c2[ob]); np.add.at(qxy, oa, s2[ob])
        np.add.at(qxx, ob, c2[oa]); np.add.at(qxy, ob, s2[oa])
        has = (qxx**2 + qxy**2) > 1e-12
        theta_local[has] = 0.5 * np.arctan2(qxy[has], qxx[has])
        return fx, fy, torque, theta_local

    def step(self, dt_h: float) -> None:
        cfg = self.cfg
        if self.n_cells == 0:
            self._update_matrix(dt_h)
            return

        mask = self.permitted_mask()
        fx, fy, torque, theta_local = self._steric_and_alignment()
        ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)

        # gradients that pull cells into the lesion
        gy_s, gx_s = np.gradient(self.stiffness_kPa, self.grid_step)
        gy_p, gx_p = np.gradient(self.sim.profibrotic, self.sim.grid_step_um)

        jx = np.clip((self.x / self.sim.grid_step_um).astype(int), 0, self.sim.nx - 1)
        jy = np.clip((self.y / self.sim.grid_step_um).astype(int), 0, self.sim.ny - 1)

        # Overdamped motion: xi * v = F. Substrate friction rises as collagen
        # accumulates, so deposited matrix physically pins the cells that made
        # it rather than mobility being reduced by hand.
        local_E = self.stiffness_kPa[iy, ix]
        friction = cfg.substrate_friction * (
            1.0 + cfg.matrix_immobilization
            * np.clip((local_E - cfg.E_healthy_kPa)
                      / max(cfg.E_act_kPa - cfg.E_healthy_kPa, 1e-9), 0.0, None)
        )
        self._last_friction = friction
        self._last_mobility = cfg.substrate_friction / friction
        propulsion = np.where(self.myo, cfg.speed_um_per_h * cfg.speed_myo_factor,
                              cfg.speed_um_per_h) * cfg.substrate_friction
        force_x = (propulsion * np.cos(self.theta)
                   + cfg.steric_force_um2_per_h * fx
                   + cfg.durotaxis_um2_per_kPa_h * gx_s[iy, ix]
                   + cfg.chemotaxis_um2_per_h * gx_p[jy, jx])
        force_y = (propulsion * np.sin(self.theta)
                   + cfg.steric_force_um2_per_h * fy
                   + cfg.durotaxis_um2_per_kPa_h * gy_s[iy, ix]
                   + cfg.chemotaxis_um2_per_h * gy_p[jy, jx])
        vx = force_x / friction
        vy = force_y / friction

        new_x = np.clip(self.x + dt_h * vx, 0.5, cfg.width_um - 0.5)
        new_y = np.clip(self.y + dt_h * vy, 0.5, cfg.height_um - 0.5)

        # confinement: a move into air space is rejected
        ok = self._permitted_at(new_x, new_y, mask)
        self.x = np.where(ok, new_x, self.x)
        self.y = np.where(ok, new_y, self.y)

        dtheta = np.arctan2(np.sin(2 * (theta_local - self.theta)),
                            np.cos(2 * (theta_local - self.theta))) / 2.0
        rot_friction = friction / cfg.substrate_friction
        omega = (cfg.steric_torque_gain * torque
                 + cfg.align_rate_per_h * dtheta) / rot_friction
        noise = self.rng.normal(0, 1.0, self.n_cells) * np.sqrt(
            2 * cfg.rot_diffusion_per_h * dt_h / rot_friction
        )
        self.theta = (self.theta + dt_h * omega + noise) % np.pi
        self._apoptosis(dt_h)

        self._activate(dt_h)
        self._proliferate(dt_h, mask)
        self._update_matrix(dt_h)

    def _apoptosis(self, dt_h: float) -> None:
        """Cells die. Myofibroblasts die more slowly - that is the point.

        Resistance of the myofibroblast to apoptosis is one of the defining
        features of the fibrotic lesion: without clearance the population only
        accumulates, which is exactly what a persistent focus looks like. The
        ratio of the two death rates is therefore a control parameter, not a
        detail, and crowding adds a further contact-inhibition-like term.
        """
        cfg = self.cfg
        if self.n_cells == 0:
            return
        ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)

        sigma = cfg.coarse_grain_um / self.grid_step
        available = gaussian_filter(self.permitted_mask().astype(float), sigma)
        density = self.local_density() / np.maximum(available, 0.05)
        packing = density[iy, ix] * cfg.cell_area_um2
        crowding = cfg.crowding_death_gain * np.clip(
            packing - cfg.max_packing_fraction, 0.0, None
        )

        hazard = np.where(self.myo, cfg.myofibroblast_death_rate,
                          cfg.fibroblast_death_rate) + crowding
        survives = self.rng.random(self.n_cells) >= hazard * dt_h
        if survives.all():
            return
        self.x = self.x[survives]
        self.y = self.y[survives]
        self.theta = self.theta[survives]
        self.myo = self.myo[survives]

    def _activate(self, dt_h: float) -> None:
        """Stiffness drives myofibroblast activation; cyclic strain restrains it.

        Cyclic mechanical stretch has been reported to *reduce* fibroblast-to-
        myofibroblast differentiation, so breathing is protective at the cell
        level. A collapsed or stiffened region is mechanically shielded, loses
        that protection, and converts - which is a second positive feedback on
        top of the stiffness one, and is why the focus grows where the lung has
        stopped moving.
        """
        cfg = self.cfg
        ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)
        local_E = self.stiffness_kPa[iy, ix]
        local_strain = self.strain[iy, ix]

        hazard = cfg.activation_rate_per_h / (
            1.0 + np.exp(-(local_E - cfg.E_act_kPa) / cfg.activation_width_kPa)
        )
        protection = 1.0 + cfg.strain_protection_strength * (
            local_strain / max(cfg.tidal_strain, 1e-9)
        )
        draw = self.rng.random(self.n_cells)
        self.myo |= (~self.myo) & (draw < hazard / protection * dt_h)

    def _proliferate(self, dt_h: float, mask: np.ndarray) -> None:
        cfg = self.cfg
        # Density must be measured per *available* area: cells are squeezed into
        # the interstitium, which is much thinner than the coarse-graining
        # length, so an unnormalised density badly underestimates crowding.
        sigma = cfg.coarse_grain_um / self.grid_step
        available = gaussian_filter(mask.astype(float), sigma)
        density = self.local_density() / np.maximum(available, 0.05)
        ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
        iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)
        packing = density[iy, ix] * cfg.cell_area_um2
        space = np.clip(1.0 - packing / cfg.max_packing_fraction, 0.0, 1.0)
        born = self.rng.random(self.n_cells) < cfg.prolif_rate_per_h * space * dt_h
        if not born.any():
            return
        k = int(born.sum())
        nx = self.x[born] + self.rng.normal(0, cfg.cell_width_um, k)
        ny = self.y[born] + self.rng.normal(0, cfg.cell_width_um, k)
        nx = np.clip(nx, 0.5, cfg.width_um - 0.5)
        ny = np.clip(ny, 0.5, cfg.height_um - 0.5)
        keep = self._permitted_at(nx, ny, mask)
        if not keep.any():
            return
        self.x = np.concatenate([self.x, nx[keep]])
        self.y = np.concatenate([self.y, ny[keep]])
        self.theta = np.concatenate(
            [self.theta, (self.theta[born][keep] + self.rng.normal(0, 0.3, keep.sum())) % np.pi]
        )
        self.myo = np.concatenate([self.myo, self.myo[born][keep]])

    def local_density(self, sigma_um: float | None = None) -> np.ndarray:
        cfg = self.cfg
        counts = np.zeros((self.ny, self.nx))
        if self.n_cells:
            ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
            iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)
            np.add.at(counts, (iy, ix), 1.0)
        sigma = (sigma_um or cfg.coarse_grain_um) / self.grid_step
        return gaussian_filter(counts / self.grid_step**2, sigma)

    def _update_matrix(self, dt_h: float) -> None:
        cfg = self.cfg
        myo_density = np.zeros((self.ny, self.nx))
        if self.myo.any():
            ix = np.clip((self.x[self.myo] / self.grid_step).astype(int), 0, self.nx - 1)
            iy = np.clip((self.y[self.myo] / self.grid_step).astype(int), 0, self.ny - 1)
            np.add.at(myo_density, (iy, ix), 1.0)
        myo_density = gaussian_filter(
            myo_density / self.grid_step**2, cfg.coarse_grain_um / self.grid_step
        )
        supply = np.clip(myo_density * cfg.cell_area_um2 / cfg.max_packing_fraction,
                         0.0, 2.0)

        self.stiffness_kPa += dt_h * (
            cfg.deposition_rate_kPa_per_h * supply
            * (1.0 - self.stiffness_kPa / cfg.E_max_kPa)
            - cfg.degradation_rate_per_h * (self.stiffness_kPa - cfg.E_healthy_kPa)
        )
        np.clip(self.stiffness_kPa, cfg.E_healthy_kPa, cfg.E_max_kPa,
                out=self.stiffness_kPa)

        # collagen thickens the septum, which is what histology measures
        target = 0.5 * cfg.septal_thickness_um * (
            1.0 + cfg.septal_thickening_gain
            * (self.stiffness_kPa - cfg.E_healthy_kPa)
            / max(cfg.E_act_kPa - cfg.E_healthy_kPa, 1e-9)
        )
        self.septal_half_thickness += dt_h * cfg.septal_thickening_rate_per_h * (
            target - self.septal_half_thickness
        )

    # -------------------------------------------------------------- readouts
    def director_field(self, sigma_um: float | None = None) -> dict:
        cfg = self.cfg
        qxx = np.zeros((self.ny, self.nx))
        qxy = np.zeros((self.ny, self.nx))
        counts = np.zeros((self.ny, self.nx))
        if self.n_cells:
            ix = np.clip((self.x / self.grid_step).astype(int), 0, self.nx - 1)
            iy = np.clip((self.y / self.grid_step).astype(int), 0, self.ny - 1)
            np.add.at(qxx, (iy, ix), np.cos(2 * self.theta))
            np.add.at(qxy, (iy, ix), np.sin(2 * self.theta))
            np.add.at(counts, (iy, ix), 1.0)
        sigma = (sigma_um or cfg.coarse_grain_um) / self.grid_step
        qxx = gaussian_filter(qxx, sigma)
        qxy = gaussian_filter(qxy, sigma)
        density = gaussian_filter(counts, sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            order = np.sqrt(qxx**2 + qxy**2) / np.maximum(density, 1e-9)
        return {
            "density": density,
            "order": np.clip(np.nan_to_num(order), 0, 1),
            "theta": (0.5 * np.arctan2(qxy, qxx)) % np.pi,
        }

    def detect_defects(self, grid_step: int = 3,
                       min_distance_um: float = 40.0) -> dict:
        """+/-1/2 defects, gated on absolute packing as in the histology tool."""
        cfg = self.cfg
        field = self.director_field()
        theta, density = field["theta"], field["density"]
        ys = np.arange(0, self.ny - grid_step, grid_step)
        xs = np.arange(0, self.nx - grid_step, grid_step)
        if ys.size < 2 or xs.size < 2:
            return {"plus": np.zeros((0, 2)), "minus": np.zeros((0, 2))}

        corners = [
            theta[np.ix_(ys, xs)], theta[np.ix_(ys, xs + grid_step)],
            theta[np.ix_(ys + grid_step, xs + grid_step)],
            theta[np.ix_(ys + grid_step, xs)],
        ]
        phases = [2 * c for c in corners]
        winding = np.zeros_like(phases[0])
        for k in range(4):
            d = phases[(k + 1) % 4] - phases[k]
            winding += np.arctan2(np.sin(d), np.cos(d))
        charge = winding / (4 * np.pi)

        packing = density[np.ix_(ys, xs)] * cfg.cell_area_um2 / self.grid_step**2
        valid = packing >= cfg.min_packing_for_nematic

        out = {}
        for key, target in (("plus", 0.5), ("minus", -0.5)):
            hit = valid & (np.abs(charge - target) < 0.2)
            row, col = np.nonzero(hit)
            pts = np.column_stack([(xs[col] + grid_step / 2) * self.grid_step,
                                   (ys[row] + grid_step / 2) * self.grid_step])
            kept: list[np.ndarray] = []
            for point in pts:
                if all(np.hypot(*(point - o)) >= min_distance_um for o in kept):
                    kept.append(point)
            out[key] = np.array(kept) if kept else np.zeros((0, 2))
        return out

    def metrics(self) -> dict:
        cfg = self.cfg
        area = cfg.width_um * cfg.height_um
        defects = self.detect_defects()
        permitted = self.permitted_mask()
        packing = (
            self.n_cells * cfg.cell_area_um2
            / max(permitted.sum() * self.grid_step**2, 1.0)
        )
        ventilated = np.ones((self.ny, self.nx), dtype=bool)
        valid = self.alveolus_label >= 0
        ventilated[valid] = self.sim.alveolar_state[self.alveolus_label[valid]] == OPEN
        strain_open = self.strain[ventilated]
        return {
            "mean_strain_ventilated": float(strain_open.mean()) if strain_open.size else 0.0,
            "max_strain": float(self.strain.max()),
            "strain_amplification": float(
                strain_open.mean() / max(cfg.tidal_strain, 1e-9)
            ) if strain_open.size else 0.0,
            "mean_friction": float(np.mean(getattr(self, "_last_friction", 1.0))),
            "n_mesenchymal": self.n_cells,
            "n_myofibroblast": int(self.myo.sum()),
            "packing_in_permitted": float(packing),
            "mean_stiffness_kPa": float(self.stiffness_kPa.mean()),
            "max_stiffness_kPa": float(self.stiffness_kPa.max()),
            "mean_septal_thickness_um": float(2 * self.septal_half_thickness.mean()),
            "n_defect_plus": int(len(defects["plus"])),
            "n_defect_minus": int(len(defects["minus"])),
            "defect_density_per_mm2": float(
                (len(defects["plus"]) + len(defects["minus"])) / (area / 1e6)
            ),
        }


class CoupledSimulation:
    """Epithelium and mesenchyme running together on the same alveolar geometry."""

    def __init__(self, config):
        config.validate()
        self.cfg = config
        self.epithelium = AlveolarSimulation(config)
        self.mesenchyme = MesenchymeLayer(self.epithelium, config)
        self.history: list[dict] = []

    @property
    def time_h(self) -> float:
        return self.epithelium.time_h

    def step(self) -> None:
        mes, ep = self.mesenchyme, self.epithelium
        cfg = mes.cfg

        # --- breathing: recompute how the fixed tidal volume is shared out ---
        mes.strain = mes.strain_field()
        ep.segment_strain = mes.sample_grid(
            mes.strain,
            ep.geometry.segment_centre[:, 0], ep.geometry.segment_centre[:, 1],
        )

        # --- stretch-activated TGF-beta: each breath releases more of it on
        # --- stiffer matrix, so strain and stiffness multiply ---
        breaths_per_h = 60.0 * cfg.breaths_per_min
        release = (
            cfg.strain_tgfb_gain * mes.strain
            * (mes.stiffness_kPa / cfg.E_act_kPa)
            * breaths_per_h * 1e-4
        )
        ix = np.clip(
            (np.arange(ep.nx) * ep.grid_step_um / mes.grid_step).astype(int),
            0, mes.nx - 1,
        )
        iy = np.clip(
            (np.arange(ep.ny) * ep.grid_step_um / mes.grid_step).astype(int),
            0, mes.ny - 1,
        )
        ep.profibrotic += cfg.dt_h * release[np.ix_(iy, ix)]

        before = self.epithelium.mesenchymal_released
        aberrant_positions = self.epithelium.geometry.segment_centre[
            self.epithelium.state == ABERRANT
        ]
        self.epithelium.step()
        released = self.epithelium.mesenchymal_released - before
        if released > 0 and aberrant_positions.size:
            picks = self.mesenchyme.rng.choice(
                len(aberrant_positions), size=released, replace=True
            )
            jitter = self.mesenchyme.rng.normal(
                0, self.cfg.septal_thickness_um, (released, 2)
            )
            self.mesenchyme.add_from_emt(aberrant_positions[picks] + jitter)
        self.mesenchyme.step(self.cfg.dt_h)

    def metrics(self) -> dict:
        combined = self.epithelium.metrics()
        combined.update(self.mesenchyme.metrics())
        return combined

    def run(self, record_every_h: float = 24.0) -> "CoupledSimulation":
        cfg = self.cfg
        n_steps = int(round(cfg.total_time_h / cfg.dt_h))
        every = max(1, int(round(record_every_h / cfg.dt_h)))
        self.history.append(self.metrics())
        for index in range(1, n_steps + 1):
            self.step()
            if index % every == 0:
                self.history.append(self.metrics())
        return self
