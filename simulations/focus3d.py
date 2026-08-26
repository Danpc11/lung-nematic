"""Three-dimensional active-nematic model of a fibroblastic focus.

Why this exists
---------------
``simulations.fibrofocus`` reaches a genuine nematic phase - on the gel data it
reproduced a local order of 0.56-0.73 against 0.63-0.67 measured - but it is
two-dimensional, so its defects are points. Histological sections cut a
three-dimensional tissue, where defects are *lines*, and what a slide counts is
line-plane intersections. Calibrating that projection needs a volume that is
actually in the nematic phase.

``simulations.alveolar3d`` cannot supply it. Measured across configurations, it
either sits at 1% packing with seven alveoli - a dilute rod gas with no
orientational order - or, with one alveolus, collapses into a blob 33-108 um
across while the alignment radius is 38 um, so every cell interacts with every
other and the order parameter pins at 0.956 regardless of density. An order
parameter that does not move when density changes tenfold is not emergent order.
The physical window, packing 0.2-0.6 in a domain of at least five alignment
radii, lies between those two failures and that model does not reach it.

What this does
--------------
It lifts the 2D rod model to three dimensions, keeping the physics that made it
work and changing only what dimensionality forces:

*Orientation.* A 2D director is one angle; a 3D director is a unit vector with
head-tail symmetry, so alignment is computed from the second-rank tensor
``Q = <3 n n - I> / 2`` and the local director is its leading eigenvector.
Averaging orientation vectors would cancel antiparallel rods.

*Order parameter.* The scalar order is the leading eigenvalue of ``Q``, which
runs from 0 (isotropic) to 1 (aligned). Note this is the 3D convention: the 2D
model's ``|<exp(2 i theta)>|`` is a different quantity on the same 0-1 scale,
and the two must not be compared numerically without care.

*Nematic torque.* In 2D the rod rotates toward the local director through the
smaller of two angles. In 3D it rotates in the plane containing both vectors,
toward whichever sign of the director is nearer - again because the director has
no head.

*Rotational noise.* Added as a small random vector in the plane perpendicular to
the rod, then renormalised, which diffuses the orientation isotropically on the
sphere. Adding noise to the components directly and renormalising would bias
orientations toward the coordinate axes.

Density is controlled by logistic proliferation against a packing target rather
than a fixed cell cap, so the steady state is a volume fraction - the quantity
that decides whether a nematic phase exists at all - instead of a number that
means different things in different domain sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Focus3DConfig:
    """Parameters in physical units. Defaults inherit the 2D model where the
    quantity is dimension-independent."""

    # ---- domain and time (um, hours) ----
    size_um: float = 240.0
    dt_h: float = 0.05
    total_time_h: float = 240.0

    # ---- rods ----
    # 50 x 11 um matches the 2D model. The aspect ratio matters: Onsager's
    # criterion puts the isotropic-nematic transition near a volume fraction of
    # ~4.2 * width / length, so a ratio near 4.5 needs roughly 0.9 - achievable
    # only with soft overlap, which is why target_packing sits below that and
    # order here is driven by active alignment rather than by excluded volume.
    rod_length_um: float = 50.0
    rod_width_um: float = 11.0
    n_nodes: int = 3
    n_initial: int = 150

    # ---- motion ----
    speed_um_per_h: float = 21.4
    speed_myo_factor: float = 0.15
    # The 2D model runs at align 1.4 / noise 0.8, a ratio of 1.7. In three
    # dimensions rotational diffusion acts on two angular degrees of freedom
    # instead of one, so that ratio sits below the ordering threshold: measured
    # here, order stays at 0.04 and even falls with density. Sweeping the ratio
    # puts the transition near 10, and the defaults below sit at 20 - ordered
    # but not saturated, so the order parameter still responds to density.
    rot_diffusion_per_h: float = 0.4
    align_rate_per_h: float = 8.0
    repulsion_um_per_h: float = 60.0

    # ---- growth ----
    prolif_rate_per_h: float = 0.022
    # Logistic target as a volume fraction, not a cell count: the same number of
    # cells is a dense phase in a small box and a gas in a large one, and it was
    # a fixed cap that made the alveolar3d model impossible to place in the
    # nematic window.
    # The isotropic-nematic crossover measured on this model sits between 0.12
    # and 0.24; 0.35 is above it but short of saturation.
    target_packing: float = 0.35

    seed: int = 0

    # ---- derived ----
    alignment_radius_um: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.alignment_radius_um <= 0:
            self.alignment_radius_um = 1.5 * self.rod_width_um
        # The domain must be several interaction ranges across or every cell
        # sees every other and global alignment is trivial rather than emergent.
        if self.size_um < 5 * self.alignment_radius_um:
            raise ValueError(
                f"size_um ({self.size_um}) is under five alignment radii "
                f"({5 * self.alignment_radius_um:.0f}); order would be a "
                "finite-size artefact rather than a nematic phase"
            )

    @property
    def rod_volume_um3(self) -> float:
        return self.rod_length_um * self.rod_width_um**2

    @property
    def domain_volume_um3(self) -> float:
        return self.size_um**3


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12
    )


class Focus3DSimulation:
    """Self-propelled rods with nematic alignment in a periodic cube."""

    def __init__(self, config: Focus3DConfig | None = None) -> None:
        self.cfg = config or Focus3DConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        size = self.cfg.size_um

        self.position = self.rng.uniform(0, size, (self.cfg.n_initial, 3))
        self.director = _normalise(
            self.rng.normal(size=(self.cfg.n_initial, 3))
        )
        self.myo = np.zeros(self.cfg.n_initial, dtype=bool)
        self.time_h = 0.0

    # ------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return int(self.position.shape[0])

    @property
    def packing_fraction(self) -> float:
        return (self.n_cells * self.cfg.rod_volume_um3
                / self.cfg.domain_volume_um3)

    def _node_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Rods as a few nodes along their axis, so contact is anisotropic.

        A single point per cell would make steric repulsion isotropic and the
        rods would behave as spheres, which do not form a nematic phase at any
        density.
        """
        cfg = self.cfg
        offsets = np.linspace(-0.5, 0.5, cfg.n_nodes) * cfg.rod_length_um
        nodes = (self.position[:, None, :]
                 + offsets[None, :, None] * self.director[:, None, :])
        owner = np.repeat(np.arange(self.n_cells), cfg.n_nodes)
        return nodes.reshape(-1, 3) % cfg.size_um, owner

    # --------------------------------------------------------- interactions
    def _steric_and_alignment(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        nodes, owner = self._node_positions()
        tree = cKDTree(nodes, boxsize=cfg.size_um)

        force = np.zeros((self.n_cells, 3))
        tensor = np.zeros((self.n_cells, 3, 3))
        counts = np.zeros(self.n_cells)

        pairs = tree.query_pairs(cfg.rod_width_um, output_type="ndarray")
        if pairs.size:
            a, b = pairs[:, 0], pairs[:, 1]
            owner_a, owner_b = owner[a], owner[b]
            keep = owner_a != owner_b
            a, b, owner_a, owner_b = (a[keep], b[keep],
                                      owner_a[keep], owner_b[keep])
            if a.size:
                delta = nodes[a] - nodes[b]
                # Minimum image: the domain is periodic, and without this a pair
                # straddling the boundary would be pushed apart across the whole
                # box instead of together across the wrap.
                delta -= cfg.size_um * np.round(delta / cfg.size_um)
                distance = np.linalg.norm(delta, axis=1) + 1e-9
                overlap = np.clip(1.0 - distance / cfg.rod_width_um, 0.0, 1.0)
                unit = delta / distance[:, None]
                np.add.at(force, owner_a, overlap[:, None] * unit)
                np.add.at(force, owner_b, -overlap[:, None] * unit)

        # Alignment comes from the SAME node-contact pairs as the steric term,
        # exactly as the 2D model does it. Using centre-to-centre distance
        # instead makes the neighbourhood isotropic and throws away the very
        # anisotropy that drives nematic ordering: measured that way, order fell
        # with density (0.067 to 0.017 from packing 0.09 to 0.47) rather than
        # rising. Node contacts between elongated rods are side-by-side
        # contacts, which is the nematic interaction.
        outer = self.director[:, :, None] * self.director[:, None, :]
        if pairs.size and a.size:
            np.add.at(tensor, owner_a, outer[owner_b])
            np.add.at(tensor, owner_b, outer[owner_a])
            np.add.at(counts, owner_a, 1.0)
            np.add.at(counts, owner_b, 1.0)

        # A rod with no contacts keeps its own orientation as the target, so the
        # torque vanishes rather than pointing somewhere arbitrary.
        lonely = counts == 0
        tensor[lonely] = outer[lonely]
        counts[lonely] = 1.0
        tensor /= counts[:, None, None]

        _, vectors = np.linalg.eigh(tensor)
        local_director = vectors[..., -1]
        return force, local_director

    # -------------------------------------------------------------- stepping
    def step(self) -> None:
        cfg = self.cfg
        force, local_director = self._steric_and_alignment()

        speed = np.where(self.myo, cfg.speed_um_per_h * cfg.speed_myo_factor,
                         cfg.speed_um_per_h)
        self.position = (
            self.position
            + cfg.dt_h * (speed[:, None] * self.director
                          + cfg.repulsion_um_per_h * force)
        ) % cfg.size_um

        # Rotate toward the local director, in the plane the two span. The sign
        # is chosen so the rod turns through the acute angle: a director has no
        # head, so turning 170 degrees to "match" it would be wrong by symmetry.
        target = local_director * np.sign(
            np.sum(local_director * self.director, axis=1, keepdims=True) + 1e-12
        )
        self.director = _normalise(
            self.director
            + cfg.dt_h * cfg.align_rate_per_h * (target - self.director)
        )

        # Rotational diffusion as a kick perpendicular to the rod. Perturbing
        # the components directly and renormalising would concentrate
        # orientations along the coordinate axes.
        kick = self.rng.normal(
            0, np.sqrt(2 * cfg.rot_diffusion_per_h * cfg.dt_h),
            (self.n_cells, 3),
        )
        kick -= self.director * np.sum(kick * self.director, axis=1,
                                       keepdims=True)
        self.director = _normalise(self.director + kick)

        self._proliferate()
        self.time_h += cfg.dt_h

    def _proliferate(self) -> None:
        cfg = self.cfg
        headroom = 1.0 - self.packing_fraction / cfg.target_packing
        if headroom <= 0:
            return
        expected = self.n_cells * cfg.prolif_rate_per_h * cfg.dt_h * headroom
        n_new = int(self.rng.poisson(max(expected, 0.0)))
        if n_new == 0:
            return

        parents = self.rng.choice(self.n_cells, n_new)
        # Daughters appear alongside the parent, offset across the rod rather
        # than along it, and inherit its orientation - division in a packed
        # tissue does not randomise the axis.
        perpendicular = _normalise(self.rng.normal(size=(n_new, 3)))
        perpendicular -= self.director[parents] * np.sum(
            perpendicular * self.director[parents], axis=1, keepdims=True
        )
        offset = _normalise(perpendicular) * cfg.rod_width_um * 0.6

        self.position = np.vstack([
            self.position, (self.position[parents] + offset) % cfg.size_um
        ])
        self.director = np.vstack([self.director, self.director[parents]])
        self.myo = np.concatenate([self.myo, self.myo[parents]])

    def run(self, until_h: float | None = None) -> None:
        target = until_h if until_h is not None else self.cfg.total_time_h
        while self.time_h < target:
            self.step()

    # -------------------------------------------------------------- readouts
    def order_tensor(self) -> np.ndarray:
        """Global ``Q = <3 n n - I> / 2``."""
        if self.n_cells == 0:
            return np.zeros((3, 3))
        outer = np.einsum("ij,ik->jk", self.director, self.director)
        return 1.5 * outer / self.n_cells - 0.5 * np.eye(3)

    def global_order(self) -> float:
        """Leading eigenvalue of Q: 0 isotropic, 1 perfectly aligned.

        This is the 3D convention. The 2D model reports
        ``|<exp(2 i theta)>|``, a different quantity on the same 0-1 scale;
        comparing them numerically without saying so would be an error.
        """
        if self.n_cells == 0:
            return 0.0
        return float(np.linalg.eigvalsh(self.order_tensor())[-1])

    def director_volume(self, n_voxels: int = 48, sigma_voxels: float = 1.5):
        """Coarse-grained Q-tensor field, for ``simulations.stereology``."""
        from .stereology import director_volume_from_points

        return director_volume_from_points(
            self.position, self.director,
            (n_voxels, n_voxels, n_voxels),
            (self.cfg.size_um,) * 3,
            sigma_voxels=sigma_voxels,
        )

    def metrics(self) -> dict:
        return {
            "time_h": self.time_h,
            "n_cells": self.n_cells,
            "packing_fraction": self.packing_fraction,
            "global_order_S": self.global_order(),
            "domain_over_alignment_radius": (
                self.cfg.size_um / self.cfg.alignment_radius_um
            ),
        }
