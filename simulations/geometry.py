"""
Alveolar geometry for the epithelial stage of the fibrosis model.

The alveolar region is built as a Voronoi tessellation of a jittered hexagonal
lattice: each Voronoi cell is one alveolus, each ridge shared by two cells is
an interalveolar septum. Septa are then discretised into short segments, and
each segment carries one epithelial state. This is the substrate on which the
AT2 -> KRT8+ -> AT1 state machine runs.

The tessellation is a deliberate simplification. Real alveoli are 3D polyhedra
sharing septa with many neighbours; here a 2D section through that structure is
modelled, which is the same plane the histology sees.

Geometric targets: human alveolar diameter is of order 200 um and the healthy
interalveolar septum is a few microns thick, thickening severalfold in fibrosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Voronoi


def _polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a simple polygon."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-12:
        return points.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
    return np.array([cx, cy])


@dataclass
class Alveolus:
    """One Voronoi cell: an air space bounded by septa."""

    index: int
    vertices: np.ndarray            # polygon, closed implicitly
    centroid: np.ndarray
    area_um2: float
    septa: list[int] = field(default_factory=list)

    @property
    def equivalent_radius_um(self) -> float:
        """Radius of a disc of the same area; used in the Laplace balance."""
        return float(np.sqrt(self.area_um2 / np.pi))


@dataclass
class Septum:
    """One Voronoi ridge: the wall shared by two alveoli."""

    index: int
    start: np.ndarray
    end: np.ndarray
    alveoli: tuple[int, ...]
    segment_slice: tuple[int, int] = (0, 0)   # [start, stop) into segment arrays

    @property
    def length_um(self) -> float:
        return float(np.hypot(*(self.end - self.start)))


class AlveolarGeometry:
    """Voronoi alveoli, their septa, and the epithelial segments on the septa."""

    def __init__(
        self,
        width_um: float = 1200.0,
        height_um: float = 1200.0,
        alveolar_diameter_um: float = 200.0,
        jitter: float = 0.18,
        segment_length_um: float = 12.0,
        seed: int = 0,
    ):
        self.width_um = float(width_um)
        self.height_um = float(height_um)
        self.segment_length_um = float(segment_length_um)
        rng = np.random.default_rng(seed)

        seeds = self._hex_seeds(alveolar_diameter_um, jitter, rng)
        voronoi = Voronoi(seeds)

        self.alveoli: list[Alveolus] = []
        region_of_point: dict[int, int] = {}
        for point_index, region_index in enumerate(voronoi.point_region):
            region = voronoi.regions[region_index]
            if not region or -1 in region:
                continue                       # unbounded cell at the border
            polygon = voronoi.vertices[region]
            if (
                polygon[:, 0].min() < 0 or polygon[:, 0].max() > self.width_um
                or polygon[:, 1].min() < 0 or polygon[:, 1].max() > self.height_um
            ):
                continue                       # cell sticking out of the domain
            alveolus = Alveolus(
                index=len(self.alveoli),
                vertices=polygon,
                centroid=_polygon_centroid(polygon),
                area_um2=_polygon_area(polygon),
            )
            region_of_point[point_index] = alveolus.index
            self.alveoli.append(alveolus)

        # Septa: ridges whose two owning cells both survived the trimming.
        self.septa: list[Septum] = []
        for (pa, pb), (va, vb) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            if va == -1 or vb == -1:
                continue
            owners = tuple(
                region_of_point[p] for p in (pa, pb) if p in region_of_point
            )
            if not owners:
                continue
            septum = Septum(
                index=len(self.septa),
                start=voronoi.vertices[va],
                end=voronoi.vertices[vb],
                alveoli=owners,
            )
            if septum.length_um < self.segment_length_um:
                continue
            for owner in owners:
                self.alveoli[owner].septa.append(septum.index)
            self.septa.append(septum)

        self._build_segments()

    # ------------------------------------------------------------------ setup
    def _hex_seeds(self, diameter: float, jitter: float,
                   rng: np.random.Generator) -> np.ndarray:
        """Jittered hexagonal lattice of alveolar centres."""
        dx = diameter
        dy = diameter * np.sqrt(3) / 2
        points = []
        row = 0
        y = -dy
        while y < self.height_um + dy:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x = -dx + offset
            while x < self.width_um + dx:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
        seeds = np.array(points, dtype=float)
        seeds += rng.normal(0.0, jitter * diameter, seeds.shape)
        return seeds

    def _build_segments(self) -> None:
        """Cut every septum into epithelial segments of roughly equal length."""
        centres: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        owners: list[int] = []
        lengths: list[float] = []

        for septum in self.septa:
            n_segments = max(1, int(round(septum.length_um / self.segment_length_um)))
            start_index = len(centres)
            direction = septum.end - septum.start
            unit = direction / max(np.linalg.norm(direction), 1e-9)
            normal = np.array([-unit[1], unit[0]])
            for k in range(n_segments):
                t = (k + 0.5) / n_segments
                centres.append(septum.start + t * direction)
                normals.append(normal)
                owners.append(septum.index)
                lengths.append(septum.length_um / n_segments)
            septum.segment_slice = (start_index, len(centres))

        self.segment_centre = np.array(centres) if centres else np.zeros((0, 2))
        self.segment_normal = np.array(normals) if normals else np.zeros((0, 2))
        self.segment_septum = np.array(owners, dtype=int)
        self.segment_length = np.array(lengths)

        # alveolus index for every segment (a segment may border two alveoli;
        # both are recorded, -1 padding when the septum is on the border)
        pairs = np.full((len(self.segment_septum), 2), -1, dtype=int)
        for septum in self.septa:
            lo, hi = septum.segment_slice
            for slot, alveolus in enumerate(septum.alveoli[:2]):
                pairs[lo:hi, slot] = alveolus
        self.segment_alveoli = pairs

    # ---------------------------------------------------------------- helpers
    @property
    def n_alveoli(self) -> int:
        return len(self.alveoli)

    @property
    def n_segments(self) -> int:
        return int(self.segment_centre.shape[0])

    @property
    def alveolar_areas(self) -> np.ndarray:
        return np.array([a.area_um2 for a in self.alveoli])

    @property
    def alveolar_radii(self) -> np.ndarray:
        return np.array([a.equivalent_radius_um for a in self.alveoli])

    def segments_of_alveolus(self, index: int) -> np.ndarray:
        """Indices of the epithelial segments lining one alveolus."""
        mask = (self.segment_alveoli[:, 0] == index) | (
            self.segment_alveoli[:, 1] == index
        )
        return np.nonzero(mask)[0]

    def summary(self) -> dict:
        areas = self.alveolar_areas
        diameters = 2 * self.alveolar_radii
        return {
            "n_alveoli": self.n_alveoli,
            "n_septa": len(self.septa),
            "n_segments": self.n_segments,
            "mean_diameter_um": float(diameters.mean()) if areas.size else 0.0,
            "std_diameter_um": float(diameters.std()) if areas.size else 0.0,
            "mean_septum_length_um": float(
                np.mean([s.length_um for s in self.septa])
            ) if self.septa else 0.0,
            "total_septal_length_um": float(
                np.sum([s.length_um for s in self.septa])
            ) if self.septa else 0.0,
        }
