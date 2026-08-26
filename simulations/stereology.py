"""Relate defects counted on a 2D section to the 3D structure they cut.

The problem
-----------
In a two-dimensional nematic, defects are points and their areal density is a
property of the material. In three dimensions they are *lines* - disclinations -
and a histological section intersects them. What is counted on a slide is
therefore a number of line-plane intersections per unit area, not a point
density, and the two are different quantities with different units. Comparing a
section's defect density to a cell monolayer's is comparing a stereological
projection to the thing itself.

This is not a small correction. The classical stereological identity for
isotropic uniformly random sections through a network of curves is

    N_A = L_V / 2

where ``N_A`` is intersections per unit area and ``L_V`` is line length per unit
volume. The factor of two is the average of ``|cos|`` between the line and the
section normal over the sphere, and it holds *only* when the lines are
isotropically oriented. Disclinations in a fibroblastic focus are not isotropic:
they run along the architecture of the lesion. When they are preferentially
parallel to the section plane, a section cuts almost none of them and the count
collapses; when perpendicular, every one is counted.

So the section angle is not a nuisance to average over. It is a variable that
can change the measured density several-fold with no change in the tissue, and
the only way to calibrate it is to cut a known volume at known angles.

What this module does
---------------------
``section_sweep`` takes a 3D director field, cuts it at a range of angles, runs
the same plaquette-winding detector used on real sections, and reports how the
apparent areal density varies. ``estimate_line_density`` inverts the relation to
recover ``L_V``, and ``anisotropy_factor`` reports how far the lines are from
isotropic - which is the number that says whether ``N_A = L_V / 2`` may be used
at all.

Nothing here assumes the tissue is isotropic. That assumption is what the module
exists to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# The isotropic stereological factor: <|cos t|> over the sphere is 1/2.
ISOTROPIC_FACTOR = 0.5


def director_volume_from_points(
    positions: np.ndarray,
    directors: np.ndarray,
    shape: tuple[int, int, int],
    extent_um: tuple[float, float, float],
    sigma_voxels: float = 1.5,
) -> dict[str, np.ndarray]:
    """Coarse-grain 3D rod orientations onto a voxel grid as a Q tensor.

    Directors are head-tail symmetric in 3D as well, so they cannot be averaged
    as vectors: two antiparallel rods would cancel. The second-rank tensor
    ``Q = <n n> - I/3`` is the correct object, and its leading eigenvector
    recovers the local director.
    """
    positions = np.asarray(positions, dtype=float)
    directors = np.asarray(directors, dtype=float)
    if positions.shape[0] != directors.shape[0]:
        raise ValueError("positions and directors must have the same length")

    counts = np.zeros(shape)
    tensor = np.zeros(shape + (3, 3))

    index = np.stack([
        np.clip((positions[:, axis] / extent_um[axis] * shape[axis]).astype(int),
                0, shape[axis] - 1)
        for axis in range(3)
    ], axis=1)

    unit = directors / np.maximum(
        np.linalg.norm(directors, axis=1, keepdims=True), 1e-12
    )
    outer = unit[:, :, None] * unit[:, None, :]
    for row in range(3):
        for column in range(3):
            np.add.at(tensor[..., row, column],
                      (index[:, 0], index[:, 1], index[:, 2]),
                      outer[:, row, column])
    np.add.at(counts, (index[:, 0], index[:, 1], index[:, 2]), 1.0)

    if sigma_voxels > 0:
        counts = gaussian_filter(counts, sigma_voxels)
        for row in range(3):
            for column in range(3):
                tensor[..., row, column] = gaussian_filter(
                    tensor[..., row, column], sigma_voxels
                )

    with np.errstate(invalid="ignore", divide="ignore"):
        tensor = tensor / np.maximum(counts, 1e-9)[..., None, None]
    return {"tensor": np.nan_to_num(tensor), "density": counts}


def _director_from_tensor(tensor: np.ndarray) -> np.ndarray:
    """Leading eigenvector of each voxel's Q tensor."""
    _, vectors = np.linalg.eigh(tensor)
    return vectors[..., -1]


def section_plane(
    volume: dict[str, np.ndarray],
    polar_deg: float,
    azimuth_deg: float = 0.0,
    n_samples: int = 160,
) -> dict[str, np.ndarray]:
    """Cut the volume with a plane and project the director into it.

    A section does not see the full 3D director: it sees its projection onto the
    cut plane, which is what a microscope image of that section contains. Taking
    the in-plane components before measuring winding is therefore not an
    approximation, it is what the real measurement does.
    """
    polar = np.radians(polar_deg)
    azimuth = np.radians(azimuth_deg)
    normal = np.array([
        np.sin(polar) * np.cos(azimuth),
        np.sin(polar) * np.sin(azimuth),
        np.cos(polar),
    ])
    # Any two vectors completing an orthonormal frame will do; the choice only
    # rotates the resulting image, which winding is invariant to.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    shape = np.array(volume["density"].shape, dtype=float)
    centre = shape / 2.0
    # Stay inside the inscribed sphere so the plane never leaves the volume,
    # whatever its orientation; sampling outside would read zero-padding as
    # tissue and manufacture defects at the boundary.
    half = float(shape.min()) / 2.0 * 0.8
    grid = np.linspace(-half, half, n_samples)
    uu, vv = np.meshgrid(grid, grid, indexing="ij")

    points = (centre[None, None, :]
              + uu[..., None] * axis_u[None, None, :]
              + vv[..., None] * axis_v[None, None, :])
    voxel = np.clip(np.rint(points).astype(int), 0,
                    (shape - 1).astype(int)[None, None, :])

    tensor = volume["tensor"][voxel[..., 0], voxel[..., 1], voxel[..., 2]]
    density = volume["density"][voxel[..., 0], voxel[..., 1], voxel[..., 2]]
    director = _director_from_tensor(tensor)

    in_plane_u = np.einsum("...i,i->...", director, axis_u)
    in_plane_v = np.einsum("...i,i->...", director, axis_v)
    theta = (0.5 * np.arctan2(2 * in_plane_u * in_plane_v,
                              in_plane_u**2 - in_plane_v**2)) % np.pi

    # How much of the director actually lies in the plane. Where a rod is nearly
    # perpendicular to the cut its in-plane projection is tiny and its apparent
    # orientation is dominated by noise, so this doubles as a confidence weight.
    in_plane_fraction = np.hypot(in_plane_u, in_plane_v)
    return {
        "theta": theta,
        "density": density,
        "in_plane_fraction": in_plane_fraction,
        "normal": normal,
    }


def count_section_defects(
    section: dict[str, np.ndarray],
    grid_step: int = 3,
    min_density_ratio: float = 0.3,
    min_in_plane: float = 0.3,
) -> dict[str, float]:
    """Plaquette winding on a section, as the histology pipeline does it.

    ``min_in_plane`` discards regions where the director is nearly normal to the
    cut. Their apparent in-plane angle is essentially random, and including them
    would generate spurious winding that scales with how obliquely the tissue
    was sectioned - exactly the artefact this module is meant to quantify rather
    than commit.

    ``min_density_ratio`` is deliberately an ABSOLUTE threshold, taken relative
    to the volume's mean density rather than as a quantile of this section. A
    quantile always removes a fixed share of the plane whether or not any of it
    is empty: on a uniformly filled test volume a 0.4 quantile discarded 58% of
    the field and every defect with it, reporting zero where four were present.
    ``simulations.fibrofocus.render.detect_defects`` avoids the same trap for
    the same reason.
    """
    theta = section["theta"]
    density = section["density"]
    reference = float(np.mean(density[density > 0])) if np.any(density > 0) else 0.0
    valid = (density >= min_density_ratio * reference) & (
        section["in_plane_fraction"] >= min_in_plane
    )

    rows = np.arange(0, theta.shape[0] - grid_step, grid_step)
    cols = np.arange(0, theta.shape[1] - grid_step, grid_step)
    if rows.size < 2 or cols.size < 2:
        return {"n_plus": 0, "n_minus": 0, "n_total": 0, "valid_fraction": 0.0}

    corners = [
        theta[np.ix_(rows, cols)],
        theta[np.ix_(rows, cols + grid_step)],
        theta[np.ix_(rows + grid_step, cols + grid_step)],
        theta[np.ix_(rows + grid_step, cols)],
    ]
    winding = np.zeros_like(corners[0])
    for index in range(4):
        step = corners[(index + 1) % 4] - corners[index]
        winding += (step + np.pi / 2) % np.pi - np.pi / 2
    charge = winding / (2 * np.pi)

    ok = [valid[np.ix_(rows + offset_r, cols + offset_c)]
          for offset_r, offset_c in ((0, 0), (0, grid_step),
                                     (grid_step, grid_step), (grid_step, 0))]
    usable = np.all(ok, axis=0)

    plus = int(np.sum((charge > 0.25) & usable))
    minus = int(np.sum((charge < -0.25) & usable))
    return {
        "n_plus": plus,
        "n_minus": minus,
        "n_total": plus + minus,
        "valid_fraction": float(usable.mean()),
    }


def section_sweep(
    volume: dict[str, np.ndarray],
    extent_um: tuple[float, float, float],
    polar_angles_deg=(0, 15, 30, 45, 60, 75, 90),
    n_azimuths: int = 4,
    n_samples: int = 160,
) -> pd.DataFrame:
    """Apparent areal defect density as a function of section angle.

    The spread across angles is the whole point. If the density were independent
    of the cut, a histological count would be a property of the tissue; the
    extent to which it is not is the size of the correction that any
    section-to-monolayer comparison needs.
    """
    shape = np.array(volume["density"].shape, dtype=float)
    voxel_um = np.array(extent_um) / shape
    half = float(shape.min()) / 2.0 * 0.8
    # The sampled plane is a square of side 2*half voxels; convert with the mean
    # voxel size, which is exact for the cubic grids used here.
    side_um = 2 * half * float(voxel_um.mean())
    area_mm2 = (side_um / 1000.0) ** 2

    rows = []
    for polar in polar_angles_deg:
        for azimuth in np.linspace(0, 180, n_azimuths, endpoint=False):
            section = section_plane(volume, polar, azimuth, n_samples)
            counts = count_section_defects(section)
            rows.append({
                "polar_deg": float(polar),
                "azimuth_deg": float(azimuth),
                "n_defects": counts["n_total"],
                "n_plus": counts["n_plus"],
                "n_minus": counts["n_minus"],
                "valid_fraction": counts["valid_fraction"],
                "areal_density_mm2": counts["n_total"] / area_mm2,
                "mean_in_plane_fraction": float(
                    section["in_plane_fraction"].mean()
                ),
                "charge_imbalance": (
                    abs(counts["n_plus"] - counts["n_minus"]) / counts["n_total"]
                    if counts["n_total"] else np.nan
                ),
            })
    return pd.DataFrame(rows)


def anisotropy_factor(sweep: pd.DataFrame) -> dict[str, float]:
    """How far the apparent density is from angle-independent.

    ``N_A = L_V / 2`` requires isotropically oriented lines. This reports the
    ratio between the most and least productive section angles: at 1 the
    identity may be used, and the further above 1 the more a single histological
    density describes the cut rather than the tissue.
    """
    by_angle = sweep.groupby("polar_deg").areal_density_mm2.mean()
    if by_angle.empty or by_angle.min() <= 0:
        return {"ratio": float("nan"), "best_polar_deg": float("nan"),
                "worst_polar_deg": float("nan"),
                "mean_density_mm2": float(by_angle.mean())
                if not by_angle.empty else float("nan")}
    return {
        "ratio": float(by_angle.max() / by_angle.min()),
        "best_polar_deg": float(by_angle.idxmax()),
        "worst_polar_deg": float(by_angle.idxmin()),
        "mean_density_mm2": float(by_angle.mean()),
    }


def estimate_line_density(sweep: pd.DataFrame) -> dict[str, float]:
    """Recover disclination line length per unit volume from a section sweep.

    Uses ``L_V = 2 * <N_A>`` with the average taken over solid angle, weighting
    each polar angle by ``sin(theta)`` because equal steps in polar angle do not
    sample the sphere uniformly - a plain mean over the angles in the sweep
    would over-weight the poles and bias the estimate.

    The estimate is only as good as the isotropy it assumes; read
    ``anisotropy_factor`` first and treat a large ratio as a reason to report
    the sweep rather than this single number.
    """
    by_angle = sweep.groupby("polar_deg").areal_density_mm2.mean()
    angles = np.radians(by_angle.index.to_numpy(dtype=float))
    weights = np.sin(angles)
    if weights.sum() <= 0:
        weighted = float(by_angle.mean())
    else:
        weighted = float(np.average(by_angle.to_numpy(), weights=weights))
    return {
        "mean_areal_density_mm2": weighted,
        "line_density_mm_per_mm3": weighted / ISOTROPIC_FACTOR,
        "isotropy_assumed": True,
    }
