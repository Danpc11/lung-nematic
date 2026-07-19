from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class AnalysisConfig:
    # Nuclear segmentation
    min_nucleus_area_px: int = 20
    max_nucleus_area_px: int = 700
    min_major_axis_px: float = 4.0
    max_major_axis_px: float = 60.0
    min_minor_axis_px: float = 2.5
    max_aspect_ratio: float = 6.0
    min_aspect_ratio_for_orientation: float = 1.35

    # Nematic field
    sigmas_px: tuple[float, ...] = (40.0, 55.0, 70.0, 85.0)
    field_grid_step_px: int = 64
    min_local_order_for_display: float = 0.12
    density_quantile: float = 0.45

    # Defect detection
    defect_grid_step_px: int = 24
    min_edge_distance_px: int = 30
    defect_cluster_radius_px: float = 70.0
    min_scales_for_persistence: int = 2

    # Integer (+/-1) defect layer (opt-in). Detected on an N-point ring so the
    # full 2*pi winding of an aster/vortex/saddle is resolved, which a 4-corner
    # plaquette cannot do (its per-edge phase step sits at the +/-pi branch cut).
    detect_integer_defects: bool = False
    integer_defect_loop_radius_px: int = 30
    integer_defect_loop_points: int = 8
    # Every ring enclosing an integer defect registers it, so raw candidates
    # must be collapsed for a count to mean "defects". Defaults to twice the ring
    # radius (the diameter of the disc one defect illuminates); set to 0
    # to keep every candidate.
    integer_min_separation_px: float = 60.0

    # Collagen / fused fields
    collagen_inner_scale_px: float = 1.5
    mask_normalized_smoothing: bool = False

    # Analysis selection
    field_type: str = "nuclear"
    run_null: bool = False
    run_colocalization: bool = False

    # Null model
    n_permutations: int = 199
    null_mode: str = "shuffle"
    null_downsample: int = 2

    # Colocalization
    n_bootstrap: int = 2000
    colocalization_annulus_inner_frac: float = 1.0
    colocalization_annulus_outer_frac: float = 2.0

    # Reproducibility
    random_seed: int = 42

    # Physical scale
    default_microns_per_pixel: float | None = 0.114679  # 1 / 8.72 px per µm

    # Output
    save_diagnostic_panel: bool = True
    save_intermediate_arrays: bool = False
    save_defect_maps: bool = False
    defect_map_window_px: int = 220

    def validate(self) -> None:
        if self.min_nucleus_area_px <= 0:
            raise ValueError("min_nucleus_area_px must be positive.")
        if self.max_nucleus_area_px <= self.min_nucleus_area_px:
            raise ValueError(
                "max_nucleus_area_px must exceed min_nucleus_area_px."
            )
        if not self.sigmas_px:
            raise ValueError("sigmas_px must contain at least one value.")
        if any(value <= 0 for value in self.sigmas_px):
            raise ValueError("All sigmas_px values must be positive.")
        if not 0 <= self.density_quantile < 1:
            raise ValueError("density_quantile must be in [0, 1).")
        if self.min_scales_for_persistence < 1:
            raise ValueError(
                "min_scales_for_persistence must be at least 1."
            )
        if self.min_scales_for_persistence > len(self.sigmas_px):
            raise ValueError(
                "min_scales_for_persistence cannot exceed the number "
                "of smoothing scales."
            )
        if (
            self.default_microns_per_pixel is not None
            and self.default_microns_per_pixel <= 0
        ):
            raise ValueError(
                "default_microns_per_pixel must be positive or null."
            )

        # Pixel-based radii and steps must be usable.
        for name in (
            "field_grid_step_px",
            "defect_grid_step_px",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.min_edge_distance_px < 0:
            raise ValueError("min_edge_distance_px must be non-negative.")
        if self.defect_cluster_radius_px <= 0:
            raise ValueError("defect_cluster_radius_px must be positive.")
        if self.integer_min_separation_px < 0:
            raise ValueError("integer_min_separation_px must be non-negative.")
        if self.integer_defect_loop_radius_px < 1:
            raise ValueError("integer_defect_loop_radius_px must be at least 1.")
        if self.integer_defect_loop_points < 6:
            raise ValueError(
                "integer_defect_loop_points must be at least 6 to resolve a "
                "full +/-1 winding without branch-cut ambiguity."
            )
        if self.defect_map_window_px < 16:
            raise ValueError("defect_map_window_px must be at least 16.")
        if self.collagen_inner_scale_px <= 0:
            raise ValueError("collagen_inner_scale_px must be positive.")

        # Analysis selection.
        if self.field_type not in {"nuclear", "collagen", "fused"}:
            raise ValueError(
                "field_type must be 'nuclear', 'collagen' or 'fused'."
            )
        if self.null_mode not in {"shuffle", "uniform"}:
            raise ValueError("null_mode must be 'shuffle' or 'uniform'.")
        for name in ("n_permutations", "n_bootstrap", "null_downsample"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")

        # Colocalization annulus.
        if self.colocalization_annulus_inner_frac < 0:
            raise ValueError(
                "colocalization_annulus_inner_frac must be non-negative."
            )
        if (
            self.colocalization_annulus_outer_frac
            <= self.colocalization_annulus_inner_frac
        ):
            raise ValueError(
                "colocalization_annulus_outer_frac must exceed the inner "
                "fraction."
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sigmas_px"] = list(self.sigmas_px)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        valid_names = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - valid_names)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")

        normalized = dict(data)
        if "sigmas_px" in normalized:
            normalized["sigmas_px"] = tuple(
                float(value) for value in normalized["sigmas_px"]
            )

        config = cls(**normalized)
        config.validate()
        return config


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AnalysisConfig.from_dict(data)


def save_config(config: AnalysisConfig, path: str | Path) -> None:
    config.validate()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def load_default_config() -> AnalysisConfig:
    """Load the default configuration shipped inside the installed package.

    Works regardless of the current working directory, so it is safe after a
    plain ``pip install git+...`` where the repository's ``config/`` folder is
    not present.
    """
    from importlib.resources import files

    resource = files("lung_nematic.data").joinpath("default_config.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    return AnalysisConfig.from_dict(data)
