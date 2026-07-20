"""
Worked example: NHLF stiffness series, and the cross-comparison with histology.

Run from the repository root once the gel frames and histology images are on
disk. It does three things, in the order the analysis should be read:

  1. analyses the phase-contrast gel frames into a per-image table and draws
     each director field;
  2. fits the order(stiffness) calibration from the gels, where stiffness is
     known;
  3. analyses the histology with the existing collagen pipeline and places each
     tissue image on the *same* order axis, then reads an estimated stiffness
     off the gel calibration - as a range, never a point.

Only dimensionless quantities cross between the two systems: nematic order,
coherence, and correlation length in object-diameter units. Defect densities in
mm^-2 do not, because the two systems are at different pixel scales; supply
`microns_per_pixel` per system before comparing those.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from lung_nematic.config import load_default_config
from lung_nematic.phase_contrast import analyze_gel_series


# --------------------------------------------------------------------------
# 1-2. the gels
# --------------------------------------------------------------------------
def run_gels(output_dir: str = "results/gels") -> pd.DataFrame:
    config = replace(
        load_default_config(),
        sigmas_px=(18.0, 25.0, 32.0),   # a few cell widths, in this pixel scale
        density_quantile=0.30,
        min_scales_for_persistence=2,
        defect_grid_step_px=12,
        min_edge_distance_px=25,
        field_grid_step_px=40,
    )

    # map each substrate stiffness to its frame(s); lists allow replicates
    gels = {
        5.0: "data/gels/5k_b2_2.tif",
        23.0: "data/gels/23k_c2_1.tif",
    }

    table = analyze_gel_series(gels, config, output_dir=output_dir, draw_fields=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output_dir) / "gel_metrics.csv", index=False)

    print("Per-frame gel metrics")
    print(table[[
        "stiffness_kPa", "global_nematic_order_S", "local_S_median",
        "correlation_length_px", "n_defects_total",
    ]].to_string(index=False))

    if "calibration" in table.attrs:
        calibration = table.attrs["calibration"]
        print(f"\norder(E) fit: R^2 = {table.attrs['calibration_r_squared']:.3f}")
        print("  stiffer substrate -> higher order -> fewer defects, as expected"
              " for an active nematic")
    return table


# --------------------------------------------------------------------------
# 3. place histology on the same axis
# --------------------------------------------------------------------------
def compare_with_histology(
    gel_table: pd.DataFrame,
    histology_paths: list[str],
    output_dir: str = "results/histology",
) -> pd.DataFrame:
    from lung_nematic.pipeline import analyze_image

    calibration = gel_table.attrs.get("calibration")
    config = load_default_config()

    rows = []
    for path in histology_paths:
        summary = analyze_image(
            path,
            metadata={
                "image_id": Path(path).stem,
                "filename": Path(path).name,
                "relative_path": Path(path).name,
                "group": "histology",
                "microns_per_pixel": None,
            },
            output_root=output_dir,
            config=config,
            field_type="collagen",     # continuous eosin: many samples per window
            run_null=True,
        )
        order = summary.get("local_S_median", float("nan"))
        row = {"image": Path(path).stem, "local_S_median": order}

        # the inverse step: an ESTIMATE with an interval, flagged if extrapolated
        if calibration is not None and order == order:
            estimate = calibration.estimate_stiffness(order, order_uncertainty=0.05)
            row.update({
                "estimated_E_kPa": estimate["stiffness_kPa"],
                "E_low_kPa": estimate["low_kPa"],
                "E_high_kPa": estimate["high_kPa"],
                "in_calibrated_range": estimate["in_calibrated_range"],
            })
        rows.append(row)

    table = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output_dir) / "histology_on_gel_axis.csv", index=False)

    print("\nHistology placed on the gel order axis")
    print(table.to_string(index=False))
    print(
        "\nEstimated stiffness is an interpolation bounded by the gel scatter,"
        "\nnot a measurement. Report the interval, and treat any row with"
        "\nin_calibrated_range = False as extrapolation."
    )
    return table


if __name__ == "__main__":
    gel_table = run_gels()
    histology = [
        "data/histology/23-15_1.jpg",
        "data/histology/23-15_13.jpg",
        # ... the rest of the series
    ]
    if all(Path(p).exists() for p in histology):
        compare_with_histology(gel_table, histology)
    else:
        print("\n(histology comparison skipped: point the paths at your images)")
