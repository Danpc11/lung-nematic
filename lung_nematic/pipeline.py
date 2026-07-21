from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .collagen_field import (
    compute_collagen_field,
    detect_multiscale_collagen_defects,
)
from .colocalization import run_colocalization as _run_colocalization
from .config import AnalysisConfig
from .defect_maps import render_defect_maps
from .defects import detect_multiscale_defects
from .fused_field import detect_multiscale_fused_defects
from .io_utils import json_safe as _json_safe, read_rgb
from .metrics import summarize_image
from .null_model import (
    run_collagen_null_model,
    run_null_model,
    save_null_histogram,
)
from .preprocessing import make_tissue_mask
from .segmentation import segment_nuclei, select_oriented_nuclei
from .visualization import save_diagnostic_panel, save_overlay


def _safe_identifier(value: object) -> str:
    text = str(value)
    for character in ("/", "\\", ":", " "):
        text = text.replace(character, "_")
    return text




def _detect_for_field(
    field_type: str,
    oriented_nuclei: pd.DataFrame,
    eosin: np.ndarray,
    tissue_mask: np.ndarray,
    config: AnalysisConfig,
):
    if field_type == "nuclear":
        return detect_multiscale_defects(oriented_nuclei, tissue_mask, config)
    if field_type == "collagen":
        return detect_multiscale_collagen_defects(
            eosin, tissue_mask, config, config.collagen_inner_scale_px
        )
    return detect_multiscale_fused_defects(
        oriented_nuclei, eosin, tissue_mask, config, config.collagen_inner_scale_px
    )


def analyze_image(
    image_path: str | Path,
    metadata: dict,
    output_root: str | Path,
    config: AnalysisConfig,
    *,
    field_type: str | None = None,
    run_null: bool | None = None,
    run_colocalization: bool | None = None,
) -> dict:
    """
    Run the full analysis for one image and save all outputs.

    ``field_type`` ("nuclear", "collagen", "fused"), ``run_null`` and
    ``run_colocalization`` default to the values in ``config`` when left as
    None, so the CLI drives them through the config while callers (e.g. the
    Colab notebook) may override them per call. This is the single engine both
    interfaces use.
    """
    config.validate()

    field_type = config.field_type if field_type is None else field_type
    run_null = config.run_null if run_null is None else run_null
    run_colocalization = (
        config.run_colocalization
        if run_colocalization is None
        else run_colocalization
    )
    if field_type not in {"nuclear", "collagen", "fused"}:
        raise ValueError(f"Unknown field_type: {field_type}")

    image_path = Path(image_path)
    output_root = Path(output_root)
    rgb = read_rgb(image_path)
    tissue_mask, hed = make_tissue_mask(rgb)
    eosin = hed[:, :, 1]

    labels, nuclei = segment_nuclei(tissue_mask, hed, config)
    oriented_nuclei = select_oriented_nuclei(nuclei, config)

    defects, fields, raw_detections = _detect_for_field(
        field_type, oriented_nuclei, eosin, tissue_mask, config
    )

    representative_sigma = float(config.sigmas_px[len(config.sigmas_px) // 2])
    field = fields[representative_sigma]

    safe_id = _safe_identifier(metadata["image_id"])
    image_output = output_root / safe_id
    image_output.mkdir(parents=True, exist_ok=True)
    tag = f"{safe_id}_{field_type}"

    nuclei_export = nuclei.copy()
    nuclei_export.insert(0, "image_id", metadata["image_id"])
    nuclei_export.insert(1, "group", metadata["group"])
    nuclei_export.to_csv(image_output / f"{safe_id}_nuclei.csv", index=False)

    defects_export = defects.copy()
    defects_export.insert(0, "image_id", metadata["image_id"])
    defects_export.insert(1, "group", metadata["group"])
    defects_export.insert(2, "field", field_type)
    defects_export.to_csv(image_output / f"{tag}_defects.csv", index=False)

    if not raw_detections.empty:
        raw_detections.to_csv(
            image_output / f"{tag}_raw_defect_detections.csv", index=False
        )

    overlay_path = image_output / f"{tag}_overlay.png"
    save_overlay(
        rgb, tissue_mask, field, defects, overlay_path, config,
        title=f"{metadata['image_id']} | {metadata['group']} | {field_type}",
    )

    if config.save_diagnostic_panel:
        save_diagnostic_panel(
            rgb, tissue_mask, labels, field, defects,
            image_output / f"{tag}_diagnostic_panel.png",
            title=f"{metadata['image_id']} | {metadata['group']} | {field_type}",
        )

    if config.save_intermediate_arrays:
        np.savez_compressed(
            image_output / f"{tag}_field_arrays.npz",
            tissue_mask=tissue_mask, labels=labels,
            density=field["density"], order=field["order"], theta=field["theta"],
        )

    summary = summarize_image(
        metadata=metadata,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        nuclei=nuclei,
        oriented_nuclei=oriented_nuclei,
        field=field,
        defects=defects,
        density_quantile=config.density_quantile,
        representative_sigma_px=representative_sigma,
    )
    summary["field_type"] = field_type
    summary["overlay_path"] = str(overlay_path)

    if run_null:
        summary.update(
            _run_null(field_type, oriented_nuclei, eosin, tissue_mask, config,
                      image_output, tag)
        )

    if run_colocalization:
        coloc = _run_colocalization(
            defects, field, tissue_mask, config,
            representative_sigma_px=representative_sigma,
            n_bootstrap=config.n_bootstrap, seed=config.random_seed,
        )
        pd.DataFrame({
            "core_null": coloc["core_null"],
            "annulus_null": coloc["annulus_null"],
        }).to_csv(image_output / f"{tag}_colocalization_null.csv", index=False)
        for key, value in coloc.items():
            if key.endswith("_null"):
                continue
            summary[f"coloc_{key}"] = value

    if config.save_defect_maps and not defects.empty:
        integer_field = None
        if (defects["charge"].abs() == 1.0).any():
            if field_type == "collagen":
                integer_field = field
            else:
                integer_field = compute_collagen_field(
                    eosin, representative_sigma,
                    config.collagen_inner_scale_px,
                    tissue_mask=tissue_mask,
                    mask_normalized=config.mask_normalized_smoothing,
                )
        maps = render_defect_maps(
            rgb, field, defects, image_output / "defect_maps", tag, config,
            integer_field=integer_field,
        )
        if not maps.empty:
            maps.to_csv(
                image_output / f"{tag}_defect_maps.csv", index=False
            )

    with (image_output / f"{tag}_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe(summary), handle, indent=2, ensure_ascii=False,
            allow_nan=False,
        )

    return summary


def _run_null(
    field_type, oriented_nuclei, eosin, tissue_mask, config, image_output, tag
) -> dict:
    if field_type == "fused":
        return {"null_note": "null model is not defined for the fused field"}

    if field_type == "nuclear":
        result = run_null_model(
            oriented_nuclei, tissue_mask, config,
            n_permutations=config.n_permutations,
            downsample=config.null_downsample,
            mode=config.null_mode, seed=config.random_seed,
            n_jobs=config.null_n_jobs,
        )
    else:
        result = run_collagen_null_model(
            eosin, tissue_mask, config,
            n_permutations=config.n_permutations,
            downsample=config.null_downsample,
            inner_scale_px=config.collagen_inner_scale_px,
            seed=config.random_seed,
            n_jobs=config.null_n_jobs,
        )

    save_null_histogram(
        result, image_output / f"{tag}_null_hist.png", title=tag
    )
    pd.DataFrame({"null_total": result["null_totals"]}).to_csv(
        image_output / f"{tag}_null_totals.csv", index=False
    )
    summary_keys = {}
    for key, value in result.items():
        if key == "null_totals":
            continue
        name = key if key.startswith("null_") else f"null_{key}"
        summary_keys[name] = value
    return summary_keys
