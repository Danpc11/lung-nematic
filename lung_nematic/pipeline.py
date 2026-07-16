from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .defects import detect_multiscale_defects
from .io_utils import read_rgb
from .metrics import summarize_image
from .preprocessing import make_tissue_mask
from .segmentation import segment_nuclei, select_oriented_nuclei
from .visualization import save_diagnostic_panel, save_overlay


def _safe_identifier(value: object) -> str:
    text = str(value)
    for character in ("/", "\\\\", ":", " "):
        text = text.replace(character, "_")
    return text


def analyze_image(
    image_path: str | Path,
    metadata: dict,
    output_root: str | Path,
    config: AnalysisConfig,
) -> dict:
    """
    Run the full analysis for one image and save all outputs.
    """
    config.validate()

    image_path = Path(image_path)
    output_root = Path(output_root)
    rgb = read_rgb(image_path)
    tissue_mask, hed = make_tissue_mask(rgb)

    labels, nuclei = segment_nuclei(
        tissue_mask,
        hed,
        config,
    )
    oriented_nuclei = select_oriented_nuclei(
        nuclei,
        config,
    )

    defects, fields, raw_detections = (
        detect_multiscale_defects(
            oriented_nuclei,
            tissue_mask,
            config,
        )
    )

    representative_sigma = float(
        config.sigmas_px[len(config.sigmas_px) // 2]
    )
    field = fields[representative_sigma]

    safe_id = _safe_identifier(metadata["image_id"])
    image_output = output_root / safe_id
    image_output.mkdir(parents=True, exist_ok=True)

    nuclei_export = nuclei.copy()
    nuclei_export.insert(0, "image_id", metadata["image_id"])
    nuclei_export.insert(1, "group", metadata["group"])
    nuclei_export.to_csv(
        image_output / f"{safe_id}_nuclei.csv",
        index=False,
    )

    defects_export = defects.copy()
    defects_export.insert(0, "image_id", metadata["image_id"])
    defects_export.insert(1, "group", metadata["group"])
    defects_export.to_csv(
        image_output / f"{safe_id}_defects.csv",
        index=False,
    )

    if not raw_detections.empty:
        raw_detections.to_csv(
            image_output / f"{safe_id}_raw_defect_detections.csv",
            index=False,
        )

    overlay_path = (
        image_output / f"{safe_id}_nematic_overlay.png"
    )
    save_overlay(
        rgb,
        tissue_mask,
        field,
        defects,
        overlay_path,
        config,
        title=f"{metadata['image_id']} | {metadata['group']}",
    )

    if config.save_diagnostic_panel:
        save_diagnostic_panel(
            rgb,
            tissue_mask,
            labels,
            field,
            defects,
            image_output / f"{safe_id}_diagnostic_panel.png",
            title=f"{metadata['image_id']} | {metadata['group']}",
        )

    if config.save_intermediate_arrays:
        np.savez_compressed(
            image_output / f"{safe_id}_field_arrays.npz",
            tissue_mask=tissue_mask,
            labels=labels,
            density=field["density"],
            order=field["order"],
            theta=field["theta"],
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
    summary["overlay_path"] = str(overlay_path)

    with (
        image_output / f"{safe_id}_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    return summary
