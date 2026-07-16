from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import AnalysisConfig
from .io_utils import (
    discover_images,
    load_metadata,
    resolve_metadata,
)
from .pipeline import analyze_image


def analyze_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig,
    metadata_csv: str | Path | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze every supported image under input_dir recursively.

    Returns
    -------
    summary:
        One row per successfully analyzed image.
    errors:
        One row per failed image.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_csv)
    image_paths = discover_images(input_dir)

    summaries: list[dict] = []
    errors: list[dict] = []

    for image_path in tqdm(
        image_paths,
        desc="Analyzing histology images",
    ):
        try:
            image_metadata = resolve_metadata(
                image_path,
                metadata,
                config.default_microns_per_pixel,
            )
            summary = analyze_image(
                image_path,
                image_metadata,
                output_dir,
                config,
            )
            summaries.append(summary)
        except Exception as error:
            errors.append(
                {
                    "filename": image_path.name,
                    "path": str(image_path),
                    "error": repr(error),
                }
            )
            if not continue_on_error:
                raise

    summary_df = pd.DataFrame(summaries)
    errors_df = pd.DataFrame(errors)

    summary_df.to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
    )
    if not errors_df.empty:
        errors_df.to_csv(
            output_dir / "processing_errors.csv",
            index=False,
        )

    return summary_df, errors_df


def summarize_by_group(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    metric_columns = [
        "global_nematic_order_S",
        "local_S_median",
        "n_defects_total",
        "n_plus_half",
        "n_minus_half",
        "net_topological_charge",
        "defect_density_mm2",
        "mean_defect_confidence",
    ]
    available = [
        column
        for column in metric_columns
        if column in summary_df.columns
    ]

    return (
        summary_df
        .groupby("group")[available]
        .agg(["count", "mean", "median", "std"])
    )
