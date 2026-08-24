from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import AnalysisConfig
from .io_utils import (
    discover_images,
    load_metadata,
    resolve_metadata,
    safe_identifier,
)
from .pipeline import analyze_image


def _check_identifier_collisions(identifiers: list[str]) -> None:
    """Reject identifiers that collide once normalized to directory names.

    The raw ``image_id`` is not what ends up on disk: ``safe_identifier`` folds
    path-significant characters to ``_`` first. Checking the raw values would
    therefore accept ``case/01`` and ``case_01`` as distinct and then let both
    write into ``<output>/case_01``, where the second silently overwrites the
    first. Uniqueness has to be enforced on the same string the filesystem sees.

    ``safe_identifier`` also raises for reserved names ("", ".", ".."), so this
    is where an ``image_id`` that would escape the output root is caught -
    before any image is processed rather than midway through a batch.
    """
    normalized = [safe_identifier(identifier) for identifier in identifiers]
    collisions = {
        name: sorted(
            {
                raw
                for raw, norm in zip(identifiers, normalized)
                if norm == name
            }
        )
        for name in set(normalized)
        if normalized.count(name) > 1
    }
    if collisions:
        raise ValueError(
            "image_id values must be unique after normalization to directory "
            f"names; colliding output names: {collisions}. Note that distinct "
            "ids such as 'case/01' and 'case_01' both normalize to 'case_01'. "
            "Provide a metadata CSV with unique image_id (or relative_path) "
            "entries."
        )


def _merge_field_tables(output_dir: Path, stem: str) -> pd.DataFrame:
    """Concatenate every per-field ``<stem>_<field>.csv`` into ``<stem>.csv``.

    Batch-level reports used to be written to a single fixed filename, so
    running the CLI twice over one output directory with different ``--field``
    values left per-image directories holding both fields while the batch
    summary described only the last one, with no marker of the overwrite. Each
    run now owns a per-field file and the combined view is rebuilt from whatever
    per-field files exist, which makes repeated runs additive instead of
    destructive.
    """
    frames = [
        pd.read_csv(path)
        for path in sorted(output_dir.glob(f"{stem}_*.csv"))
    ]
    frames = [frame for frame in frames if not frame.empty]
    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    combined_path = output_dir / f"{stem}.csv"
    if combined.empty:
        if combined_path.exists():
            combined_path.unlink()
    else:
        combined.to_csv(combined_path, index=False)
    return combined


def analyze_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig,
    metadata_csv: str | Path | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze every supported image under input_dir recursively.

    ``output_dir`` may not be ``input_dir`` and, if it is nested inside it, is
    excluded from image discovery: the pipeline writes PNG overlays, diagnostic
    panels and defect maps, and those would otherwise be re-analysed as
    histology on a second run.

    Returns
    -------
    summary:
        One row per successfully analyzed image (this run's field only).
    errors:
        One row per failed image (this run's field only).
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir == input_dir:
        raise ValueError(
            "output_dir must not be the same directory as input_dir: the "
            "overlays and diagnostic panels written there would be discovered "
            "as input images on the next run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_csv)
    image_paths = discover_images(input_dir, exclude_dirs=[output_dir])

    # Resolve metadata up front and require unique image_id values, so two
    # images (e.g. control/sample01.tif and fibrosis/sample01.tif) cannot
    # silently overwrite each other's results.
    resolved = [
        resolve_metadata(
            image_path,
            metadata,
            config.default_microns_per_pixel,
            root=input_dir,
        )
        for image_path in image_paths
    ]
    _check_identifier_collisions(
        [str(item["image_id"]) for item in resolved]
    )

    summaries: list[dict] = []
    errors: list[dict] = []

    for image_path, image_metadata in tqdm(
        list(zip(image_paths, resolved)),
        desc="Analyzing histology images",
    ):
        try:
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
                    "field_type": config.field_type,
                    "error": repr(error),
                }
            )
            if not continue_on_error:
                raise

    summary_df = pd.DataFrame(summaries)
    errors_df = pd.DataFrame(errors)

    # Per-field files are authoritative; the unsuffixed file is a rebuilt view.
    field = config.field_type
    summary_path = output_dir / f"summary_metrics_{field}.csv"
    summary_df.to_csv(summary_path, index=False)

    errors_path = output_dir / f"processing_errors_{field}.csv"
    if not errors_df.empty:
        errors_df.to_csv(errors_path, index=False)
    elif errors_path.exists():
        # Do not leave failures from an earlier run beside a clean new summary.
        errors_path.unlink()

    _merge_field_tables(output_dir, "summary_metrics")
    _merge_field_tables(output_dir, "processing_errors")

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
        # The integer layer is per-image but was not aggregated, so +/-1
        # defects were invisible in the group summary even when detected.
        "n_plus_one",
        "n_minus_one",
        "net_topological_charge",
        "defect_density_mm2",
        "defect_density_integer_mm2",
        "defect_density_all_mm2",
        "mean_defect_confidence",
    ]
    available = [
        column
        for column in metric_columns
        if column in summary_df.columns
    ]

    # Aggregating across orientation fields would average nuclear and collagen
    # measurements into one meaningless row, so field_type joins the key
    # whenever the caller supplies it.
    keys = [
        column
        for column in ("field_type", "group")
        if column in summary_df.columns
    ]
    if not keys:
        raise KeyError(
            "summary frame must contain a 'group' column to aggregate."
        )

    key = keys[0] if len(keys) == 1 else keys
    return summary_df.groupby(key)[available].agg(
        ["count", "mean", "median", "std"]
    )
