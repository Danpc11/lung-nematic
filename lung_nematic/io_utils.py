from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def discover_images(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)


def load_metadata(metadata_csv: str | Path | None) -> pd.DataFrame:
    if metadata_csv is None:
        return pd.DataFrame()

    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")

    metadata = pd.read_csv(path)
    has_filename = "filename" in metadata.columns
    has_relpath = "relative_path" in metadata.columns
    if not (has_filename or has_relpath):
        raise ValueError(
            "metadata.csv must include a 'filename' or 'relative_path' column."
        )
    if has_filename:
        metadata["filename"] = metadata["filename"].astype("string")
    if has_relpath:
        metadata["relative_path"] = metadata["relative_path"].astype("string")
    return metadata


def resolve_metadata(
    image_path: str | Path,
    metadata: pd.DataFrame,
    default_microns_per_pixel: float | None = None,
    root: str | Path | None = None,
) -> dict:
    path = Path(image_path)
    relative = None
    if root is not None:
        try:
            relative = path.relative_to(Path(root)).as_posix()
        except ValueError:
            relative = None

    resolved = {
        "filename": path.name,
        "image_id": path.stem,
        "group": path.parent.name,
        "relative_path": relative,
        "microns_per_pixel": default_microns_per_pixel,
    }

    if not metadata.empty:
        matches = metadata.iloc[0:0]
        if "relative_path" in metadata.columns and relative is not None:
            matches = metadata.loc[metadata["relative_path"] == relative]
        # A CSV may offer both lookup columns while leaving relative_path blank
        # for some rows. In that case, or when no relative path matches, fall
        # back to the documented filename lookup instead of silently discarding
        # the row's calibration and grouping metadata.
        if matches.empty and "filename" in metadata.columns:
            matches = metadata.loc[metadata["filename"] == path.name]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple metadata rows match image: {relative or path.name}"
            )
        if len(matches) == 1:
            row = matches.iloc[0]
            for key in ("image_id", "group", "microns_per_pixel"):
                if key in metadata.columns and pd.notna(row.get(key)):
                    resolved[key] = row[key]

    try:
        mpp = float(resolved["microns_per_pixel"])
        resolved["microns_per_pixel"] = mpp if mpp > 0 else None
    except (TypeError, ValueError):
        resolved["microns_per_pixel"] = None

    return resolved


def read_rgb(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
