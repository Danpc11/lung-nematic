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
        return pd.DataFrame()

    metadata = pd.read_csv(path)
    if "filename" not in metadata.columns:
        raise ValueError("metadata.csv must include a 'filename' column.")

    metadata["filename"] = metadata["filename"].astype(str)
    return metadata


def resolve_metadata(
    image_path: str | Path,
    metadata: pd.DataFrame,
    default_microns_per_pixel: float | None = None,
) -> dict:
    path = Path(image_path)
    resolved = {
        "filename": path.name,
        "image_id": path.stem,
        "group": path.parent.name,
        "microns_per_pixel": default_microns_per_pixel,
    }

    if not metadata.empty:
        matches = metadata.loc[metadata["filename"] == path.name]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple metadata rows match filename: {path.name}"
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
