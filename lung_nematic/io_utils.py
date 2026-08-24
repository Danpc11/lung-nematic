from __future__ import annotations

import math
import zlib
from collections.abc import Iterable
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

# Characters that cannot appear in a directory name on at least one supported
# platform. ``/`` and ``\`` would create nesting, ``:`` is a stream separator on
# NTFS, and the rest are outright illegal there. Spaces are folded too so output
# paths stay shell-safe.
_INVALID_NAME_CHARS = set('/\\:*?"<>|') | {" "}

# Names that resolve to something other than a fresh child directory.
_RESERVED_NAMES = {"", ".", ".."}


def safe_identifier(value: object) -> str:
    """Return ``value`` as a directory-safe name, or raise.

    Every character that is illegal or path-significant on some supported
    platform is folded to ``_``. The result is then checked against the reserved
    names: ``".."`` would make ``output_root / safe_id`` resolve to the *parent*
    of the output root and scatter results outside it, while ``""`` and ``"."``
    would resolve to the output root itself and drop the per-image files loose
    among the batch-level ones. Those are refused rather than normalized,
    because there is no correct directory to silently pick instead.

    Note that this folding is lossy: distinct identifiers such as ``case/01``
    and ``case_01`` both map to ``case_01``. Callers that build a batch must
    therefore check uniqueness on the *normalized* name (see
    ``lung_nematic.batch.analyze_folder``), not on the raw ``image_id``.
    """
    raw = str(value)
    # Surrounding whitespace must go before folding, not after: folding first
    # would turn "case " into "case_" and "   " into "___", inventing distinct
    # directory names out of padding instead of collapsing it.
    text = "".join(
        "_" if character in _INVALID_NAME_CHARS or ord(character) < 32
        else character
        for character in raw.strip()
    )
    # NTFS silently strips trailing dots, so "case01." and "case01" would be the
    # same directory there but not here. Strip them ourselves so the collision is
    # visible on every platform.
    text = text.rstrip(".")
    if text in _RESERVED_NAMES:
        raise ValueError(
            f"image_id {raw!r} does not yield a usable directory name "
            "(it normalizes to a reserved path). Provide a metadata CSV with "
            "an explicit, non-empty image_id."
        )
    return text


def derived_seed(base_seed: int, *parts: object) -> int:
    """Deterministic per-item seed derived from ``base_seed`` and ``parts``.

    A single seed reused verbatim for every image makes the permutation nulls
    reproducible but *not independent*: each image is shuffled with the same
    random stream. That dependence is invisible in a per-image p-value and only
    bites when the p-values are later combined across a cohort. Deriving a seed
    per (image, field) keeps full reproducibility while decorrelating the
    streams. CRC32 is used rather than ``hash()`` because the latter is salted
    per process and would not be reproducible across runs.
    """
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return (int(base_seed) * 1_000_003 + zlib.crc32(key)) % (2**31 - 1)


def discover_images(
    input_dir: str | Path,
    exclude_dirs: Iterable[str | Path] = (),
) -> list[Path]:
    """Recursively collect supported images under ``input_dir``.

    ``exclude_dirs`` removes entire subtrees from the search. This matters
    because the pipeline writes PNG overlays, diagnostic panels and defect maps,
    all of which have supported extensions: if the output tree lives inside the
    input tree, a second run would pick up the first run's renderings and
    analyse them as histology. They do not fail - the annotations are burned
    into the raster, so tissue segmentation succeeds - it simply produces
    plausible numbers from synthetic images.
    """
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    excluded = [Path(directory).resolve() for directory in exclude_dirs]

    images = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(directory) for directory in excluded):
            continue
        images.append(path)
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


def json_safe(obj):
    """Recursively convert to strict-JSON types (non-finite floats -> null).

    Shared so every module that serialises results - the pipeline, the adaptive
    notebook cell, anything writing a JSON summary - produces standards-compliant
    JSON. NaN and inf become null, and numpy scalars become their Python
    equivalents, so ``json.dumps(json_safe(x), allow_nan=False)`` never raises.
    """
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj
