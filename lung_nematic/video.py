"""Read time-lapse videos without confusing playback with acquisition time.

Original TIFF frames should always be preferred. H.264 is spatially and
temporally lossy, and its block structure can add horizontal or vertical signal
to a structure-tensor director field. Container FPS is therefore returned for
provenance only: it is a playback rate and must never replace the independently
recorded acquisition interval.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_video_frames(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> Iterator[np.ndarray]:
    """Yield RGB frames in encoded order, decoding the video only once."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    reader = imageio.get_reader(str(path), format="ffmpeg")
    yielded = 0
    try:
        for index, frame in enumerate(reader):
            if index % stride:
                continue
            rgb = np.asarray(frame)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=2)
            elif rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            yield rgb
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        reader.close()


def video_metadata(path: str | Path) -> dict[str, object]:
    """Return container metadata; FPS is explicitly not acquisition timing."""
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        size = metadata.get("size") or metadata.get("source_size")
        frame_count = metadata.get("nframes")
        if frame_count in (None, float("inf")):
            try:
                frame_count = reader.count_frames()
            except (RuntimeError, NotImplementedError):
                frame_count = None
        return {
            "size": tuple(size) if size is not None else None,
            "frame_count": int(frame_count) if frame_count is not None else None,
            "codec": metadata.get("codec"),
            "container_fps": metadata.get("fps"),
        }
    finally:
        reader.close()


def axis_bias(theta: np.ndarray, mask: np.ndarray | None = None,
              tolerance_deg: float = 10.0) -> dict[str, float | bool]:
    """Flag excess directors near compression-prone image axes.

    Under a uniform nematic angle distribution, the expected fraction within
    ``tolerance_deg`` of either 0 or pi/2 is ``4*tolerance/pi``. A binomial
    five-sigma excess is intentionally conservative: spatial pixels are
    correlated, so an ordinary binomial significance test would overstate the
    evidence. This is a codec-artifact warning, not a biological hypothesis
    test.
    """
    values = np.asarray(theta, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    angles = np.mod(values[valid], np.pi)
    if not angles.size:
        return {"axis_fraction": np.nan, "uniform_expected": np.nan,
                "axis_excess": np.nan, "flagged": False}
    tolerance = np.deg2rad(tolerance_deg)
    distance_zero = np.minimum(angles, np.pi - angles)
    distance_vertical = np.abs(angles - np.pi / 2)
    fraction = float(np.mean((distance_zero <= tolerance)
                             | (distance_vertical <= tolerance)))
    expected = float(4 * tolerance / np.pi)
    # Effective sample size is capped because neighbouring field pixels share
    # the same smoothing kernel and are not independent observations.
    effective_n = min(int(angles.size), 1000)
    noise = np.sqrt(expected * (1 - expected) / effective_n)
    return {"axis_fraction": fraction, "uniform_expected": expected,
            "axis_excess": fraction - expected,
            "flagged": bool(fraction > expected + 5 * noise)}
