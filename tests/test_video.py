"""Video input preserves encoded frame order and reports codec provenance."""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest

from lung_nematic.video import axis_bias, read_video_frames, video_metadata


def test_mp4_round_trip_preserves_count_order_and_dimensions(tmp_path):
    path = tmp_path / "NHLF_5kPa.mp4"
    frames = [np.full((32, 48, 3), value, dtype=np.uint8)
              for value in (20, 80, 140, 200)]
    imageio.mimwrite(path, frames, fps=7, codec="libx264",
                     macro_block_size=1)

    decoded = list(read_video_frames(path))
    metadata = video_metadata(path)
    assert len(decoded) == 4
    assert all(frame.shape == (32, 48, 3) for frame in decoded)
    assert np.all(np.diff([frame.mean() for frame in decoded]) > 0)
    assert metadata["size"] == (48, 32)
    assert metadata["frame_count"] == 4
    assert metadata["container_fps"] == 7.0


def test_video_stride_and_limit_apply_in_encoded_order(tmp_path):
    path = tmp_path / "NHLF_23kPa.mp4"
    frames = [np.full((16, 16, 3), value, dtype=np.uint8)
              for value in range(0, 200, 20)]
    imageio.mimwrite(path, frames, fps=5, codec="libx264",
                     macro_block_size=1)
    decoded = list(read_video_frames(path, max_frames=3, stride=2))
    assert len(decoded) == 3
    means = [frame.mean() for frame in decoded]
    assert np.all(np.diff(means) > 0)
    assert means == pytest.approx([0, 40, 80], abs=3)


def test_axis_bias_flags_axis_locked_but_not_uniform_angles():
    uniform = np.linspace(0, np.pi, 10000, endpoint=False).reshape(100, 100)
    locked = np.zeros((100, 100))
    assert axis_bias(uniform)["flagged"] is False
    assert axis_bias(locked)["flagged"] is True
