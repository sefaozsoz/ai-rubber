"""Smoke tests for frame extraction (no GPU needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_utils import MAX_PROCESS_WIDTH, extract_frames, read_first_frame_rgb


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Write a tiny 10-frame synthetic video."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_frames_counts_and_size(sample_video: Path, tmp_path: Path):
    info = extract_frames(sample_video, tmp_path / "frames")
    assert info.frame_count == 10
    assert info.width == 320 and info.height == 240
    assert len(list(info.frames_dir.glob("*.jpg"))) == 10


def test_wide_video_is_downscaled(tmp_path: Path):
    path = tmp_path / "wide.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (1920, 1080))
    for _ in range(3):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()

    info = extract_frames(path, tmp_path / "frames")
    assert info.width <= MAX_PROCESS_WIDTH
    assert info.width % 2 == 0 and info.height % 2 == 0


def test_first_frame_is_rgb(sample_video: Path, tmp_path: Path):
    info = extract_frames(sample_video, tmp_path / "frames")
    frame = read_first_frame_rgb(info)
    assert frame.shape == (240, 320, 3)


def test_missing_video_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        extract_frames(tmp_path / "yok.mp4", tmp_path / "frames")
