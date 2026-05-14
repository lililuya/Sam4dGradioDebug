from pathlib import Path

import cv2
import numpy as np

from debug_core.views.preview_utils import build_preview_frame, read_video_frame_rgb


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    assert writer.isOpened()
    try:
        frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
        frame0[:, :] = [0, 0, 255]
        frame1 = np.zeros((8, 8, 3), dtype=np.uint8)
        frame1[:, :] = [0, 255, 0]
        writer.write(frame0)
        writer.write(frame1)
    finally:
        writer.release()


def test_read_video_frame_rgb_returns_expected_shape(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_tiny_video(video_path)

    frame = read_video_frame_rgb(video_path, 0)

    assert frame.shape == (8, 8, 3)
    assert frame.dtype == np.uint8


def test_build_preview_frame_draws_bbox_overlays(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_tiny_video(video_path)

    preview = build_preview_frame(
        video_path=video_path,
        frame_index=0,
        auto_bbox=[1, 1, 4, 4],
        manual_bbox=[2, 2, 6, 6],
        pending_bbox=[0, 0, 7, 7],
    )

    assert preview.shape == (8, 8, 3)
    assert np.any(preview[0, 0] != preview[3, 3])
