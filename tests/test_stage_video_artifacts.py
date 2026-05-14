from pathlib import Path

import cv2
import numpy as np

from debug_core.views.stage_video_artifacts import (
    build_mask_preview_video,
    build_tracking_preview_video,
)


def _write_frame(path: Path, color_bgr: tuple[int, int, int]) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    cv2.imwrite(str(path), frame)


def _write_mask(path: Path, value: int) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = value
    cv2.imwrite(str(path), mask)


def test_build_mask_preview_video_writes_mp4_from_mask_sequence(tmp_path: Path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _write_mask(mask_dir / "00000000.png", 1)
    _write_mask(mask_dir / "00000001.png", 1)

    output_path = tmp_path / "mask_preview.mp4"
    written = build_mask_preview_video(mask_dir=mask_dir, output_path=output_path, fps=2.0)

    assert written == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_build_tracking_preview_video_writes_overlay_mp4(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    _write_frame(image_dir / "00000000.jpg", (0, 0, 255))
    _write_frame(image_dir / "00000001.jpg", (0, 255, 0))
    _write_mask(mask_dir / "00000000.png", 1)
    _write_mask(mask_dir / "00000001.png", 1)

    output_path = tmp_path / "tracking_preview.mp4"
    written = build_tracking_preview_video(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_path=output_path,
        fps=2.0,
    )

    assert written == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
