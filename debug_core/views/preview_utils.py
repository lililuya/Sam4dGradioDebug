from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalize_path_text(path: str | Path | None) -> str:
    return "" if path in {None, ""} else str(Path(path))


def read_video_frame_rgb(video_path: str | Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for preview: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Unable to read frame {frame_index} from video: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _draw_bbox(frame_rgb: np.ndarray, bbox_xyxy: list[int] | None, color: tuple[int, int, int]) -> None:
    if not bbox_xyxy or len(bbox_xyxy) < 4:
        return
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy[:4]]
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, thickness=1)


def build_preview_frame(
    *,
    video_path: str | Path,
    frame_index: int,
    auto_bbox: list[int] | None = None,
    manual_bbox: list[int] | None = None,
    pending_bbox: list[int] | None = None,
) -> np.ndarray:
    frame_rgb = read_video_frame_rgb(video_path, frame_index).copy()
    _draw_bbox(frame_rgb, auto_bbox, (255, 215, 0))
    _draw_bbox(frame_rgb, manual_bbox, (0, 255, 255))
    _draw_bbox(frame_rgb, pending_bbox, (255, 0, 255))
    return frame_rgb
