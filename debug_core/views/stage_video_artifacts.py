from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _list_sequence_files(directory: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def _open_video_writer(*, output_path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1.0)),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open preview video writer: {output_path}")
    return writer


def build_mask_preview_video(*, mask_dir: str | Path, output_path: str | Path, fps: float) -> Path | None:
    mask_files = _list_sequence_files(mask_dir, (".png", ".jpg", ".jpeg"))
    if not mask_files:
        return None

    first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        return None

    output = Path(output_path)
    writer = _open_video_writer(
        output_path=output,
        frame_size=(int(first_mask.shape[1]), int(first_mask.shape[0])),
        fps=float(fps),
    )
    try:
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            color_mask = cv2.applyColorMap(np.clip(mask.astype(np.uint8) * 32, 0, 255), cv2.COLORMAP_JET)
            writer.write(color_mask)
    finally:
        writer.release()
    return output


def build_tracking_preview_video(
    *,
    image_dir: str | Path,
    mask_dir: str | Path,
    output_path: str | Path,
    fps: float,
) -> Path | None:
    image_files = _list_sequence_files(image_dir, (".jpg", ".jpeg", ".png"))
    if not image_files:
        return None

    output = Path(output_path)
    first_frame = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        return None

    writer = _open_video_writer(
        output_path=output,
        frame_size=(int(first_frame.shape[1]), int(first_frame.shape[0])),
        fps=float(fps),
    )
    try:
        for image_file in image_files:
            frame = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            mask_path = Path(mask_dir) / f"{image_file.stem}.png"
            overlay = frame.copy()
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    color_mask = cv2.applyColorMap(np.clip(mask.astype(np.uint8) * 32, 0, 255), cv2.COLORMAP_JET)
                    active = mask > 0
                    overlay[active] = cv2.addWeighted(frame, 0.35, color_mask, 0.65, 0.0)[active]
            writer.write(overlay)
    finally:
        writer.release()
    return output
