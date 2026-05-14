from __future__ import annotations

import json
from pathlib import Path

import cv2


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _count_video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open clip video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"Unable to determine frame count for clip video: {path}")
    return frame_count


def load_clip_package(path: str | Path) -> dict:
    clip_dir = Path(path).resolve()
    clip_video = clip_dir / "clip.mp4"
    meta_path = clip_dir / "meta.json"
    track_path = clip_dir / "track.json"
    for required_path in (clip_video, meta_path, track_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required clip package file: {required_path.name}")

    meta = _read_json(meta_path)
    track = _read_json(track_path)
    records = list(track.get("records") or [])
    if not records:
        raise ValueError("Clip package track.json has no records")

    frame_count = int(meta.get("frame_count") or 0) or _count_video_frames(clip_video)
    fps = float(meta.get("fps") or 0.0)
    return {
        "clip_dir": str(clip_dir),
        "clip_id": str(meta.get("clip_id") or clip_dir.name),
        "sample_uuid": str(meta.get("sample_uuid") or ""),
        "source_path": str(meta.get("source_path") or clip_video),
        "clip_video_path": str(clip_video),
        "frame_count": frame_count,
        "fps": fps,
        "clip_track_records": records,
        "frame_stems": [f"{index:08d}" for index in range(frame_count)],
        "meta": meta,
        "track": track,
    }
