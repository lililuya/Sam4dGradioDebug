import json
from pathlib import Path

import cv2
import numpy as np

from debug_core.stages.load_clip_package import load_clip_package


def test_load_clip_package_reads_required_metadata(tmp_path: Path):
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    writer = cv2.VideoWriter(str(clip_dir / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        writer.release()
    (clip_dir / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": "sampleuuid_face01_seg001",
                "sample_uuid": "sampleuuid",
                "source_path": "/dataset/source.mp4",
                "fps": 2.0,
                "frame_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        json.dumps(
            {
                "records": [
                    {"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "landmarks": [[1, 1], [2, 2]], "score": 0.9},
                    {"frame_index_in_clip": 1, "bbox_xyxy": [1, 1, 4, 4], "landmarks": [[1, 1], [2, 2]], "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )

    context = load_clip_package(clip_dir)

    assert context["clip_id"] == "sampleuuid_face01_seg001"
    assert context["sample_uuid"] == "sampleuuid"
    assert context["frame_count"] == 2
    assert len(context["clip_track_records"]) == 2


def test_load_clip_package_rejects_missing_track_json(tmp_path: Path):
    clip_dir = tmp_path / "broken_clip"
    clip_dir.mkdir()
    (clip_dir / "clip.mp4").write_bytes(b"")
    (clip_dir / "meta.json").write_text("{}", encoding="utf-8")

    try:
        load_clip_package(clip_dir)
    except FileNotFoundError as exc:
        assert "track.json" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
