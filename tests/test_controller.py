from pathlib import Path

import cv2
import numpy as np

from debug_core.controllers.debug_pipeline_controller import DebugPipelineController


def _build_clip_context(tmp_path: Path) -> dict:
    return {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }


def _build_preview_clip_context(tmp_path: Path) -> dict:
    clip_dir = tmp_path / "clip_pkg"
    clip_dir.mkdir()
    clip_video_path = clip_dir / "clip.mp4"
    writer = cv2.VideoWriter(str(clip_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    assert writer.isOpened()
    try:
        for color in ([0, 0, 255], [0, 255, 0]):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[:, :] = color
            writer.write(frame)
    finally:
        writer.release()
    return {
        "clip_dir": str(clip_dir),
        "clip_video_path": str(clip_video_path),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }


def test_controller_loads_sample_and_stores_auto_proposal(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    session = controller.bootstrap_session(_build_clip_context(tmp_path))
    assert session["sample_context"]["clip_id"] == "sampleuuid_face01_seg001"
    assert session["auto_selection_proposal"]["bbox_xyxy"] == [1, 1, 4, 4]


def test_controller_apply_manual_selection_replaces_effective_selection(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_clip_context(tmp_path))
    selection = controller.set_manual_selection(track_id=5, start_frame_idx=1, bbox_xyxy=[3, 3, 7, 7])
    assert selection["track_id"] == 5
    assert controller.session["effective_selection"]["track_id"] == 5


def test_controller_run_tracking_creates_new_revision_and_records_paths(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_preview_clip_context(tmp_path))
    result = controller.run_tracking_stage()
    assert result["run_dir"].name == "run_v001"
    assert result["selection_used"]["bbox_xyxy"] == [1, 1, 4, 4]
    assert Path(result["debug_metrics_dir"]).is_dir()
    assert Path(result["image_dir"]).is_dir()
    assert Path(result["working_mask_dir"]).is_dir()
    assert (Path(result["image_dir"]) / "00000000.jpg").is_file()
    assert (Path(result["working_mask_dir"]) / "00000000.png").is_file()


def test_controller_run_body4d_creates_body4d_revision(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_clip_context(tmp_path))
    controller.run_tracking_stage()
    result = controller.run_body4d_stage()
    assert result["run_dir"].endswith("run_v001")
    assert result["rendered_video_path"].endswith("4d.mp4")
    assert result["recorded_frame_outputs_path"].endswith("body4d_frame_records.pkl")


def test_controller_run_wan_export_requires_upstream_results(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_clip_context(tmp_path))
    try:
        controller.run_wan_export_stage()
    except RuntimeError as exc:
        assert "body4d" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError")


def test_controller_register_preview_click_builds_pending_bbox_on_second_click(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_preview_clip_context(tmp_path))

    first = controller.register_preview_click(frame_idx=0, x=1, y=2)
    second = controller.register_preview_click(frame_idx=0, x=6, y=7)

    assert first["pending_bbox"] is None
    assert second["pending_bbox"] == [1, 2, 6, 7]


def test_controller_get_preview_returns_image_array(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    controller.bootstrap_session(_build_preview_clip_context(tmp_path))

    preview = controller.get_preview(frame_idx=0)

    assert preview.shape == (8, 8, 3)
