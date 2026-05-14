from pathlib import Path

import cv2
import numpy as np

from app import (
    _apply_clicked_bbox_for_ui,
    _apply_manual_selection_for_ui,
    _run_body4d_for_ui,
    _run_tracking_for_ui,
    _handle_preview_click_for_ui,
    _load_sample_bundle_for_ui,
    _load_sample_for_ui,
    _preview_frame_for_ui,
    build_app,
)


def _write_ui_debug_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "debug_mode:",
                "  preserve_all_intermediate: true",
                "runtime: {}",
                "wan_export: {}",
                "debug: {}",
                "refine: {}",
                "reprompt: {}",
                "upstream:",
                "  repo_root: ../../sam-body4d-master",
                "  refined_config_path: ../../sam-body4d-master/configs/body4d_refined.yaml",
                "  enable_real_runtime: false",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_ui_test_clip(clip_dir: Path) -> None:
    writer = cv2.VideoWriter(str(clip_dir / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    assert writer.isOpened()
    try:
        for color in ([0, 0, 255], [0, 255, 0]):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[:, :] = color
            writer.write(frame)
    finally:
        writer.release()


def test_build_app_returns_gradio_blocks():
    demo = build_app()
    assert demo is not None
    assert demo.__class__.__name__ == "Blocks"


def test_load_sample_for_ui_returns_clip_metadata(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    sample_info, proposal = _load_sample_for_ui(str(clip_dir), str(config_path), str(tmp_path))

    assert sample_info["clip_id"] == "sampleuuid_face01_seg001"
    assert proposal["bbox_xyxy"] == [1, 1, 4, 4]


def test_load_sample_bundle_for_ui_returns_preview_and_selection_fields(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    outputs = _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))

    sample_info, proposal, _, preview, track_id, start_frame_idx, x1, y1, x2, y2, status = outputs
    assert sample_info["clip_id"] == "sampleuuid_face01_seg001"
    assert proposal["bbox_xyxy"] == [1, 1, 4, 4]
    assert preview.shape == (8, 8, 3)
    assert track_id == 1
    assert start_frame_idx == 0
    assert [x1, y1, x2, y2] == [1, 1, 4, 4]
    assert "auto proposal" in status.lower()


def test_apply_manual_selection_for_ui_returns_selection_and_preview(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    selection, preview, status = _apply_manual_selection_for_ui(str(tmp_path), 3, 1, 2, 2, 6, 6)

    assert selection["track_id"] == 3
    assert selection["bbox_xyxy"] == [2, 2, 6, 6]
    assert preview.shape == (8, 8, 3)
    assert "manual" in status.lower()


def test_preview_frame_for_ui_updates_image_for_selected_frame(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    preview, status = _preview_frame_for_ui(str(tmp_path), 1)

    assert preview.shape == (8, 8, 3)
    assert "frame 1" in status.lower()


def test_handle_preview_click_for_ui_returns_pending_bbox_after_second_click(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    _, x1, y1, x2, y2, status_first = _handle_preview_click_for_ui(str(tmp_path), 0, 1, 2)
    _, x1b, y1b, x2b, y2b, status_second = _handle_preview_click_for_ui(str(tmp_path), 0, 6, 7)

    assert [x1, y1, x2, y2] == [0, 0, 0, 0]
    assert "first corner" in status_first.lower()
    assert [x1b, y1b, x2b, y2b] == [1, 2, 6, 7]
    assert "pending bbox" in status_second.lower()


def test_apply_clicked_bbox_for_ui_promotes_pending_bbox_to_manual_selection(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    _handle_preview_click_for_ui(str(tmp_path), 0, 1, 2)
    _handle_preview_click_for_ui(str(tmp_path), 0, 6, 7)
    selection, preview, status = _apply_clicked_bbox_for_ui(str(tmp_path), 4, 0)

    assert selection["track_id"] == 4
    assert selection["bbox_xyxy"] == [1, 2, 6, 7]
    assert preview.shape == (8, 8, 3)
    assert "clicked bbox" in status.lower()


def test_run_tracking_for_ui_returns_previewable_video_paths(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    result = _run_tracking_for_ui(str(tmp_path))

    assert Path(result["effective_mask_video_path"]).is_file()
    assert Path(result["tracking_preview_video_path"]).is_file()


def test_run_body4d_for_ui_returns_rendered_video_path(tmp_path: Path):
    config_path = _write_ui_debug_config(tmp_path / "debug.yaml")
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    _write_ui_test_clip(clip_dir)
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":2}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    _load_sample_bundle_for_ui(str(clip_dir), str(config_path), str(tmp_path))
    _run_tracking_for_ui(str(tmp_path))
    result = _run_body4d_for_ui(str(tmp_path))

    assert Path(result["rendered_video_path"]).is_file()


def test_build_app_exposes_stage_control_components():
    demo = build_app()
    keys = set(demo.debug_components.keys())
    assert "run_tracking_button" in keys
    assert "run_body4d_button" in keys
    assert "run_wan_export_button" in keys
    assert "frame_slider" in keys
    assert "apply_clicked_bbox_button" in keys
    assert "tracking_mask_video" in keys
    assert "tracking_preview_video" in keys
    assert "body4d_video" in keys
