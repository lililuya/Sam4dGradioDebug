from pathlib import Path

from debug_core.config import load_debug_config
from debug_core.session import create_debug_session
from debug_core.stages import run_wan_export as run_wan_export_stage_module


def _write_runtime_debug_config(
    *,
    path: Path,
    repo_root: Path,
    refined_config_path: Path,
    enable_real_runtime: bool,
) -> Path:
    path.write_text(
        "\n".join(
            [
                "debug_mode:",
                "  preserve_all_intermediate: true",
                "runtime: {}",
                "wan_export:",
                "  enable: true",
                "  output_dir: outputs_export",
                "  metadata_output_dir: outputs_export/progress_json",
                "debug: {}",
                "refine: {}",
                "reprompt: {}",
                "upstream:",
                f"  repo_root: {repo_root.as_posix()}",
                f"  refined_config_path: {refined_config_path.as_posix()}",
                f"  enable_real_runtime: {'true' if enable_real_runtime else 'false'}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _build_tracking_result(tmp_path: Path) -> dict:
    images_dir = tmp_path / "tracking_images"
    masks_dir = tmp_path / "tracking_masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    (images_dir / "00000000.jpg").write_bytes(b"image")
    (masks_dir / "00000000.png").write_bytes(b"mask")
    return {
        "image_dir": str(images_dir),
        "working_mask_dir": str(masks_dir),
        "refined_mask_dir": str(masks_dir),
        "clip_dir": str(tmp_path / "clip_pkg"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "input_video": str(tmp_path / "clip_pkg" / "clip.mp4"),
        "selection_used": {"track_id": 1, "start_frame_idx": 0, "bbox_xyxy": [1, 1, 4, 4]},
        "obj_ids": [1],
        "fps": 2.0,
        "frame_count": 1,
        "real_runtime_used": False,
    }


def _build_body4d_result(tmp_path: Path) -> dict:
    recorded_path = tmp_path / "body4d_frame_records.pkl"
    recorded_path.write_bytes(b"records")
    rendered_video_path = tmp_path / "4d.mp4"
    rendered_video_path.write_bytes(b"video")
    return {
        "run_dir": str(tmp_path / "body4d_run"),
        "recorded_frame_outputs_path": str(recorded_path),
        "rendered_video_path": str(rendered_video_path),
        "real_runtime_used": False,
    }


def test_run_wan_export_stage_marks_stub_result_when_real_runtime_is_disabled(tmp_path: Path):
    config_path = _write_runtime_debug_config(
        path=tmp_path / "debug.yaml",
        repo_root=tmp_path / "missing_repo",
        refined_config_path=tmp_path / "missing_repo" / "configs" / "body4d_refined.yaml",
        enable_real_runtime=False,
    )
    debug_config = load_debug_config(config_path)
    runtime_session = create_debug_session(
        clip_id="sampleuuid_face01_seg001",
        sample_uuid="sampleuuid",
        source_path="/dataset/source.mp4",
        output_root=tmp_path / "outputs",
    )

    result = run_wan_export_stage_module.run_wan_export_stage(
        runtime_session=runtime_session,
        tracking_result=_build_tracking_result(tmp_path),
        body4d_result=_build_body4d_result(tmp_path),
        debug_config=debug_config,
    )

    assert result["real_runtime_used"] is False
    assert result["stub_reason"] == "real_runtime_disabled"
    assert Path(result["target_mp4"]).is_file()


def test_run_wan_export_stage_prefers_real_runtime_branch_when_upstream_is_available(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "sam-body4d-master"
    refined_config_path = repo_root / "configs" / "body4d_refined.yaml"
    refined_config_path.parent.mkdir(parents=True)
    refined_config_path.write_text("runtime: {}\nwan_export: {}\n", encoding="utf-8")
    config_path = _write_runtime_debug_config(
        path=tmp_path / "debug.yaml",
        repo_root=repo_root,
        refined_config_path=refined_config_path,
        enable_real_runtime=True,
    )
    debug_config = load_debug_config(config_path)
    runtime_session = create_debug_session(
        clip_id="sampleuuid_face01_seg001",
        sample_uuid="sampleuuid",
        source_path="/dataset/source.mp4",
        output_root=tmp_path / "outputs",
    )
    tracking_result = _build_tracking_result(tmp_path)
    body4d_result = _build_body4d_result(tmp_path)

    def _fake_real_wan_export_runner(*, tracking_result, body4d_result, run_dir, debug_config):
        target_dir = Path(run_dir) / "sampleuuid_face01_seg001_target1"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_names = {
            "target_mp4": "target.mp4",
            "src_pose_mp4": "src_pose.mp4",
            "src_face_mp4": "src_face.mp4",
            "src_bg_mp4": "src_bg.mp4",
            "src_mask_mp4": "src_mask.mp4",
            "src_mask_detail_mp4": "src_mask_detail.mp4",
            "src_ref_png": "src_ref.png",
            "meta_json": "meta.json",
            "pose_meta_sequence_json": "pose_meta_sequence.json",
            "smpl_sequence_json": "smpl_sequence.json",
        }
        result = {
            "run_dir": str(run_dir),
            "tracking_result": dict(tracking_result),
            "body4d_result": dict(body4d_result),
            "target_dir": str(target_dir),
            "real_runtime_used": True,
        }
        for key, filename in output_names.items():
            path = target_dir / filename
            path.write_bytes(b"output")
            result[key] = str(path)
        return result

    monkeypatch.setattr(run_wan_export_stage_module, "_run_real_wan_export_stage", _fake_real_wan_export_runner)

    result = run_wan_export_stage_module.run_wan_export_stage(
        runtime_session=runtime_session,
        tracking_result=tracking_result,
        body4d_result=body4d_result,
        debug_config=debug_config,
    )

    assert result["real_runtime_used"] is True
    assert Path(result["target_mp4"]).is_file()
    assert Path(result["pose_meta_sequence_json"]).is_file()
