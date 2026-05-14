from pathlib import Path

from debug_core.config import load_debug_config
from debug_core.session import create_debug_session
from debug_core.stages import run_refine_4d as run_refine_4d_stage_module


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
                "wan_export: {}",
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
    images_src = tmp_path / "tracking_images"
    masks_src = tmp_path / "tracking_masks"
    images_src.mkdir()
    masks_src.mkdir()
    (images_src / "00000000.jpg").write_bytes(b"image-bytes")
    (masks_src / "00000000.png").write_bytes(b"mask-bytes")
    return {
        "image_dir": str(images_src),
        "working_mask_dir": str(masks_src),
        "refined_mask_dir": str(masks_src),
        "selection_used": {"track_id": 1, "start_frame_idx": 0, "bbox_xyxy": [1, 1, 4, 4]},
    }


def test_run_body4d_stage_marks_stub_result_when_real_runtime_is_disabled(tmp_path: Path):
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

    result = run_refine_4d_stage_module.run_body4d_stage(
        runtime_session=runtime_session,
        tracking_result=_build_tracking_result(tmp_path),
        debug_config=debug_config,
    )

    assert result["real_runtime_used"] is False
    assert result["stub_reason"] == "real_runtime_disabled"
    assert Path(result["rendered_video_path"]).is_file()


def test_run_body4d_stage_prefers_real_runtime_branch_when_upstream_is_available(tmp_path: Path, monkeypatch):
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

    def _fake_real_body4d_runner(*, tracking_result, run_dir, frame_record_path, debug_config):
        Path(frame_record_path).write_bytes(b"frame-records")
        rendered_frames_dir = Path(run_dir) / "rendered_frames"
        mesh_dir = Path(run_dir) / "mesh_4d_individual"
        focal_dir = Path(run_dir) / "focal_4d_individual"
        for path in (rendered_frames_dir, mesh_dir, focal_dir):
            path.mkdir(parents=True, exist_ok=True)
        rendered_video_path = Path(run_dir) / "4d.mp4"
        rendered_video_path.write_bytes(b"video")
        return {
            "run_dir": str(run_dir),
            "input_tracking_run": dict(tracking_result),
            "rendered_video_path": str(rendered_video_path),
            "rendered_frames_dir": str(rendered_frames_dir),
            "mesh_dir": str(mesh_dir),
            "focal_dir": str(focal_dir),
            "recorded_frame_outputs_path": str(frame_record_path),
            "real_runtime_used": True,
        }

    monkeypatch.setattr(run_refine_4d_stage_module, "_run_real_body4d_stage", _fake_real_body4d_runner)

    result = run_refine_4d_stage_module.run_body4d_stage(
        runtime_session=runtime_session,
        tracking_result=tracking_result,
        debug_config=debug_config,
    )

    assert result["real_runtime_used"] is True
    assert Path(result["recorded_frame_outputs_path"]).is_file()
    assert Path(result["rendered_video_path"]).is_file()
