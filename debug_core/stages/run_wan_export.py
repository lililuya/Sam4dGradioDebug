from __future__ import annotations

import sys
from pathlib import Path

from debug_core.adapters.recording_frame_writer import load_recorded_frame_outputs
from debug_core.adapters.refined_runtime_adapter import resolve_upstream_runtime_paths
from debug_core.config import DebugAppConfig
from debug_core.adapters.wan_export_adapter import write_debug_export_stub


def _should_use_real_wan_export_runtime(debug_config: DebugAppConfig | None) -> bool:
    return debug_config is not None and bool(debug_config.upstream.get("enable_real_runtime", True))


def _run_stub_wan_export_stage(
    *,
    tracking_result: dict,
    body4d_result: dict,
    run_dir: Path,
    stub_reason: str,
) -> dict:
    outputs = write_debug_export_stub(Path(run_dir))
    return {
        "run_dir": str(run_dir),
        "tracking_result": dict(tracking_result),
        "body4d_result": dict(body4d_result),
        "target_dir": outputs["target_dir"],
        "target_mp4": outputs["target.mp4"],
        "src_pose_mp4": outputs["src_pose.mp4"],
        "src_face_mp4": outputs["src_face.mp4"],
        "src_bg_mp4": outputs["src_bg.mp4"],
        "src_mask_mp4": outputs["src_mask.mp4"],
        "src_mask_detail_mp4": outputs["src_mask_detail.mp4"],
        "src_ref_png": outputs["src_ref.png"],
        "meta_json": outputs["meta.json"],
        "pose_meta_sequence_json": outputs["pose_meta_sequence.json"],
        "smpl_sequence_json": outputs["smpl_sequence.json"],
        "real_runtime_used": False,
        "stub_reason": str(stub_reason),
    }


def _run_real_wan_export_stage(
    *,
    tracking_result: dict,
    body4d_result: dict,
    run_dir: Path,
    debug_config: DebugAppConfig,
) -> dict:
    upstream_paths = resolve_upstream_runtime_paths(debug_config)
    repo_root = upstream_paths["repo_root"]
    refined_config_path = upstream_paths["refined_config_path"]

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from scripts import offline_app_refined as refined_module
    from scripts.wan_sample_export import WanSampleExporter

    recorded_frame_outputs_path = str(body4d_result.get("recorded_frame_outputs_path") or "").strip()
    if not recorded_frame_outputs_path:
        raise RuntimeError("Real wan export stage requires body4d recorded_frame_outputs_path.")

    recorded_records = load_recorded_frame_outputs(recorded_frame_outputs_path)
    if not recorded_records:
        raise RuntimeError("Recorded body4d frame outputs are empty.")

    cfg = refined_module.load_refined_config(str(refined_config_path))
    app = refined_module.RefinedOfflineApp(str(refined_config_path), config=cfg)

    sample_id = str(tracking_result.get("clip_id") or tracking_result.get("sample_uuid") or Path(run_dir).name)
    source_video_path = str(tracking_result.get("input_video") or "").strip() or None
    images_dir = str(body4d_result.get("input_images_dir") or tracking_result.get("image_dir") or "").strip()
    masks_dir = str(body4d_result.get("input_masks_dir") or tracking_result.get("working_mask_dir") or "").strip()
    if not images_dir or not masks_dir:
        raise RuntimeError("Real wan export stage requires images_dir and masks_dir.")

    runtime_app = app._ensure_base_app()
    wan_writer = WanSampleExporter(
        sample_id=sample_id,
        output_dir=str(run_dir),
        images_dir=images_dir,
        masks_dir=masks_dir,
        source_video_path=source_video_path,
        config=cfg.wan_export,
        sample_uuid=str(tracking_result.get("sample_uuid") or "").strip() or None,
        clip_id=str(tracking_result.get("clip_id") or "").strip() or None,
        source_reference=str(tracking_result.get("source_path") or source_video_path or "").strip() or None,
        clip_track_records=list(tracking_result.get("clip_track_records") or []),
        mesh_faces=getattr(runtime_app.sam3_3d_body_model, "faces", None),
    )
    for record in recorded_records:
        wan_writer(
            record.get("image_path"),
            record.get("mask_output"),
            record.get("id_current"),
        )

    written_targets = wan_writer.finalize()
    if not written_targets:
        raise RuntimeError("Wan export replay did not produce any target directories.")

    target_dir = Path(written_targets[0])
    return {
        "run_dir": str(run_dir),
        "tracking_result": dict(tracking_result),
        "body4d_result": dict(body4d_result),
        "target_dir": str(target_dir),
        "target_mp4": str(target_dir / "target.mp4"),
        "src_pose_mp4": str(target_dir / "src_pose.mp4"),
        "src_face_mp4": str(target_dir / "src_face.mp4"),
        "src_bg_mp4": str(target_dir / "src_bg.mp4"),
        "src_mask_mp4": str(target_dir / "src_mask.mp4"),
        "src_mask_detail_mp4": str(target_dir / "src_mask_detail.mp4"),
        "src_ref_png": str(target_dir / "src_ref.png"),
        "meta_json": str(target_dir / "meta.json"),
        "pose_meta_sequence_json": str(target_dir / "pose_meta_sequence.json"),
        "smpl_sequence_json": str(target_dir / "smpl_sequence.json"),
        "real_runtime_used": True,
        "upstream_repo_root": str(repo_root),
        "upstream_refined_config_path": str(refined_config_path),
        "upstream_module_name": str(getattr(refined_module, "__name__", "")),
    }


def run_wan_export_stage(
    *,
    runtime_session,
    tracking_result: dict,
    body4d_result: dict,
    debug_config: DebugAppConfig | None = None,
) -> dict:
    run_dir = runtime_session.next_revision_dir("wan_export")
    if not _should_use_real_wan_export_runtime(debug_config):
        return _run_stub_wan_export_stage(
            tracking_result=tracking_result,
            body4d_result=body4d_result,
            run_dir=Path(run_dir),
            stub_reason="real_runtime_disabled",
        )

    upstream_paths = resolve_upstream_runtime_paths(debug_config)
    repo_root = upstream_paths["repo_root"]
    refined_config_path = upstream_paths["refined_config_path"]
    if not repo_root.is_dir() or not refined_config_path.is_file():
        return _run_stub_wan_export_stage(
            tracking_result=tracking_result,
            body4d_result=body4d_result,
            run_dir=Path(run_dir),
            stub_reason="upstream_runtime_unavailable",
        )

    try:
        return _run_real_wan_export_stage(
            tracking_result=tracking_result,
            body4d_result=body4d_result,
            run_dir=Path(run_dir),
            debug_config=debug_config,
        )
    except Exception as exc:
        result = _run_stub_wan_export_stage(
            tracking_result=tracking_result,
            body4d_result=body4d_result,
            run_dir=Path(run_dir),
            stub_reason=f"real_runtime_failed:{type(exc).__name__}",
        )
        result["real_runtime_error"] = str(exc)
        return result
