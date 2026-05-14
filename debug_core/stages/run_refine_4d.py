from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

from debug_core.adapters.body4d_runtime_adapter import (
    copy_tracking_inputs_to_body4d_run,
    promote_rendered_video_to_standard_path,
)
from debug_core.adapters.recording_frame_writer import RecordedFrameWriter
from debug_core.config import DebugAppConfig
from debug_core.adapters.refined_runtime_adapter import (
    build_compat_sample_summaries,
    resolve_upstream_runtime_paths,
)


def _ensure_body4d_stub_outputs(*, run_dir: Path) -> dict[str, str]:
    rendered_video_path = run_dir / "4d_stub.mp4"
    rendered_frames_dir = run_dir / "rendered_frames"
    mesh_dir = run_dir / "mesh_4d_individual"
    focal_dir = run_dir / "focal_4d_individual"
    for path in (rendered_frames_dir, mesh_dir, focal_dir):
        path.mkdir(parents=True, exist_ok=True)
    rendered_video_path.write_bytes(b"")
    standardized_video_path = promote_rendered_video_to_standard_path(
        run_dir=run_dir,
        rendered_video_path=rendered_video_path,
    )
    return {
        "rendered_video_path": str(standardized_video_path),
        "rendered_frames_dir": str(rendered_frames_dir),
        "mesh_dir": str(mesh_dir),
        "focal_dir": str(focal_dir),
    }


def _should_use_real_body4d_runtime(debug_config: DebugAppConfig | None) -> bool:
    return debug_config is not None and bool(debug_config.upstream.get("enable_real_runtime", True))


def _run_stub_body4d_stage(
    *,
    tracking_result: dict,
    run_dir: Path,
    frame_record_path: Path,
    stub_reason: str,
) -> dict:
    copied_inputs = copy_tracking_inputs_to_body4d_run(
        tracking_result=tracking_result,
        run_dir=run_dir,
    )
    writer = RecordedFrameWriter(output_path=frame_record_path)
    finalized_outputs = writer.finalize()
    stub_outputs = _ensure_body4d_stub_outputs(run_dir=run_dir)
    return {
        "run_dir": str(run_dir),
        "input_tracking_run": dict(tracking_result),
        "input_images_dir": str(copied_inputs["images_dir"]),
        "input_masks_dir": str(copied_inputs["masks_dir"]),
        "rendered_video_path": stub_outputs["rendered_video_path"],
        "rendered_frames_dir": stub_outputs["rendered_frames_dir"],
        "mesh_dir": stub_outputs["mesh_dir"],
        "focal_dir": stub_outputs["focal_dir"],
        "recorded_frame_outputs_path": str(finalized_outputs[0]),
        "real_runtime_used": False,
        "stub_reason": str(stub_reason),
    }


def _run_real_body4d_stage(
    *,
    tracking_result: dict,
    run_dir: Path,
    frame_record_path: Path,
    debug_config: DebugAppConfig,
) -> dict:
    upstream_paths = resolve_upstream_runtime_paths(debug_config)
    repo_root = upstream_paths["repo_root"]
    refined_config_path = upstream_paths["refined_config_path"]

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from scripts import offline_app_refined as refined_module

    cfg = refined_module.load_refined_config(str(refined_config_path))
    app = refined_module.RefinedOfflineApp(str(refined_config_path), config=cfg)
    app.reprompt_thresholds = refined_module.build_reprompt_thresholds(cfg)

    input_path = str(tracking_result.get("clip_dir") or tracking_result.get("input_video") or "").strip()
    if not input_path:
        raise RuntimeError("Real body4d stage requires clip_dir or input_video in tracking_result.")

    sample = app.prepare_input(input_path, str(run_dir), skip_existing=False)
    app.sample_config = cfg
    app.sample_summary = {"status": "body4d_only"}
    compat_summaries = build_compat_sample_summaries(app, sample)
    app.sample_summary["fps_summary"] = compat_summaries["fps_summary"]
    app.sample_summary["bitrate_summary"] = compat_summaries["bitrate_summary"]
    app.sample_summary["clip_duration_summary"] = compat_summaries["clip_duration_summary"]

    obj_ids = [int(value) for value in list(tracking_result.get("obj_ids") or [])]
    if not obj_ids:
        obj_ids = [int((tracking_result.get("selection_used") or {}).get("track_id", 1) or 1)]

    app.prepare_sample_output(str(run_dir), obj_ids)
    copied_inputs = copy_tracking_inputs_to_body4d_run(
        tracking_result=tracking_result,
        run_dir=run_dir,
    )

    runtime_app = app._ensure_base_app()
    align_dtypes = getattr(refined_module, "_align_completion_pipeline_dtypes", None)
    if callable(align_dtypes):
        align_dtypes(runtime_app)

    autocast_disabled = getattr(refined_module, "_autocast_disabled", None)
    autocast_context = autocast_disabled() if callable(autocast_disabled) else nullcontext()
    frame_writer = RecordedFrameWriter(output_path=frame_record_path)

    context = refined_module.load_base_offline_module().build_4d_context(
        input_dir=str(run_dir),
        output_dir=str(run_dir),
        runtime=runtime_app.RUNTIME,
        sam3_3d_body_model=runtime_app.sam3_3d_body_model,
        pipeline_mask=runtime_app.pipeline_mask,
        pipeline_rgb=runtime_app.pipeline_rgb,
        depth_model=runtime_app.depth_model,
        predictor=runtime_app.predictor,
        generator=runtime_app.generator,
        frame_writer=frame_writer,
    )
    with autocast_context:
        rendered_video_path = refined_module.load_base_offline_module().run_4d_pipeline_from_context(context)

    standardized_video_path = promote_rendered_video_to_standard_path(
        run_dir=run_dir,
        rendered_video_path=rendered_video_path,
    )
    return {
        "run_dir": str(run_dir),
        "input_tracking_run": dict(tracking_result),
        "input_images_dir": str(copied_inputs["images_dir"]),
        "input_masks_dir": str(copied_inputs["masks_dir"]),
        "rendered_video_path": str(standardized_video_path),
        "rendered_frames_dir": str(run_dir / "rendered_frames"),
        "mesh_dir": str(run_dir / "mesh_4d_individual"),
        "focal_dir": str(run_dir / "focal_4d_individual"),
        "recorded_frame_outputs_path": str(frame_record_path),
        "real_runtime_used": True,
        "upstream_repo_root": str(repo_root),
        "upstream_refined_config_path": str(refined_config_path),
        "upstream_module_name": str(getattr(refined_module, "__name__", "")),
    }


def run_body4d_stage(*, runtime_session, tracking_result: dict, debug_config: DebugAppConfig | None = None) -> dict:
    run_dir = runtime_session.next_revision_dir("body4d")
    frame_record_path = Path(run_dir) / "body4d_frame_records.pkl"

    if not _should_use_real_body4d_runtime(debug_config):
        return _run_stub_body4d_stage(
            tracking_result=tracking_result,
            run_dir=Path(run_dir),
            frame_record_path=frame_record_path,
            stub_reason="real_runtime_disabled",
        )

    upstream_paths = resolve_upstream_runtime_paths(debug_config)
    repo_root = upstream_paths["repo_root"]
    refined_config_path = upstream_paths["refined_config_path"]
    if not repo_root.is_dir() or not refined_config_path.is_file():
        return _run_stub_body4d_stage(
            tracking_result=tracking_result,
            run_dir=Path(run_dir),
            frame_record_path=frame_record_path,
            stub_reason="upstream_runtime_unavailable",
        )

    try:
        return _run_real_body4d_stage(
            tracking_result=tracking_result,
            run_dir=Path(run_dir),
            frame_record_path=frame_record_path,
            debug_config=debug_config,
        )
    except Exception as exc:
        result = _run_stub_body4d_stage(
            tracking_result=tracking_result,
            run_dir=Path(run_dir),
            frame_record_path=frame_record_path,
            stub_reason=f"real_runtime_failed:{type(exc).__name__}",
        )
        result["real_runtime_error"] = str(exc)
        return result
