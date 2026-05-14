from __future__ import annotations
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from debug_core.config import DebugAppConfig


def resolve_upstream_runtime_paths(config: DebugAppConfig) -> dict:
    base_dir = Path(config.path).resolve().parent
    repo_root = Path(str(config.upstream.get("repo_root", ""))).expanduser()
    refined_config_path = Path(str(config.upstream.get("refined_config_path", ""))).expanduser()
    if not repo_root.is_absolute():
        repo_root = (base_dir / repo_root).resolve()
    else:
        repo_root = repo_root.resolve()
    if not refined_config_path.is_absolute():
        refined_config_path = (base_dir / refined_config_path).resolve()
    else:
        refined_config_path = refined_config_path.resolve()
    return {
        "repo_root": repo_root,
        "refined_config_path": refined_config_path,
        "enable_real_runtime": bool(config.upstream.get("enable_real_runtime", True)),
    }


def _run_tracking_stub(*, sample_context: dict, selection: dict, run_dir: Path) -> dict:
    debug_metrics_dir = run_dir / "debug_metrics"
    image_dir = run_dir / "images"
    raw_mask_dir = run_dir / "masks_raw"
    refined_mask_dir = run_dir / "masks_refined"
    working_mask_dir = run_dir / "masks"
    overlay_dir = run_dir / "overlays"
    for path in (debug_metrics_dir, image_dir, raw_mask_dir, refined_mask_dir, working_mask_dir, overlay_dir):
        path.mkdir(parents=True, exist_ok=True)

    clip_video_path = str(sample_context.get("clip_video_path") or sample_context.get("input_video") or "").strip()
    frame_stems = [str(value) for value in list(sample_context.get("frame_stems") or [])]
    if clip_video_path and frame_stems:
        capture = cv2.VideoCapture(clip_video_path)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"unable to open video for tracking stub: {clip_video_path}")
            for frame_stem in frame_stems:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    break
                cv2.imwrite(str(image_dir / f"{frame_stem}.jpg"), frame_bgr)
                mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                cv2.imwrite(str(raw_mask_dir / f"{frame_stem}.png"), mask)
                cv2.imwrite(str(refined_mask_dir / f"{frame_stem}.png"), mask)
                cv2.imwrite(str(working_mask_dir / f"{frame_stem}.png"), mask)
        finally:
            capture.release()

    return {
        "run_dir": run_dir,
        "selection_used": dict(selection),
        "image_dir": str(image_dir),
        "raw_mask_dir": str(raw_mask_dir),
        "refined_mask_dir": str(refined_mask_dir),
        "working_mask_dir": str(working_mask_dir),
        "overlay_dir": str(overlay_dir),
        "debug_metrics_dir": str(debug_metrics_dir),
        "frame_metrics": [],
        "real_runtime_used": False,
        "stub_reason": "real_runtime_disabled_or_unavailable",
        "obj_ids": [int(selection.get("track_id", 1) or 1)],
        "start_frame_idx": int(selection.get("start_frame_idx", 0) or 0),
        "input_video": clip_video_path,
        "input_type": "clip_package" if sample_context.get("clip_dir") else "",
        "frame_count": int(sample_context.get("frame_count", 0) or 0),
        "fps": float(sample_context.get("fps", 0.0) or 0.0),
        "clip_dir": str(sample_context.get("clip_dir") or ""),
        "clip_id": str(sample_context.get("clip_id") or ""),
        "sample_uuid": str(sample_context.get("sample_uuid") or ""),
        "source_path": str(sample_context.get("source_path") or ""),
    }


def _should_use_real_runtime(debug_config: DebugAppConfig | None) -> bool:
    return debug_config is not None and bool(debug_config.upstream.get("enable_real_runtime", True))


def _import_upstream_refined_module(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from scripts import offline_app_refined as refined_module

    return refined_module


def _create_runtime_profile(selection: dict) -> dict:
    return {
        "debug_override": {
            "track_id": int(selection.get("track_id", 1) or 1),
            "start_frame_idx": int(selection.get("start_frame_idx", 0) or 0),
            "bbox_xyxy": [int(value) for value in list(selection.get("bbox_xyxy") or [])[:4]],
        }
    }


def build_compat_sample_summaries(app, sample: dict) -> dict:
    fps_builder = getattr(app, "_build_sample_fps_summary", None)
    bitrate_builder = getattr(app, "_build_sample_bitrate_summary", None)
    duration_builder = getattr(app, "_build_clip_duration_summary", None)

    fps_fallback = {
        "source_fps": float(sample.get("fps", 0.0) or 0.0),
        "source_fps_source": "sample_metadata",
        "rendered_4d_fps": float(sample.get("fps", 0.0) or 0.0),
        "wan_target_fps": None,
    }
    if callable(fps_builder):
        try:
            fps_summary = dict(fps_builder(sample))
        except Exception:
            fps_summary = dict(fps_fallback)
    else:
        fps_summary = dict(fps_fallback)

    bitrate_fallback = {
        "source_bitrate": None,
        "source_bitrate_source": "unavailable",
        "rendered_4d_bitrate": None,
    }
    if callable(bitrate_builder):
        try:
            bitrate_summary = dict(bitrate_builder(sample))
        except Exception:
            bitrate_summary = dict(bitrate_fallback)
    else:
        bitrate_summary = dict(bitrate_fallback)

    frame_count = int(sample.get("frame_count", 0) or 0)
    fps_value = float(fps_summary.get("source_fps", 0.0) or sample.get("fps", 0.0) or 0.0)
    duration_seconds = 0.0
    if frame_count > 0 and fps_value > 0.0:
        duration_seconds = float(frame_count) / float(fps_value)
    duration_fallback = {
        "frame_count": frame_count,
        "fps": fps_value,
        "duration_seconds": float(duration_seconds),
        "max_clip_len_seconds": 0.0,
        "input_type": str(sample.get("input_type") or ""),
        "clip_dir": str(sample.get("clip_dir") or ""),
    }
    if callable(duration_builder):
        try:
            clip_duration_summary = dict(duration_builder(sample))
        except Exception:
            clip_duration_summary = dict(duration_fallback)
    else:
        clip_duration_summary = dict(duration_fallback)

    return {
        "fps_summary": fps_summary,
        "bitrate_summary": bitrate_summary,
        "clip_duration_summary": clip_duration_summary,
    }


def _read_video_frame(video_path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"unable to read frame {frame_index} from video: {video_path}")
    return frame_bgr


def _initialize_real_tracking_app(
    *,
    refined_module,
    refined_config_path: Path,
    run_dir: Path,
    sample_context: dict,
) -> tuple[object, dict]:
    cfg = refined_module.load_refined_config(str(refined_config_path))
    app = refined_module.RefinedOfflineApp(str(refined_config_path), config=cfg)
    app.reprompt_thresholds = refined_module.build_reprompt_thresholds(cfg)
    sample = app.prepare_input(str(sample_context["clip_dir"]), str(run_dir), skip_existing=False)
    app.sample_config = cfg
    app.sample_summary = {"status": "tracking_only"}
    compat_summaries = build_compat_sample_summaries(app, sample)
    app.sample_summary["fps_summary"] = compat_summaries["fps_summary"]
    app.sample_summary["bitrate_summary"] = compat_summaries["bitrate_summary"]
    app.sample_summary["clip_duration_summary"] = compat_summaries["clip_duration_summary"]
    return app, sample


def _prepare_manual_tracking_prompt(*, app, sample: dict, selection: dict):
    runtime_app = app._ensure_base_app()
    track_id = int(selection.get("track_id", 1) or 1)
    start_frame_idx = int(selection.get("start_frame_idx", 0) or 0)
    bbox_xyxy = [float(value) for value in list(selection.get("bbox_xyxy") or [])[:4]]
    if len(bbox_xyxy) != 4:
        raise RuntimeError("manual tracking selection requires bbox_xyxy with four values")

    frame_bgr = _read_video_frame(str(sample["input_video"]), start_frame_idx)
    frame_height, frame_width = frame_bgr.shape[:2]
    rel_box = np.array(
        [[
            bbox_xyxy[0] / float(max(frame_width, 1)),
            bbox_xyxy[1] / float(max(frame_height, 1)),
            bbox_xyxy[2] / float(max(frame_width, 1)),
            bbox_xyxy[3] / float(max(frame_height, 1)),
        ]],
        dtype=np.float32,
    )

    inference_state = runtime_app.predictor.init_state(video_path=sample["input_video"])
    runtime_app.predictor.clear_all_points_in_video(inference_state)
    runtime_app.RUNTIME["inference_state"] = inference_state
    runtime_app.RUNTIME["out_obj_ids"] = []
    runtime_app.RUNTIME["video_fps"] = float(sample.get("fps") or 25.0)

    _, runtime_app.RUNTIME["out_obj_ids"], _, _ = runtime_app.predictor.add_new_points_or_box(
        inference_state=runtime_app.RUNTIME["inference_state"],
        frame_idx=int(start_frame_idx),
        obj_id=int(track_id),
        box=rel_box,
    )
    app.initial_targets = {
        "obj_ids": list(runtime_app.RUNTIME["out_obj_ids"]),
        "start_frame_idx": int(start_frame_idx),
        "boxes_xyxy": [[float(value) for value in bbox_xyxy]],
    }
    app.sample_summary["initial_targets"] = {
        "obj_ids": list(runtime_app.RUNTIME["out_obj_ids"]),
        "start_frame_idx": int(start_frame_idx),
    }
    return runtime_app


def _collect_video_segments(*, refined_module, runtime_app, start_frame_idx: int, frame_count: int) -> dict[int, dict[int, np.ndarray]]:
    from scripts.offline_tracking_compat import unpack_propagate_output

    video_segments: dict[int, dict[int, np.ndarray]] = {}
    did_preflight = False
    for reverse in (False, True):
        for propagate_output in runtime_app.predictor.propagate_in_video(
            runtime_app.RUNTIME["inference_state"],
            start_frame_idx=int(start_frame_idx),
            max_frame_num_to_track=int(frame_count),
            reverse=bool(reverse),
            propagate_preflight=not did_preflight,
        ):
            did_preflight = True
            frame_idx, obj_ids, _low_res_masks, video_res_masks = unpack_propagate_output(propagate_output)
            video_segments[int(frame_idx)] = {
                int(out_obj_id): (video_res_masks[index] > 0.0).cpu().numpy()
                for index, out_obj_id in enumerate(obj_ids)
            }
    return video_segments


def _write_tracking_images_and_raw_masks(*, refined_module, app, sample: dict, video_segments: dict[int, dict[int, np.ndarray]]) -> list[np.ndarray]:
    frame_stems = list(sample.get("frames") or [])
    raw_masks: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(sample["input_video"]))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video for tracking export: {sample['input_video']}")
        for frame_index, frame_stem in enumerate(frame_stems):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"unable to read frame {frame_index} for tracking export")
            cv2.imwrite(os.path.join(app.output_paths["images"], f"{frame_stem}.jpg"), frame_bgr)
            frame_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            for obj_id, out_mask in (video_segments.get(frame_index) or {}).items():
                binary = (out_mask[0] > 0).astype(np.uint8) * 255
                frame_mask[binary == 255] = int(obj_id)
            raw_masks.append(frame_mask)
            refined_module.save_indexed_mask(frame_mask, os.path.join(app.output_paths["masks_raw"], f"{frame_stem}.png"))
    finally:
        capture.release()
    return raw_masks


def _refine_and_materialize_tracking_masks(*, app, sample: dict, raw_masks: list[np.ndarray]) -> dict:
    frame_stems = list(sample.get("frames") or [])
    raw_chunk = {
        "chunk_id": 0,
        "frame_indices": list(range(len(frame_stems))),
        "frame_stems": frame_stems,
        "raw_masks": raw_masks,
    }
    chunk_descriptor = {
        "chunk_id": 0,
        "start_frame": 0,
        "end_frame": max(0, len(frame_stems) - 1),
        "frames": frame_stems,
    }
    refined_chunk = app.refine_chunk_masks(raw_chunk)
    final_chunk = app.maybe_reprompt_chunk(chunk_descriptor, refined_chunk, app.initial_targets)
    app.write_chunk_outputs(chunk_descriptor, raw_chunk, final_chunk)
    return final_chunk


def run_tracking_with_override(
    *,
    sample_context: dict,
    selection: dict,
    run_dir: Path,
    debug_config: DebugAppConfig | None = None,
) -> dict:
    if not _should_use_real_runtime(debug_config):
        return _run_tracking_stub(sample_context=sample_context, selection=selection, run_dir=run_dir)

    upstream_paths = resolve_upstream_runtime_paths(debug_config)
    repo_root = upstream_paths["repo_root"]
    refined_config_path = upstream_paths["refined_config_path"]
    if not repo_root.is_dir() or not refined_config_path.is_file():
        return _run_tracking_stub(sample_context=sample_context, selection=selection, run_dir=run_dir)

    refined_module = _import_upstream_refined_module(repo_root)
    runtime_profile = _create_runtime_profile(selection)
    app, sample = _initialize_real_tracking_app(
        refined_module=refined_module,
        refined_config_path=refined_config_path,
        run_dir=run_dir,
        sample_context=sample_context,
    )
    runtime_app = _prepare_manual_tracking_prompt(app=app, sample=sample, selection=selection)
    app.prepare_sample_output(str(run_dir), app.initial_targets.get("obj_ids", [int(selection.get("track_id", 1) or 1)]))
    video_segments = _collect_video_segments(
        refined_module=refined_module,
        runtime_app=runtime_app,
        start_frame_idx=int(selection.get("start_frame_idx", 0) or 0),
        frame_count=int(sample.get("frame_count", 0) or 0),
    )
    raw_masks = _write_tracking_images_and_raw_masks(
        refined_module=refined_module,
        app=app,
        sample=sample,
        video_segments=video_segments,
    )
    final_chunk = _refine_and_materialize_tracking_masks(app=app, sample=sample, raw_masks=raw_masks)
    app.finalize_sample()

    overlay_dir = run_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "selection_used": dict(selection),
        "raw_mask_dir": str(app.output_paths["masks_raw"]),
        "refined_mask_dir": str(app.output_paths["masks_refined"]),
        "working_mask_dir": str(Path(str(run_dir)) / "masks"),
        "image_dir": str(app.output_paths["images"]),
        "overlay_dir": str(overlay_dir),
        "debug_metrics_dir": str(app.output_paths.get("debug_metrics", run_dir / "debug_metrics")),
        "frame_metrics": list(final_chunk.get("frame_metrics", [])),
        "runtime_profile": runtime_profile,
        "upstream_repo_root": str(repo_root),
        "upstream_refined_config_path": str(refined_config_path),
        "upstream_module_name": str(getattr(refined_module, "__name__", "")),
        "real_runtime_used": True,
        "obj_ids": [int(value) for value in list(app.initial_targets.get("obj_ids", []))],
        "start_frame_idx": int(app.initial_targets.get("start_frame_idx", int(selection.get("start_frame_idx", 0) or 0))),
        "input_video": str(sample.get("input_video") or ""),
        "input_type": str(sample.get("input_type") or ""),
        "frame_count": int(sample.get("frame_count", 0) or 0),
        "fps": float(sample.get("fps") or 0.0),
        "clip_dir": str(sample.get("clip_dir") or ""),
        "clip_id": str(sample.get("clip_id") or ""),
        "sample_uuid": str(sample.get("sample_uuid") or ""),
        "source_path": str(sample.get("source_path") or ""),
    }
