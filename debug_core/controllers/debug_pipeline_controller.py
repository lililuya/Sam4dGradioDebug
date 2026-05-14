from __future__ import annotations

from pathlib import Path

from debug_core.config import DebugAppConfig
from debug_core.session import create_debug_session
from debug_core.stages.run_refine_4d import run_body4d_stage
from debug_core.stages.run_tracking import run_tracking_stage
from debug_core.stages.run_wan_export import run_wan_export_stage
from debug_core.stages.select_target import apply_manual_selection, propose_target_selection
from debug_core.views.preview_utils import build_preview_frame


class DebugPipelineController:
    def __init__(self, *, output_root: str | Path):
        self.output_root = Path(output_root)
        self.session: dict | None = None

    def bootstrap_session(self, sample_context: dict, debug_config: DebugAppConfig | None = None) -> dict:
        runtime_session = create_debug_session(
            clip_id=str(sample_context["clip_id"]),
            sample_uuid=str(sample_context["sample_uuid"]),
            source_path=str(sample_context["source_path"]),
            output_root=self.output_root,
        )
        proposal = propose_target_selection(sample_context)
        self.session = {
            "runtime_session": runtime_session,
            "sample_context": dict(sample_context),
            "debug_config": debug_config,
            "auto_selection_proposal": proposal,
            "manual_selection": None,
            "effective_selection": proposal,
            "pending_click_points": [],
            "pending_bbox": None,
        }
        return self.session

    def set_manual_selection(self, *, track_id: int, start_frame_idx: int, bbox_xyxy: list[int]) -> dict:
        if self.session is None:
            raise RuntimeError("No active session")
        proposal = dict(self.session["auto_selection_proposal"])
        selection = apply_manual_selection(
            proposal,
            track_id=track_id,
            start_frame_idx=start_frame_idx,
            bbox_xyxy=bbox_xyxy,
        )
        self.session["manual_selection"] = selection
        self.session["effective_selection"] = selection
        self.session["pending_click_points"] = []
        self.session["pending_bbox"] = None
        return selection

    def register_preview_click(self, *, frame_idx: int, x: int, y: int) -> dict:
        if self.session is None:
            raise RuntimeError("No active session")
        points = list(self.session.get("pending_click_points") or [])
        points.append({"frame_idx": int(frame_idx), "x": int(x), "y": int(y)})
        pending_bbox = None
        if len(points) >= 2:
            first = points[-2]
            second = points[-1]
            if int(first["frame_idx"]) == int(second["frame_idx"]):
                pending_bbox = [
                    min(int(first["x"]), int(second["x"])),
                    min(int(first["y"]), int(second["y"])),
                    max(int(first["x"]), int(second["x"])),
                    max(int(first["y"]), int(second["y"])),
                ]
                points = []
        self.session["pending_click_points"] = points
        self.session["pending_bbox"] = pending_bbox
        return {"pending_points": points, "pending_bbox": pending_bbox}

    def get_preview(self, *, frame_idx: int):
        if self.session is None:
            raise RuntimeError("No active session")
        sample_context = self.session["sample_context"]
        auto_bbox = (self.session.get("auto_selection_proposal") or {}).get("bbox_xyxy")
        manual_bbox = (self.session.get("effective_selection") or {}).get("bbox_xyxy")
        pending_bbox = self.session.get("pending_bbox")
        return build_preview_frame(
            video_path=sample_context["clip_video_path"],
            frame_index=int(frame_idx),
            auto_bbox=auto_bbox,
            manual_bbox=manual_bbox,
            pending_bbox=pending_bbox,
        )

    def run_tracking_stage(self) -> dict:
        if self.session is None:
            raise RuntimeError("No active session")
        result = run_tracking_stage(
            runtime_session=self.session["runtime_session"],
            sample_context=self.session["sample_context"],
            selection=self.session["effective_selection"],
            debug_config=self.session.get("debug_config"),
        )
        self.session["tracking_result"] = result
        return result

    def run_body4d_stage(self) -> dict:
        if self.session is None or "tracking_result" not in self.session:
            raise RuntimeError("Tracking stage must succeed before body4d")
        result = run_body4d_stage(
            runtime_session=self.session["runtime_session"],
            tracking_result=self.session["tracking_result"],
            debug_config=self.session.get("debug_config"),
        )
        self.session["body4d_result"] = result
        return result

    def run_wan_export_stage(self) -> dict:
        if self.session is None or "body4d_result" not in self.session:
            raise RuntimeError("Body4D stage must succeed before wan export")
        result = run_wan_export_stage(
            runtime_session=self.session["runtime_session"],
            tracking_result=self.session["tracking_result"],
            body4d_result=self.session["body4d_result"],
            debug_config=self.session.get("debug_config"),
        )
        self.session["wan_export_result"] = result
        return result
