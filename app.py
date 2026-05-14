from __future__ import annotations

from pathlib import Path

import gradio as gr
from gradio.events import SelectData

from debug_core.config import load_debug_config
from debug_core.controllers.debug_pipeline_controller import DebugPipelineController
from debug_core.stages.load_clip_package import load_clip_package


APP_CONTROLLER: DebugPipelineController | None = None


def _get_controller(output_root: str) -> DebugPipelineController:
    global APP_CONTROLLER
    output_root_path = Path(output_root)
    if APP_CONTROLLER is None or APP_CONTROLLER.output_root != output_root_path:
        APP_CONTROLLER = DebugPipelineController(output_root=output_root_path)
    return APP_CONTROLLER


def _load_sample_for_ui(clip_dir: str, config_path: str, output_root: str):
    debug_config = load_debug_config(config_path)
    controller = _get_controller(output_root)
    sample_context = load_clip_package(clip_dir)
    session = controller.bootstrap_session(sample_context, debug_config=debug_config)
    return session["sample_context"], session["auto_selection_proposal"]


def _load_sample_bundle_for_ui(clip_dir: str, config_path: str, output_root: str):
    sample_info, proposal = _load_sample_for_ui(clip_dir, config_path, output_root)
    controller = _get_controller(output_root)
    preview = controller.get_preview(frame_idx=int(proposal["start_frame_idx"]))
    bbox_xyxy = list(proposal.get("bbox_xyxy") or [0, 0, 0, 0])
    status = "Loaded clip package and applied auto proposal."
    return (
        sample_info,
        proposal,
        None,
        preview,
        int(proposal.get("track_id", 1) or 1),
        int(proposal.get("start_frame_idx", 0) or 0),
        int(bbox_xyxy[0]),
        int(bbox_xyxy[1]),
        int(bbox_xyxy[2]),
        int(bbox_xyxy[3]),
        status,
    )


def _apply_manual_selection_for_ui(
    output_root: str,
    track_id: int,
    start_frame_idx: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
):
    controller = _get_controller(output_root)
    selection = controller.set_manual_selection(
        track_id=int(track_id),
        start_frame_idx=int(start_frame_idx),
        bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)],
    )
    preview = controller.get_preview(frame_idx=int(start_frame_idx))
    status = "Applied manual selection override."
    return selection, preview, status


def _preview_frame_for_ui(output_root: str, frame_idx: int):
    controller = _get_controller(output_root)
    preview = controller.get_preview(frame_idx=int(frame_idx))
    status = f"Preview updated for frame {int(frame_idx)}."
    return preview, status


def _handle_preview_click_for_ui(output_root: str, frame_idx: int, x: int, y: int):
    controller = _get_controller(output_root)
    click_result = controller.register_preview_click(frame_idx=int(frame_idx), x=int(x), y=int(y))
    pending_bbox = list(click_result.get("pending_bbox") or [])
    preview = controller.get_preview(frame_idx=int(frame_idx))
    if pending_bbox:
        status = f"Pending bbox ready from clicks: {pending_bbox}."
        return preview, int(pending_bbox[0]), int(pending_bbox[1]), int(pending_bbox[2]), int(pending_bbox[3]), status
    status = "Registered first corner click. Click the opposite corner to finish the bbox."
    return preview, 0, 0, 0, 0, status


def _handle_preview_select_event(output_root: str, frame_idx: int, evt: SelectData):
    index = getattr(evt, "index", None)
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        controller = _get_controller(output_root)
        preview = controller.get_preview(frame_idx=int(frame_idx))
        return preview, 0, 0, 0, 0, "Click event did not include image coordinates."
    return _handle_preview_click_for_ui(output_root, int(frame_idx), int(index[0]), int(index[1]))


def _apply_clicked_bbox_for_ui(output_root: str, track_id: int, start_frame_idx: int):
    controller = _get_controller(output_root)
    if controller.session is None:
        raise RuntimeError("No active session")
    pending_bbox = list(controller.session.get("pending_bbox") or [])
    if len(pending_bbox) < 4:
        raise RuntimeError("No pending bbox from image clicks. Click two corners first.")
    selection = controller.set_manual_selection(
        track_id=int(track_id),
        start_frame_idx=int(start_frame_idx),
        bbox_xyxy=[int(value) for value in pending_bbox[:4]],
    )
    preview = controller.get_preview(frame_idx=int(start_frame_idx))
    status = "Applied clicked bbox as manual selection."
    return selection, preview, status


def _slider_update_for_loaded_sample(sample_info: dict, proposal: dict):
    frame_count = int(sample_info.get("frame_count", 1) or 1)
    start_frame_idx = int(proposal.get("start_frame_idx", 0) or 0)
    return gr.update(minimum=0, maximum=max(0, frame_count - 1), value=start_frame_idx)


def _run_tracking_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_tracking_stage()


def _run_body4d_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_body4d_stage()


def _run_wan_export_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_wan_export_stage()


def build_app():
    controller = DebugPipelineController(output_root=Path("outputs_debug"))
    with gr.Blocks(title="Sam4dGradioDebug") as demo:
        gr.Markdown("# Sam4dGradioDebug")
        with gr.Row():
            with gr.Column(scale=1):
                clip_dir = gr.Textbox(label="Clip Package Directory")
                config_path = gr.Textbox(label="Config Path", value="configs/debug_default.yaml")
                output_root = gr.Textbox(label="Session Output Root", value="outputs_debug")
                preserve_all = gr.Checkbox(label="Preserve All Intermediates", value=True)
                load_button = gr.Button("Load Sample")
            with gr.Column(scale=2):
                sample_info = gr.JSON(label="Loaded Sample Info")
                proposal_info = gr.JSON(label="Target Selection Proposal")
                manual_selection_info = gr.JSON(label="Manual Selection")
                current_frame = gr.Image(label="Current Frame Preview", interactive=True)

        with gr.Accordion("Target Selection", open=True):
            with gr.Row():
                track_id = gr.Number(label="Track ID", value=1, precision=0)
                start_frame_idx = gr.Number(label="Start Frame", value=0, precision=0)
                frame_slider = gr.Slider(label="Preview Frame", minimum=0, maximum=0, value=0, step=1)
            with gr.Row():
                x1 = gr.Number(label="BBox X1", value=0, precision=0)
                y1 = gr.Number(label="BBox Y1", value=0, precision=0)
                x2 = gr.Number(label="BBox X2", value=0, precision=0)
                y2 = gr.Number(label="BBox Y2", value=0, precision=0)
            with gr.Row():
                apply_manual_button = gr.Button("Apply Manual Selection")
                apply_clicked_bbox_button = gr.Button("Apply Clicked BBox")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("Tracking / Mask", open=True):
            run_tracking_button = gr.Button("Run Tracking / Mask")
            tracking_result = gr.JSON(label="Tracking Result")

        with gr.Accordion("4D", open=False):
            run_body4d_button = gr.Button("Run 4D")
            body4d_result = gr.JSON(label="Body4D Result")

        with gr.Accordion("Wan Export", open=False):
            run_wan_export_button = gr.Button("Run Wan Export")
            wan_export_result = gr.JSON(label="Wan Export Result")

        load_button.click(
            fn=_load_sample_bundle_for_ui,
            inputs=[clip_dir, config_path, output_root],
            outputs=[sample_info, proposal_info, manual_selection_info, current_frame, track_id, start_frame_idx, x1, y1, x2, y2, status_text],
        )
        load_button.click(
            fn=lambda clip_dir_value, config_path_value, output_root_value: _slider_update_for_loaded_sample(
                *_load_sample_for_ui(clip_dir_value, config_path_value, output_root_value)
            ),
            inputs=[clip_dir, config_path, output_root],
            outputs=[frame_slider],
        )
        apply_manual_button.click(
            fn=_apply_manual_selection_for_ui,
            inputs=[output_root, track_id, start_frame_idx, x1, y1, x2, y2],
            outputs=[manual_selection_info, current_frame, status_text],
        )
        apply_clicked_bbox_button.click(
            fn=_apply_clicked_bbox_for_ui,
            inputs=[output_root, track_id, start_frame_idx],
            outputs=[manual_selection_info, current_frame, status_text],
        )
        frame_slider.change(
            fn=_preview_frame_for_ui,
            inputs=[output_root, frame_slider],
            outputs=[current_frame, status_text],
        )
        current_frame.select(
            fn=_handle_preview_select_event,
            inputs=[output_root, frame_slider],
            outputs=[current_frame, x1, y1, x2, y2, status_text],
        )
        run_tracking_button.click(fn=_run_tracking_for_ui, inputs=[output_root], outputs=[tracking_result])
        run_body4d_button.click(fn=_run_body4d_for_ui, inputs=[output_root], outputs=[body4d_result])
        run_wan_export_button.click(fn=_run_wan_export_for_ui, inputs=[output_root], outputs=[wan_export_result])

        demo.controller = controller
        demo.debug_components = {
            "clip_dir": clip_dir,
            "config_path": config_path,
            "output_root": output_root,
            "preserve_all": preserve_all,
            "load_button": load_button,
            "sample_info": sample_info,
            "proposal_info": proposal_info,
            "manual_selection_info": manual_selection_info,
            "current_frame": current_frame,
            "track_id": track_id,
            "start_frame_idx": start_frame_idx,
            "frame_slider": frame_slider,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "apply_manual_button": apply_manual_button,
            "apply_clicked_bbox_button": apply_clicked_bbox_button,
            "status_text": status_text,
            "run_tracking_button": run_tracking_button,
            "run_body4d_button": run_body4d_button,
            "run_wan_export_button": run_wan_export_button,
            "tracking_result": tracking_result,
            "body4d_result": body4d_result,
            "wan_export_result": wan_export_result,
        }
    return demo


if __name__ == "__main__":
    build_app().launch()
