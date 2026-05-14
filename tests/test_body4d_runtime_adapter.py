from pathlib import Path

from debug_core.adapters.body4d_runtime_adapter import (
    copy_tracking_inputs_to_body4d_run,
    promote_rendered_video_to_standard_path,
    resolve_tracking_mask_input_dir,
)


def test_resolve_tracking_mask_input_dir_prefers_working_mask_dir():
    tracking_result = {
        "working_mask_dir": "working_masks",
        "refined_mask_dir": "refined_masks",
    }
    assert resolve_tracking_mask_input_dir(tracking_result) == "working_masks"


def test_copy_tracking_inputs_to_body4d_run_copies_images_and_masks(tmp_path: Path):
    images_src = tmp_path / "tracking_images"
    masks_src = tmp_path / "tracking_masks"
    images_src.mkdir()
    masks_src.mkdir()
    (images_src / "00000000.jpg").write_bytes(b"image-bytes")
    (masks_src / "00000000.png").write_bytes(b"mask-bytes")
    run_dir = tmp_path / "body4d_run"
    run_dir.mkdir()

    copy_tracking_inputs_to_body4d_run(
        tracking_result={
            "image_dir": str(images_src),
            "working_mask_dir": str(masks_src),
            "refined_mask_dir": str(masks_src),
        },
        run_dir=run_dir,
    )

    assert (run_dir / "images" / "00000000.jpg").is_file()
    assert (run_dir / "masks" / "00000000.png").is_file()


def test_promote_rendered_video_to_standard_path_renames_timestamped_output(tmp_path: Path):
    run_dir = tmp_path / "body4d_run"
    run_dir.mkdir()
    timestamped_video = run_dir / "4d_123.mp4"
    timestamped_video.write_bytes(b"video")

    promoted = promote_rendered_video_to_standard_path(run_dir=run_dir, rendered_video_path=timestamped_video)

    assert promoted == run_dir / "4d.mp4"
    assert promoted.is_file()
