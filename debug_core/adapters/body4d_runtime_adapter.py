from __future__ import annotations

import shutil
from pathlib import Path


def resolve_tracking_mask_input_dir(tracking_result: dict) -> str:
    working_mask_dir = str(tracking_result.get("working_mask_dir") or "").strip()
    if working_mask_dir:
        return working_mask_dir

    refined_mask_dir = str(tracking_result.get("refined_mask_dir") or "").strip()
    if refined_mask_dir:
        return refined_mask_dir

    raw_mask_dir = str(tracking_result.get("raw_mask_dir") or "").strip()
    if raw_mask_dir:
        return raw_mask_dir

    raise RuntimeError("Tracking result does not contain a usable mask directory.")


def _copy_tree_contents(*, source_dir: Path, target_dir: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing required directory: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(path for path in source_dir.iterdir() if path.is_file()):
        shutil.copy2(source_path, target_dir / source_path.name)


def copy_tracking_inputs_to_body4d_run(*, tracking_result: dict, run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    images_dir = run_dir / "images"
    masks_dir = run_dir / "masks"

    image_source_dir = Path(str(tracking_result.get("image_dir") or "")).resolve()
    mask_source_dir = Path(resolve_tracking_mask_input_dir(tracking_result)).resolve()

    _copy_tree_contents(source_dir=image_source_dir, target_dir=images_dir)
    _copy_tree_contents(source_dir=mask_source_dir, target_dir=masks_dir)
    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
    }


def promote_rendered_video_to_standard_path(*, run_dir: str | Path, rendered_video_path: str | Path | None) -> Path:
    run_dir = Path(run_dir)
    standard_path = run_dir / "4d.mp4"
    if rendered_video_path is None:
        return standard_path

    rendered_video_path = Path(rendered_video_path)
    if rendered_video_path.resolve() == standard_path.resolve():
        return standard_path

    standard_path.parent.mkdir(parents=True, exist_ok=True)
    if standard_path.exists():
        standard_path.unlink()
    rendered_video_path.replace(standard_path)
    return standard_path
