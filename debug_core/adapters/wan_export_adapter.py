from __future__ import annotations

from pathlib import Path


def write_debug_export_stub(run_dir: Path) -> dict:
    target_dir = run_dir / "target_output"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in (
        "target.mp4",
        "src_pose.mp4",
        "src_face.mp4",
        "src_bg.mp4",
        "src_mask.mp4",
        "src_mask_detail.mp4",
        "src_ref.png",
        "meta.json",
        "pose_meta_sequence.json",
        "smpl_sequence.json",
    ):
        path = target_dir / name
        path.write_bytes(b"")
        outputs[name] = str(path)
    return {"target_dir": str(target_dir), **outputs}
