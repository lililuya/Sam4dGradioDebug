from __future__ import annotations

from pathlib import Path


def ensure_layout(session_dir: Path) -> None:
    for name in (
        "logs",
        "load",
        "target_selection",
        "tracking",
        "body4d",
        "wan_export",
    ):
        (session_dir / name).mkdir(parents=True, exist_ok=True)


def next_revision_dir(stage_root: Path) -> Path:
    existing = sorted(path for path in stage_root.glob("run_v*") if path.is_dir())
    next_index = len(existing) + 1
    path = stage_root / f"run_v{next_index:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path
