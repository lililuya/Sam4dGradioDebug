from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class StageRecord:
    status: str = "idle"
    active_revision: str | None = None
    last_error_path: Path | None = None


@dataclass(slots=True)
class DebugSessionState:
    clip_id: str
    sample_uuid: str
    source_path: str
    session_id: str
    session_dir: Path
    stage_status: dict[str, StageRecord] = field(default_factory=dict)
