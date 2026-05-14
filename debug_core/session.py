from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from debug_core.artifacts import ensure_layout, next_revision_dir


@dataclass(slots=True)
class DebugSession:
    clip_id: str
    sample_uuid: str
    source_path: str
    session_id: str
    session_dir: Path
    session_meta_path: Path

    def next_revision_dir(self, stage_name: str) -> Path:
        return next_revision_dir(self.session_dir / stage_name)


def create_debug_session(*, clip_id: str, sample_uuid: str, source_path: str, output_root: str | Path) -> DebugSession:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_id = f"session_{timestamp}"
    session_dir = Path(output_root) / clip_id / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    ensure_layout(session_dir)
    session_meta_path = session_dir / "session_meta.json"
    session_meta_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "sample_uuid": sample_uuid,
                "source_path": source_path,
                "session_id": session_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return DebugSession(
        clip_id=clip_id,
        sample_uuid=sample_uuid,
        source_path=source_path,
        session_id=session_id,
        session_dir=session_dir,
        session_meta_path=session_meta_path,
    )
