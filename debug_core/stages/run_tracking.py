from __future__ import annotations

from pathlib import Path

from debug_core.config import DebugAppConfig
from debug_core.adapters.refined_runtime_adapter import run_tracking_with_override


def run_tracking_stage(
    *,
    runtime_session,
    sample_context: dict,
    selection: dict,
    debug_config: DebugAppConfig | None = None,
) -> dict:
    run_dir = runtime_session.next_revision_dir("tracking")
    return run_tracking_with_override(
        sample_context=sample_context,
        selection=selection,
        run_dir=Path(run_dir),
        debug_config=debug_config,
    )
