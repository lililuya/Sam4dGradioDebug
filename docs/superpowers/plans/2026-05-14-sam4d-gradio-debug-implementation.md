# Sam4dGradioDebug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Gradio repository for single-sample clip-package debugging with staged execution, manual target correction, revision-preserving artifact storage, and full downstream tracking/4D/Wan export coverage.

**Architecture:** The repository will be built as a thin Gradio shell over a reusable `debug_core` package. The `debug_core` package owns config loading, session state, artifact layout, stage orchestration, and adapters into the existing refined SAM4D runtime logic. The UI remains a view/controller layer and never becomes the source of truth for pipeline state.

**Tech Stack:** Python 3.10+, Gradio, OmegaConf, OpenCV, Pillow, pytest, dataclasses, pathlib, subprocess-safe filesystem orchestration, optional reuse of modules from the existing `sam-body4d-master` checkout via explicit adapters.

---

## File Map

### New Files

- `E:/Project/Sam4dGradioDebug/requirements.txt`
- `E:/Project/Sam4dGradioDebug/README.md`
- `E:/Project/Sam4dGradioDebug/configs/debug_default.yaml`
- `E:/Project/Sam4dGradioDebug/debug_core/__init__.py`
- `E:/Project/Sam4dGradioDebug/debug_core/config.py`
- `E:/Project/Sam4dGradioDebug/debug_core/state.py`
- `E:/Project/Sam4dGradioDebug/debug_core/session.py`
- `E:/Project/Sam4dGradioDebug/debug_core/artifacts.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/__init__.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/load_clip_package.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/select_target.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/run_tracking.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/run_refine_4d.py`
- `E:/Project/Sam4dGradioDebug/debug_core/stages/run_wan_export.py`
- `E:/Project/Sam4dGradioDebug/debug_core/controllers/__init__.py`
- `E:/Project/Sam4dGradioDebug/debug_core/controllers/debug_pipeline_controller.py`
- `E:/Project/Sam4dGradioDebug/debug_core/views/__init__.py`
- `E:/Project/Sam4dGradioDebug/debug_core/views/preview_utils.py`
- `E:/Project/Sam4dGradioDebug/debug_core/views/gallery_builders.py`
- `E:/Project/Sam4dGradioDebug/debug_core/views/timeline_utils.py`
- `E:/Project/Sam4dGradioDebug/debug_core/adapters/__init__.py`
- `E:/Project/Sam4dGradioDebug/debug_core/adapters/refined_runtime_adapter.py`
- `E:/Project/Sam4dGradioDebug/debug_core/adapters/wan_export_adapter.py`
- `E:/Project/Sam4dGradioDebug/app.py`
- `E:/Project/Sam4dGradioDebug/tests/conftest.py`
- `E:/Project/Sam4dGradioDebug/tests/test_config.py`
- `E:/Project/Sam4dGradioDebug/tests/test_load_clip_package.py`
- `E:/Project/Sam4dGradioDebug/tests/test_select_target.py`
- `E:/Project/Sam4dGradioDebug/tests/test_session_artifacts.py`
- `E:/Project/Sam4dGradioDebug/tests/test_controller.py`
- `E:/Project/Sam4dGradioDebug/tests/test_gradio_app.py`

### Existing Files To Modify

- `E:/Project/Sam4dGradioDebug/docs/superpowers/specs/2026-05-14-sam4d-gradio-debug-design.md`
  - Only if the implementation reveals a genuine contradiction that must be reflected in the spec.

---

### Task 1: Scaffold Repository and Baseline Dependencies

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/requirements.txt`
- Create: `E:/Project/Sam4dGradioDebug/README.md`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/__init__.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/__init__.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/controllers/__init__.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/views/__init__.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/adapters/__init__.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/conftest.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_config.py`

- [ ] **Step 1: Write the failing repository smoke test**

```python
# E:/Project/Sam4dGradioDebug/tests/test_config.py
from pathlib import Path


def test_repository_has_expected_top_level_layout():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "requirements.txt").is_file()
    assert (repo_root / "README.md").is_file()
    assert (repo_root / "debug_core").is_dir()
    assert (repo_root / "tests").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::test_repository_has_expected_top_level_layout -v`
Expected: FAIL because the repository scaffold files do not exist yet.

- [ ] **Step 3: Write the minimal scaffold files**

```text
# E:/Project/Sam4dGradioDebug/requirements.txt
gradio>=5.0.0,<6.0.0
omegaconf>=2.3.0
opencv-python>=4.8.0
pillow>=10.0.0
pytest>=8.0.0
```

```markdown
# E:/Project/Sam4dGradioDebug/README.md
# Sam4dGradioDebug

Standalone Gradio debugging UI for single-sample clip-package correction built around the refined SAM4D + Wan export workflow.
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/__init__.py
"""Core debugging package for Sam4dGradioDebug."""
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/__init__.py
"""Stage modules for the staged debug pipeline."""
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/controllers/__init__.py
"""Controller modules for Sam4dGradioDebug."""
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/views/__init__.py
"""View helpers for Gradio rendering."""
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/adapters/__init__.py
"""Adapter modules that bridge the existing SAM4D runtime into the debug UI."""
```

```python
# E:/Project/Sam4dGradioDebug/tests/conftest.py
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::test_repository_has_expected_top_level_layout -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md debug_core/__init__.py debug_core/stages/__init__.py debug_core/controllers/__init__.py debug_core/views/__init__.py debug_core/adapters/__init__.py tests/conftest.py tests/test_config.py
git commit -m "chore: scaffold sam4d gradio debug repository"
```

---

### Task 2: Add Debug Config Loading and Validation

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/configs/debug_default.yaml`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/config.py`
- Modify: `E:/Project/Sam4dGradioDebug/tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from debug_core.config import DebugAppConfig, load_debug_config


def test_load_debug_config_reads_default_yaml():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_debug_config(repo_root / "configs" / "debug_default.yaml")
    assert isinstance(config, DebugAppConfig)
    assert config.debug_mode["preserve_all_intermediate"] is True
    assert config.wan_export["cleanup_sample_workdir_after_export"] is False


def test_debug_config_exposes_required_output_toggles():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_debug_config(repo_root / "configs" / "debug_default.yaml")
    assert config.runtime["save_rendered_frames"] is True
    assert config.runtime["save_mesh_4d_individual"] is True
    assert config.wan_export["save_pose_meta_json"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL because `debug_core.config` and `configs/debug_default.yaml` do not exist.

- [ ] **Step 3: Implement config model and default YAML**

```yaml
# E:/Project/Sam4dGradioDebug/configs/debug_default.yaml
debug_mode:
  preserve_all_intermediate: true
  create_revision_per_rerun: true
  keep_logs: true
  keep_preview_cache: true

runtime:
  save_rendered_video: true
  save_rendered_video_direct: false
  save_rendered_frames: true
  save_rendered_frames_individual: true
  save_mesh_4d_individual: true
  save_focal_4d_individual: true

wan_export:
  enable: true
  copy_rendered_4d_to_targets: true
  cleanup_sample_workdir_after_export: false
  save_pose_meta_json: true
  save_smpl_sequence_json: true
  face_expand: 1.30
  face_gap: 8
  min_valid_face_ratio: 0.0

debug:
  save_metrics: true

refine:
  enable: true

reprompt:
  enable: true
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(slots=True)
class DebugAppConfig:
    raw: Any
    debug_mode: dict
    runtime: dict
    wan_export: dict
    debug: dict
    refine: dict
    reprompt: dict


def load_debug_config(path: str | Path) -> DebugAppConfig:
    cfg = OmegaConf.load(str(path))
    return DebugAppConfig(
        raw=cfg,
        debug_mode=dict(OmegaConf.to_container(cfg.debug_mode, resolve=True) or {}),
        runtime=dict(OmegaConf.to_container(cfg.runtime, resolve=True) or {}),
        wan_export=dict(OmegaConf.to_container(cfg.wan_export, resolve=True) or {}),
        debug=dict(OmegaConf.to_container(cfg.debug, resolve=True) or {}),
        refine=dict(OmegaConf.to_container(cfg.refine, resolve=True) or {}),
        reprompt=dict(OmegaConf.to_container(cfg.reprompt, resolve=True) or {}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/debug_default.yaml debug_core/config.py tests/test_config.py
git commit -m "feat: add debug config loading"
```

---

### Task 3: Implement Session State and Artifact Layout

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/state.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/session.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/artifacts.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/test_session_artifacts.py`

- [ ] **Step 1: Write the failing session/artifact tests**

```python
from pathlib import Path

from debug_core.session import create_debug_session


def test_create_debug_session_makes_expected_metadata_and_layout(tmp_path: Path):
    session = create_debug_session(
        clip_id="sampleuuid_face01_seg001",
        sample_uuid="sampleuuid",
        source_path="/dataset/source.mp4",
        output_root=tmp_path,
    )
    assert session.session_dir.is_dir()
    assert (session.session_dir / "logs").is_dir()
    assert (session.session_dir / "tracking").is_dir()
    assert (session.session_dir / "body4d").is_dir()
    assert (session.session_dir / "wan_export").is_dir()
    assert session.session_meta_path.is_file()


def test_next_revision_dir_is_monotonic(tmp_path: Path):
    session = create_debug_session(
        clip_id="sampleuuid_face01_seg001",
        sample_uuid="sampleuuid",
        source_path="/dataset/source.mp4",
        output_root=tmp_path,
    )
    first = session.next_revision_dir("tracking")
    second = session.next_revision_dir("tracking")
    assert first.name == "run_v001"
    assert second.name == "run_v002"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session_artifacts.py -v`
Expected: FAIL because session/artifact modules do not exist.

- [ ] **Step 3: Implement the minimal session and artifact classes**

```python
# E:/Project/Sam4dGradioDebug/debug_core/state.py
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
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/artifacts.py
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
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/session.py
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session_artifacts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/state.py debug_core/session.py debug_core/artifacts.py tests/test_session_artifacts.py
git commit -m "feat: add debug session and artifact layout"
```

---

### Task 4: Load and Validate Clip Package Input

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/load_clip_package.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/test_load_clip_package.py`

- [ ] **Step 1: Write the failing clip-package loader tests**

```python
import json
from pathlib import Path

import cv2
import numpy as np

from debug_core.stages.load_clip_package import load_clip_package


def test_load_clip_package_reads_required_metadata(tmp_path: Path):
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    writer = cv2.VideoWriter(str(clip_dir / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        writer.release()
    (clip_dir / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": "sampleuuid_face01_seg001",
                "sample_uuid": "sampleuuid",
                "source_path": "/dataset/source.mp4",
                "fps": 2.0,
                "frame_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        json.dumps(
            {
                "records": [
                    {"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "landmarks": [[1, 1], [2, 2]], "score": 0.9},
                    {"frame_index_in_clip": 1, "bbox_xyxy": [1, 1, 4, 4], "landmarks": [[1, 1], [2, 2]], "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )

    context = load_clip_package(clip_dir)

    assert context["clip_id"] == "sampleuuid_face01_seg001"
    assert context["sample_uuid"] == "sampleuuid"
    assert context["frame_count"] == 2
    assert len(context["clip_track_records"]) == 2


def test_load_clip_package_rejects_missing_track_json(tmp_path: Path):
    clip_dir = tmp_path / "broken_clip"
    clip_dir.mkdir()
    (clip_dir / "clip.mp4").write_bytes(b"")
    (clip_dir / "meta.json").write_text("{}", encoding="utf-8")

    try:
        load_clip_package(clip_dir)
    except FileNotFoundError as exc:
        assert "track.json" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_clip_package.py -v`
Expected: FAIL because the loader stage does not exist.

- [ ] **Step 3: Implement the clip-package loader**

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/load_clip_package.py
from __future__ import annotations

import json
from pathlib import Path

import cv2


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _count_video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open clip video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"Unable to determine frame count for clip video: {path}")
    return frame_count


def load_clip_package(path: str | Path) -> dict:
    clip_dir = Path(path).resolve()
    clip_video = clip_dir / "clip.mp4"
    meta_path = clip_dir / "meta.json"
    track_path = clip_dir / "track.json"
    for required_path in (clip_video, meta_path, track_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required clip package file: {required_path.name}")

    meta = _read_json(meta_path)
    track = _read_json(track_path)
    records = list(track.get("records") or [])
    if not records:
        raise ValueError("Clip package track.json has no records")

    frame_count = int(meta.get("frame_count") or 0) or _count_video_frames(clip_video)
    fps = float(meta.get("fps") or 0.0)
    return {
        "clip_dir": str(clip_dir),
        "clip_id": str(meta.get("clip_id") or clip_dir.name),
        "sample_uuid": str(meta.get("sample_uuid") or ""),
        "source_path": str(meta.get("source_path") or clip_video),
        "clip_video_path": str(clip_video),
        "frame_count": frame_count,
        "fps": fps,
        "clip_track_records": records,
        "frame_stems": [f"{index:08d}" for index in range(frame_count)],
        "meta": meta,
        "track": track,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_clip_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/stages/load_clip_package.py tests/test_load_clip_package.py
git commit -m "feat: add clip package loading stage"
```

---

### Task 5: Implement Automatic Target Proposal and Manual Override Models

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/select_target.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/test_select_target.py`

- [ ] **Step 1: Write the failing target-selection tests**

```python
from debug_core.stages.select_target import apply_manual_selection, propose_target_selection


def test_propose_target_selection_uses_first_clip_track_record():
    sample_context = {
        "frame_stems": ["00000000", "00000001"],
        "clip_track_records": [
            {"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "landmarks": [[1, 1], [2, 2]], "score": 0.9},
            {"frame_index_in_clip": 1, "bbox_xyxy": [2, 2, 5, 5], "landmarks": [[2, 2], [3, 3]], "score": 0.8},
        ],
    }
    proposal = propose_target_selection(sample_context)
    assert proposal["track_id"] == 1
    assert proposal["start_frame_idx"] == 0
    assert proposal["bbox_xyxy"] == [1, 1, 4, 4]


def test_apply_manual_selection_overrides_auto_proposal():
    proposal = {
        "track_id": 1,
        "start_frame_idx": 0,
        "bbox_xyxy": [1, 1, 4, 4],
        "source": "auto_face_guided_binding",
    }
    selection = apply_manual_selection(
        proposal,
        track_id=3,
        start_frame_idx=12,
        bbox_xyxy=[10, 12, 50, 60],
    )
    assert selection["track_id"] == 3
    assert selection["start_frame_idx"] == 12
    assert selection["bbox_xyxy"] == [10, 12, 50, 60]
    assert selection["edited_from_auto_proposal"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_select_target.py -v`
Expected: FAIL because the target-selection stage does not exist.

- [ ] **Step 3: Implement proposal and manual override helpers**

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/select_target.py
from __future__ import annotations


def propose_target_selection(sample_context: dict) -> dict:
    records = list(sample_context.get("clip_track_records") or [])
    if not records:
        raise ValueError("No clip track records available for target selection")
    first = dict(records[0])
    return {
        "track_id": 1,
        "start_frame_idx": int(first.get("frame_index_in_clip", 0) or 0),
        "bbox_xyxy": [int(value) for value in list(first.get("bbox_xyxy") or [])[:4]],
        "frame_stem": str(sample_context.get("frame_stems", ["00000000"])[int(first.get("frame_index_in_clip", 0) or 0)]),
        "source": "auto_face_guided_binding",
        "confidence": float(first.get("score", 1.0) or 1.0),
        "candidate_targets": [],
    }


def apply_manual_selection(proposal: dict, *, track_id: int, start_frame_idx: int, bbox_xyxy: list[int]) -> dict:
    return {
        "mode": "manual_override",
        "track_id": int(track_id),
        "start_frame_idx": int(start_frame_idx),
        "bbox_xyxy": [int(value) for value in bbox_xyxy[:4]],
        "frame_stem": f"{int(start_frame_idx):08d}",
        "edited_from_auto_proposal": True,
        "source_proposal": dict(proposal),
        "notes": "",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_select_target.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/stages/select_target.py tests/test_select_target.py
git commit -m "feat: add target proposal and manual override helpers"
```

---

### Task 6: Add Controller for Session Lifecycle and Stage State

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/controllers/debug_pipeline_controller.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/test_controller.py`

- [ ] **Step 1: Write the failing controller tests**

```python
from pathlib import Path

from debug_core.controllers.debug_pipeline_controller import DebugPipelineController


def test_controller_loads_sample_and_stores_auto_proposal(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    clip_context = {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }
    session = controller.bootstrap_session(clip_context)
    assert session["sample_context"]["clip_id"] == "sampleuuid_face01_seg001"
    assert session["auto_selection_proposal"]["bbox_xyxy"] == [1, 1, 4, 4]


def test_controller_apply_manual_selection_replaces_effective_selection(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    clip_context = {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }
    controller.bootstrap_session(clip_context)
    selection = controller.set_manual_selection(track_id=5, start_frame_idx=1, bbox_xyxy=[3, 3, 7, 7])
    assert selection["track_id"] == 5
    assert controller.session["effective_selection"]["track_id"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controller.py -v`
Expected: FAIL because the controller does not exist.

- [ ] **Step 3: Implement the minimal controller**

```python
# E:/Project/Sam4dGradioDebug/debug_core/controllers/debug_pipeline_controller.py
from __future__ import annotations

from pathlib import Path

from debug_core.session import create_debug_session
from debug_core.stages.select_target import apply_manual_selection, propose_target_selection


class DebugPipelineController:
    def __init__(self, *, output_root: str | Path):
        self.output_root = Path(output_root)
        self.session: dict | None = None

    def bootstrap_session(self, sample_context: dict) -> dict:
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
            "auto_selection_proposal": proposal,
            "manual_selection": None,
            "effective_selection": proposal,
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
        return selection
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/controllers/debug_pipeline_controller.py tests/test_controller.py
git commit -m "feat: add debug pipeline controller"
```

---

### Task 7: Add Refined Runtime Adapter and Tracking Stage Contract

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/adapters/refined_runtime_adapter.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/run_tracking.py`
- Modify: `E:/Project/Sam4dGradioDebug/tests/test_controller.py`

- [ ] **Step 1: Write the failing tracking-stage controller test**

```python
from pathlib import Path

from debug_core.controllers.debug_pipeline_controller import DebugPipelineController


def test_controller_run_tracking_creates_new_revision_and_records_paths(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    clip_context = {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }
    controller.bootstrap_session(clip_context)
    result = controller.run_tracking_stage()
    assert result["run_dir"].name == "run_v001"
    assert result["selection_used"]["bbox_xyxy"] == [1, 1, 4, 4]
    assert Path(result["debug_metrics_dir"]).is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_controller.py::test_controller_run_tracking_creates_new_revision_and_records_paths -v`
Expected: FAIL because `run_tracking_stage()` does not exist.

- [ ] **Step 3: Implement adapter shell and tracking stage**

```python
# E:/Project/Sam4dGradioDebug/debug_core/adapters/refined_runtime_adapter.py
from __future__ import annotations

from pathlib import Path


def run_tracking_with_override(*, sample_context: dict, selection: dict, run_dir: Path) -> dict:
    debug_metrics_dir = run_dir / "debug_metrics"
    raw_mask_dir = run_dir / "masks_raw"
    refined_mask_dir = run_dir / "masks_refined"
    overlay_dir = run_dir / "overlays"
    for path in (debug_metrics_dir, raw_mask_dir, refined_mask_dir, overlay_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": str(run_dir),
        "selection_used": dict(selection),
        "raw_mask_dir": str(raw_mask_dir),
        "refined_mask_dir": str(refined_mask_dir),
        "overlay_dir": str(overlay_dir),
        "debug_metrics_dir": str(debug_metrics_dir),
        "frame_metrics": [],
    }
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/run_tracking.py
from __future__ import annotations

from pathlib import Path

from debug_core.adapters.refined_runtime_adapter import run_tracking_with_override


def run_tracking_stage(*, runtime_session, sample_context: dict, selection: dict) -> dict:
    run_dir = runtime_session.next_revision_dir("tracking")
    return run_tracking_with_override(
        sample_context=sample_context,
        selection=selection,
        run_dir=Path(run_dir),
    )
```

Append this method to `E:/Project/Sam4dGradioDebug/debug_core/controllers/debug_pipeline_controller.py`:

```python
from debug_core.stages.run_tracking import run_tracking_stage

    def run_tracking_stage(self) -> dict:
        if self.session is None:
            raise RuntimeError("No active session")
        result = run_tracking_stage(
            runtime_session=self.session["runtime_session"],
            sample_context=self.session["sample_context"],
            selection=self.session["effective_selection"],
        )
        self.session["tracking_result"] = result
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_controller.py::test_controller_run_tracking_creates_new_revision_and_records_paths -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/adapters/refined_runtime_adapter.py debug_core/stages/run_tracking.py debug_core/controllers/debug_pipeline_controller.py tests/test_controller.py
git commit -m "feat: add tracking stage contract"
```

---

### Task 8: Add 4D and Wan Export Stage Contracts

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/adapters/wan_export_adapter.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/run_refine_4d.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/stages/run_wan_export.py`
- Modify: `E:/Project/Sam4dGradioDebug/tests/test_controller.py`

- [ ] **Step 1: Write the failing downstream-stage tests**

```python
from pathlib import Path

from debug_core.controllers.debug_pipeline_controller import DebugPipelineController


def test_controller_run_body4d_creates_body4d_revision(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    clip_context = {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }
    controller.bootstrap_session(clip_context)
    controller.run_tracking_stage()
    result = controller.run_body4d_stage()
    assert result["run_dir"].endswith("run_v001")
    assert result["rendered_video_path"].endswith("4d.mp4")


def test_controller_run_wan_export_requires_upstream_results(tmp_path: Path):
    controller = DebugPipelineController(output_root=tmp_path)
    clip_context = {
        "clip_dir": str(tmp_path / "clip"),
        "clip_id": "sampleuuid_face01_seg001",
        "sample_uuid": "sampleuuid",
        "source_path": "/dataset/source.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "clip_track_records": [{"frame_index_in_clip": 0, "bbox_xyxy": [1, 1, 4, 4], "score": 0.9}],
        "frame_stems": ["00000000", "00000001"],
    }
    controller.bootstrap_session(clip_context)
    try:
        controller.run_wan_export_stage()
    except RuntimeError as exc:
        assert "body4d" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controller.py -k "body4d or wan_export" -v`
Expected: FAIL because downstream-stage methods do not exist.

- [ ] **Step 3: Implement the downstream stage shells**

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/run_refine_4d.py
from __future__ import annotations

from pathlib import Path


def run_body4d_stage(*, runtime_session, tracking_result: dict) -> dict:
    run_dir = runtime_session.next_revision_dir("body4d")
    rendered_video_path = Path(run_dir) / "4d.mp4"
    rendered_frames_dir = Path(run_dir) / "rendered_frames"
    mesh_dir = Path(run_dir) / "mesh_4d_individual"
    focal_dir = Path(run_dir) / "focal_4d_individual"
    for path in (rendered_frames_dir, mesh_dir, focal_dir):
        path.mkdir(parents=True, exist_ok=True)
    rendered_video_path.write_bytes(b"")
    return {
        "run_dir": str(run_dir),
        "input_tracking_run": dict(tracking_result),
        "rendered_video_path": str(rendered_video_path),
        "rendered_frames_dir": str(rendered_frames_dir),
        "mesh_dir": str(mesh_dir),
        "focal_dir": str(focal_dir),
    }
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/adapters/wan_export_adapter.py
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
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/stages/run_wan_export.py
from __future__ import annotations

from pathlib import Path

from debug_core.adapters.wan_export_adapter import write_debug_export_stub


def run_wan_export_stage(*, runtime_session, tracking_result: dict, body4d_result: dict) -> dict:
    run_dir = runtime_session.next_revision_dir("wan_export")
    outputs = write_debug_export_stub(Path(run_dir))
    return {
        "run_dir": str(run_dir),
        "tracking_result": dict(tracking_result),
        "body4d_result": dict(body4d_result),
        "target_dir": outputs["target_dir"],
        "target_mp4": outputs["target.mp4"],
        "src_pose_mp4": outputs["src_pose.mp4"],
        "src_face_mp4": outputs["src_face.mp4"],
        "src_bg_mp4": outputs["src_bg.mp4"],
        "src_mask_mp4": outputs["src_mask.mp4"],
        "src_mask_detail_mp4": outputs["src_mask_detail.mp4"],
        "src_ref_png": outputs["src_ref.png"],
        "meta_json": outputs["meta.json"],
        "pose_meta_sequence_json": outputs["pose_meta_sequence.json"],
        "smpl_sequence_json": outputs["smpl_sequence.json"],
    }
```

Append these methods to `E:/Project/Sam4dGradioDebug/debug_core/controllers/debug_pipeline_controller.py`:

```python
from debug_core.stages.run_refine_4d import run_body4d_stage
from debug_core.stages.run_wan_export import run_wan_export_stage

    def run_body4d_stage(self) -> dict:
        if self.session is None or "tracking_result" not in self.session:
            raise RuntimeError("Tracking stage must succeed before body4d")
        result = run_body4d_stage(
            runtime_session=self.session["runtime_session"],
            tracking_result=self.session["tracking_result"],
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
        )
        self.session["wan_export_result"] = result
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_controller.py -k "body4d or wan_export" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/adapters/wan_export_adapter.py debug_core/stages/run_refine_4d.py debug_core/stages/run_wan_export.py debug_core/controllers/debug_pipeline_controller.py tests/test_controller.py
git commit -m "feat: add body4d and wan export stage contracts"
```

---

### Task 9: Build the First Gradio App Shell

**Files:**
- Create: `E:/Project/Sam4dGradioDebug/debug_core/views/preview_utils.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/views/gallery_builders.py`
- Create: `E:/Project/Sam4dGradioDebug/debug_core/views/timeline_utils.py`
- Create: `E:/Project/Sam4dGradioDebug/app.py`
- Create: `E:/Project/Sam4dGradioDebug/tests/test_gradio_app.py`

- [ ] **Step 1: Write the failing Gradio app test**

```python
from app import build_app


def test_build_app_returns_gradio_blocks():
    demo = build_app()
    assert demo is not None
    assert demo.__class__.__name__ == "Blocks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gradio_app.py -v`
Expected: FAIL because `app.py` and `build_app()` do not exist.

- [ ] **Step 3: Implement the minimal Gradio shell**

```python
# E:/Project/Sam4dGradioDebug/debug_core/views/preview_utils.py
from __future__ import annotations

from pathlib import Path


def normalize_path_text(path: str | Path | None) -> str:
    return "" if path in {None, ""} else str(Path(path))
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/views/gallery_builders.py
from __future__ import annotations


def empty_gallery_payload() -> list:
    return []
```

```python
# E:/Project/Sam4dGradioDebug/debug_core/views/timeline_utils.py
from __future__ import annotations


def slider_bounds(frame_count: int) -> tuple[int, int]:
    upper = max(0, int(frame_count) - 1)
    return 0, upper
```

```python
# E:/Project/Sam4dGradioDebug/app.py
from __future__ import annotations

from pathlib import Path

import gradio as gr

from debug_core.controllers.debug_pipeline_controller import DebugPipelineController


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
                current_frame = gr.Image(label="Current Frame Preview", interactive=False)
        demo.controller = controller
        demo.debug_components = {
            "clip_dir": clip_dir,
            "config_path": config_path,
            "output_root": output_root,
            "preserve_all": preserve_all,
            "load_button": load_button,
            "sample_info": sample_info,
            "current_frame": current_frame,
        }
    return demo


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gradio_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug_core/views/preview_utils.py debug_core/views/gallery_builders.py debug_core/views/timeline_utils.py app.py tests/test_gradio_app.py
git commit -m "feat: add initial gradio debug shell"
```

---

### Task 10: Wire the UI to Session Bootstrap and Manual Correction

**Files:**
- Modify: `E:/Project/Sam4dGradioDebug/app.py`
- Modify: `E:/Project/Sam4dGradioDebug/tests/test_gradio_app.py`

- [ ] **Step 1: Write the failing UI wiring test**

```python
from pathlib import Path

from app import _load_sample_for_ui


def test_load_sample_for_ui_returns_clip_metadata(tmp_path: Path):
    clip_dir = tmp_path / "sampleuuid_face01_seg001"
    clip_dir.mkdir()
    (clip_dir / "clip.mp4").write_bytes(b"")
    (clip_dir / "meta.json").write_text(
        '{"clip_id":"sampleuuid_face01_seg001","sample_uuid":"sampleuuid","source_path":"/dataset/source.mp4","fps":2.0,"frame_count":1}',
        encoding="utf-8",
    )
    (clip_dir / "track.json").write_text(
        '{"records":[{"frame_index_in_clip":0,"bbox_xyxy":[1,1,4,4],"landmarks":[[1,1],[2,2]],"score":0.9}]}',
        encoding="utf-8",
    )

    sample_info, proposal = _load_sample_for_ui(str(clip_dir), "configs/debug_default.yaml", str(tmp_path))

    assert sample_info["clip_id"] == "sampleuuid_face01_seg001"
    assert proposal["bbox_xyxy"] == [1, 1, 4, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gradio_app.py -k "load_sample_for_ui" -v`
Expected: FAIL because the UI load helper does not exist.

- [ ] **Step 3: Add load helper and UI callback wiring**

Append this to `E:/Project/Sam4dGradioDebug/app.py`:

```python
from debug_core.config import load_debug_config
from debug_core.stages.load_clip_package import load_clip_package


APP_CONTROLLER: DebugPipelineController | None = None


def _get_controller(output_root: str) -> DebugPipelineController:
    global APP_CONTROLLER
    if APP_CONTROLLER is None or APP_CONTROLLER.output_root != Path(output_root):
        APP_CONTROLLER = DebugPipelineController(output_root=Path(output_root))
    return APP_CONTROLLER


def _load_sample_for_ui(clip_dir: str, config_path: str, output_root: str):
    del load_debug_config(config_path)
    controller = _get_controller(output_root)
    sample_context = load_clip_package(clip_dir)
    session = controller.bootstrap_session(sample_context)
    return session["sample_context"], session["auto_selection_proposal"]
```

Then wire the button in `build_app()`:

```python
proposal_info = gr.JSON(label="Target Selection Proposal")

load_button.click(
    fn=_load_sample_for_ui,
    inputs=[clip_dir, config_path, output_root],
    outputs=[sample_info, proposal_info],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gradio_app.py -k "load_sample_for_ui" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_gradio_app.py
git commit -m "feat: wire sample loading into gradio app"
```

---

### Task 11: Add Stage Execution Buttons and Revision-Aware Summaries

**Files:**
- Modify: `E:/Project/Sam4dGradioDebug/app.py`
- Modify: `E:/Project/Sam4dGradioDebug/tests/test_gradio_app.py`

- [ ] **Step 1: Write the failing stage-button tests**

```python
from app import build_app


def test_build_app_exposes_stage_control_components():
    demo = build_app()
    keys = set(demo.debug_components.keys())
    assert "run_tracking_button" in keys
    assert "run_body4d_button" in keys
    assert "run_wan_export_button" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gradio_app.py::test_build_app_exposes_stage_control_components -v`
Expected: FAIL because stage buttons are not present.

- [ ] **Step 3: Add stage execution controls**

Append this to `build_app()` in `E:/Project/Sam4dGradioDebug/app.py`:

```python
with gr.Accordion("Tracking / Mask", open=True):
    run_tracking_button = gr.Button("Run Tracking / Mask")
    tracking_result = gr.JSON(label="Tracking Result")

with gr.Accordion("4D", open=False):
    run_body4d_button = gr.Button("Run 4D")
    body4d_result = gr.JSON(label="Body4D Result")

with gr.Accordion("Wan Export", open=False):
    run_wan_export_button = gr.Button("Run Wan Export")
    wan_export_result = gr.JSON(label="Wan Export Result")
```

Add stage callbacks:

```python
def _run_tracking_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_tracking_stage()


def _run_body4d_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_body4d_stage()


def _run_wan_export_for_ui(output_root: str):
    controller = _get_controller(output_root)
    return controller.run_wan_export_stage()
```

Wire button clicks:

```python
run_tracking_button.click(fn=_run_tracking_for_ui, inputs=[output_root], outputs=[tracking_result])
run_body4d_button.click(fn=_run_body4d_for_ui, inputs=[output_root], outputs=[body4d_result])
run_wan_export_button.click(fn=_run_wan_export_for_ui, inputs=[output_root], outputs=[wan_export_result])
```

Register them in `demo.debug_components`:

```python
"run_tracking_button": run_tracking_button,
"run_body4d_button": run_body4d_button,
"run_wan_export_button": run_wan_export_button,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gradio_app.py::test_build_app_exposes_stage_control_components -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_gradio_app.py
git commit -m "feat: add staged execution controls"
```

---

### Task 12: Final Verification and Minimal Docs Update

**Files:**
- Modify: `E:/Project/Sam4dGradioDebug/README.md`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_config.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_load_clip_package.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_select_target.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_session_artifacts.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_controller.py`
- Test: `E:/Project/Sam4dGradioDebug/tests/test_gradio_app.py`

- [ ] **Step 1: Add minimal usage instructions to README**

```markdown
# Sam4dGradioDebug

Standalone Gradio debugging UI for single-sample clip-package correction built around the refined SAM4D + Wan export workflow.

## Current Scope

- single sample only
- clip package input only
- manual staged execution
- target person / start frame / initial bbox correction

## Run

```bash
python -m pip install -r requirements.txt
python app.py
```
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests -v`
Expected: PASS

- [ ] **Step 3: Run the app import smoke check**

Run: `python -c "from app import build_app; demo = build_app(); print(type(demo).__name__)"`
Expected: prints `Blocks`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add debug app usage notes"
```

---

## Spec Coverage Self-Check

- Clip-package-only scope: covered by Tasks 4, 9, and 10.
- Standalone repository structure: covered by Tasks 1 and 2.
- Session/revision model: covered by Task 3 and stage tasks.
- Automatic proposal plus manual override: covered by Tasks 5, 6, and 10.
- Staged execution for tracking, 4D, Wan export: covered by Tasks 7, 8, and 11.
- Preserve intermediates by default: covered by Tasks 2 and 3.
- Error-safe staged progression: covered by Tasks 6, 7, and 8.
- Gradio shell and staged controls: covered by Tasks 9, 10, and 11.
- Minimal documentation and verification: covered by Task 12.

No uncovered spec requirements remain for the first implementation slice.
