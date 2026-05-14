# Sam4dGradioDebug Design

## Summary

`Sam4dGradioDebug` is a standalone Gradio-based debugging repository for **single-sample, clip-package-first correction workflows** built around the current refined SAM4D + Wan export pipeline.

The repository is intentionally **not** a batch-processing system and **not** a production-cleanup path. It is a visual correction tool for difficult samples where a user needs to:

- load one `clip package`
- inspect the automatically proposed target selection
- manually override target person, start frame, and initial box
- rerun downstream stages independently
- preserve every important intermediate artifact
- compare retries/revisions without overwriting history

The first release only supports:

- **single sample**
- **clip package input only**
- **manual staged execution**
- **lightweight human correction**
- **full pipeline coverage**
  - target selection
  - tracking/mask generation
  - 4D generation
  - Wan export


## Goals

The first version must:

1. Load a valid clip package directory and normalize it into a single debug session.
2. Show automatic target-selection proposals derived from clip metadata and existing refined logic.
3. Let the user manually override:
   - target person
   - start frame
   - initial bbox
4. Run the pipeline in explicit stages rather than one opaque batch call.
5. Preserve all critical intermediate artifacts by default.
6. Keep every retry/rerun as a separate revision instead of overwriting prior results.
7. Allow the user to inspect intermediate outputs visually and structurally.
8. Surface enough logs and structured metadata to support manual diagnosis.
9. Avoid mutating original clip-package source files during debugging.


## Non-Goals

The first version explicitly does **not** attempt to solve:

- multi-sample batch processing
- queue-based concurrent jobs
- browser automation or remote orchestration
- direct editing of original clip-package files
- per-frame face bbox editing
- per-frame hand-drawn mask editing
- full database-backed history
- user accounts / permission systems
- production-optimized cleanup or disk-minimizing defaults


## Primary User Workflow

The intended workflow is:

1. User opens the Gradio app.
2. User selects a clip package directory.
3. The app creates a new debug session directory.
4. The app loads clip metadata and renders preview frames.
5. The app computes and displays an automatic target-selection proposal.
6. The user either accepts the proposal or manually adjusts:
   - target identity
   - start frame
   - initial bbox
7. The user runs tracking/mask generation.
8. The user inspects frame-wise masks, overlays, and metrics.
9. If needed, the user adjusts target selection or stage parameters and reruns tracking as a new revision.
10. When tracking looks correct, the user runs 4D generation.
11. When 4D looks correct, the user runs Wan export.
12. The user reviews final outputs plus preserved intermediates and can export a debugging snapshot if needed.


## Input Contract

Version 1 supports only a **clip package directory**.

The repository assumes the input directory includes at minimum:

- `clip.mp4`
- `meta.json`
- `track.json`

Relevant clip metadata must include, directly or indirectly:

- `clip_id`
- `sample_uuid`
- `source_path`
- `fps`
- `frame_count`

Relevant track metadata must include:

- `records`

Each record is expected to expose fields such as:

- `frame_index_in_clip`
- `bbox_xyxy`
- `landmarks`
- `score`

The repository should treat the clip package as **read-only input**.


## Repository Architecture

The repository should be structured as a dedicated debug tool rather than a repackaged batch runner.

Recommended structure:

```text
Sam4dGradioDebug/
|- app.py
|- requirements.txt
|- README.md
|- configs/
|  `- debug_default.yaml
|- docs/
|  `- superpowers/
|     `- specs/
|- debug_core/
|  |- config.py
|  |- state.py
|  |- session.py
|  |- artifacts.py
|  |- controllers/
|  |  `- debug_pipeline_controller.py
|  |- stages/
|  |  |- load_clip_package.py
|  |  |- select_target.py
|  |  |- run_tracking.py
|  |  |- run_refine_4d.py
|  |  `- run_wan_export.py
|  |- views/
|  |  |- preview_utils.py
|  |  |- gallery_builders.py
|  |  `- timeline_utils.py
|  `- adapters/
|     |- refined_runtime_adapter.py
|     `- wan_export_adapter.py
`- outputs_debug/
```

### Architectural Principles

- `app.py` owns Gradio layout and event wiring only.
- `debug_core/controllers/debug_pipeline_controller.py` owns session orchestration.
- `debug_core/stages/` own stage-level business logic.
- `debug_core/adapters/` bridge reused logic from the current main repository into the standalone debug repository.
- `outputs_debug/` stores persistent session and revision artifacts.


## Execution Model

The repository should use a strong explicit session object rather than hidden global process state.

Each loaded sample creates a `DebugSession` that contains:

- clip package identity
- session output root
- current effective config snapshot
- automatic selection proposal
- active manual selection override
- stage statuses
- current active revision for each stage
- artifact registry for generated outputs
- error records and logs

Conceptual model:

```python
DebugSession = {
    "session_id": ...,
    "clip_dir": ...,
    "clip_id": ...,
    "sample_uuid": ...,
    "source_path": ...,
    "frame_count": ...,
    "fps": ...,
    "clip_track_records": [...],
    "config_snapshot_path": ...,
    "auto_selection_proposal": {...} | None,
    "manual_selection": {...} | None,
    "stage_status": {...},
    "active_revisions": {...},
    "artifacts": {...},
    "errors": [...],
}
```

The first version should support exactly **one active session at a time** within the UI.


## Gradio UI Layout

The UI should prioritize visual correction over form-heavy configuration.

Recommended layout:

### 1. Sample Load Panel

Controls:

- clip package path
- config path
- session output root
- preserve all intermediates toggle
- load sample button

Displays after load:

- clip id
- sample uuid
- source path
- fps
- frame count
- track-record count

### 2. Target Selection and Manual Correction Panel

Controls:

- frame slider
- candidate target selector
- start-frame numeric input
- bbox numeric inputs (`x1`, `y1`, `x2`, `y2`)
- apply manual override button
- reset to automatic proposal button

Preview:

- current frame
- clip face bbox overlays
- auto-selected proposal overlay
- manual bbox overlay
- optional candidate body detections

Interaction mode:

- default visual clicking / box drawing
- numeric box refinement remains available for precision correction

### 3. Tracking / Mask Stage Panel

Controls:

- run tracking/mask button
- rerun current stage button
- refine enable toggle
- reprompt diagnostics toggle
- chunk size
- initial search frames

Outputs:

- raw mask gallery
- refined mask gallery
- overlay video or gallery
- per-frame metrics table
- stage logs

### 4. 4D Stage Panel

Controls:

- run 4D button
- rerun 4D button
- rendered video toggles
- rendered frame preservation toggles

Outputs:

- `4d.mp4`
- rendered frame gallery
- stage logs
- summary statistics

### 5. Wan Export Panel

Controls:

- run Wan export button
- rerun Wan export button
- export-specific toggles:
  - save pose-meta json
  - save SMPL sequence
  - copy rendered 4D to targets
  - cleanup workdir
  - face expand / face gap / min valid face ratio

Outputs:

- `target.mp4`
- `src_pose.mp4`
- `src_face.mp4`
- `src_bg.mp4`
- `src_mask.mp4`
- `src_mask_detail.mp4`
- `src_ref.png`
- `meta.json`
- `pose_meta_sequence.json`
- `smpl_sequence.json`

### 6. Logs and Artifact Browser

Displays:

- live stage logs
- current session directory summary
- latest error snapshot
- active revision list
- artifact path browser
- optional debug-snapshot export action


## Stage Decomposition

The full pipeline should be decomposed into explicit stage functions.

### Stage 0: `load_clip_package`

Responsibilities:

- validate clip package presence
- read `clip.mp4`, `meta.json`, `track.json`
- normalize metadata into `sample_context`
- create the debug session
- cache basic preview information

Output:

```python
sample_context = {
    "clip_dir": ...,
    "clip_id": ...,
    "sample_uuid": ...,
    "source_path": ...,
    "frame_count": ...,
    "fps": ...,
    "clip_track_records": [...],
    "frame_stems": [...],
}
```

### Stage 1: `propose_target_selection`

Responsibilities:

- compute automatic target proposal
- reuse clip metadata and current face-guided selection logic where applicable
- produce candidate proposals for UI review

Output:

```python
auto_selection_proposal = {
    "track_id": ...,
    "start_frame_idx": ...,
    "bbox_xyxy": [...],
    "source": "auto_face_guided_binding",
    "confidence": ...,
    "candidate_targets": [...],
}
```

### Stage 2: `apply_manual_target_selection`

Responsibilities:

- accept user overrides
- convert them into a stable structured selection snapshot
- make the override the effective downstream target-selection input

Output:

```python
manual_selection = {
    "mode": "manual_override",
    "track_id": ...,
    "start_frame_idx": ...,
    "bbox_xyxy": [...],
    "frame_stem": ...,
    "edited_from_auto_proposal": True,
    "notes": "",
}
```

This override must be stored inside the session and written to JSON.

### Stage 3: `run_tracking_and_masks`

Responsibilities:

- initialize tracking from either:
  - automatic proposal
  - manual override
- run tracking and mask generation
- optionally refine masks
- preserve frame-level debug metrics

Important rule:

If the user provides a manual start frame and bbox, the stage must **not silently re-run automatic target guessing**.

Output:

```python
tracking_run_result = {
    "run_dir": ...,
    "selection_used": ...,
    "raw_mask_dir": ...,
    "refined_mask_dir": ...,
    "overlay_dir": ...,
    "debug_metrics_dir": ...,
    "frame_metrics": ...,
}
```

### Stage 4: `run_body4d`

Responsibilities:

- consume a specific tracking revision
- generate 4D outputs from that revision
- preserve rendered artifacts

Output:

```python
body4d_run_result = {
    "run_dir": ...,
    "input_tracking_run": ...,
    "rendered_video_path": ...,
    "rendered_frames_dir": ...,
    "mesh_dir": ...,
    "focal_dir": ...,
}
```

### Stage 5: `run_wan_export`

Responsibilities:

- consume a specific tracking revision and a specific 4D revision
- perform final Wan export
- preserve all requested export artifacts

Output:

```python
wan_export_run_result = {
    "run_dir": ...,
    "target_dir": ...,
    "target_mp4": ...,
    "src_pose_mp4": ...,
    "src_face_mp4": ...,
    "src_bg_mp4": ...,
    "src_mask_mp4": ...,
    "src_mask_detail_mp4": ...,
    "src_ref_png": ...,
    "meta_json": ...,
    "pose_meta_sequence_json": ...,
    "smpl_sequence_json": ...,
}
```


## Manual Correction Scope

Version 1 should support exactly three correction axes:

1. target person override
2. start-frame override
3. initial bbox override

These changes must:

- affect downstream stage execution
- be stored as structured session-local state
- never mutate the original clip package

Version 1 should **not** support per-frame manual face-box edits.


## Session and Revision Storage

The repository should favor preservation and reproducibility over compactness.

Suggested structure:

```text
outputs_debug/
`- <clip_id>/
   `- session_<timestamp>/
      |- session_meta.json
      |- config_snapshot.yaml
      |- user_edits.json
      |- logs/
      |- load/
      |- target_selection/
      |- tracking/
      |  |- run_v001/
      |  `- run_v002/
      |- body4d/
      |  |- run_v001/
      |  `- run_v002/
      `- wan_export/
         |- run_v001/
         `- run_v002/
```

### Revision Rules

- each meaningful rerun creates a new revision
- stage reruns must not overwrite historical successful runs by default
- the UI tracks one active revision per stage
- the user can inspect earlier revisions


## Default Debug Configuration

The repository should use a dedicated debug-first config, not the current production-like defaults.

Suggested debug defaults:

```yaml
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

debug:
  save_metrics: true

refine:
  enable: true

reprompt:
  enable: true
```

### Config Behavior Principles

- preserve by default
- clean up only by explicit user choice
- separate common controls from advanced storage controls
- mark destructive toggles clearly in the UI


## Error Handling

This repository should optimize for diagnosis rather than silent continuation.

### Rules

1. Stage failure must not destroy the session.
2. Successful upstream results remain available after downstream failure.
3. Each stage failure writes a structured error snapshot.
4. The UI shows both:
   - concise summary
   - full debug context
5. Each stage validates required inputs before heavy execution begins.

Example structured error snapshot:

```json
{
  "stage": "run_tracking_and_masks",
  "status": "failed",
  "error_type": "RuntimeError",
  "error_message": "...",
  "traceback_path": "...",
  "config_snapshot_path": "...",
  "selection_snapshot_path": "...",
  "recorded_at": "..."
}
```


## Logging and Artifact Inspection

The UI should expose:

- live textual logs
- stage-specific log files
- session metadata summary
- active artifact paths
- recent error snapshots
- optional debug snapshot export

The first version should include a debug-snapshot export action that bundles:

- current session metadata
- config snapshot
- manual selection snapshot
- active-stage outputs
- latest logs
- latest error snapshot if present


## Testing Strategy

Version 1 requires tests at three layers.

### 1. Core Logic Tests

Test pure logic in `debug_core/`:

- clip package normalization
- proposal generation
- manual override application
- revision naming and selection
- artifact path resolution
- dependency validation

### 2. Controller / Orchestration Tests

Test `debug_pipeline_controller.py`:

- stage ordering
- state propagation
- manual selection override propagation
- rerun revision creation
- failure retention behavior

### 3. Lightweight UI Integration Tests

Test:

- app startup
- component initialization
- stage-button callback wiring
- structured output contracts

Version 1 does not need heavyweight browser automation.


## First-Version Success Criteria

Version 1 is considered successful if all of the following are true:

1. A valid clip package loads successfully.
2. Automatic target proposal is visible in the UI.
3. The user can manually set target person, start frame, and initial bbox.
4. Tracking/mask can run independently from the UI.
5. Tracking outputs can be inspected frame-wise.
6. 4D can run from a chosen tracking revision.
7. Wan export can run from the active valid upstream revisions.
8. Session and revision artifacts are preserved by default.
9. Stage failure does not destroy prior results or the active session.


## Implementation Boundaries for the First Build

The first build should stop once the following are in place:

- standalone repository scaffold
- debug config system
- session model
- clip package loading
- target proposal + manual override
- tracking/mask staged execution
- 4D staged execution
- Wan export staged execution
- revision-preserving artifact management
- basic log/error UI

The first build should intentionally defer:

- batch mode
- multi-user concurrency
- per-frame manual face edits
- manual mask painting
- source clip-package mutation
- database history


## Recommended Implementation Order

To minimize integration risk:

1. scaffold standalone repository
2. add config + session model
3. implement clip-package load stage
4. implement target proposal and manual override
5. implement tracking/mask stage
6. implement artifact browsing for tracking outputs
7. implement 4D stage
8. implement Wan export stage
9. polish logs, revision handling, and debug snapshot export


## Open Implementation Notes

- The repository should reuse stable logic from the main SAM4D codebase wherever possible through explicit adapters rather than ad-hoc copy-paste inside UI callbacks.
- The UI should never be the source of truth for session state; persistent session objects and revision manifests should be authoritative.
- Intermediate results should be represented both visually and structurally so that debugging is not trapped inside UI-only state.


## Final Recommendation

`Sam4dGradioDebug` should be treated as a **dedicated visual correction tool** for difficult clip-package samples, not as another wrapper around the batch runner.

Its core identity is:

- single-sample
- clip-package-first
- staged
- correction-oriented
- evidence-preserving

That positioning should guide both repository structure and future scope decisions.
