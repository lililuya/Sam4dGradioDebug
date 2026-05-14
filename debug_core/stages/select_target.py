from __future__ import annotations


def propose_target_selection(sample_context: dict) -> dict:
    records = list(sample_context.get("clip_track_records") or [])
    if not records:
        raise ValueError("No clip track records available for target selection")
    first = dict(records[0])
    start_frame_idx = int(first.get("frame_index_in_clip", 0) or 0)
    frame_stems = list(sample_context.get("frame_stems") or ["00000000"])
    frame_stem = frame_stems[start_frame_idx] if start_frame_idx < len(frame_stems) else f"{start_frame_idx:08d}"
    return {
        "track_id": 1,
        "start_frame_idx": start_frame_idx,
        "bbox_xyxy": [int(value) for value in list(first.get("bbox_xyxy") or [])[:4]],
        "frame_stem": str(frame_stem),
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
