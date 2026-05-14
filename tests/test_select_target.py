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
