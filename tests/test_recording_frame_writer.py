from pathlib import Path

import numpy as np

from debug_core.adapters.recording_frame_writer import (
    RecordedFrameWriter,
    load_recorded_frame_outputs,
)


def test_recorded_frame_writer_persists_replayable_records(tmp_path: Path):
    output_path = tmp_path / "frame_records.pkl"
    writer = RecordedFrameWriter(output_path=output_path)

    writer(
        str(tmp_path / "00000000.jpg"),
        [{"pred_keypoints_2d": np.array([[1.0, 2.0, 1.0]], dtype=np.float32)}],
        [4],
    )
    writer(
        str(tmp_path / "00000001.jpg"),
        [{"pred_keypoints_2d": np.array([[3.0, 4.0, 1.0]], dtype=np.float32)}],
        [4],
    )
    finalized_outputs = writer.finalize()
    assert finalized_outputs == [output_path]

    records = load_recorded_frame_outputs(output_path)

    assert len(records) == 2
    assert records[0]["frame_stem"] == "00000000"
    assert records[0]["id_current"] == [4]
    assert records[0]["mask_output"][0]["pred_keypoints_2d"].shape == (1, 3)
