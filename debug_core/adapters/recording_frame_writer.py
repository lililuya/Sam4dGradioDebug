from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _to_pickle_safe(value: Any):
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {str(key): _to_pickle_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_pickle_safe(item) for item in value]
    return value


class RecordedFrameWriter:
    def __init__(self, *, output_path: str | Path):
        self.output_path = Path(output_path)
        self._records: list[dict] = []

    def __call__(self, image_path, mask_output, id_current):
        frame_stem = os.path.splitext(os.path.basename(str(image_path)))[0]
        self._records.append(
            {
                "frame_stem": frame_stem,
                "image_path": str(image_path),
                "mask_output": _to_pickle_safe(mask_output),
                "id_current": [int(value) for value in list(id_current or [])],
            }
        )
        return None

    def finalize(self) -> list[Path]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("wb") as handle:
            pickle.dump(self._records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [self.output_path]


def load_recorded_frame_outputs(path: str | Path) -> list[dict]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return list(payload or [])
