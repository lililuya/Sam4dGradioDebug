from __future__ import annotations


def slider_bounds(frame_count: int) -> tuple[int, int]:
    upper = max(0, int(frame_count) - 1)
    return 0, upper
