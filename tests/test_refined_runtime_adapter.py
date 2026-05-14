from pathlib import Path

from debug_core.adapters.refined_runtime_adapter import (
    build_compat_sample_summaries,
    resolve_upstream_runtime_paths,
)
from debug_core.config import load_debug_config


def test_resolve_upstream_runtime_paths_expands_relative_paths(tmp_path: Path):
    repo_root = tmp_path / "sam-body4d-master"
    repo_root.mkdir()
    config_dir = tmp_path / "debug_repo" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "debug.yaml"
    config_path.write_text(
        "\n".join(
            [
                "debug_mode:",
                "  preserve_all_intermediate: true",
                "runtime: {}",
                "wan_export: {}",
                "debug: {}",
                "refine: {}",
                "reprompt: {}",
                "upstream:",
                "  repo_root: ../../sam-body4d-master",
                "  refined_config_path: ../../sam-body4d-master/configs/body4d_refined.yaml",
            ]
        ),
        encoding="utf-8",
    )

    config = load_debug_config(config_path)
    paths = resolve_upstream_runtime_paths(config)

    assert paths["repo_root"] == repo_root.resolve()
    assert paths["refined_config_path"] == (repo_root / "configs" / "body4d_refined.yaml").resolve()


def test_build_compat_sample_summaries_falls_back_when_upstream_methods_are_missing():
    class LegacyLikeApp:
        def _build_sample_fps_summary(self, sample: dict) -> dict:
            return {"source_fps": float(sample["fps"]), "from_method": "fps"}

    sample = {
        "fps": 25.0,
        "frame_count": 50,
        "input_type": "clip_package",
        "clip_dir": "/dataset/clip_pkg",
    }

    summaries = build_compat_sample_summaries(LegacyLikeApp(), sample)

    assert summaries["fps_summary"]["from_method"] == "fps"
    assert summaries["bitrate_summary"]["source_bitrate"] is None
    assert summaries["clip_duration_summary"]["duration_seconds"] == 2.0


def test_build_compat_sample_summaries_falls_back_when_upstream_builder_raises():
    class LegacyLikeApp:
        def _build_sample_fps_summary(self, sample: dict) -> dict:
            return {"source_fps": float(sample["fps"]), "from_method": "fps"}

        def _build_clip_duration_summary(self, sample: dict) -> dict:
            raise AttributeError("'WanExportConfig' object has no attribute 'max_clip_len_seconds'")

    sample = {
        "fps": 25.0,
        "frame_count": 50,
        "input_type": "clip_package",
        "clip_dir": "/dataset/clip_pkg",
    }

    summaries = build_compat_sample_summaries(LegacyLikeApp(), sample)

    assert summaries["fps_summary"]["from_method"] == "fps"
    assert summaries["clip_duration_summary"]["duration_seconds"] == 2.0
    assert summaries["clip_duration_summary"]["max_clip_len_seconds"] == 0.0
