from pathlib import Path

from debug_core.adapters.refined_runtime_adapter import resolve_upstream_runtime_paths
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
