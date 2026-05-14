from pathlib import Path

from debug_core.config import DebugAppConfig, load_debug_config


def test_repository_has_expected_top_level_layout():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "requirements.txt").is_file()
    assert (repo_root / "README.md").is_file()
    assert (repo_root / "debug_core").is_dir()
    assert (repo_root / "tests").is_dir()


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


def test_debug_config_exposes_upstream_repository_settings():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_debug_config(repo_root / "configs" / "debug_default.yaml")
    assert config.upstream["repo_root"] == "../../sam-body4d-master"
    assert "configs/body4d_refined" in config.upstream["refined_config_path"]
