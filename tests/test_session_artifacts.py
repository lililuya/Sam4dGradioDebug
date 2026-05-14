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
