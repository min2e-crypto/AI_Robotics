"""share_model/weights.py: 가중치 기록 파일과 내려받기 (ADR-019). GitHub에는 접속하지 않는다."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weights", ROOT / "share_model" / "weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


def make_models(root: Path) -> Path:
    for rel, data in (("1_vehicle/best.pt", b"vehicle"), ("2_state/best.pt", b"state")):
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_bytes(data)
    return root


def test_asset_names_are_unique_per_folder():
    assert weights.asset_name("1_vehicle/best.pt") == "1_vehicle-best.pt"
    assert weights.asset_name("2_state/best.pt") != weights.asset_name("3_fire/best.pt")


def test_dry_run_publish_lists_files_without_writing(tmp_path):
    models = make_models(tmp_path / "models")
    manifest = weights.publish("models-v1", models, models / "weights.json", "team/repo", dry_run=True)
    assert (manifest["repo"], manifest["release"]) == ("team/repo", "models-v1")
    assert set(manifest["files"]) == {"1_vehicle/best.pt", "2_state/best.pt"}
    assert not (models / "weights.json").exists()


def test_status_tells_missing_and_changed(tmp_path):
    models = make_models(tmp_path / "models")
    manifest = {"release": "models-v1", "files": weights.scan(models)}
    (models / "1_vehicle/best.pt").write_bytes(b"retrained")
    (models / "2_state/best.pt").unlink()
    assert weights.status(manifest, models) == {"1_vehicle/best.pt": "changed", "2_state/best.pt": "missing"}


def test_fetch_downloads_only_what_differs(tmp_path):
    remote = make_models(tmp_path / "remote")
    manifest = {"repo": "team/repo", "release": "models-v1", "files": weights.scan(remote)}
    models = make_models(tmp_path / "models")
    (models / "2_state/best.pt").write_bytes(b"old")
    asked = []

    def download(repo, tag, asset, dest):
        asked.append((repo, tag, asset))
        (dest / asset).write_bytes(b"state")

    assert weights.fetch(manifest, models, download) == ["2_state/best.pt"]
    assert asked == [("team/repo", "models-v1", "2_state-best.pt")]
    assert set(weights.status(manifest, models).values()) == {"ok"}


def test_fetch_keeps_old_file_when_hash_differs(tmp_path):
    models = make_models(tmp_path / "models")
    manifest = {"repo": "team/repo", "release": "models-v1", "files": {"2_state/best.pt": "0" * 64}}
    with pytest.raises(SystemExit):
        weights.fetch(manifest, models, lambda repo, tag, asset, dest: (dest / asset).write_bytes(b"broken"))
    assert (models / "2_state/best.pt").read_bytes() == b"state"


def test_repo_is_read_from_https_and_ssh_remotes():
    assert weights.parse_repo("https://github.com/team/repo.git\n") == "team/repo"
    assert weights.parse_repo("git@github.com:team/repo.git") == "team/repo"
    assert weights.parse_repo("https://github.com/team/repo") == "team/repo"
    with pytest.raises(SystemExit):
        weights.parse_repo("https://example.com/team/repo.git")


def test_release_url_needs_no_login():
    assert (weights.release_url("team/repo", "models-v1", "2_state-best.pt")
            == "https://github.com/team/repo/releases/download/models-v1/2_state-best.pt")
