"""push / pull against an in-memory stand-in for the Hugging Face hub (no network)."""

import shutil
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


def FakeResponse():
    import httpx
    return httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))

from nsg_sim.cli import main


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    store = tmp_path / "hub"

    class Api:
        def create_repo(self, repo_id, **kw):
            (store / repo_id).mkdir(parents=True, exist_ok=True)

        def upload_folder(self, repo_id, folder_path, allow_patterns, **kw):
            for name in allow_patterns:
                shutil.copy(Path(folder_path) / name, store / repo_id / name)

    def download(repo_id, filename, **kw):
        path = store / repo_id / filename
        if not path.exists():
            cls = RepositoryNotFoundError if not path.parent.exists() else EntryNotFoundError
            raise cls("missing", response=FakeResponse())
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return store


def test_push_then_pull(built, fake_hub, data_dir, tmp_path, monkeypatch):
    name = "triage_sampled_test"
    assert main(["push", name, "--org", "test"]) == 0
    assert (fake_hub / "test" / name / "records.jsonl").exists()
    assert main(["push", name, "--org", "test"]) == 0  # identical: no-op

    other = tmp_path / "other_data"
    monkeypatch.setenv("NSG_DATA_DIR", str(other))
    assert main(["pull", name, "--org", "test"]) == 0
    assert (other / name / "records.jsonl").read_bytes() == (data_dir / name / "records.jsonl").read_bytes()


def test_push_refuses_different_content(built, fake_hub):
    name = "triage_sampled_test"
    main(["push", name, "--org", "test2"])
    manifest = fake_hub / "test2" / name / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"sha256": "', '"sha256": "0'))
    with pytest.raises(SystemExit):
        main(["push", name, "--org", "test2"])
