"""`nsg push NAME` / `nsg pull NAME`: share built datasets through the Hugging Face hub.

Each dataset lives in its own private dataset repo `<org>/<name>` (default org from NSG_HF_ORG,
else spar-nsg-simulators) holding records.jsonl + manifest.json. Datasets are immutable here too:
push refuses when the repo already holds different content, pull refuses to replace a local
dataset with different content, and pulled files are checked against the manifest hash.
Authentication: `huggingface-cli login` or HF_TOKEN in .env.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_ORG = "spar-nsg-simulators"
FILES = ("records.jsonl", "manifest.json")


def register(sub) -> None:
    p = sub.add_parser("push", help="upload a built dataset to the Hugging Face hub")
    p.add_argument("names", nargs="+")
    p.add_argument("--org", default=None, help=f"hub organisation (default NSG_HF_ORG or {DEFAULT_ORG})")
    p.add_argument("--public", action="store_true", help="create the repo as public (default private)")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("pull", help="download a dataset from the Hugging Face hub into data/")
    p.add_argument("names", nargs="+")
    p.add_argument("--org", default=None)
    p.add_argument("--revision", default=None, help="hub commit / tag (default latest)")
    p.set_defaults(func=cmd_pull)


def _org(args) -> str:
    return args.org or os.environ.get("NSG_HF_ORG", DEFAULT_ORG)


def _remote_manifest(api, repo_id: str, revision: str | None = None) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        path = hf_hub_download(repo_id, "manifest.json", repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    return json.loads(Path(path).read_text())


def cmd_push(args) -> int:
    from huggingface_hub import HfApi
    from nsg_sim.storage import dataset_dir, load_dataset
    api = HfApi()
    for name in args.names:
        _, manifest = load_dataset(name)  # verifies the local hash
        repo_id = f"{_org(args)}/{name}"
        remote = _remote_manifest(api, repo_id)
        if remote is not None:
            if remote.get("sha256") == manifest["sha256"]:
                print(f"{name}: already on the hub ({repo_id}), nothing to do")
                continue
            raise SystemExit(f"{name}: {repo_id} holds different content (sha {remote.get('sha256', '?')[:12]} vs "
                             f"{manifest['sha256'][:12]}). Datasets are immutable: rename the recipe.")
        api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(dataset_dir(name)),
                          allow_patterns=list(FILES), commit_message=f"{name} sha {manifest['sha256'][:12]}")
        print(f"{name}: pushed to https://huggingface.co/datasets/{repo_id}")
    return 0


def cmd_pull(args) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    from nsg_sim.storage import dataset_dir, load_dataset, sha256
    api = HfApi()
    for name in args.names:
        repo_id = f"{_org(args)}/{name}"
        remote = _remote_manifest(api, repo_id, args.revision)
        if remote is None:
            raise SystemExit(f"{name}: no dataset at {repo_id}")
        target = dataset_dir(name)
        if (target / "manifest.json").exists():
            local = json.loads((target / "manifest.json").read_text())
            if local.get("sha256") == remote["sha256"]:
                print(f"{name}: up to date")
                continue
            raise SystemExit(f"{name}: a different local dataset exists ({target}); move it away first")
        with tempfile.TemporaryDirectory() as tmp:
            for f in FILES:
                src = hf_hub_download(repo_id, f, repo_type="dataset", revision=args.revision)
                shutil.copy(src, Path(tmp) / f)
            if sha256((Path(tmp) / "records.jsonl").read_bytes()) != remote["sha256"]:
                raise SystemExit(f"{name}: downloaded records do not match the manifest hash")
            target.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                shutil.copy(Path(tmp) / f, target / f)
        load_dataset(name)
        print(f"{name}: pulled {remote['n_records']} records into {target}")
    return 0
