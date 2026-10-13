"""Built datasets on disk: data/<name>/records.jsonl + manifest.json. Built datasets are immutable.

Writing a dataset whose name exists is allowed only when the content is identical (same sha256);
a changed recipe or generator must get a new name. The data root defaults to <repo>/data and can
be moved with the NSG_DATA_DIR environment variable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import nsg_sim

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("NSG_DATA_DIR", REPO_ROOT / "data"))


def dataset_dir(name: str) -> Path:
    return data_root() / name


def records_bytes(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=False) + "\n" for r in records).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def save_dataset(name: str, records: list[dict], manifest: dict, overwrite: bool = False) -> Path:
    data = records_bytes(records)
    digest = sha256(data)
    target = dataset_dir(name)
    if (target / "manifest.json").exists() and not overwrite:
        existing = json.loads((target / "manifest.json").read_text())
        if existing.get("sha256") == digest:
            return target  # identical rebuild: nothing to do
        raise FileExistsError(
            f"dataset {name!r} already exists with different content (sha {existing.get('sha256', '?')[:12]} "
            f"vs {digest[:12]}). Built datasets are immutable: give the changed recipe a new name.")
    target.mkdir(parents=True, exist_ok=True)
    (target / "records.jsonl").write_bytes(data)
    manifest = {
        **manifest,
        "name": name,
        "sha256": digest,
        "n_records": len(records),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "nsg_sim_version": nsg_sim.__version__,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return target


def load_manifest(name: str) -> dict:
    path = dataset_dir(name) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset {name!r} is not built (no {path}); run `nsg build` or `nsg pull`")
    return json.loads(path.read_text())


def load_dataset(name: str, verify: bool = True) -> tuple[list[dict], dict]:
    manifest = load_manifest(name)
    data = (dataset_dir(name) / "records.jsonl").read_bytes()
    if verify and sha256(data) != manifest["sha256"]:
        raise ValueError(f"dataset {name!r}: records.jsonl does not match its manifest hash (was it edited?)")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return records, manifest


def list_datasets() -> list[dict]:
    root = data_root()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*/manifest.json")):
        m = json.loads(path.read_text())
        out.append({k: m.get(k) for k in ("name", "n_records", "sha256", "created", "description")})
    return out
