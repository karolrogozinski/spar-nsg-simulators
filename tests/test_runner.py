import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nsg_sim.analysis.metrics import load_pairs, nsg, accuracy, group
from nsg_sim.config import ExperimentConfig
from nsg_sim.runner import run_experiment


def experiment(**kw):
    base = {"name": "t", "dataset": "triage_full_v1", "filter": {"evidence_conditions": ["none"]},
            "sample": {"per_cell": 1}, "simulators": [{"backend": "baseline", "model": "follow_g"}]}
    return ExperimentConfig.model_validate({**base, **kw})


def test_baselines_give_expected_metrics(built, tmp_path):
    cfg = experiment(simulators=[{"backend": "baseline", "model": m} for m in ("follow_g", "true_f", "copy_reference")])
    out = run_experiment(cfg, root=tmp_path, log=lambda *_: None)
    cells = group(load_pairs([out]), ["simulator", "explanation_type", "level"])
    assert all(accuracy(c) == 1 for (sim, *_), c in cells.items() if sim == "true_f")
    assert all(nsg(c) == 1 for (sim, etype, lvl), c in cells.items() if sim == "follow_g" and etype == "faithful")
    copy = [p for (sim, *_), c in cells.items() if sim == "copy_reference" for p in c]
    assert 0.3 < accuracy(copy) < 0.7


def test_dry_run_writes_nothing_and_max_requests_guards(built, tmp_path):
    cfg = experiment(simulators=[{"backend": "ollama", "model": "m", "base_url": "http://127.0.0.1:9/v1"}])
    out = run_experiment(cfg, dry_run=True, root=tmp_path, log=lambda *_: None)
    assert not out.exists()
    with pytest.raises(RuntimeError):
        run_experiment(experiment(simulators=cfg.model_dump()["simulators"], run={"max_requests": 3}),
                       dry_run=True, root=tmp_path, log=lambda *_: None)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server():
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).parent / "fake_openai_server.py"), str(port), "5"])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()
    proc.wait()


def test_api_backend_with_errors_then_resume(built, tmp_path, fake_server):
    sim = {"backend": "ollama", "model": "fake", "base_url": fake_server}
    cfg = experiment(simulators=[sim], filter={"evidence_conditions": ["none"], "levels": ["P0"]},
                     run={"retries": 0, "concurrency": 4})
    out = run_experiment(cfg, root=tmp_path, log=lambda *_: None)
    rows = [json.loads(line) for line in next(out.glob("*.jsonl")).read_text().splitlines()]
    assert any(r["error"] for r in rows) and any(r["prediction"] for r in rows)
    # second run retries only the failed rows; with retries enabled everything completes
    cfg = experiment(simulators=[sim], filter={"evidence_conditions": ["none"], "levels": ["P0"]}, run={"retries": 3})
    for _ in range(3):
        run_experiment(cfg, root=tmp_path, log=lambda *_: None)
    pairs = load_pairs([out])
    assert pairs and all(p["valid"] for p in pairs)


def test_run_refuses_changed_dataset(built, tmp_path):
    out = run_experiment(experiment(), root=tmp_path, log=lambda *_: None)
    meta = json.loads((out / "run.json").read_text())
    meta["dataset_sha256"] = "0" * 64
    (out / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        run_experiment(experiment(), root=tmp_path, log=lambda *_: None)


def test_nsg_run_writes_summary_and_figure(built, tmp_path):
    import csv

    import yaml

    from nsg_sim.cli import main
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(yaml.safe_dump({"name": "report_test", "dataset": "triage_simple_v1",
                                   "simulators": [{"backend": "baseline", "model": "follow_g"}]}))
    assert main(["run", str(cfg), "--runs-dir", str(tmp_path / "runs")]) == 0
    out = tmp_path / "runs" / "report_test"
    rows = list(csv.DictReader((out / "summary.csv").open()))
    faithful = [r for r in rows if r["explanation_type"] == "faithful"]
    assert faithful and all(float(r["nsg"]) == 1.0 for r in faithful)
    assert (out / "nsg.png").stat().st_size > 0
