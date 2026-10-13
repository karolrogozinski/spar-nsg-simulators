from pathlib import Path

import pytest
from pydantic import ValidationError

from nsg_sim.cli import main
from nsg_sim.config import DatasetConfig, ExperimentConfig, load_yaml

from conftest import CONFIGS


def test_all_shipped_configs_validate(capsys):
    paths = [str(p) for p in sorted((CONFIGS / "datasets").glob("*.yaml")) + sorted((CONFIGS / "experiments").glob("*.yaml"))]
    assert main(["validate", *paths]) == 0, capsys.readouterr().out


def test_typo_in_dataset_key_is_an_error():
    raw = load_yaml(CONFIGS / "datasets" / "triage_full_v1.yaml")
    raw["counterfactuals"]["n_per_refrence"] = 4
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(raw)


def test_typo_in_experiment_key_is_an_error():
    raw = load_yaml(CONFIGS / "experiments" / "triage_simple_smoke.yaml")
    raw["simulators"][0]["reasoning"] = "low"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_unknown_domain_feature_in_rule_is_an_error(tmp_path):
    raw = load_yaml(CONFIGS / "datasets" / "triage_full_v1.yaml")
    raw["rules"][0]["rule"] = [["temperatur", ">=", 38]]
    path = tmp_path / "bad.yaml"
    import yaml
    path.write_text(yaml.safe_dump(raw))
    assert main(["validate", str(path)]) == 1


def test_preview_prints_rules_and_saves_nothing(capsys, data_dir):
    assert main(["preview", str(CONFIGS / "datasets" / "triage_full_v1.yaml"), "--rule", "T04"]) == 0
    out = capsys.readouterr().out
    assert "=== T04" in out and "omission" in out and "nothing saved" in out
