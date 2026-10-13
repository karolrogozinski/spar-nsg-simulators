"""Shared fixtures. Datasets are built into a temporary data root, never into data/.

Besides the shipped recipes (configs/datasets/triage_simple_v1.yaml, triage_full_v1.yaml) the tests build recipes that
exercise the other sources and transforms: sampled rules with exact balance, rule chains, the
mirror and abstract twins, and a custom (non-tabular) domain from tests/fixtures/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nsg_sim.builder import build_dataset
from nsg_sim.config import DatasetConfig, load_dataset_config

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

TEST_RECIPES = [
    {"name": "triage_sampled_test", "domain": "triage", "rule_sampler": {"preset": "medium", "n_rules": 6},
     "explanations": {"types": ["faithful", "omission", "substitution"]},
     "counterfactuals": {"n_reference": 2, "n_per_reference": 8, "balance": "exact"}},
    {"name": "triage_chains_test", "domain": "triage",
     "rule_builder": {"kind": "chains", "and_sizes": [1, 2, 4], "structures": ["or", "unless", "at_least"],
                      "chains": {"A": {"conditions": [["temperature", ">=", 38], ["heart_rate", ">=", 110],
                                                      ["pain", ">=", 7], ["oxygen", "<", 94]],
                                       "unless": ["arrival", "==", "referral"]}}},
     "counterfactuals": {"n_reference": 2, "n_per_reference": 4}},
    {"name": "triage_counter_test", "derive_from": "triage_full_v1", "transform": "mirror"},
    {"name": "triage_abstract_test", "derive_from": "triage_full_v1", "transform": "abstract"},
    {"name": "sentence_test", "domain": "sentence_example", "rule_sampler": {"preset": "medium", "n_rules": 4, "ops": [">=", "<=", "=="]},
     "explanations": {"types": ["faithful", "omission", "substitution", "distractor"], "levels": {"distractor": ["P0"]}},
     "counterfactuals": {"n_reference": 2, "n_per_reference": 6}},
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    old = {k: os.environ.get(k) for k in ("NSG_DATA_DIR", "NSG_DOMAIN_PATH")}
    os.environ["NSG_DATA_DIR"] = str(path)
    os.environ["NSG_DOMAIN_PATH"] = str(FIXTURES)
    yield path
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def built(data_dir):
    """name -> (records, manifest) for the shipped recipes and every recipe above."""
    out = {}
    for name in ("triage_simple_v1", "triage_full_v1"):
        cfg = load_dataset_config(CONFIGS / "datasets" / f"{name}.yaml")
        out[cfg.name] = build_dataset(cfg)
    for raw in TEST_RECIPES:
        cfg = DatasetConfig.model_validate(raw)
        out[cfg.name] = build_dataset(cfg)
    return out
