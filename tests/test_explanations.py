"""Edit sizes (omission / substitution n, wrong_threshold shift) and the direction of substitution proxies."""

import random

import pytest

from nsg_sim.builder import build_dataset
from nsg_sim.config import DatasetConfig, Explanations, ExperimentConfig
from nsg_sim.domain.tabular import load_domain
from nsg_sim.explanations import build_variants
from nsg_sim.rules.ast import from_spec
from nsg_sim.rules.sources import RuleItem, _from_entry
from nsg_sim.config import RuleEntry
from nsg_sim.runner import select_records

TRIAGE = load_domain("triage")
FOUR = {"all": [["temperature", ">=", 38], ["heart_rate", ">=", 110], ["pain", ">=", 7], ["oxygen", "<", 94]]}
SIZED = {"types": ["faithful", "omission", "substitution", "wrong_threshold"],
         "omission": {"n": [1, 2, 3]}, "substitution": {"n": [1, 2]}, "wrong_threshold": {"shift": [0.1, 0.3]}}


def variants(spec, explanations=SIZED, **pins):
    item = RuleItem("R", from_spec(spec), **pins)
    return {v.key: v for v in build_variants(item, TRIAGE, Explanations.model_validate(explanations), random.Random(0))}


def test_omission_sizes_are_nested():
    vs = variants(FOUR)
    kept = [set(map(repr, vs[f"omission@{n}"].g.conditions())) for n in (1, 2, 3)]
    assert len(kept[0]) == 3 and len(kept[1]) == 2 and len(kept[2]) == 1
    assert kept[0] > kept[1] > kept[2]


def test_too_large_edits_are_skipped():
    vs = variants(["temperature", ">=", 39])
    assert not any(k.startswith("omission") for k in vs)            # cannot drop the only condition
    assert "substitution@1" in vs and "substitution@2" not in vs    # only one condition to replace


def test_threshold_shifts_move_the_same_threshold_further():
    vs = variants(FOUR)
    small, large = vs["wrong_threshold@0.1"].extras, vs["wrong_threshold@0.3"].extras
    assert small["edited"] == large["edited"]
    old = small["edited"].value
    assert (small["shifted"].value - old) * (large["shifted"].value - old) > 0          # same direction
    assert abs(large["shifted"].value - old) > abs(small["shifted"].value - old)


@pytest.mark.parametrize("spec", [["temperature", ">=", 39], ["oxygen", "<", 92], ["pain", ">=", 7],
                                  ["temperature", "<", 37], ["oxygen", ">=", 95]])
def test_substitution_proxy_points_the_same_way(spec):
    old = from_spec(spec)
    proxy = variants(spec, {"types": ["substitution"]})["substitution"].extras["proxy"][0]

    def supports_yes(c):
        side = "high" if c.op in (">=", ">") else "low"
        return side == TRIAGE.feature(c.feature).yes_when

    assert supports_yes(proxy) == supports_yes(old), (old, proxy)


def test_pinned_lists_come_first():
    entry = RuleEntry(id="R", rule=FOUR, omit=[["pain", ">=", 7], ["oxygen", "<", 94]],
                      substitute=[["temperature", ">=", 38], ["symptom_days", ">=", 5]])
    item = _from_entry(entry)
    vs = {v.key: v for v in build_variants(item, TRIAGE, Explanations.model_validate(SIZED), random.Random(0))}
    assert [c.feature for c in vs["omission@2"].extras["edited"]] == ["pain", "oxygen"]
    assert vs["substitution@1"].extras["proxy"][0].feature == "symptom_days"


def test_sized_dataset_records_and_sampling(data_dir):
    cfg = DatasetConfig.model_validate({"name": "sizes_test", "domain": "triage",
                                        "rules": [{"id": "R1", "rule": FOUR}], "explanations": SIZED,
                                        "counterfactuals": {"n_reference": 2, "n_per_reference": 4}})
    records, manifest = build_dataset(cfg)
    assert len({r["record_id"] for r in records}) == len(records)
    sizes = {(r["explanation_type"], r.get("edit_size")) for r in records}
    assert {("omission", 1), ("omission", 2), ("omission", 3), ("substitution", 1),
            ("wrong_threshold", 0.1), ("wrong_threshold", 0.3)} <= sizes
    assert manifest["stats"]["skipped_variants"] == ["R1: substitution@2"]   # one free numeric feature only
    # per_cell samples every size separately
    exp = ExperimentConfig.model_validate({"name": "x", "dataset": "sizes_test", "sample": {"per_cell": 1},
                                           "filter": {"levels": ["P0"]},
                                           "simulators": [{"backend": "baseline", "model": "follow_g"}]})
    picked = {(r["explanation_type"], r.get("edit_size")) for r in select_records(records, exp)}
    assert picked == {s for s in sizes}
    # a mirror twin keeps sizes and labels
    twin, _ = build_dataset(DatasetConfig(name="sizes_test_counter", derive_from="sizes_test", transform="mirror"))
    assert [r.get("edit_size") for r in twin] == [r.get("edit_size") for r in records]


def test_default_recipes_have_no_edit_size(built):
    records, _ = built["triage_full_v1"]
    assert all("edit_size" not in r for r in records)


# --- omission targets (EXP-009) ----------------------------------------------------

TARGETS = {"types": ["faithful", "omission"],
           "omission": {"target": ["aligned", "opposed", "neutral"], "strong": ["temperature", "oxygen"]}}
MIXED = {"all": [["oxygen", ">=", 90], ["chronic", "==", True], ["temperature", ">=", 38]]}


def test_omission_targets_drop_the_condition_of_their_kind():
    vs = variants(MIXED, TARGETS)
    dropped = {k: {repr(c) for c in from_spec(MIXED).conditions()} - {repr(c) for c in v.g.conditions()}
               for k, v in vs.items() if k != "faithful"}
    assert dropped == {"omission@aligned": {repr(from_spec(["temperature", ">=", 38]))},   # fever -> doctor: expected
                       "omission@opposed": {repr(from_spec(["oxygen", ">=", 90]))},        # high saturation -> doctor
                       "omission@neutral": {repr(from_spec(["chronic", "==", True]))}}
    assert all(v.target == k.split("@")[1] for k, v in vs.items() if k != "faithful")


def test_omission_target_without_a_match_is_skipped():
    vs = variants({"all": [["temperature", ">=", 38], ["chronic", "==", True]]}, TARGETS)
    assert set(vs) == {"faithful", "omission@aligned", "omission@neutral"}


def test_omission_targets_need_strong_features_and_single_drops():
    with pytest.raises(ValueError):
        Explanations.model_validate({"types": ["omission"], "omission": {"target": ["aligned"]}})
    with pytest.raises(ValueError):
        Explanations.model_validate({"types": ["omission"], "omission": {"target": ["neutral"], "n": [1, 2]}})


def test_shared_counterfactuals_and_partial_twin(data_dir):
    cfg = DatasetConfig.model_validate({
        "name": "targets_test", "domain": "triage", "rules": [{"id": "T1", "rule": MIXED}],
        "explanations": {**TARGETS, "levels": {"faithful": ["P0"], "omission": ["P0"]}},
        "counterfactuals": {"n_reference": 2, "n_per_reference": 8, "balance": "exact", "shared": True}})
    records, _ = build_dataset(cfg)
    by_variant = {}
    for r in records:
        by_variant.setdefault(r["cluster_id"].split("|")[1], []).append((r["input_x"], r["input_cx"]))
    cases = list(by_variant.values())
    assert len(by_variant) == 4 and all(c == cases[0] for c in cases)   # identical cases for every explanation
    assert {r.get("edit_target") for r in records} == {None, "aligned", "opposed", "neutral"}
    assert all(r["discriminating"] for r in records if r["explanation_type"] == "omission"
               and r["described_formal"] != r["rule_formal"] and r["label_cx"] != r["explanation_label_cx"])
    # partial twin: same records, the explanation only gains the domain's note
    twin, _ = build_dataset(DatasetConfig(name="targets_test_partial", derive_from="targets_test", transform="partial"))
    note = TRIAGE.explanation["partial_note"]
    assert [r["input_cx"] for r in twin] == [r["input_cx"] for r in records]
    assert all(t["explanation_text"] == note + r["explanation_text"] for t, r in zip(twin, records))
