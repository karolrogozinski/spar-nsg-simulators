import json
import re
from collections import Counter, defaultdict

import pytest

from nsg_sim.builder import build_dataset, domain_for_manifest
from nsg_sim.config import DatasetConfig
from nsg_sim.rules.ast import from_spec
from nsg_sim.storage import save_dataset

from conftest import TEST_RECIPES


def test_labels_follow_the_rules(built):
    for name, (records, manifest) in built.items():
        values = domain_for_manifest(manifest).feature_values
        for r in records:
            f, g = from_spec(r["rule"]), from_spec(r["described_rule"])
            x, cx = values(r["input_x"]), values(r["input_cx"])
            assert (r["label_x"], r["label_cx"]) == tuple("YES" if f.evaluate(v) else "NO" for v in (x, cx))
            assert r["explanation_label_cx"] == ("YES" if g.evaluate(cx) else "NO")
            assert r["discriminating"] == (r["label_cx"] != r["explanation_label_cx"])


def test_explanations_agree_on_reference_and_unsound_ones_change_the_rule(built):
    records, _ = built["triage_full_v1"]
    for r in records:
        if r["explanation_type"] in ("faithful", "distractor"):
            assert r["described_rule"] == r["rule"]
        elif r["explanation_type"] in ("omission", "substitution", "wrong_threshold"):
            assert r["described_rule"] != r["rule"]
        if r["explanation_type"] == "faithful":
            assert not r["discriminating"]


def test_build_is_deterministic(built, tmp_path, monkeypatch):
    cfg = DatasetConfig.model_validate(TEST_RECIPES[0])
    records, manifest = build_dataset(cfg, save=False)
    assert records == built["triage_sampled_test"][0]


def test_rebuild_is_a_noop_and_changed_content_is_refused(built):
    records, manifest = built["triage_sampled_test"]
    save_dataset(manifest["name"], records, manifest)  # identical: fine
    changed = [dict(records[0], label_cx="flipped"), *records[1:]]
    with pytest.raises(FileExistsError):
        save_dataset(manifest["name"], changed, manifest)


def test_exact_balance(built):
    records, _ = built["triage_sampled_test"]
    per_ref = defaultdict(Counter)
    for r in records:
        if r["level"] == "P0" and r["evidence_condition"] == "none":
            per_ref[(r["rule_id"], r["explanation_type"], r["item_id"].rsplit("|", 1)[0])][r["label_cx"]] += 1
    for counts in per_ref.values():
        assert counts["YES"] == counts["NO"]


@pytest.mark.parametrize("name", ["triage_counter_test", "triage_abstract_test"])
def test_derived_datasets_keep_labels_and_pairing(built, name):
    base = {r["record_id"]: r for r in built["triage_full_v1"][0]}
    for r in built[name][0]:
        b = base[r["paired_record_id"]]
        assert (r["label_x"], r["label_cx"], r["explanation_label_cx"]) == (b["label_x"], b["label_cx"], b["explanation_label_cx"])


def test_qualitative_levels_have_no_digits(built):
    for name, (records, _) in built.items():
        if name == "triage_abstract_test":
            continue  # abstract features are named by letters only; digits come from the 0-100 scale at P0/P1
        for r in records:
            if r["level"] in ("P2", "P3"):
                assert not re.search(r"\d", r["explanation_text"]), (name, r["explanation_text"])


def test_domain_from_manifest_round_trips(built):
    _, manifest = built["triage_abstract_test"]
    domain = domain_for_manifest(manifest)
    assert set(domain.features) == set(built["triage_abstract_test"][0][0]["input_x"])


def test_manifest_records_recipe_and_hash(built):
    for _, manifest in built.values():
        assert len(manifest["sha256"]) == 64 and manifest["n_records"] > 0 and "config" in manifest


def test_hand_written_rules_get_automatic_edits(built):
    records, _ = built["triage_full_v1"]
    types = {(r["rule_id"], r["explanation_type"]) for r in records}
    # T03 pins nothing: omission, substitution and wrong_threshold are picked automatically
    assert {("T03", t) for t in ("omission", "substitution", "wrong_threshold")} <= types
    # T04 pins its substitution: income of the proxy is the pinned one
    sub = next(r for r in records if r["rule_id"] == "T04" and r["explanation_type"] == "substitution")
    assert "pain >= 6" in sub["described_formal"]


def test_chain_rules_grow_one_condition_at_a_time(built):
    records, _ = built["triage_chains_test"]
    sizes = {r["rule_id"]: r["rule_tags"]["n_conditions"] for r in records}
    assert sizes == {"A_and_n1": 1, "A_and_n2": 2, "A_and_n4": 4, "A_or_n4": 4, "A_unless_n4": 4, "A_k_of_n_n4": 4}


def test_simple_example_uses_defaults(built):
    records, manifest = built["triage_simple_v1"]
    assert {r["evidence_condition"] for r in records} == {"none"}
    assert {r["level"] for r in records} == {"P0", "P1", "P2", "P3"}
    assert {r["explanation_type"] for r in records} == {"faithful", "omission", "substitution"}
