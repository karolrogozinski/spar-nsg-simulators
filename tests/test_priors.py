"""Prior elicitation: sampling outside the dataset, parsing, agreement measures."""

from nsg_sim.analysis.priors import _toward_yes, auc, direction_agreement, prior_strength, spearman
from nsg_sim.builder import domain_for_manifest
from nsg_sim.priors import parse_direction, parse_probability, sample_inputs
from nsg_sim.rules.ast import from_spec
from nsg_sim.sampling import key


def test_sampled_cases_are_never_dataset_inputs(built):
    records, manifest = built["triage_full_v1"]
    domain = domain_for_manifest(manifest)
    xs = sample_inputs(domain, records, 200, seed=1)
    used = {key(r["input_x"]) for r in records} | {key(r["input_cx"]) for r in records}
    assert len({key(x) for x in xs}) == 200 and not used & {key(x) for x in xs}
    assert xs == sample_inputs(domain, records, 200, seed=1)


def test_parsing(built):
    domain = domain_for_manifest(built["triage_full_v1"][1])
    assert parse_probability("[PROBABILITY]\n85\n\n[ANSWER]\nYES") == 0.85
    assert parse_probability("<think>[PROBABILITY] 10</think>[PROBABILITY]: **40**") == 0.4
    assert parse_probability("about 80%") is None
    assert parse_direction("[ANSWER]\nLOWER", domain.feature("oxygen")) == "low"
    assert parse_direction("[ANSWER] WITH", domain.feature("chronic")) is True
    assert parse_direction("[ANSWER]\nambulance", domain.feature("arrival")) == "ambulance"
    assert parse_direction("[ANSWER] SAME", domain.feature("sex")) == "same"
    assert parse_direction("\nHIGHER", domain.feature("pain")) == "high"            # no header (gemma-3-27b)
    assert parse_direction("I think it depends.", domain.feature("pain")) is None


def test_agreement_measures(built):
    domain = domain_for_manifest(built["triage_full_v1"][1])
    assert auc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auc([True, False], [0.5, 0.5]) == 0.5
    rule = from_spec({"all": [["temperature", ">=", 38], ["oxygen", "<", 94]], "unless": ["arrival", "==", "walk_in"]})
    beliefs = {"temperature": "high", "oxygen": "low", "arrival": "ambulance"}
    assert direction_agreement(rule, domain, beliefs) == (1.0, 3)
    flipped = {"temperature": "low", "oxygen": "low", "arrival": None}
    assert direction_agreement(rule, domain, flipped) == (0.5, 2)


def test_feature_prior_strength(built):
    domain = domain_for_manifest(built["triage_full_v1"][1])
    assert round(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 9) == 1.0
    assert round(spearman([1, 2, 2, 3], [3, 2, 2, 1]), 9) == -1.0
    oxygen, chronic, arrival = (domain.feature(n) for n in ("oxygen", "chronic", "arrival"))
    assert _toward_yes(oxygen, 85) > _toward_yes(oxygen, 95)          # yes_when: low
    assert _toward_yes(chronic, True) == 1.0 and _toward_yes(arrival, "walk_in") == 0.0
    rows = [{"simulator": "s", "dataset": "d", "rule_feature": True, "rho": r} for r in (0.4, -0.2)]
    rows.append({"simulator": "s", "dataset": "d", "rule_feature": False, "rho": 0.9})
    (out,) = prior_strength(rows)
    assert round(out["strength"], 2) == 0.3 and round(out["alignment"], 2) == 0.1 and out["n_features"] == 2
