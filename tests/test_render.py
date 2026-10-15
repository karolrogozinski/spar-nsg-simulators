import random
import re

from nsg_sim.config import RuleSampler
from nsg_sim.domain.tabular import load_domain
from nsg_sim.render.checks import check_text
from nsg_sim.render.levels import LevelChecks
from nsg_sim.render.template import inline_ok, render_rule
from nsg_sim.rules.ast import from_spec
from nsg_sim.rules.sampler import PRESETS, sample_rules

TRIAGE = load_domain("triage")


def test_flat_rules_stay_one_sentence():
    for spec in [["temperature", ">=", 39],
                 {"all": [["pain", ">=", 7], ["symptom_days", ">=", 3]]},
                 {"all": [["temperature", ">=", 38], {"any": [["heart_rate", ">=", 110], ["oxygen", "<", 94]]}]},
                 {"all": [["age", ">=", 65], ["chronic", "==", True]], "unless": ["pain", "<", 2]},
                 {"at_least": 2, "of": [["temperature", ">=", 38], ["heart_rate", ">=", 110], ["surgery", "==", True]]}]:
        rule = from_spec(spec)
        assert inline_ok(rule)
        assert "\n" not in render_rule(rule, TRIAGE, "exact")


def test_nested_rules_become_indented_lists():
    rule = from_spec({"any": [{"all": [["surgery", "==", True], ["temperature", ">=", 38]], "unless": ["age", "<", 18]},
                              ["oxygen", "<", 90]],
                      "unless": ["arrival", "==", "referral"]})
    assert not inline_ok(rule)
    assert render_rule(rule, TRIAGE, "exact") == (
        "The patient is sent to a doctor if the patient meets at least one of the following:\n"
        "- meets all of the following, unless the patient is younger than 18:\n"
        "  - had surgery recently\n"
        "  - has a temperature of at least 38°C\n"
        "- has an oxygen saturation below 90%\n"
        "unless the patient arrived with a GP referral.")


def test_negated_groups():
    rule = from_spec({"not": {"any": [["temperature", ">=", 38], ["pain", ">=", 7]]}})
    assert render_rule(rule, TRIAGE, "exact").startswith("The patient is sent to a doctor if the patient meets none of the following:\n- ")


def test_equivalent_operators_reuse_domain_wording():
    hr = TRIAGE.feature("heart_rate")
    assert hr.phrase(">", 100, "exact") == hr.phrase(">=", 105, "exact") == "has a heart rate of at least 105 bpm"
    assert TRIAGE.feature("pain").phrase("<=", 3, "exact") == "rates their pain below 4/10"
    assert TRIAGE.feature("sex").phrase("!=", "male", "exact") == "is not male"
    assert TRIAGE.feature("insurance").phrase("!=", "none", "qualitative") == "has health insurance"


def test_qualitative_words_follow_the_threshold():
    temp = TRIAGE.feature("temperature")
    assert temp.phrase(">=", 37, "qualitative") == "does not have a low body temperature"
    assert temp.phrase(">=", 38, "qualitative") == "has a slightly raised temperature"
    assert temp.phrase(">=", 39, "qualitative") == "has a fever"


def test_sampled_rules_name_every_feature_and_p2_has_no_digits():
    for preset in PRESETS:
        for item in sample_rules(RuleSampler(preset=preset, n_rules=10), TRIAGE, random.Random(preset)):
            exact, qual = render_rule(item.rule, TRIAGE, "exact"), render_rule(item.rule, TRIAGE, "qualitative")
            assert not check_text(exact, item.rule, TRIAGE, LevelChecks(require_all_features=True, forbid_other_features=True,
                                                                       require_thresholds=True)), exact
            assert not re.search(r"\d", qual), qual
            assert "equal to" not in exact, exact          # no generic fallback wording on triage


def test_explanation_wording_matches_the_profile():
    """A categorical value must read the same in explanations and in the case description: gpt-oss
    misread "did not walk in" against a profile saying "arrived on foot" (EXP-005/006)."""
    for name, f in TRIAGE.features.items():
        shown = TRIAGE.display.get(name, {})
        if f.kind != "categorical" or not shown:
            continue
        for v in f.values:
            assert str(shown[v]) in f.phrase("==", v, "exact"), (name, v, shown[v], f.phrase("==", v, "exact"))
