import itertools
import random

import pytest

from nsg_sim.config import RuleSampler
from nsg_sim.domain.tabular import load_domain
from nsg_sim.rules.ast import All, AtLeast, Any, Cond, Not, Unless, from_spec, to_spec
from nsg_sim.rules.sampler import PRESETS, sample_rules

TRIAGE = load_domain("triage")


def random_inputs(domain, n, seed=0):
    rng = random.Random(seed)
    return [domain.sample_input(rng) for _ in range(n)]


def test_evaluate_matches_formal_on_sampled_rules():
    xs = random_inputs(TRIAGE, 200)
    for preset in PRESETS:
        for item in sample_rules(RuleSampler(preset=preset, n_rules=5), TRIAGE, random.Random(preset)):
            for x in xs:
                assert item.rule.evaluate(x) == bool(eval(item.rule.formal(), {}, dict(x))), (preset, item.rule.formal())


def test_every_node_type():
    a, b, c = Cond("temperature", ">=", 38), Cond("heart_rate", ">", 100), Cond("chronic", "==", True)
    rules = [All((a, b)), Any((a, b)), AtLeast((a, b, c), 2), Not(a), Unless(All((a, b)), c), All((Any((a, b)), Not(c)))]
    for rule, x in itertools.product(rules, random_inputs(TRIAGE, 100)):
        assert rule.evaluate(x) == bool(eval(rule.formal(), {}, dict(x)))
        assert from_spec(to_spec(rule)) == rule


def test_spec_accepts_legacy_atleast_key():
    spec = {"atleast": 2, "of": [["temperature", ">=", 38], ["heart_rate", ">", 100], ["chronic", "==", False]]}
    assert from_spec(spec) == from_spec({**{k: v for k, v in spec.items() if k != "atleast"}, "at_least": 2})


def test_replace_collapses_single_items():
    a, b = Cond("temperature", ">=", 38), Cond("heart_rate", ">", 100)
    assert All((a, b)).replace(b, None) == a


@pytest.mark.parametrize("preset", list(PRESETS))
def test_sampler_respects_preset(preset):
    p = PRESETS[preset]
    items = sample_rules(RuleSampler(preset=preset, n_rules=6), TRIAGE, random.Random(1))
    assert len(items) == 6
    lo, hi = p.get("n_conditions", (1, 99))
    for item in items:
        assert lo <= len(item.rule.conditions()) <= hi
        assert len(set(item.rule.features())) == len(item.rule.conditions())  # one condition per feature


def test_sampler_is_deterministic():
    one = [i.rule for i in sample_rules(RuleSampler(preset="hard", n_rules=4), TRIAGE, random.Random(3))]
    two = [i.rule for i in sample_rules(RuleSampler(preset="hard", n_rules=4), TRIAGE, random.Random(3))]
    assert one == two
