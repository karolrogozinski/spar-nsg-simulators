import random

from nsg_sim.domain.tabular import load_domain
from nsg_sim.rules.ast import from_spec


def test_sentence_domain_builds_with_the_standard_pipeline(built):
    records, manifest = built["sentence_test"]
    domain = load_domain("sentence_example")
    assert {r["explanation_type"] for r in records} == {"faithful", "omission", "substitution", "distractor"}
    for r in records:
        assert r["description_cx"].startswith('The sentence is: "')
        f = from_spec(r["rule"])
        assert r["label_cx"] == ("YES" if f.evaluate(domain.feature_values(r["input_cx"])) else "NO")
        assert 1 <= len(domain.changed(r["input_x"], r["input_cx"])) <= 2


def test_sentence_perturbations_stay_in_range(data_dir):
    domain = load_domain("sentence_example")
    rng = random.Random(0)
    for _ in range(500):
        x = domain.sample_input(rng)
        cx = domain.perturb(x, rng, list(domain.features))
        for name, value in domain.feature_values(cx).items():
            assert value in domain.feature(name).values
