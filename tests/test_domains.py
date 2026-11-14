"""Consistency checks for every domain in configs/domains/: wording, profile, mirror maps.

Loading a domain checks little (unknown feature keys are ignored), and a wrong phrase or profile
only shows up when a dataset is built, so these tests catch it per domain, before any build.
"""

import random
import re

import pytest
import yaml

from nsg_sim.domain.tabular import DOMAINS_DIR, PLACEHOLDER, load_domain

DOMAINS = sorted(p.stem for p in DOMAINS_DIR.glob("*.yaml"))
STYLES = ("exact", "approximate", "qualitative")
FEATURE_KEYS = {"type", "range", "step", "values", "label", "phrases", "overrides", "display", "aliases",
                "sensitive", "yes_when", "mirror", "approx_unit"}


def raw(name):
    return yaml.safe_load((DOMAINS_DIR / f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", DOMAINS)
def test_feature_keys_are_known(name):
    for f, cfg in raw(name)["features"].items():
        assert not set(cfg) - FEATURE_KEYS, f"{name}.{f}: unknown keys {sorted(set(cfg) - FEATURE_KEYS)}"


@pytest.mark.parametrize("name", DOMAINS)
def test_numeric_rule_features_have_full_wording(name):
    """Sampled, mirrored and negated conditions use >= and <, at every template style."""
    for f, cfg in raw(name)["features"].items():
        if cfg["type"] != "numeric" or "yes_when" not in cfg:
            continue
        for op in (">=", "<"):
            entry = cfg.get("phrases", {}).get(op, {})
            assert all(entry.get(s) for s in STYLES), f"{name}.{f} {op}: missing styles"
            words = entry["qualitative"]
            if isinstance(words, list):
                assert "max" not in words[-1], f"{name}.{f} {op}: last qualitative entry must have no max"
                maxes = [w["max"] for w in words[:-1]]
                assert maxes == sorted(maxes), f"{name}.{f} {op}: qualitative maxes out of order"
                texts = [w["text"] for w in words]
            else:
                texts = [words]
            assert not any(re.search(r"\d|\{v\}", t) for t in texts), f"{name}.{f} {op}: digits at P2"


@pytest.mark.parametrize("name", DOMAINS)
def test_every_condition_renders(name):
    domain = load_domain(name)
    for f in domain.features.values():
        if f.kind == "numeric":
            conds = [(op, v) for op in (">=", ">", "<=", "<") for v in f.values[1:-1]]
        else:
            conds = [(op, v) for op in ("==", "!=") for v in f.values]
        for op, v in conds:
            for style in STYLES:
                text = f.phrase(op, v, style)
                assert text and " equal to " not in text, f"{name}.{f.name} {op} {v} {style}: {text!r}"


@pytest.mark.parametrize("name", DOMAINS)
def test_profile_shows_every_feature(name):
    domain = load_domain(name)
    if not getattr(domain, "profile", None):
        return
    used = {m.group(1) for m in PLACEHOLDER.finditer(domain.profile)}
    assert used == set(domain.features), f"{name}: profile placeholders {sorted(used ^ set(domain.features))}"
    rng = random.Random(0)
    for _ in range(50):
        text = domain.describe(domain.sample_input(rng))
        assert "{" not in text and "True" not in text and "False" not in text, text


@pytest.mark.parametrize("name", DOMAINS)
def test_categories_work_with_sampler_and_transforms(name):
    for f in load_domain(name).features.values():
        if f.kind != "categorical":
            continue
        assert len(f.values) <= 8, f"{name}.{f.name}: the abstract twin names at most 8 values"
        if f.yes_when is not None:
            assert set(f.yes_when) <= set(f.values), f"{name}.{f.name}: yes_when values not in values"
            if len(f.values) >= 3:   # only != is sampled (== passes too rarely): one value must be left out
                assert len(f.values) - len(f.yes_when) == 1, f"{name}.{f.name}: yes_when must leave out one value"
        if f.mirror_map:
            assert set(f.mirror_map) <= set(f.values)
            assert sorted(f.mirror_map) == sorted(f.mirror_map.values()), f"{name}.{f.name}: mirror is not a bijection"
        elif f.yes_when is not None:
            pytest.fail(f"{name}.{f.name}: a rule category needs a mirror map, or the counter twin keeps its direction")


@pytest.mark.parametrize("name", [d for d in DOMAINS if d != "triage"])   # triage keeps its exact P1 (EXP-001-010)
def test_approximate_level_rounds_thresholds(name):
    """P1 states the threshold rounded to a nearby round number ("around 40°C or more" for 39)."""
    for f in load_domain(name).features.values():
        if f.kind != "numeric" or f.yes_when is None:
            continue
        for op in (">=", "<"):
            assert "{approx}" in f.phrases[op]["approximate"], f"{name}.{f.name} {op}: P1 without rounding"
        u = f.rounding_unit()
        assert u >= 2 * (f.step or 1)
        for v in f.values[1:-1]:
            for op in (">=", ">", "<", "<="):
                b = v + (f.step or 1) if op in (">", "<=") else v
                r = f.approximate(op, v)
                assert f.low < r <= f.high and (r == b or any(r % w == 0 and abs(r - b) <= w / 2 for w in (u, 5)))
