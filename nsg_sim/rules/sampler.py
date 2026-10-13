"""Random rules from a configurable grammar, to scale difficulty up or down from YAML.

Knobs (all in `rule_sampler`, any of them overriding the chosen `preset`):
  n_rules              how many rules
  n_conditions         [min, max] conditions per rule
  connectors           weights of the node types used to combine conditions:
                         and, or, unless, at_least ({weight, k: [min, max]}), not
  max_depth            how deeply groups may nest
  condition_pass_rate  [min, max] share of inputs meeting a single condition
  rule_yes_rate        [min, max] share of random inputs the whole rule approves
  features             {include, exclude, max_per_feature}
  ops                  allowed comparison operators
A rule is drawn top-down: split the conditions over the children of a randomly chosen connector,
recurse until the depth limit, then fill leaves with conditions whose pass rate is in range.
Rules whose approval rate falls outside `rule_yes_rate` (estimated on random inputs) are redrawn.
"""

from __future__ import annotations

import random

from nsg_sim.config import RuleSampler
from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import All, Any, AtLeast, Cond, Node, Not, Unless
from nsg_sim.rules.sources import RuleItem

PRESETS = {
    "easy": dict(n_conditions=[1, 2], connectors={"and": 1}, max_depth=1),
    "medium": dict(n_conditions=[2, 4], connectors={"and": 3, "or": 1, "unless": 1}, max_depth=2),
    "hard": dict(n_conditions=[4, 8], connectors={"and": 3, "or": 2, "unless": 1, "at_least": {"weight": 1, "k": [2, 3]}},
                 max_depth=3),
    "very_hard": dict(n_conditions=[6, 10], connectors={"and": 2, "or": 2, "unless": 1, "not": 1,
                                                         "at_least": {"weight": 2, "k": [2, 4]}}, max_depth=4),
}
STRUCTURE_NAMES = {"All": "and", "Any": "or", "AtLeast": "at_least", "Unless": "unless", "Not": "not", "Cond": "single"}
DEFAULTS = dict(n_rules=6, n_conditions=[2, 4], connectors={"and": 1}, max_depth=2,
                condition_pass_rate=[0.35, 0.8], rule_yes_rate=[0.15, 0.85], features={},
                ops=[">=", "<=", "<", ">", "==", "!="])


def resolve(cfg: RuleSampler) -> dict:
    params = dict(DEFAULTS)
    if cfg.preset:
        if cfg.preset not in PRESETS:
            raise ValueError(f"unknown rule_sampler preset {cfg.preset!r}; choose from {sorted(PRESETS)}")
        params.update(PRESETS[cfg.preset])
    params.update({k: v for k, v in cfg.model_dump().items() if v is not None and k not in ("preset", "tries")})
    params["tries"] = cfg.tries
    return params


def sample_rules(cfg: RuleSampler, domain: Domain, rng: random.Random) -> list[RuleItem]:
    p = resolve(cfg)
    features = _allowed_features(domain, p["features"])
    lo, hi = p["n_conditions"]
    if hi > len(features) * p["features"].get("max_per_feature", 1):
        raise ValueError(f"rule_sampler: up to {hi} conditions but only {len(features)} usable features")
    probe = [domain.sample_input(random.Random(f"probe|{i}")) for i in range(2000)]
    items, seen = [], set()
    for idx in range(p["n_rules"]):
        for attempt in range(p["tries"]):
            n = rng.randint(lo, hi)
            pool = list(features)
            rule = _node(n, 1, p, domain, pool, rng)
            yes = sum(rule.evaluate(domain.feature_values(x)) for x in probe) / len(probe)
            if p["rule_yes_rate"][0] <= yes <= p["rule_yes_rate"][1] and rule.formal() not in seen:
                seen.add(rule.formal())
                items.append(RuleItem(f"S{idx + 1:02d}", rule, tags={
                    "preset": cfg.preset, "structure": STRUCTURE_NAMES[type(rule).__name__], "yes_rate": round(yes, 3)}))
                break
        else:
            raise RuntimeError(f"rule_sampler: no rule with approval rate in {p['rule_yes_rate']} after {p['tries']} tries; "
                               "relax rule_yes_rate / condition_pass_rate or allow more connectors")
    return items


def _allowed_features(domain: Domain, fcfg: dict) -> list[str]:
    names = list(fcfg.get("include") or domain.features)
    return [n for n in names if n not in set(fcfg.get("exclude") or [])]


def _weights(connectors: dict) -> dict[str, float]:
    return {k: (v["weight"] if isinstance(v, dict) else v) for k, v in connectors.items()
            if (v["weight"] if isinstance(v, dict) else v) > 0}


def _node(n: int, depth: int, p: dict, domain: Domain, pool: list[str], rng: random.Random) -> Node:
    if n == 1:
        cond = _condition(domain, pool, p, rng)
        if "not" in _weights(p["connectors"]) and rng.random() < _not_share(p):
            return Not(cond)
        return cond
    weights = {k: w for k, w in _weights(p["connectors"]).items() if k != "not"}
    if depth >= p["max_depth"] or not weights:
        return All(tuple(_node(1, depth + 1, p, domain, pool, rng) for _ in range(n)))
    kind = rng.choices(list(weights), list(weights.values()))[0]
    if kind == "unless":
        base = _node(n - 1, depth + 1, p, domain, pool, rng)
        return Unless(base, _node(1, depth + 1, p, domain, pool, rng))
    parts = _split(n, rng.randint(2, n), rng)
    children = tuple(_node(m, depth + 1, p, domain, pool, rng) for m in parts)
    if kind in ("and", "or"):
        cls = All if kind == "and" else Any
        # "a and (b and c)" is just "a and b and c": merge same-connector children
        return cls(tuple(i for c in children for i in (c.items if isinstance(c, cls) else (c,))))
    kmin, kmax = p["connectors"]["at_least"].get("k", [2, 2])
    k = rng.randint(min(kmin, len(children)), min(kmax, len(children)))
    return AtLeast(children, max(1, k))


def _not_share(p: dict) -> float:
    w = _weights(p["connectors"])
    return w.get("not", 0) / sum(w.values())


def _split(n: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, n), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [n])]


def _condition(domain: Domain, pool: list[str], p: dict, rng: random.Random) -> Cond:
    lo, hi = p["condition_pass_rate"]
    rng.shuffle(pool)
    for i, name in enumerate(pool):
        feat = domain.feature(name)
        options = []
        for op in p["ops"]:
            if feat.kind == "numeric" and op in (">=", "<=", "<", ">"):
                options += [(op, v) for v in feat.values[1:-1] if lo <= feat.pass_rate(op, v) <= hi]
            elif feat.kind != "numeric" and op in ("==", "!="):
                options += [(op, v) for v in feat.values if lo <= feat.pass_rate(op, v) <= hi]
        if options:
            pool.pop(i)  # one condition per feature
            op, v = rng.choice(options)
            return Cond(name, op, v)
    raise RuntimeError("rule_sampler: no feature left with a condition in the condition_pass_rate range")
