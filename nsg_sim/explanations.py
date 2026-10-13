"""Explanation variants of a rule f: the rule g each explanation describes.

  faithful         g = f
  omission         one condition of f dropped
  substitution     one condition replaced by a plausible proxy (sensitive features first)
  wrong_threshold  one numeric threshold moved
  distractor       g = f, the text also names features that played no role
Hand-written rules may pin an edit (`omit`, `substitute`, `shift`, `distractors`); everything not
pinned is chosen automatically and deterministically. A type that cannot be built for a rule (e.g.
omission of a one-condition rule) is skipped for that rule.
"""

from __future__ import annotations

import random

from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import Cond, Node
from nsg_sim.rules.sources import RuleItem


def build_variants(item: RuleItem, domain: Domain, types: list[str], rng: random.Random,
                   n_distractors: int = 2) -> dict[str, tuple[Node, dict]]:
    """type -> (g, extras). Types that cannot be built for this rule are left out."""
    f = item.rule
    out = {}
    for etype in types:
        if etype == "faithful":
            out[etype] = (f, {})
        elif etype == "omission":
            cond = item.omit if item.omit is not None else _pick_omission(f, domain, rng)
            g = f.replace(cond, None) if cond is not None else None
            if g is not None and g != f:
                out[etype] = (g, {"edited": cond})
        elif etype == "substitution":
            edit = item.substitute if item.substitute is not None else _pick_substitution(f, domain, rng)
            if edit is not None:
                out[etype] = (f.replace(*edit), {"edited": edit[0], "proxy": edit[1]})
        elif etype == "wrong_threshold":
            edit = item.shift if item.shift is not None else _pick_shift(f, domain, rng)
            if edit is not None:
                out[etype] = (f.replace(*edit), {"edited": edit[0], "shifted": edit[1]})
        elif etype == "distractor":
            names = item.distractors or _pick_distractors(f, domain, rng, n_distractors)
            if names:
                out[etype] = (f, {"distractors": names})
    for etype, (g, _) in out.items():
        if etype in ("omission", "substitution", "wrong_threshold") and g == f:
            raise ValueError(f"rule {item.id}: {etype} did not change the rule")
    return out


def _pick_omission(f: Node, domain: Domain, rng: random.Random) -> Cond | None:
    conds = f.conditions()
    if len(conds) < 2:
        return None
    sensitive = [c for c in conds if domain.feature(c.feature).sensitive]
    return rng.choice(sensitive or conds)


def _pick_substitution(f: Node, domain: Domain, rng: random.Random) -> tuple[Cond, Cond] | None:
    conds = f.conditions()
    sensitive = [c for c in conds if domain.feature(c.feature).sensitive]
    old = rng.choice(sensitive or conds)
    target = domain.feature(old.feature).pass_rate(old.op, old.value)
    used = set(f.features())
    # proxy: a numeric, non-sensitive feature outside the rule whose condition is met about as often
    candidates = []
    for name, feat in domain.features.items():
        if name in used or feat.sensitive or feat.kind != "numeric":
            continue
        for op in (">=", "<="):
            for v in feat.values[1:-1]:
                candidates.append((abs(feat.pass_rate(op, v) - target), rng.random(), Cond(name, op, v)))
    if not candidates:
        return None
    return old, min(candidates)[2]


def _pick_shift(f: Node, domain: Domain, rng: random.Random) -> tuple[Cond, Cond] | None:
    numeric = [c for c in f.conditions() if domain.feature(c.feature).kind == "numeric" and c.op in (">=", ">", "<=", "<")]
    if not numeric:
        return None
    old = rng.choice(numeric)
    values = domain.feature(old.feature).values
    i = values.index(old.value) if old.value in values else min(range(len(values)), key=lambda j: abs(values[j] - old.value))
    jump = max(1, round(0.12 * len(values)))
    options = [j for j in (i - jump, i + jump) if 0 < j < len(values) - 1]
    if not options:
        return None
    return old, Cond(old.feature, old.op, values[rng.choice(options)])


def _pick_distractors(f: Node, domain: Domain, rng: random.Random, n: int) -> list[str]:
    unused = [name for name in domain.features if name not in f.features()]
    return rng.sample(unused, min(n, len(unused)))
