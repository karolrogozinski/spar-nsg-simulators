"""Explanation variants of a rule f: the rule g each explanation describes.

  faithful         g = f
  omission         n conditions of f dropped (sensitive features first), or one condition chosen by
                   its relation to the simulators' priors (targets aligned / opposed / neutral)
  substitution     n conditions replaced by plausible proxies pointing the same way
  wrong_threshold  one numeric threshold moved by a share of the feature's range
  distractor       g = f, the text also names features that played no role
The sizes (n, the share) come from the recipe (`explanations.omission.n` etc.; default one condition
and 12% of the range). Hand-written rules may pin edits (`omit`, `substitute`, `shift`,
`distractors`); everything not pinned is chosen automatically and deterministically. A variant that
cannot be built for a rule (e.g. omitting 2 conditions of a 2-condition rule) is skipped for that rule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nsg_sim.config import DEFAULT_SHIFT
from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import Cond, Node, Not, Unless
from nsg_sim.rules.sources import RuleItem



@dataclass
class Variant:
    """One explanation of a rule: its type, how big the edit is, and the rule g it describes."""

    key: str                    # unique per rule: "omission", or "omission@2" when sizes are configured
    etype: str
    size: float | int | None    # conditions changed (omission, substitution) or share of range moved (wrong_threshold)
    g: Node
    extras: dict
    target: str | None = None   # omission target (aligned / opposed / neutral) when the recipe asks for them


def explanation_label(rec: dict) -> str:
    """The explanation type, with the edit size when the dataset has several ("omission n=2")."""
    if rec.get("edit_target"):
        return f"{rec['explanation_type']} {rec['edit_target']}"
    size = rec.get("edit_size")
    if size is None:
        return rec["explanation_type"]
    return f"{rec['explanation_type']} {'shift' if rec['explanation_type'] == 'wrong_threshold' else 'n'}={size}"


def variant_key(etype: str, size) -> str:
    return etype if size is None else f"{etype}@{size}"


def build_variants(item: RuleItem, domain: Domain, explanations, rng: random.Random) -> list[Variant]:
    """Every explanation variant of a rule. Variants that cannot be built for this rule are left out.

    `explanations` is the recipe's `explanations` section (types, edit sizes, number of distractors).
    For several sizes of one type the edits are nested: omitting 2 conditions drops the 1-condition
    omission's condition plus one more, so a bigger size is strictly more unfaithful.
    """
    f = item.rule
    out = []
    for etype in explanations.types:
        seed = rng.random()  # one seed per type: every size edits the same conditions in the same order
        if explanations.targets(etype):
            for target in explanations.targets(etype):
                cond = _pick_target(f, domain, target, explanations.omission.strong, random.Random(f"{seed}|{target}"))
                g = f.replace(cond, None) if cond is not None else None
                if g is not None:
                    out.append(Variant(f"{etype}@{target}", etype, None, g, {"edited": [cond]}, target))
            continue
        for size in explanations.sizes(etype) or [None]:
            g, extras = _edit(f, item, domain, etype, size, explanations, random.Random(seed))
            if g is None:
                continue
            if etype in ("omission", "substitution", "wrong_threshold") and g == f:
                raise ValueError(f"rule {item.id}: {etype} did not change the rule")
            out.append(Variant(variant_key(etype, size), etype, size, g, extras))
    return out


def _edit(f: Node, item: RuleItem, domain: Domain, etype: str, size, explanations, rng: random.Random):
    if etype == "faithful":
        return f, {}
    if etype == "distractor":
        names = item.distractors or _pick_distractors(f, domain, rng, explanations.n_distractors)
        return (f, {"distractors": names}) if names else (None, {})
    if etype == "omission":
        n = size or 1
        order = _edit_order(f, domain, item.omit, rng)
        if n >= len(f.conditions()) or len(order) < n:
            return None, {}
        g = f
        for cond in order[:n]:
            g = g.replace(cond, None)
        return (g, {"edited": order[:n]}) if g is not None else (None, {})
    if etype == "substitution":
        n = size or 1
        pairs = _pick_substitutions(f, domain, item.substitute, n, rng)
        if len(pairs) < n:
            return None, {}
        g = f
        for old, new in pairs:
            g = g.replace(old, new)
        return g, {"edited": [old for old, _ in pairs], "proxy": [new for _, new in pairs]}
    if etype == "wrong_threshold":
        shares = explanations.wrong_threshold.shift
        edit = (item.shift if size is None and item.shift is not None else
                _pick_shift(f, domain, item.shift, size or DEFAULT_SHIFT, max(shares), rng))
        return (f.replace(*edit), {"edited": edit[0], "shifted": edit[1]}) if edit else (None, {})
    raise ValueError(etype)


def _edit_order(f: Node, domain: Domain, pinned: list[Cond], rng: random.Random) -> list[Cond]:
    """Conditions in the order they get edited: pinned ones first, then sensitive features, then the rest."""
    conds = f.conditions()
    rest = [c for c in conds if c not in pinned]
    rng.shuffle(rest)
    rest.sort(key=lambda c: not domain.feature(c.feature).sensitive)
    return [c for c in pinned if c in conds] + rest


def _pick_target(f: Node, domain: Domain, target: str, strong: list[str], rng: random.Random) -> Cond | None:
    """A condition of f to omit by its relation to the simulators' priors (None if the rule has none).

    aligned: a `strong` feature whose condition points the way its `yes_when` says (meeting it makes YES
    more plausible); opposed: a `strong` feature pointing the other way; neutral: a feature not in `strong`.
    Needs a rule without negations or exceptions, where meeting any condition pushes towards YES.
    """
    if any(isinstance(n, (Not, Unless)) for n in _nodes(f)):
        raise ValueError(f"omission targets need rules without negations or exceptions: {f.formal()}")

    def kind(c: Cond) -> str | None:
        if c.feature not in strong:
            return "neutral"
        return {True: "aligned", False: "opposed"}.get(domain.feature(c.feature).supports_yes(c.op, c.value))

    matches = [c for c in f.conditions() if kind(c) == target]
    return rng.choice(matches) if matches else None


def _nodes(node: Node):
    yield node
    for child in getattr(node, "items", ()) or ():
        yield from _nodes(child)
    for attr in ("item", "base", "exception"):
        if getattr(node, attr, None) is not None:
            yield from _nodes(getattr(node, attr))


def _selects(c: Cond) -> str | None:
    """Which end of a numeric feature a condition keeps: "high" for >= / >, "low" for <= / <."""
    return {">=": "high", ">": "high", "<=": "low", "<": "low"}.get(c.op)


def _pick_substitutions(f: Node, domain: Domain, pinned: list[tuple[Cond, Cond]], n: int,
                        rng: random.Random) -> list[tuple[Cond, Cond]]:
    pairs = [p for p in pinned if p[0] in f.conditions()][:n]
    used = set(f.features()) | {new.feature for _, new in pairs}
    for old in _edit_order(f, domain, [old for old, _ in pairs], rng)[len(pairs):]:
        if len(pairs) == n:
            break
        proxy = _proxy(old, domain, used, rng)
        if proxy is not None:
            pairs.append((old, proxy))
            used.add(proxy.feature)
    return pairs


def _proxy(old: Cond, domain: Domain, used: set, rng: random.Random) -> Cond | None:
    """A numeric, non-sensitive feature outside the rule whose condition is met about as often, and that
    points the same way: if `old` makes YES more plausible, so does the proxy.

    The way is read from the features' `yes_when` when both have it; otherwise the proxy keeps the end
    of the range the old condition keeps (">=" stays ">=", "<" stays "<").
    """
    old_feat = domain.feature(old.feature)
    target = old_feat.pass_rate(old.op, old.value)
    side = _selects(old)
    supports_yes = old_feat.supports_yes(old.op, old.value)
    candidates = []
    for name, feat in domain.features.items():
        if name in used or feat.sensitive or feat.kind != "numeric":
            continue
        for op in (">=", "<"):
            proxy_side = _selects(Cond(name, op, 0))
            if supports_yes is not None and feat.yes_when:
                if feat.supports_yes(op, 0) != supports_yes:
                    continue
            elif side and proxy_side != side:
                continue
            for v in feat.values[1:-1]:
                candidates.append((abs(feat.pass_rate(op, v) - target), rng.random(), Cond(name, op, v)))
    return min(candidates)[2] if candidates else None


def _pick_shift(f: Node, domain: Domain, pinned: tuple[Cond, Cond] | None, share: float, max_share: float,
                rng: random.Random) -> tuple[Cond, Cond] | None:
    """Move one numeric threshold by `share` of the feature's range.

    The condition and the direction are drawn once per rule (or taken from the pinned shift) and chosen
    so that the largest configured share still fits in the range: every share then moves the same
    threshold the same way, only further.
    """
    numeric = [c for c in f.conditions() if domain.feature(c.feature).kind == "numeric" and _selects(c)]
    if not numeric:
        return None

    def index(c):
        values = domain.feature(c.feature).values
        return values.index(c.value) if c.value in values else min(range(len(values)), key=lambda j: abs(values[j] - c.value))

    def fits(c, d, s):
        values = domain.feature(c.feature).values
        return 0 < index(c) + d * max(1, round(s * len(values))) < len(values) - 1

    if pinned is not None and pinned[0] in numeric:
        old, direction = pinned[0], 1 if pinned[1].value > pinned[0].value else -1
    else:
        old, direction = rng.choice(numeric), rng.choice([-1, 1])
        roomy = [c for c in numeric if fits(c, 1, max_share) or fits(c, -1, max_share)]
        if old not in roomy and roomy:
            old = rng.choice(roomy)
    if not fits(old, direction, max_share) and fits(old, -direction, max_share):
        direction = -direction
    for d in (direction, -direction):  # the other way only when this share runs off the range
        if fits(old, d, share):
            values = domain.feature(old.feature).values
            return old, Cond(old.feature, old.op, values[index(old) + d * max(1, round(share * len(values)))])
    return None


def _pick_distractors(f: Node, domain: Domain, rng: random.Random, n: int) -> list[str]:
    unused = [name for name in domain.features if name not in f.features()]
    return rng.sample(unused, min(n, len(unused)))
