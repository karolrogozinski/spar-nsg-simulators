"""Rule-prior agreement: how much each rule matches what a simulator believes about the domain.

  auc            AUC of the simulator's P(YES) (mean over wordings) against the rule's labels on the
                 elicited cases: 0.5 unrelated, 1 the rule is the simulator's prior, < 0.5 the opposite
  dir_agreement  share of the rule's conditions that point the way the simulator's direction beliefs say
                 (an exception or a negation points the other way); conditions on features with no clear
                 belief are left out
  consistency    mean correlation of P(YES) between question wordings (is the prior measure stable?)

Per feature (feature_table): Spearman rho between the simulator's P(YES) and the feature over the
elicited cases, signed so that > 0 means "agrees with the domain's yes_when". A domain's prior strength
is the mean |rho| over its rule features (EXP-011: the x axis across domains).
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from nsg_sim.builder import domain_for_manifest
from nsg_sim.priors import parse_direction
from nsg_sim.rules.ast import Cond, Not, Unless, from_spec
from nsg_sim.storage import load_dataset


def load(run_dir: Path) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    inputs = json.loads((run_dir / "inputs.json").read_text())
    rows = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        latest = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            latest[(r["view"], r["kind"], str(r["item"]), r["wording"])] = r
        for r in latest.values():
            rows[r["simulator"]].append(r)
    return inputs, rows


def decision_probs(rows: list[dict], view: str, n: int) -> tuple[list[float | None], float | None]:
    by = defaultdict(dict)
    for r in rows:
        if r["view"] == view and r["kind"] == "decision" and r.get("probability") is not None:
            by[int(r["item"])][r["wording"]] = r["probability"]
    probs = [sum(by[i].values()) / len(by[i]) if by[i] else None for i in range(n)]
    wordings = sorted({w for d in by.values() for w in d})
    corrs = []
    for a, b in itertools.combinations(wordings, 2):
        pairs = [(d[a], d[b]) for d in by.values() if a in d and b in d]
        if len(pairs) > 2:
            corrs.append(_corr(pairs))
    corrs = [c for c in corrs if c is not None]
    return probs, (sum(corrs) / len(corrs) if corrs else None)


def beliefs(rows: list[dict], view: str, domain) -> dict:
    """feature -> majority belief over wordings (None when there is no majority or it is "same")."""
    votes = defaultdict(list)
    for r in rows:
        if r["view"] == view and r["kind"] == "direction" and not r.get("error"):
            votes[r["item"]].append(parse_direction(r["raw_response"], domain.feature(r["item"])))
    out = {}
    for feature, vs in votes.items():
        (top, count), = Counter(map(repr, vs)).most_common(1)
        value = next(v for v in vs if repr(v) == top)
        out[feature] = value if count * 2 > len(vs) and value not in (None, "same") else None
    return out


def auc(labels: list[bool], scores: list[float]) -> float | None:
    pos = [s for lab, s in zip(labels, scores) if lab]
    neg = [s for lab, s in zip(labels, scores) if not lab]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def _corr(pairs):
    n = len(pairs)
    mx, my = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    sx = sum((a - mx) ** 2 for a, _ in pairs) ** 0.5
    sy = sum((b - my) ** 2 for _, b in pairs) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy)


def _polarities(node, want=True):
    if isinstance(node, Cond):
        return [(node, want)]
    if isinstance(node, Not):
        return _polarities(node.item, not want)
    if isinstance(node, Unless):
        return _polarities(node.base, want) + _polarities(node.exception, not want)
    return [p for i in node.items for p in _polarities(i, want)]


def direction_agreement(rule, domain, belief: dict) -> tuple[float | None, int]:
    known = agree = 0
    for cond, want in _polarities(rule):
        b = belief.get(cond.feature)
        if b is None:
            continue
        feat = domain.feature(cond.feature)
        believed = replace(feat, yes_when=[b] if feat.kind == "categorical" else b)
        supports = believed.supports_yes(cond.op, cond.value)
        if supports is None:
            continue
        known += 1
        agree += supports == want
    return (agree / known if known else None), known


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):   # ties share their mean rank
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return _corr(list(zip(_ranks(xs), _ranks(ys)))) if len(xs) > 2 else None


def _toward_yes(feature, value) -> float | None:
    """The feature value as a number that grows toward the domain's yes_when (None: no direction)."""
    if feature.yes_when is None:
        return None
    if feature.kind == "numeric":
        return value if feature.yes_when == "high" else -value
    if feature.kind == "bool":
        return float(value == feature.yes_when)
    return float(value in feature.yes_when)


def _directed(manifest: dict) -> dict:
    """View feature name -> (base feature with yes_when, value -> base value). An abstract view keeps the
    base directions: numbers are rescaled monotonically and categories renamed in order."""
    from nsg_sim.storage import load_manifest
    from nsg_sim.transforms import AbstractMap
    derived = manifest.get("derived")
    if derived and derived["transform"] == "abstract":
        amap = AbstractMap(domain_for_manifest(load_manifest(derived["from"])))
        out = {}
        for name, letter in amap.names.items():
            f = amap.base.feature(name)
            back = {amap.value(name, v): v for v in f.values}
            out[letter] = (f, back.get)
        return out
    return {name: (f, lambda v: v) for name, f in domain_for_manifest(manifest).features.items()}


def _rule_features(records: list[dict]) -> set[str]:
    rules = {r["rule_id"]: from_spec(r["rule"]) for r in records}
    return {c.feature for rule in rules.values() for c, _ in _polarities(rule)}


def feature_table(run_dir: Path, datasets: list[str]) -> list[dict]:
    """Per simulator x domain view x feature: rho of P(YES) with the feature, signed toward yes_when.
    A mirror twin shares its base's view, so give base and abstract datasets. `rule_feature` marks the
    features the dataset's rules use."""
    inputs, rows = load(run_dir)
    out = []
    for name in datasets:
        records, manifest = load_dataset(name)
        domain = domain_for_manifest(manifest)
        xs = inputs.get(domain.name)
        if xs is None:
            raise ValueError(f"{run_dir} has no elicited cases for the {domain.name!r} view of {name}")
        used, directed = _rule_features(records), _directed(manifest)
        for sim, srows in rows.items():
            probs, _ = decision_probs(srows, domain.name, len(xs))
            for fname, (base, back) in directed.items():
                pairs = [(_toward_yes(base, back(x[fname])), p) for x, p in zip(xs, probs) if p is not None]
                rho = None if base.yes_when is None else spearman([a for a, _ in pairs], [p for _, p in pairs])
                out.append({"simulator": sim, "dataset": name, "feature": fname, "rule_feature": fname in used,
                            "rho": rho, "n_cases": len(pairs)})
    return out


def prior_strength(features: list[dict]) -> list[dict]:
    """Per simulator x dataset: mean |rho| and mean signed rho over the rule features."""
    by = defaultdict(list)
    for r in features:
        if r["rule_feature"] and r["rho"] is not None:
            by[(r["simulator"], r["dataset"])].append(r["rho"])
    return [{"simulator": s, "dataset": d, "strength": sum(map(abs, v)) / len(v),
             "alignment": sum(v) / len(v), "n_features": len(v)} for (s, d), v in sorted(by.items())]


def rule_table(run_dir: Path, datasets: list[str]) -> list[dict]:
    inputs, rows = load(run_dir)
    out = []
    for name in datasets:
        records, manifest = load_dataset(name)
        domain = domain_for_manifest(manifest)
        view = domain.name
        if view not in inputs:
            raise ValueError(f"{run_dir} has no elicited cases for the {view!r} view of {name}")
        xs = inputs[view]
        rules = {}
        for r in records:
            rules.setdefault(r["rule_id"], from_spec(r["rule"]))
        for sim, srows in rows.items():
            probs, consistency = decision_probs(srows, view, len(xs))
            belief = beliefs(srows, view, domain)
            for rid, rule in rules.items():
                pairs = [(rule.evaluate(domain.feature_values(x)), p) for x, p in zip(xs, probs) if p is not None]
                d, n_known = direction_agreement(rule, domain, belief)
                out.append({"simulator": sim, "dataset": name, "variant": manifest.get("derived", {}).get("transform", "base"),
                            "rule_id": rid, "auc": auc([a for a, _ in pairs], [p for _, p in pairs]),
                            "dir_agreement": d, "n_directions": n_known, "consistency": consistency,
                            "n_cases": len(pairs)})
    return out
