"""Paired dataset transforms: every derived record comes from one base record and keeps its labels.

  mirror    every input is reflected within each feature's range (numeric v -> low + high - v,
            booleans flipped, categorical values swapped via the feature's `mirror` map), and every
            rule condition is mirrored to match, so decisions are unchanged while the direction of
            each condition is reversed (higher credit score -> approve becomes lower -> approve)
  abstract  features renamed "feature A".., "flag B".., "category C".. (numbers rescaled
            monotonically to 0-100, categories to alpha / beta / ...), with generic wording, so
            domain knowledge no longer predicts the decision
"""

from __future__ import annotations

from nsg_sim.domain.base import Domain
from nsg_sim.domain.features import Feature
from nsg_sim.domain.tabular import TabularDomain
from nsg_sim.render import render_explanation
from nsg_sim.render.levels import Level
from nsg_sim.rules.ast import Cond, from_spec, to_spec

GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- mirror --------------------------------------------------------------------

def mirror_input(domain: Domain, x: dict) -> dict:
    return {k: domain.feature(k).reflect(v) for k, v in x.items()}


def mirror_cond(domain: Domain, c: Cond) -> Cond:
    op, value = domain.feature(c.feature).mirror_condition(c.op, c.value)
    return Cond(c.feature, op, value)


# --- abstract ------------------------------------------------------------------

class AbstractMap:
    """Renaming and rescaling of a domain's features."""

    def __init__(self, domain: Domain):
        self.base = domain
        self.names = {name: LETTERS[i] for i, name in enumerate(domain.features)}
        features = {}
        for name, f in domain.features.items():
            letter = self.names[name]
            if f.kind == "numeric":
                features[letter] = Feature(letter, "numeric", f"feature {letter}", [self.value(name, v) for v in f.values], step=None)
            elif f.kind == "bool":
                features[letter] = Feature(letter, "bool", f"flag {letter}", [False, True])
            else:
                features[letter] = Feature(letter, "categorical", f"category {letter}", [self.value(name, v) for v in f.values])
        self.domain = TabularDomain(f"{domain.name}_abstract", features, domain.task, domain.explanation)

    def value(self, name: str, v):
        f = self.base.feature(name)
        if f.kind == "numeric":
            return round((v - f.low) / (f.high - f.low) * 100)
        if f.kind == "bool":
            return v
        return GREEK[f.values.index(v)]

    def input(self, x: dict) -> dict:
        return {self.names[k]: self.value(k, v) for k, v in x.items()}

    def cond(self, c: Cond) -> Cond:
        return Cond(self.names[c.feature], c.op, self.value(c.feature, c.value))


# --- apply ---------------------------------------------------------------------

def transform_domain(domain: Domain, transform: str) -> Domain:
    return domain if transform == "mirror" else AbstractMap(domain).domain


def derive_records(base_records: list[dict], base_domain: Domain, transform: str, levels: list[Level],
                   dataset_name: str, llm=None) -> tuple[list[dict], Domain]:
    if transform == "mirror":
        domain = base_domain
        map_input = lambda x: mirror_input(base_domain, x)  # noqa: E731
        map_cond = lambda c: mirror_cond(base_domain, c)    # noqa: E731
        map_feature = lambda name: name                      # noqa: E731
    elif transform == "abstract":
        amap = AbstractMap(base_domain)
        domain, map_input, map_cond, map_feature = amap.domain, amap.input, amap.cond, amap.names.get
    else:
        raise ValueError(f"unknown transform {transform!r}")
    by_id = {lvl.id: lvl for lvl in levels}
    texts: dict[tuple, tuple] = {}
    out = []
    for r in base_records:
        f = from_spec(r["rule"]).map_conditions(map_cond)
        g = from_spec(r["described_rule"]).map_conditions(map_cond)
        distractors = [map_feature(d) for d in r["distractors"]] if r["distractors"] else None
        tkey = (r["rule_id"], r["explanation_type"], r.get("edit_size"), r["level"])
        if tkey not in texts:
            lvl = by_id[r["level"]]
            formal = g.formal() if lvl.formal else None
            if formal and distractors:
                formal += "  # also reviewed: " + ", ".join(distractors)
            texts[tkey] = (render_explanation(g, domain, lvl, distractors, llm), formal)
        text, formal = texts[tkey]
        x, cx = map_input(r["input_x"]), map_input(r["input_cx"])
        evidence = [{**e, "input": map_input(e["input"]), "description": domain.describe(map_input(e["input"]))}
                    for e in r["evidence"]]
        out.append({**r,
                    "dataset": dataset_name,
                    "rule": to_spec(f), "rule_formal": f.formal(),
                    "described_rule": to_spec(g), "described_formal": g.formal(),
                    "distractors": distractors,
                    "input_x": x, "description_x": domain.describe(x),
                    "input_cx": cx, "description_cx": domain.describe(cx),
                    "explanation_text": text, "explanation_formal": formal,
                    "evidence": evidence,
                    "variant": transform,
                    "paired_record_id": r["record_id"]})
    return out, domain
