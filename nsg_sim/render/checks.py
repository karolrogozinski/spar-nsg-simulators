"""Automatic checks of explanation texts against a level's `checks` settings."""

from __future__ import annotations

import re

from nsg_sim.domain.base import Domain
from nsg_sim.render.levels import LevelChecks
from nsg_sim.rules.ast import Node


def check_text(text: str, rule: Node, domain: Domain, checks: LevelChecks) -> list[str]:
    """Problems found in `text` (empty list = passes). The LLM judge is run separately."""
    problems = []
    low = text.lower()
    if checks.forbid_digits and re.search(r"\d", text):
        problems.append("contains digits")
    if checks.require_thresholds:
        for c in rule.conditions():
            feat = domain.feature(c.feature)
            if feat.kind == "numeric" and not any(re.search(rf"(?<![\d.]){re.escape(str(v))}(?!\d|\.\d)", text)  # "$80k" counts as 80
                                                  for v in feat.equivalent_thresholds(c.op, c.value)):
                problems.append(f"threshold {c.value} of {c.feature} missing")
    if checks.require_all_features:
        for name in rule.features():
            if not _mentions(low, domain.feature(name)):
                problems.append(f"feature {domain.feature(name).label!r} not mentioned")
    if checks.forbid_other_features:
        used = set(rule.features())
        for name, feat in domain.features.items():
            if name not in used and _mentions(low, feat):
                problems.append(f"mentions unrelated feature {feat.label!r}")
    if checks.max_words is not None and len(text.split()) > checks.max_words:
        problems.append(f"longer than {checks.max_words} words")
    return problems


def _mentions(low_text: str, feature) -> bool:
    return any(re.search(rf"\b{re.escape(n)}\b", low_text) for n in feature.names())
