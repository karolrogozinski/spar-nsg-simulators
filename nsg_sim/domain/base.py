"""Domain: what an input looks like, how to sample and perturb it, and how to talk about it.

`TabularDomain` (domain/tabular.py) covers inputs that are a dict of feature values and is
configured entirely from YAML. For other inputs (e.g. sentences whose features are computed
from the text) subclass `Domain` and override the methods marked "override".
"""

from __future__ import annotations

import random

from nsg_sim.domain.features import Feature
from nsg_sim.text import join_nl

DEFAULT_TASK = {
    "actor": "decision maker",       # e.g. "triage nurse"
    "actor_short": "decision maker",  # e.g. "nurse"
    "subject": "case",               # e.g. "patient"
    "decision": "The decision maker decides YES or NO.",
}
DEFAULT_EXPLANATION = {
    "intro": "The answer is YES if the {subject} ",
    "unless": ", unless the {subject} ",
    "features_only": "The decision was driven mainly by the {subject}'s {features}.",
    "distractor": "The {subject}'s {features} were also reviewed but did not change the outcome.",
}


class Domain:
    """Base class. `features` maps name -> Feature for every feature rules may use."""

    name: str
    features: dict[str, Feature]
    task: dict
    explanation: dict

    def __init__(self, name: str, features: dict[str, Feature], task: dict | None = None,
                 explanation: dict | None = None):
        self.name = name
        self.features = features
        self.task = {**DEFAULT_TASK, **(task or {})}
        explanation = {**DEFAULT_EXPLANATION, **(explanation or {})}
        self.explanation = {k: v.replace("{subject}", self.task["subject"]) for k, v in explanation.items()}

    # --- override for non-tabular inputs ---------------------------------------

    def sample_input(self, rng: random.Random) -> dict:
        """A random input. Override."""
        return {name: rng.choice(f.values) for name, f in self.features.items()}

    def perturb(self, x: dict, rng: random.Random, focus: list[str]) -> dict:
        """A nearby input: 1-2 features changed, mostly among `focus` (the rule's features). Override."""
        k = rng.choice([1, 2])
        pool = focus if rng.random() < 0.8 else list(self.features)
        cx = dict(x)
        for name in rng.sample(pool, min(k, len(pool))):
            cx[name] = rng.choice([v for v in self.features[name].values if v != x[name]])
        return cx

    def describe(self, x: dict) -> str:
        """Natural-language description of an input. Override."""
        raise NotImplementedError

    def feature_values(self, x: dict) -> dict:
        """Values the rules are evaluated on. Override when features are computed from the input."""
        return x

    # --- shared ----------------------------------------------------------------

    def feature(self, name: str) -> Feature:
        return self.features[name]

    def changed(self, x: dict, cx: dict) -> list[str]:
        return [k for k in x if x[k] != cx[k]]

    def generic_description(self, x: dict) -> str:
        """Fallback profile: numeric features with their ranges, then flags and categories."""
        values = self.feature_values(x)
        numeric = [f for f in self.features.values() if f.kind == "numeric"]
        parts = [f"{f.label} = {values[f.name]}" for f in numeric]
        ranges = {(f.low, f.high) for f in numeric}
        if len(ranges) == 1:
            lo, hi = next(iter(ranges))
            note = f" (each feature ranges from {lo} to {hi})"
        else:
            note = " (" + ", ".join(f"{f.label}: {f.low}-{f.high}" for f in numeric) + ")"
        flags = [f"{f.label} is {'on' if values[f.name] else 'off'}" for f in self.features.values() if f.kind == "bool"]
        categories = [f"{f.label} is {values[f.name]}" for f in self.features.values() if f.kind == "categorical"]
        others = flags + categories
        text = f"The {self.task['subject']} has {join_nl(parts, 'and')}{note}" if parts else f"The {self.task['subject']}"
        if others:
            text += ("; " if parts else " has ") + join_nl(others, "and")
        return text + "."
