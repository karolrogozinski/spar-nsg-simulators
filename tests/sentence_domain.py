"""A non-tabular domain for the tests, and the pattern to copy for new input types:
short sentences whose features are computed from the words.

An input is the sentence's words by part of speech, e.g.
    {"adj": ["happy"], "noun": ["cats"], "verb": ["sleep"], "adv": ["quietly"], "question": False}
which reads "Happy cats sleep quietly." Rules are written over the computed features
(n_adj, n_noun, n_verb, n_adv, is_question), declared in YAML like tabular features, so
phrases, levels, explanation types and the rule sampler work unchanged.

This is the pattern for any new input type: subclass Domain and override
  sample_input  - a random input
  perturb       - a nearby input (here: add or drop one word of a part of speech, or flip ?/.)
  feature_values- the values rules are evaluated on
  describe      - how the input is shown to the simulator
and point a domain YAML at the class with `class: module:ClassName` (see fixtures/sentence_example.yaml).
"""

from __future__ import annotations

import random

from nsg_sim.domain.base import Domain
from nsg_sim.domain.features import feature_from_config
from nsg_sim.text import join_nl

POS = ("adj", "noun", "verb", "adv")
COUNT = {f"n_{p}": p for p in POS}


class SentenceDomain(Domain):
    def __init__(self, name: str, features: dict, words: dict, task: dict | None = None,
                 explanation: dict | None = None, description: str | None = None):
        feats = {k: feature_from_config(k, v) for k, v in features.items()}
        unknown = set(feats) - set(COUNT) - {"is_question"}
        if unknown:
            raise ValueError(f"domain {name}: features {sorted(unknown)} cannot be computed from a sentence")
        super().__init__(name, feats, task, explanation)
        self.words = {p: list(words[p]) for p in POS}
        self.description = description

    def _range(self, pos: str) -> list[int]:
        f = self.features.get(f"n_{pos}")
        return f.values if f else [0]

    def sample_input(self, rng: random.Random) -> dict:
        x = {p: rng.sample(self.words[p], rng.choice(self._range(p))) for p in POS}
        x["question"] = rng.random() < 0.5 if "is_question" in self.features else False
        return x

    def perturb(self, x: dict, rng: random.Random, focus: list[str]) -> dict:
        cx = {k: list(v) if isinstance(v, list) else v for k, v in x.items()}
        pool = focus if rng.random() < 0.8 else list(self.features)
        for name in rng.sample(pool, min(rng.choice([1, 2]), len(pool))):
            if name == "is_question":
                cx["question"] = not cx["question"]
                continue
            pos, allowed = COUNT[name], self._range(COUNT[name])
            n = len(cx[pos])
            options = [m for m in (n - 1, n + 1) if m in allowed]
            if not options:
                continue
            if rng.choice(options) > n:
                cx[pos].append(rng.choice([w for w in self.words[pos] if w not in cx[pos]]))
            else:
                cx[pos].pop(rng.randrange(n))
        return cx

    def feature_values(self, x: dict) -> dict:
        return {**{f"n_{p}": len(x[p]) for p in POS}, "is_question": x["question"]}

    def changed(self, x: dict, cx: dict) -> list[str]:
        a, b = self.feature_values(x), self.feature_values(cx)
        return [k for k in self.features if a[k] != b[k]]

    def sentence(self, x: dict) -> str:
        groups = [join_nl(x[p], "and") for p in POS if x[p]]
        text = " ".join(groups) or "..."
        return text[0].upper() + text[1:] + ("?" if x["question"] else ".")

    def describe(self, x: dict) -> str:
        return f'The sentence is: "{self.sentence(x)}"'
