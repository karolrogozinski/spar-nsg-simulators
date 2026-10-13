"""LLM renderer: write an explanation for a rule following a level's natural-language description.

The model gets the rule (as an exact description, so it does not need to read code), the
domain's feature names, the level description and optional examples, and returns only the
explanation text. Texts are generated once, at dataset build time, at temperature 0 and cached
on disk by prompt hash (data/_cache/llm_render/), so rebuilding never re-queries the model.
Each text is checked against the level's `checks`; failures are retried with the problems
listed, and the build stops if a level keeps failing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from nsg_sim.config import Simulator
from nsg_sim.domain.base import Domain
from nsg_sim.render.checks import check_text
from nsg_sim.render.levels import Level
from nsg_sim.render.template import render_distractor, render_rule
from nsg_sim.rules.ast import Node
from nsg_sim.storage import data_root

PROMPT = """You write explanations of decisions for a research dataset.

Context: {decision}
The {actor_short} decides with this exact rule:
{exact}

Features the rule uses: {features}.
{distractor_note}
Write the explanation the {actor_short} gives, following this description of the required level of detail:
{description}
{examples}
Within that level of detail, everything the explanation says must agree with the rule above: do not add
features, conditions or directions the rule does not have, and do not reverse any of them.
Write the explanation only, as plain text, without quotes or preamble."""

JUDGE = """Rule: {exact}
Explanation: {text}

Does the explanation describe this rule without contradicting it (it may be less precise, but every
statement it makes must agree with the rule)? Answer with one word: YES or NO."""


class LLMRenderer:
    def __init__(self):
        self.backends = {}
        self.loop = asyncio.new_event_loop()  # one loop for the renderer's lifetime (clients are bound to it)

    def _backend(self, model_cfg: dict):
        from nsg_sim.backends.base import make_backend
        sim = Simulator.model_validate({"temperature": 0.0, **model_cfg})
        key = json.dumps(sim.model_dump(mode="json"), sort_keys=True)
        if key not in self.backends:
            self.backends[key] = make_backend(sim)
        return self.backends[key]

    def _complete(self, model_cfg: dict, prompt: str) -> str:
        cache = data_root() / "_cache" / "llm_render" / (hashlib.sha256(
            (json.dumps(model_cfg, sort_keys=True) + prompt).encode()).hexdigest() + ".json")
        if cache.exists():
            return json.loads(cache.read_text())["text"]
        from nsg_sim.backends.base import Request
        resp = self.loop.run_until_complete(self._backend(model_cfg).complete(Request(prompt)))
        if resp.error or not resp.text:
            raise RuntimeError(f"LLM renderer failed: {resp.error or 'empty response'}")
        text = resp.text.strip().strip('"').strip()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": model_cfg, "prompt": prompt, "text": text}) + "\n")
        return text

    def render(self, rule: Node, domain: Domain, level: Level, distractors: list[str] | None = None) -> str:
        exact = render_distractor(rule, domain, "exact", distractors) if distractors else render_rule(rule, domain, "exact")
        examples = "".join(f"\nExample.\nRule: {e['rule']}\nExplanation: {e['text']}\n" for e in level.examples)
        prompt = PROMPT.format(
            decision=domain.task["decision"], actor_short=domain.task["actor_short"], exact=exact,
            features=", ".join(domain.feature(f).label for f in rule.features()),
            distractor_note=("The explanation should also mention, as reviewed but irrelevant: "
                             + ", ".join(domain.feature(d).label for d in distractors) + ".\n") if distractors else "",
            description=level.description.strip(), examples=examples)
        problems: list[str] = []
        for attempt in range(level.retries + 1):
            text = self._complete(level.model, prompt + ("" if not problems else
                                  "\n\nA previous attempt had these problems, avoid them: " + "; ".join(problems)))
            problems = check_text(text, rule, domain, level.checks)
            if not problems and level.checks.judge:
                verdict = self._complete(level.model, JUDGE.format(exact=exact, text=text)).upper()
                if not verdict.startswith("YES"):
                    problems = ["judge: the text does not describe the rule"]
            if not problems:
                return text
        raise ValueError(f"level {level.id}: could not produce a valid text for {rule.formal()}: {problems}")
