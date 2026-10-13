"""Turn a rule into explanation text at a specificity level (template or LLM renderer)."""

from __future__ import annotations

from nsg_sim.domain.base import Domain
from nsg_sim.render.levels import Level
from nsg_sim.render.template import render_distractor, render_rule
from nsg_sim.rules.ast import Node


def render_explanation(rule: Node, domain: Domain, level: Level, distractors: list[str] | None = None,
                       llm=None) -> str:
    if level.renderer == "template":
        if distractors:
            return render_distractor(rule, domain, level.style, distractors)
        return render_rule(rule, domain, level.style)
    if llm is None:
        raise RuntimeError(f"level {level.id} uses the llm renderer but no LLM renderer was set up")
    return llm.render(rule, domain, level, distractors)
