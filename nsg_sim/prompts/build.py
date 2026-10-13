"""Simulator prompts: record + domain task + template -> text, with a leak check.

Only these record fields reach a prompt: description_x, label_x, explanation_text /
explanation_formal, evidence[*].description / label, description_cx. Rules, counterfactual
labels, `discriminating` and `contradicts_explanation` never do.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).with_name("templates")


@lru_cache(maxsize=None)
def load_template(name: str) -> dict:
    path = Path(name) if Path(name).suffix else TEMPLATES_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text())


def _fields(task: dict) -> dict:
    actor_short, subject = task["actor_short"], task["subject"]
    return {**task, "Actor_short": actor_short[:1].upper() + actor_short[1:], "Subject": subject[:1].upper() + subject[1:],
            "ACTOR_SHORT": actor_short.upper(), "SUBJECT": subject.upper()}


def build_prompt(record: dict, task: dict, template: str = "nsg_paper", instruction: str = "follow",
                 explanation_format: str = "text", with_explanation: bool = True) -> str:
    t = load_template(template)
    fields = _fields(task)
    fmt = lambda key, **kw: t[key].format(**fields, **kw)  # noqa: E731
    evidence = record.get("evidence") or []

    setup = [fmt("setup_with_explanation" if with_explanation else "setup_no_explanation")]
    if evidence:
        setup.append(fmt("setup_evidence", n=len(setup) + 1))
    setup.append(fmt("setup_counterfactual", n=len(setup) + 1))

    reference = fmt("reference", description=record["description_x"], label=record["label_x"])
    if with_explanation:
        text = record.get("explanation_formal") if explanation_format == "formal" else None
        reference += fmt("explanation", explanation=text or record["explanation_text"])

    if instruction == "follow":
        intro = f"{fmt('intro_task')} {fmt('intro_follow')}\n\n{task['decision']}"
        # as in the paper, the "follow" sentence is only added when there is an explanation
        instr = fmt("instruction_follow") + (fmt("instruction_follow_explanation") if with_explanation else "")
    elif instruction == "neutral":
        intro = f"{fmt('intro_task')} {fmt('intro_neutral')}\n\n{task['decision']}"
        instr = fmt("instruction_neutral")
    else:
        raise ValueError(f"unknown instruction {instruction!r}")

    sections = [intro, "\n".join(setup), instr, reference]
    if evidence:
        items = "\n\n".join(fmt("evidence_item", i=i, description=e["description"], label=e["label"])
                            for i, e in enumerate(evidence, 1))
        sections.append(fmt("evidence_header") + "\n" + items)
    sections.append(fmt("counterfactual", description=record["description_cx"]))
    sections.append(fmt("answer_format"))
    prompt = "\n\n".join(sections)
    check_no_leak(record, prompt)
    return prompt


def check_no_leak(record: dict, prompt: str) -> None:
    """Fail if a prompt reveals the true rule or analysis-only fields."""
    for word in ("contradicts_explanation", "discriminating"):
        assert word not in prompt, f"{record['record_id']}: prompt contains {word!r}"
    # the true rule may only appear when the explanation states it (faithful / distractor)
    if record["described_formal"] != record["rule_formal"]:
        assert record["rule_formal"] not in prompt, f"{record['record_id']}: true rule leaked into the prompt"
