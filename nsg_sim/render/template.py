"""Rule -> explanation text from the domain's phrases (deterministic, no API).

Styles: exact, approximate, qualitative (per-condition phrases of the given style) and
features (only names the features, no structure or direction).

Rules whose structure reads unambiguously in one sentence are written inline: top-level "and"
items are listed ("A, B and C"), a trailing "or" group reads "either A or B", "at least k" groups
read "meets at least two of these conditions: A; B; C", and an exception reads ", unless the
<subject> X". Anything nested deeper (a group inside "or" / "at least" / "not", an exception
inside a group, a group as the exception) is written as an indented list, so the scope of every
connector is visible:

    The patient is sent to a doctor if the patient meets at least one of the following:
    - meets all of the following:
      - had surgery recently
      - has a temperature of at least 38°C
    - has an oxygen saturation below 90%
    unless the patient arrived with a GP referral.
"""

from __future__ import annotations

from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import All, Any, AtLeast, Cond, Node, Not, Unless
from nsg_sim.text import join_nl, number_word

NEGATED_OP = {">=": "<", ">": "<=", "<=": ">", "<": ">=", "==": "!=", "!=": "=="}


def render_rule(rule: Node, domain: Domain, style: str) -> str:
    if style == "features":
        names = [domain.feature(f).label for f in rule.features()]
        return domain.explanation["features_only"].format(features=join_nl(names, "and"))
    if not inline_ok(rule):
        return _structured(rule, domain, style)
    if isinstance(rule, Unless):
        return (domain.explanation["intro"] + _top(rule.base, domain, style)
                + domain.explanation["unless"] + _item(rule.exception, domain, style) + ".")
    return domain.explanation["intro"] + _top(rule, domain, style) + "."


def render_distractor(rule: Node, domain: Domain, style: str, distractors: list[str]) -> str:
    names = [domain.feature(f).label for f in distractors]
    text = render_rule(rule, domain, style)
    return text + ("\n" if "\n" in text else " ") + domain.explanation["distractor"].format(features=join_nl(names, "and"))


def _top(node: Node, domain: Domain, style: str) -> str:
    if isinstance(node, All):
        return join_nl([_item(i, domain, style) for i in node.items], "and")
    return _item(node, domain, style)


def _item(node: Node, domain: Domain, style: str) -> str:
    if isinstance(node, Cond):
        return domain.feature(node.feature).phrase(node.op, node.value, style)
    if isinstance(node, Any):
        return "either " + join_nl([_item(i, domain, style) for i in node.items], "or")
    if isinstance(node, All):
        return "both " + join_nl([_item(i, domain, style) for i in node.items], "and")
    if isinstance(node, AtLeast):
        conditions = "; ".join(_item(i, domain, style) for i in node.items)
        return f"meets at least {number_word(node.k)} of these conditions: {conditions}"
    if isinstance(node, Not):
        if isinstance(node.item, Cond):  # say the opposite condition directly
            c = node.item
            return domain.feature(c.feature).phrase(NEGATED_OP[c.op], c.value, style)
        return "does not meet the following: " + _item(node.item, domain, style)
    if isinstance(node, Unless):
        return (_item(node.base, domain, style) + domain.explanation["unless"].rstrip()
                + " " + _item(node.exception, domain, style))
    raise TypeError(node)


# --- structure ---------------------------------------------------------------------

def _simple(node: Node) -> bool:
    return isinstance(node, Cond) or (isinstance(node, Not) and isinstance(node.item, Cond))


def inline_ok(rule: Node) -> bool:
    """True when one sentence states the rule without ambiguity about what each connector covers."""
    if isinstance(rule, Unless):
        if not _simple(rule.exception):
            return False
        rule = rule.base
    if _simple(rule):
        return True
    if isinstance(rule, (Any, AtLeast)):
        return all(_simple(i) for i in rule.items)
    if isinstance(rule, All):
        *head, last = rule.items
        return all(_simple(i) for i in head) and (
            _simple(last) or (isinstance(last, Any) and all(_simple(i) for i in last.items)))
    return False


def _structured(rule: Node, domain: Domain, style: str) -> str:
    if not isinstance(rule, Unless):
        return domain.explanation["intro"] + "\n".join(_lines(rule, domain, style, 0))
    # a top-level exception goes on its own unindented line, so it clearly covers the whole list
    exc = _lines(rule.exception, domain, style, 0)
    lines = _lines(rule.base, domain, style, 0) + [domain.explanation["unless"].lstrip(", ") + exc[0]] + exc[1:]
    return domain.explanation["intro"] + "\n".join(lines) + "."


def _header(node: Node) -> str | None:
    if isinstance(node, All):
        return "meets all of the following:"
    if isinstance(node, Any):
        return "meets at least one of the following:"
    if isinstance(node, AtLeast):
        return f"meets at least {number_word(node.k)} of the following:"
    if isinstance(node, Not) and isinstance(node.item, All):
        return "does not meet all of the following:"
    if isinstance(node, Not) and isinstance(node.item, Any):
        return "meets none of the following:"
    if isinstance(node, Not) and isinstance(node.item, AtLeast):
        return f"meets fewer than {number_word(node.item.k)} of the following:"
    return None


def _lines(node: Node, domain: Domain, style: str, level: int) -> list[str]:
    """First line is the node's own text (no bullet); the rest are its indented bullet children."""
    if _simple(node):
        return [_item(node, domain, style)]
    pad = "  " * level
    if isinstance(node, Unless):
        unless = domain.explanation["unless"].rstrip()
        if _simple(node.exception):
            exc = _item(node.exception, domain, style)
            if _simple(node.base):
                return [_item(node.base, domain, style) + unless + " " + exc]
            lines = _lines(node.base, domain, style, level)
            return [lines[0].rstrip(":") + unless + " " + exc + ":"] + lines[1:]
        exc = _lines(node.exception, domain, style, level)
        return _lines(node.base, domain, style, level) + [pad + unless.lstrip(", ") + " " + exc[0]] + exc[1:]
    header = _header(node)
    if header is None:  # not of something unusual (e.g. not of an exception)
        return ["does not meet the following:"] + _bullet(_lines(node.item, domain, style, level + 1), pad)
    items = node.item.items if isinstance(node, Not) else node.items
    lines = [header]
    for child in items:
        lines += _bullet(_lines(child, domain, style, level + 1), pad)
    return lines


def _bullet(child_lines: list[str], pad: str) -> list[str]:
    return [pad + "- " + child_lines[0]] + child_lines[1:]
