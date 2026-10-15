"""Feature types of a tabular domain: values, condition phrases, pass rates and mirroring.

A feature is one of
  numeric      integer grid from `range` with `step` (e.g. credit score 300-850 by 10)
  bool         True / False
  categorical  a fixed list of values

Phrases turn a condition (feature, op, value) into text at a specificity style:
  exact        "has a credit score of at least 600"
  approximate  "has a credit score of roughly 600 or higher"
  qualitative  "has a reasonably good credit score"
Domains may give their own phrases per operator and style; anything missing falls back to
generic defaults built from the feature's label, so a new domain can start with no phrases at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

def compare(value, op: str, threshold) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"unknown operator {op!r}")


@dataclass
class Feature:
    name: str
    kind: str                       # numeric | bool | categorical
    label: str                      # human name, e.g. "credit score"
    values: list                    # the full domain, in order
    step: int | None = None         # numeric grid step
    phrases: dict = field(default_factory=dict)
    overrides: list = field(default_factory=list)   # [{op, value, text}] exact-condition texts
    mirror_map: dict = field(default_factory=dict)  # categorical value swaps used by `mirror`
    sensitive: bool = False         # e.g. gender, region: candidates for hidden causes
    aliases: list = field(default_factory=list)     # other words that name the feature in texts (text checks)
    yes_when: object = None         # what makes a YES more plausible: "high" / "low" (numeric),
                                    # true / false (bool), a list of values (categorical)

    def names(self) -> list[str]:
        """Words that count as mentioning the feature: its label and aliases."""
        return [self.label.lower(), *(a.lower() for a in self.aliases)]

    def equivalent_thresholds(self, op: str, value) -> list:
        """Thresholds that state the same condition on the grid (e.g. "> 75" is ">= 80" with step 5)."""
        if self.kind != "numeric" or not self.step:
            return [value]
        alt = EQUIVALENT_OP.get(op)
        return [value] + ([value + alt[1] * self.step] if alt else [])

    # --- values -----------------------------------------------------------

    @property
    def low(self):
        return self.values[0]

    @property
    def high(self):
        return self.values[-1]

    def pass_rate(self, op: str, value) -> float:
        """Share of the feature's values meeting the condition (inputs are sampled uniformly)."""
        return sum(compare(v, op, value) for v in self.values) / len(self.values)

    def reflect(self, value):
        """Mirror a value within the feature's range (numeric), flip it (bool) or swap it (categorical)."""
        if self.kind == "numeric":
            return self.low + self.high - value
        if self.kind == "bool":
            return not value
        return self.mirror_map.get(value, value)

    def mirror_condition(self, op: str, value) -> tuple[str, object]:
        """Condition met by reflect(x) exactly when the original condition is met by x.

        Numeric results use "<" and ">=" so mirrored thresholds read naturally on the grid,
        e.g. income >= 50 (range 15-150, step 5) -> income < 120.
        """
        if self.kind == "numeric":
            r, step = self.reflect(value), self.step or 1
            return {">=": ("<", r + step), ">": ("<", r), "<=": (">=", r), "<": (">=", r + step),
                    "==": ("==", r), "!=": ("!=", r)}[op]
        return op, self.reflect(value)

    def supports_yes(self, op: str, value) -> bool | None:
        """Whether meeting `feature op value` makes a YES more plausible (None: no `yes_when` to tell)."""
        if self.yes_when is None:
            return None
        if self.kind == "numeric":
            side = {">=": "high", ">": "high", "<=": "low", "<": "low"}.get(op)
            return None if side is None else side == self.yes_when
        if self.kind == "bool":
            if op not in ("==", "!="):
                return None
            return ((value is True) == (op == "==")) == self.yes_when
        if op == "==":
            return value in self.yes_when
        if op == "!=":  # clear only when every remaining value points the same way
            rest = {v for v in self.values if v != value}
            if rest <= set(self.yes_when):
                return True
            if not rest & set(self.yes_when):
                return False
        return None

    # --- phrases ----------------------------------------------------------

    def phrase(self, op: str, value, style: str) -> str:
        """Text for the condition `feature op value` at a specificity style."""
        for o in self.overrides:
            if o["op"] == op and o["value"] == value:
                return o["text"]
        if self.kind == "bool":
            truthy = (value is True) == (op == "==")
            return self.phrases.get("true" if truthy else "false") or (
                f"has {self.label} on" if truthy else f"has {self.label} off")
        if self.kind == "categorical":
            by_value = self.phrases.get("values", {})
            if op == "==" and value in by_value:
                return by_value[value]
            not_value = self.phrases.get("not_values", {})
            if op == "!=" and value in not_value:
                return not_value[value]
            template = self.phrases.get(op)
            if isinstance(template, str):
                return template.format(v=value)
            if isinstance(template, dict) and template.get("exact"):
                return template["exact"].format(v=value)
            return f"has {self.label} equal to {value}" if op == "==" else f"does not have {self.label} equal to {value}"
        return self._numeric_phrase(op, value, style)

    def _numeric_phrase(self, op: str, value, style: str) -> str:
        entry = self.phrases.get(op, {})
        if not entry and self.step:
            # on a grid of values "> v" is ">= v+step" and "<= v" is "< v+step": reuse the domain's
            # wording for the equivalent condition rather than falling back to generic text
            alt_op, shift = EQUIVALENT_OP[op] if op in EQUIVALENT_OP else (None, 0)
            alt_value = value + shift * self.step
            if alt_op in self.phrases and self.low <= alt_value <= self.high:
                return self.phrase(alt_op, alt_value, style)
        template = entry.get(style) if isinstance(entry, dict) else None
        if isinstance(template, list):  # threshold-calibrated choices: first entry whose bounds hold
            template = next(t["text"] for t in template
                            if t.get("min", value) <= value <= t.get("max", value))
        if template:
            return template.format(v=value)
        return default_numeric_phrase(self.label, op, value, style, self.low, self.high)


EQUIVALENT_OP = {">": (">=", 1), ">=": (">", -1), "<=": ("<", 1), "<": ("<=", -1)}


def default_numeric_phrase(label: str, op: str, value, style: str, low, high) -> str:
    """Generic wording; the qualitative words depend on where the threshold sits in the range."""
    if style == "exact":
        return {">=": f"has {label} of at least {value}", ">": f"has {label} above {value}",
                "<=": f"has {label} of at most {value}", "<": f"has {label} below {value}",
                "==": f"has {label} of exactly {value}", "!=": f"does not have {label} of {value}"}[op]
    if style == "approximate":
        return {">=": f"has {label} of roughly {value} or more", ">": f"has {label} above roughly {value}",
                "<=": f"has {label} of at most roughly {value}", "<": f"has {label} below roughly {value}",
                "==": f"has {label} of about {value}", "!=": f"does not have {label} of about {value}"}[op]
    middle = (low + high) / 2
    if op in (">=", ">"):
        return f"does not have a low {label}" if value < middle else f"has a fairly high {label}"
    if op in ("<=", "<"):
        return f"does not have a high {label}" if value > middle else f"has a fairly low {label}"
    return f"has {label} around {value}" if op == "==" else f"does not have {label} around {value}"


def feature_from_config(name: str, cfg: dict) -> Feature:
    kind = cfg["type"]
    if kind == "numeric":
        low, high = cfg["range"]
        step = cfg.get("step", 1)
        values = list(range(low, high + 1, step))
        if values[-1] != high:
            raise ValueError(f"feature {name}: range {low}-{high} is not a multiple of step {step}")
    elif kind == "bool":
        values, step = [False, True], None
    elif kind == "categorical":
        values, step = list(cfg["values"]), None
    else:
        raise ValueError(f"feature {name}: unknown type {kind!r}")
    return Feature(
        name=name, kind=kind, label=cfg.get("label", name.replace("_", " ")), values=values, step=step,
        phrases=cfg.get("phrases", {}), overrides=cfg.get("overrides", []),
        mirror_map=cfg.get("mirror", {}), sensitive=cfg.get("sensitive", False),
        aliases=list(cfg.get("aliases", [])), yes_when=_yes_when(name, kind, cfg.get("yes_when")),
    )


def _yes_when(name: str, kind: str, value):
    if value is None:
        return None
    ok = {"numeric": value in ("high", "low"), "bool": isinstance(value, bool),
          "categorical": isinstance(value, list)}[kind]
    if not ok:
        raise ValueError(f"feature {name}: yes_when must be 'high' / 'low' (numeric), true / false (bool) "
                         f"or a list of values (categorical), got {value!r}")
    return value
