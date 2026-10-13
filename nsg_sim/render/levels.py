"""Specificity levels of a dataset: any number, each rendered by a template style or an LLM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

TEMPLATE_STYLES = ("exact", "approximate", "qualitative", "features")


class LevelChecks(BaseModel):
    """Automatic checks applied to a level's texts (mainly for LLM-written ones)."""

    model_config = ConfigDict(extra="forbid")
    forbid_digits: bool = False
    require_thresholds: bool = False      # every numeric threshold of the rule appears in the text
    require_all_features: bool = False    # every feature of the rule is named
    forbid_other_features: bool = False   # no feature outside the rule is named
    max_words: int | None = None
    judge: bool = False                   # ask an LLM whether the text still describes the rule


class Level(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    renderer: str = "template"            # template | llm
    style: str | None = None              # template renderer: exact / approximate / qualitative / features
    formal: bool | None = None            # also store the rule's formal expression (default: exact style)
    # llm renderer
    model: dict | None = None             # {backend, model, ...} as in experiment simulators
    description: str | None = None
    examples: list[dict] = Field(default_factory=list)   # [{rule: <text or spec>, text: ...}]
    checks: LevelChecks = Field(default_factory=LevelChecks)
    retries: int = 2

    @model_validator(mode="after")
    def _check(self):
        if self.renderer == "template":
            if self.style not in TEMPLATE_STYLES:
                raise ValueError(f"level {self.id}: template levels need style in {TEMPLATE_STYLES}")
        elif self.renderer == "llm":
            if not self.model or not self.description:
                raise ValueError(f"level {self.id}: llm levels need `model` and `description`")
        else:
            raise ValueError(f"level {self.id}: unknown renderer {self.renderer!r}")
        if self.formal is None:
            self.formal = self.style == "exact"
        return self


DEFAULT_LEVELS = [
    Level(id="P0", style="exact"),
    Level(id="P1", style="approximate"),
    Level(id="P2", style="qualitative", checks=LevelChecks(forbid_digits=True)),
    Level(id="P3", style="features", checks=LevelChecks(forbid_digits=True)),
]
