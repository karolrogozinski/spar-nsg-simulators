"""YAML configuration schemas. Unknown keys are errors, so typos fail before anything runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any as AnyType, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nsg_sim.render.levels import DEFAULT_LEVELS, Level

EXPLANATION_TYPES = ("faithful", "omission", "substitution", "wrong_threshold", "distractor")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- datasets ------------------------------------------------------------------

class RuleEntry(Strict):
    """A hand-written rule. Optional edits pin the unfaithful variants (otherwise chosen automatically)."""

    id: str
    rule: AnyType                                   # rule spec (see rules/ast.py)
    omit: list | None = None                        # condition(s) dropped by `omission`: one condition or a list
    substitute: list[list] | None = None            # [old condition, proxy condition], or a list of such pairs
    shift: list[list] | None = None                 # [old condition, condition with another threshold]
    distractors: list[str] | None = None            # features mentioned by `distractor`
    tags: dict = Field(default_factory=dict)        # free metadata copied into records (e.g. level: 2)


class Chain(Strict):
    conditions: list[list]                          # ordered conditions [feature, op, value]
    unless: list | None = None                      # negative condition for the `unless` structure


class ChainBuilder(Strict):
    """Rules from ordered condition chains: nested "and" rules of growing size + other structures."""

    kind: Literal["chains"] = "chains"
    chains: dict[str, Chain]
    and_sizes: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    structures: list[Literal["or", "unless", "at_least"]] = Field(default_factory=lambda: ["or", "unless", "at_least"])
    structure_size: int = 4
    at_least_k: int = 2


class RuleSampler(Strict):
    """Random rules from a configurable grammar; see rules/sampler.py and its presets."""

    preset: str | None = None
    n_rules: int | None = None
    n_conditions: list[int] | None = None           # [min, max]
    connectors: dict[str, AnyType] | None = None    # {and: w, or: w, unless: w, at_least: {weight, k}, not: w}
    max_depth: int | None = None
    condition_pass_rate: list[float] | None = None  # [min, max] share of inputs meeting one condition
    rule_yes_rate: list[float] | None = None        # [min, max] share of inputs approved by the rule
    features: dict[str, AnyType] | None = None      # {include: [...], exclude: [...], max_per_feature: 1}
    ops: list[str] | None = None
    direction: Literal["any", "intuitive", "counter"] | None = None  # intuitive: every condition points the
                                                    # way the features' `yes_when` says (needs yes_when)
    tries: int = 200


class EditCount(Strict):
    """How many conditions an omission / substitution changes; one explanation variant per value."""

    n: list[int] = Field(default_factory=lambda: [1])

    @model_validator(mode="after")
    def _check(self):
        if not self.n or any(k < 1 for k in self.n) or len(set(self.n)) != len(self.n):
            raise ValueError(f"n must be distinct numbers >= 1, got {self.n}")
        return self


DEFAULT_SHIFT = 0.12


class ThresholdShift(Strict):
    """How far wrong_threshold moves a threshold, as a share of the feature's range; one variant per value."""

    shift: list[float] = Field(default_factory=lambda: [DEFAULT_SHIFT])

    @model_validator(mode="after")
    def _check(self):
        if not self.shift or any(not 0 < s < 1 for s in self.shift) or len(set(self.shift)) != len(self.shift):
            raise ValueError(f"shift must be distinct shares between 0 and 1, got {self.shift}")
        return self


class Explanations(Strict):
    types: list[str] = Field(default_factory=lambda: ["faithful"])
    levels: dict[str, list[str]] = Field(default_factory=dict)  # type -> levels it is built for (default: all)
    n_distractors: int = 2
    omission: EditCount = Field(default_factory=EditCount)
    substitution: EditCount = Field(default_factory=EditCount)
    wrong_threshold: ThresholdShift = Field(default_factory=ThresholdShift)

    @model_validator(mode="after")
    def _check(self):
        for t in self.types:
            if t not in EXPLANATION_TYPES:
                raise ValueError(f"unknown explanation type {t!r}; choose from {EXPLANATION_TYPES}")
        return self

    def sizes(self, etype: str) -> list | None:
        """Edit sizes to build for a type, or None when the type uses the single default edit."""
        if etype in ("omission", "substitution"):
            n = getattr(self, etype).n
            return None if n == [1] else n
        if etype == "wrong_threshold":
            shift = self.wrong_threshold.shift
            return None if shift == [DEFAULT_SHIFT] else shift
        return None


class Counterfactuals(Strict):
    n_reference: int = 2                            # reference inputs per rule (alternating YES / NO)
    n_per_reference: int = 8
    balance: Literal["loose", "exact"] = "loose"
    # loose: round-robin over (discriminating, label) buckets that exist
    # exact: every reference gets n/2 YES and n/2 NO counterfactuals (references are resampled until possible)
    pool_size: int = 3000


class Evidence(Strict):
    n: int = 5
    contradicting: list[int] = Field(default_factory=lambda: [0, 1, 3, 5])
    types: list[str] = Field(default_factory=lambda: ["omission", "substitution"])
    other_types_contradicting: list[int] = Field(default_factory=list)  # e.g. [0]: k0 sets for the other types
    levels: list[str] = Field(default_factory=lambda: ["P0"])


class DatasetConfig(Strict):
    name: str
    description: str = ""
    domain: str | None = None
    seed: int = 0
    rules: list[RuleEntry] | None = None
    rule_builder: ChainBuilder | None = None
    rule_sampler: RuleSampler | None = None
    explanations: Explanations = Field(default_factory=Explanations)
    counterfactuals: Counterfactuals = Field(default_factory=Counterfactuals)
    levels: list[Level] = Field(default_factory=lambda: [lvl.model_copy() for lvl in DEFAULT_LEVELS])
    evidence: Evidence | None = None
    # derived datasets: paired transform of another built dataset
    derive_from: str | None = None
    transform: Literal["mirror", "abstract"] | None = None

    @model_validator(mode="after")
    def _check(self):
        sources = [s for s in (self.rules, self.rule_builder, self.rule_sampler) if s is not None]
        if self.derive_from:
            if not self.transform or sources:
                raise ValueError("derived datasets take `derive_from` + `transform` and no rules")
        else:
            if self.domain is None or len(sources) != 1:
                raise ValueError("give a `domain` and exactly one of `rules`, `rule_builder`, `rule_sampler`")
        ids = [lvl.id for lvl in self.levels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate level ids {ids}")
        for t, lvls in self.explanations.levels.items():
            if not set(lvls) <= set(ids):
                raise ValueError(f"explanations.levels[{t}] uses unknown levels {sorted(set(lvls) - set(ids))}")
        return self


# --- experiments ------------------------------------------------------------------

class Simulator(Strict):
    backend: str                                     # ollama | openrouter | openai | vllm | baseline | ...
    model: str
    label: str | None = None                         # name in results (default: model[@reasoning_effort])
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    base_url: str | None = None
    params: dict = Field(default_factory=dict)       # extra backend-specific request fields
    tags: dict = Field(default_factory=dict)         # labels for analysis, e.g. {family: gemma3, size_b: 12};
                                                     # results get them as model.family, model.size_b

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"{self.model}@{self.reasoning_effort}" if self.reasoning_effort else self.model


class PromptConfig(Strict):
    template: str = "nsg_paper"
    instruction: Literal["follow", "neutral"] = "follow"
    explanation_format: Literal["text", "formal"] = "text"


class Filter(Strict):
    explanation_types: list[str] | None = None
    levels: list[str] | None = None
    evidence_conditions: list[str] | None = None
    rule_ids: list[str] | None = None
    where: dict[str, list] = Field(default_factory=dict)   # any other record field -> allowed values


class Sample(Strict):
    per_cell: int | None = None                      # items per (rule, type, evidence condition), all levels kept
    seed: int = 0
    limit: int | None = None


class RunSettings(Strict):
    concurrency: int = 4
    max_tokens: int = 2048
    max_requests: int | None = None                  # stop before sending more than this many requests
    timeout: float = 300
    retries: int = 2


class ExperimentConfig(Strict):
    name: str
    description: str = ""
    dataset: str
    dataset_sha: str | None = None                   # optional pin: refuse to run on other content (prefix ok)
    filter: Filter = Field(default_factory=Filter)
    sample: Sample = Field(default_factory=Sample)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    with_baseline: bool = True                       # also run prompts without the explanation (needed for NSG)
    simulators: list[Simulator]
    run: RunSettings = Field(default_factory=RunSettings)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))
