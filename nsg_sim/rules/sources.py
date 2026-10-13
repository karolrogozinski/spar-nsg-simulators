"""Where a dataset's rules come from: a hand-written list, condition chains or the random sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from nsg_sim.config import ChainBuilder, DatasetConfig, RuleEntry
from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import All, Any, AtLeast, Cond, Node, Unless, from_spec


@dataclass
class RuleItem:
    id: str
    rule: Node
    omit: list[Cond] = field(default_factory=list)                    # pinned omissions, in order
    substitute: list[tuple[Cond, Cond]] = field(default_factory=list)  # pinned (old, proxy) pairs, in order
    shift: tuple[Cond, Cond] | None = None
    distractors: list[str] | None = None
    tags: dict = field(default_factory=dict)


def rules_from_config(cfg: DatasetConfig, domain: Domain) -> list[RuleItem]:
    if cfg.rules is not None:
        items = [_from_entry(e) for e in cfg.rules]
    elif cfg.rule_builder is not None:
        items = _from_chains(cfg.rule_builder)
    else:
        from nsg_sim.rules.sampler import sample_rules
        items = sample_rules(cfg.rule_sampler, domain, random.Random(f"{cfg.seed}|rules"))
    for item in items:
        validate_rule(item.rule, domain, item.id)
        item.tags.setdefault("n_conditions", item.rule.size)
        item.tags.setdefault("depth", item.rule.depth)
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate rule ids: {ids}")
    return items


def validate_rule(rule: Node, domain: Domain, rule_id: str) -> None:
    for c in rule.conditions():
        if c.feature not in domain.features:
            raise ValueError(f"rule {rule_id}: unknown feature {c.feature!r}")
        f = domain.feature(c.feature)
        if f.kind != "numeric" and c.value not in f.values:
            raise ValueError(f"rule {rule_id}: {c.feature} has no value {c.value!r} (values: {f.values})")


def _cond(spec) -> Cond:
    node = from_spec(spec)
    if not isinstance(node, Cond):
        raise ValueError(f"expected a condition [feature, op, value], got {spec!r}")
    return node


def _from_entry(e: RuleEntry) -> RuleItem:
    return RuleItem(
        id=e.id, rule=from_spec(e.rule),
        omit=[_cond(c) for c in (e.omit if e.omit and isinstance(e.omit[0], list) else [e.omit] if e.omit else [])],
        substitute=[(_cond(a), _cond(b)) for a, b in
                    (e.substitute if e.substitute and isinstance(e.substitute[0][0], list) else
                     [e.substitute] if e.substitute else [])],
        shift=(_cond(e.shift[0]), _cond(e.shift[1])) if e.shift else None,
        distractors=e.distractors, tags=dict(e.tags),
    )


def _from_chains(b: ChainBuilder) -> list[RuleItem]:
    items = []
    for chain_id, chain in b.chains.items():
        conds = [_cond(c) for c in chain.conditions]
        for k in b.and_sizes:
            rule = conds[0] if k == 1 else All(tuple(conds[:k]))
            items.append(RuleItem(f"{chain_id}_and_n{k}", rule, tags={"chain": chain_id, "structure": "and"}))
        k = b.structure_size
        head = conds[:k]
        for structure in b.structures:
            if structure == "or":
                rule = All(tuple(head[:-2]) + (Any(tuple(head[-2:])),))
            elif structure == "unless":
                if chain.unless is None:
                    raise ValueError(f"chain {chain_id}: the unless structure needs `unless`")
                base = head[0] if k == 2 else All(tuple(head[:-1]))
                rule = Unless(base, _cond(chain.unless))
            else:
                rule = AtLeast(tuple(head), b.at_least_k)
            name = "k_of_n" if structure == "at_least" else structure
            items.append(RuleItem(f"{chain_id}_{name}_n{k}", rule, tags={"chain": chain_id, "structure": name}))
    return items
