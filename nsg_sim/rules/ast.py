"""Decision rules as a small tree: conditions joined by and / or / not / at-least-k / unless.

YAML / JSON form (`from_spec`, `to_spec`):
  [credit_score, ">=", 600]                      a condition
  {all: [...]}                                   every item holds
  {any: [...]}                                   at least one item holds
  {at_least: 2, of: [...]}                       at least k items hold
  {not: ...}                                     the item does not hold
  {all: [...], unless: [region, "==", east]}     any node may carry an `unless` exception
Nodes are immutable; edits (used by explanation transforms) return new trees.
"""

from __future__ import annotations

from dataclasses import dataclass

from nsg_sim.domain.features import compare


class Node:
    def evaluate(self, x: dict) -> bool:
        raise NotImplementedError

    def formal(self) -> str:
        """Python expression over the feature names; eval(formal, {}, x) == evaluate(x)."""
        raise NotImplementedError

    def conditions(self) -> list["Cond"]:
        raise NotImplementedError

    def features(self) -> list[str]:
        out = []
        for c in self.conditions():
            if c.feature not in out:
                out.append(c.feature)
        return out

    @property
    def size(self) -> int:
        return len(self.conditions())

    @property
    def depth(self) -> int:
        raise NotImplementedError

    def replace(self, old: "Cond", new: "Cond | None") -> "Node | None":
        """Tree with condition `old` replaced by `new` (None removes it; empty groups collapse)."""
        raise NotImplementedError

    def map_conditions(self, fn) -> "Node":
        """Tree with every condition c replaced by fn(c)."""
        raise NotImplementedError


def _wrapped(node: Node) -> str:
    return node.formal() if isinstance(node, (Cond, All, Not)) else f"({node.formal()})"


@dataclass(frozen=True)
class Cond(Node):
    feature: str
    op: str
    value: object

    def evaluate(self, x):
        return compare(x[self.feature], self.op, self.value)

    def formal(self):
        return f"{self.feature} {self.op} {self.value!r}"

    def conditions(self):
        return [self]

    @property
    def depth(self):
        return 0

    def replace(self, old, new):
        return new if self == old else self

    def map_conditions(self, fn):
        return fn(self)


def _collapse(cls, items: list, **kw) -> Node | None:
    items = [i for i in items if i is not None]
    if not items:
        return None
    if cls in (All, Any) and len(items) == 1:
        return items[0]
    return cls(tuple(items), **kw)


@dataclass(frozen=True)
class All(Node):
    items: tuple

    def evaluate(self, x):
        return all(i.evaluate(x) for i in self.items)

    def formal(self):
        return " and ".join(_wrapped(i) for i in self.items)

    def conditions(self):
        return [c for i in self.items for c in i.conditions()]

    @property
    def depth(self):
        return 1 + max(i.depth for i in self.items)

    def replace(self, old, new):
        return _collapse(All, [i.replace(old, new) for i in self.items])

    def map_conditions(self, fn):
        return All(tuple(i.map_conditions(fn) for i in self.items))


@dataclass(frozen=True)
class Any(Node):
    items: tuple

    def evaluate(self, x):
        return any(i.evaluate(x) for i in self.items)

    def formal(self):
        return " or ".join(_wrapped(i) for i in self.items)

    def conditions(self):
        return [c for i in self.items for c in i.conditions()]

    @property
    def depth(self):
        return 1 + max(i.depth for i in self.items)

    def replace(self, old, new):
        return _collapse(Any, [i.replace(old, new) for i in self.items])

    def map_conditions(self, fn):
        return Any(tuple(i.map_conditions(fn) for i in self.items))


@dataclass(frozen=True)
class AtLeast(Node):
    items: tuple
    k: int

    def evaluate(self, x):
        return sum(i.evaluate(x) for i in self.items) >= self.k

    def formal(self):
        # booleans add up in eval: a counting rule over conditions, no arithmetic on feature values
        return "((" + ") + (".join(i.formal() for i in self.items) + f")) >= {self.k}"

    def conditions(self):
        return [c for i in self.items for c in i.conditions()]

    @property
    def depth(self):
        return 1 + max(i.depth for i in self.items)

    def replace(self, old, new):
        items = [i for i in (i.replace(old, new) for i in self.items) if i is not None]
        if not items:
            return None
        return AtLeast(tuple(items), min(self.k, len(items)))

    def map_conditions(self, fn):
        return AtLeast(tuple(i.map_conditions(fn) for i in self.items), self.k)


@dataclass(frozen=True)
class Not(Node):
    item: Node

    def evaluate(self, x):
        return not self.item.evaluate(x)

    def formal(self):
        return f"not ({self.item.formal()})"

    def conditions(self):
        return self.item.conditions()

    @property
    def depth(self):
        return 1 + self.item.depth

    def replace(self, old, new):
        item = self.item.replace(old, new)
        return None if item is None else Not(item)

    def map_conditions(self, fn):
        return Not(self.item.map_conditions(fn))


@dataclass(frozen=True)
class Unless(Node):
    base: Node
    exception: Node

    def evaluate(self, x):
        return self.base.evaluate(x) and not self.exception.evaluate(x)

    def formal(self):
        return f"{_wrapped(self.base)} and not ({self.exception.formal()})"

    def conditions(self):
        return self.base.conditions() + self.exception.conditions()

    @property
    def depth(self):
        return 1 + max(self.base.depth, self.exception.depth)

    def replace(self, old, new):
        base, exc = self.base.replace(old, new), self.exception.replace(old, new)
        if exc is None:
            return base
        if base is None:
            return Not(exc)
        return Unless(base, exc)

    def map_conditions(self, fn):
        return Unless(self.base.map_conditions(fn), self.exception.map_conditions(fn))


# --- (de)serialisation ------------------------------------------------------

def from_spec(spec) -> Node:
    if isinstance(spec, Node):
        return spec
    if isinstance(spec, (list, tuple)):
        if len(spec) != 3:
            raise ValueError(f"a condition is [feature, op, value], got {spec!r}")
        return Cond(spec[0], spec[1], spec[2])
    if not isinstance(spec, dict):
        raise ValueError(f"cannot read rule {spec!r}")
    spec = dict(spec)
    unless = spec.pop("unless", None)
    if "all" in spec:
        node = _group(All, spec["all"])
    elif "any" in spec:
        node = _group(Any, spec["any"])
    elif "at_least" in spec or "atleast" in spec:  # "atleast" is the legacy generators' spelling
        k = spec.get("at_least", spec.get("atleast"))
        node = AtLeast(tuple(from_spec(i) for i in spec["of"]), int(k))
    elif "not" in spec:
        node = Not(from_spec(spec["not"]))
    else:
        raise ValueError(f"unknown rule node {spec!r}")
    return Unless(node, from_spec(unless)) if unless is not None else node


def _group(cls, items) -> Node:
    nodes = tuple(from_spec(i) for i in items)
    return nodes[0] if len(nodes) == 1 else cls(nodes)


def to_spec(node: Node):
    if isinstance(node, Cond):
        return [node.feature, node.op, node.value]
    if isinstance(node, All):
        return {"all": [to_spec(i) for i in node.items]}
    if isinstance(node, Any):
        return {"any": [to_spec(i) for i in node.items]}
    if isinstance(node, AtLeast):
        return {"at_least": node.k, "of": [to_spec(i) for i in node.items]}
    if isinstance(node, Not):
        return {"not": to_spec(node.item)}
    if isinstance(node, Unless):
        base = to_spec(node.base)
        if not isinstance(base, dict) or "unless" in base:
            base = {"all": [base]}
        return {**base, "unless": to_spec(node.exception)}
    raise TypeError(node)
