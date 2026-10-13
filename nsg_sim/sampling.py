"""Reference inputs, counterfactuals and evidence, for any domain.

A reference input x must get the same decision from the true rule f and from every explanation
rule g (an unfaithful explanation looks right on the case it explains). Counterfactuals are
distinct perturbations of x (1-2 features changed), bucketed by (f != g, f) and chosen round-robin,
so the set is balanced over labels and discriminating cases where possible.
"""

from __future__ import annotations

import json
import random

from nsg_sim.domain.base import Domain
from nsg_sim.rules.ast import Node

BUCKET_ORDER = [(True, True), (True, False), (False, True), (False, False)]


def key(x: dict) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def evaluate(rule: Node, domain: Domain, x: dict) -> bool:
    return rule.evaluate(domain.feature_values(x))


def candidate_buckets(domain: Domain, x: dict, f: Node, g: Node | None, rng: random.Random,
                      focus: list[str], pool_size: int) -> dict[tuple, list[dict]]:
    buckets: dict[tuple, list[dict]] = {}
    seen = {key(x)}
    for _ in range(pool_size):
        cx = domain.perturb(x, rng, focus)
        k = key(cx)
        if k in seen:
            continue
        seen.add(k)
        fx = evaluate(f, domain, cx)
        disc = g is not None and fx != evaluate(g, domain, cx)
        buckets.setdefault((disc, fx), []).append(cx)
    return buckets


def take_round_robin(buckets: dict[tuple, list[dict]], keys: list[tuple], n: int) -> list[dict]:
    keys = [k for k in keys if k in buckets]
    out = []
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop(0))
    return out


def balanced_possible(buckets: dict[tuple, list[dict]], n: int) -> bool:
    """Enough counterfactuals of each label for an exact n/2 + n/2 split."""
    return all(sum(len(buckets.get((d, lab), [])) for d in (True, False)) >= n // 2 for lab in (True, False))


def select_counterfactuals(buckets: dict[tuple, list[dict]], n: int, balance: str) -> list[dict]:
    if balance == "exact":
        half = n // 2
        yes = take_round_robin(buckets, [(True, True), (False, True)], half)
        no = take_round_robin(buckets, [(True, False), (False, False)], n - half)
        out = []
        for i in range(max(len(yes), len(no))):  # interleave so the order does not encode the label
            out += [lst[i] for lst in (yes, no) if i < len(lst)]
        return out
    return take_round_robin(buckets, BUCKET_ORDER, n)


def sample_reference(domain: Domain, f: Node, gs: list[Node], target: bool, rng: random.Random,
                     tries: int = 200_000) -> dict:
    for _ in range(tries):
        x = domain.sample_input(rng)
        if all(evaluate(r, domain, x) == target for r in [f, *gs]):
            return x
    raise RuntimeError(f"no reference input with decision {target} for rule {f.formal()}")


def make_evidence(domain: Domain, buckets: dict[tuple, list[dict]], rng: random.Random, n: int,
                  k_levels: list[int], f: Node, g: Node, exclude: list[dict]) -> dict[int, list[dict]]:
    """Nested evidence sets: k -> n inputs labelled by f, k of which contradict g (f != g).

    Start from n inputs consistent with g (balanced labels) and overwrite k fixed random positions
    with contradicting inputs; the k=1 set is the k=0 set with one swap, k=3 extends it, etc.
    """
    consistent = take_round_robin(buckets, [(False, True), (False, False)], n)
    contra = take_round_robin(buckets, [(True, True), (True, False)], max(k_levels))
    if len(contra) < max(k_levels):  # nearby contradicting inputs ran out: draw random ones
        seen = {key(e) for e in exclude + consistent + contra + [e for b in buckets.values() for e in b]}
        for _ in range(200_000):
            if len(contra) >= max(k_levels):
                break
            e = domain.sample_input(rng)
            if key(e) not in seen and evaluate(f, domain, e) != evaluate(g, domain, e):
                seen.add(key(e))
                contra.append(e)
    if len(consistent) < n or len(contra) < max(k_levels):
        raise RuntimeError(f"not enough evidence: {len(consistent)} consistent, {len(contra)} contradicting")
    positions = rng.sample(range(n), n)
    sets = {}
    for k in k_levels:
        ev = list(consistent)
        for i in range(k):
            ev[positions[i]] = contra[i]
        sets[k] = ev
    return sets
