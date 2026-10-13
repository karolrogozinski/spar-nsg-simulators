"""Dataset recipe (YAML) -> records + manifest.

Determinism: every random choice uses its own generator seeded from (seed, rule id, ...), so
the same recipe and code always give the same records, and changing one rule does not reshuffle
the others. Reference inputs are shared by all explanation types of a rule, and counterfactuals
by all specificity levels of a (rule, type, reference), which allows paired comparisons.
"""

from __future__ import annotations

import json
import random
from collections import Counter

from nsg_sim.config import DatasetConfig
from nsg_sim.domain.base import Domain
from nsg_sim.domain.tabular import TabularDomain, load_domain
from nsg_sim.explanations import build_variants
from nsg_sim.render import render_explanation
from nsg_sim.render.checks import check_text
from nsg_sim.rules.ast import from_spec, to_spec
from nsg_sim.rules.sources import rules_from_config
from nsg_sim.sampling import (balanced_possible, candidate_buckets, evaluate, make_evidence,
                              sample_reference, select_counterfactuals)

def label(value: bool) -> str:
    return "YES" if value else "NO"


def build_records(cfg: DatasetConfig, llm=None) -> tuple[list[dict], dict, Domain]:
    """Records, manifest fields and domain for a non-derived recipe."""
    domain = load_domain(cfg.domain)
    items = rules_from_config(cfg, domain)
    level_ids = [lvl.id for lvl in cfg.levels]
    cf, ev_cfg = cfg.counterfactuals, cfg.evidence
    records = []
    for item in items:
        f = item.rule
        variants = build_variants(item, domain, cfg.explanations.types, random.Random(f"{cfg.seed}|variants|{item.id}"),
                                  cfg.explanations.n_distractors)
        gs = [g for g, _ in variants.values()]
        focus = [name for name in domain.features if any(name in r.features() for r in [f, *gs])]
        # texts depend only on (type, level): render once per rule
        texts = {}
        for etype, (g, extras) in variants.items():
            for lvl in cfg.levels:
                if lvl.id in cfg.explanations.levels.get(etype, level_ids):
                    text = render_explanation(g, domain, lvl, extras.get("distractors"), llm)
                    problems = check_text(text, g, domain, lvl.checks) if lvl.renderer == "template" else []
                    if problems:
                        raise ValueError(f"rule {item.id} {etype} {lvl.id}: {problems}: {text}")
                    formal = g.formal() if lvl.formal else None
                    if formal and extras.get("distractors"):
                        formal += "  # also reviewed: " + ", ".join(extras["distractors"])
                    texts[(etype, lvl.id)] = (text, formal)

        for ref in range(cf.n_reference):
            target = ref % 2 == 0  # alternate YES, NO references
            x = _reference(domain, f, gs, target, cfg, item.id, ref, focus)
            for etype, (g, extras) in variants.items():
                rng = random.Random(f"{cfg.seed}|cx|{item.id}|{etype}|{ref}")
                buckets = candidate_buckets(domain, x, f, None if g == f else g, rng, focus, cf.pool_size)
                cxs = select_counterfactuals(buckets, cf.n_per_reference, cf.balance)
                evidence_sets, k_levels = {}, []
                if ev_cfg:
                    k_levels = ev_cfg.contradicting if etype in ev_cfg.types else ev_cfg.other_types_contradicting
                if k_levels:
                    evidence_sets = make_evidence(domain, buckets, rng, ev_cfg.n, k_levels, f, g, [x, *cxs])
                for lvl in cfg.levels:
                    if (etype, lvl.id) not in texts:
                        continue
                    text, formal = texts[(etype, lvl.id)]
                    conditions = [("none", None, [])]
                    if evidence_sets and lvl.id in ev_cfg.levels:
                        conditions += [(f"k{k}", k, evidence_sets[k]) for k in k_levels]
                    for cond, n_contra, ev in conditions:
                        evidence = [{"input": e, "description": domain.describe(e), "label": label(evaluate(f, domain, e)),
                                     "contradicts_explanation": evaluate(f, domain, e) != evaluate(g, domain, e)}
                                    for e in ev]
                        for j, cx in enumerate(cxs):
                            f_cx, g_cx = evaluate(f, domain, cx), evaluate(g, domain, cx)
                            item_id = f"{item.id}|{etype}|r{ref}|c{j}"
                            records.append({
                                "dataset": cfg.name,
                                "record_id": f"{item_id}|{lvl.id}|{cond}",
                                "item_id": item_id,
                                "cluster_id": f"{item.id}|{etype}|r{ref}",
                                "rule_id": item.id,
                                "rule_tags": item.tags,
                                "explanation_type": etype,
                                "level": lvl.id,
                                "evidence_condition": cond,
                                "n_contradicting": n_contra,
                                "rule": to_spec(f),
                                "rule_formal": f.formal(),
                                "described_rule": to_spec(g),
                                "described_formal": g.formal(),
                                "distractors": extras.get("distractors"),
                                "input_x": x,
                                "description_x": domain.describe(x),
                                "label_x": label(target),
                                "input_cx": cx,
                                "description_cx": domain.describe(cx),
                                "label_cx": label(f_cx),
                                "explanation_label_cx": label(g_cx),
                                "discriminating": f_cx != g_cx,
                                "explanation_text": text,
                                "explanation_formal": formal,
                                "evidence": evidence,
                                "variant": None,
                                "paired_record_id": None,
                                "seed": cfg.seed,
                            })
    manifest = {"description": cfg.description, "config": cfg.model_dump(mode="json"),
                "domain": _domain_info(domain), "levels": [lvl.model_dump(mode="json") for lvl in cfg.levels],
                "stats": stats(records)}
    return records, manifest, domain


def _reference(domain, f, gs, target, cfg, rule_id, ref, focus) -> dict:
    rng = random.Random(f"{cfg.seed}|ref|{rule_id}|{ref}")
    cf = cfg.counterfactuals
    if cf.balance != "exact":
        return sample_reference(domain, f, gs, target, rng)
    for _ in range(500):  # resample until 1-2 changes can reach both decisions
        x = sample_reference(domain, f, gs, target, rng)
        if balanced_possible(candidate_buckets(domain, x, f, None, rng, focus, cf.pool_size), cf.n_per_reference):
            return x
    raise RuntimeError(f"rule {rule_id}: no reference input with balanced counterfactuals")


def _domain_info(domain: Domain) -> dict:
    return {"name": domain.name, "task": domain.task, "explanation": domain.explanation}


def stats(records: list[dict]) -> dict:
    cells = Counter((r["explanation_type"], r["level"], r["evidence_condition"]) for r in records)
    none = [r for r in records if r["evidence_condition"] == "none"]
    return {
        "records_per_cell": {"|".join(k): v for k, v in sorted(cells.items())},
        "yes_share": round(sum(r["label_cx"] == "YES" for r in none) / max(1, len(none)), 3),
        "discriminating_share": round(sum(r["discriminating"] for r in none) / max(1, len(none)), 3),
        "n_rules": len({r["rule_id"] for r in records}),
    }


def validate_records(records: list[dict], domain: Domain, cfg: DatasetConfig | None = None) -> None:
    """Consistency checks; raises AssertionError with the offending record id."""
    cx_sets: dict[tuple, dict] = {}
    for r in records:
        rid = r["record_id"]
        f, g = from_spec(r["rule"]), from_spec(r["described_rule"])
        x, cx = r["input_x"], r["input_cx"]
        assert label(evaluate(f, domain, x)) == r["label_x"] == label(evaluate(g, domain, x)), f"{rid}: f/g disagree on x"
        assert label(evaluate(f, domain, cx)) == r["label_cx"], rid
        assert label(evaluate(g, domain, cx)) == r["explanation_label_cx"], rid
        assert r["discriminating"] == (r["label_cx"] != r["explanation_label_cx"]), rid
        if r["explanation_type"] in ("faithful", "distractor"):
            assert f == g, rid
        if isinstance(domain, TabularDomain):
            assert 1 <= len(domain.changed(x, cx)) <= 2, f"{rid}: counterfactual changes {domain.changed(x, cx)}"
        for e in r["evidence"]:
            assert e["label"] == label(evaluate(f, domain, e["input"])), rid
        assert sum(e["contradicts_explanation"] for e in r["evidence"]) == (r["n_contradicting"] or 0), rid
        cx_sets.setdefault((r["cluster_id"], r["evidence_condition"]), {}).setdefault(r["level"], []).append(r["item_id"])
    for k, by_level in cx_sets.items():
        lists = list(by_level.values())
        assert all(lst == lists[0] for lst in lists), f"{k}: counterfactuals differ across levels"
    if cfg is not None and cfg.counterfactuals.balance == "exact":
        per_ref = Counter()
        for r in records:
            if r["evidence_condition"] == "none" and r["level"] == cfg.levels[0].id:
                per_ref[(r["cluster_id"], r["label_cx"])] += 1
        for (cluster, lab), n in per_ref.items():
            assert n * 2 == cfg.counterfactuals.n_per_reference, f"{cluster}: {n} {lab} counterfactuals"


# --- whole datasets ------------------------------------------------------------

def domain_for_manifest(manifest: dict) -> Domain:
    """The domain a built dataset uses (derived datasets re-apply their transform to the base domain)."""
    from nsg_sim.storage import load_manifest
    from nsg_sim.transforms import transform_domain
    derived = manifest.get("derived")
    if derived:
        return transform_domain(domain_for_manifest(load_manifest(derived["from"])), derived["transform"])
    return load_domain(manifest["domain"]["name"])


def build_dataset(cfg: DatasetConfig, llm=None, save: bool = True) -> tuple[list[dict], dict]:
    """Build (or derive), validate and optionally save a dataset. Returns records and manifest."""
    from nsg_sim.render.levels import Level
    from nsg_sim.storage import load_dataset, save_dataset
    from nsg_sim.transforms import derive_records

    if cfg.derive_from:
        base_records, base_manifest = load_dataset(cfg.derive_from)
        base_domain = domain_for_manifest(base_manifest)
        levels = [Level.model_validate(lvl) for lvl in base_manifest["levels"]]
        records, domain = derive_records(base_records, base_domain, cfg.transform, levels, cfg.name, llm)
        validate_records(records, domain)
        for r, b in zip(records, base_records):
            assert (r["label_x"], r["label_cx"], r["explanation_label_cx"]) == (b["label_x"], b["label_cx"], b["explanation_label_cx"]), \
                f"{r['record_id']}: labels changed by the {cfg.transform} transform"
        manifest = {"description": cfg.description, "config": cfg.model_dump(mode="json"),
                    "domain": _domain_info(domain), "levels": base_manifest["levels"],
                    "derived": {"from": cfg.derive_from, "transform": cfg.transform, "base_sha256": base_manifest["sha256"]},
                    "stats": stats(records)}
    else:
        records, manifest, domain = build_records(cfg, llm)
        validate_records(records, domain, cfg)
    if save:
        path = save_dataset(cfg.name, records, manifest)
        manifest = json.loads((path / "manifest.json").read_text())
    return records, manifest
