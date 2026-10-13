"""Load run results joined with dataset records, and compute accuracy / NSG with cluster bootstrap CIs.

Each "pair" is one dataset record answered by one simulator, with and (when run) without the
explanation. Counterfactuals of the same reference input are correlated, so intervals resample
whole clusters (rule, explanation type, reference input) rather than single records.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from nsg_sim.explanations import explanation_label
from nsg_sim.prompts.parse import parse_answer
from nsg_sim.storage import load_dataset


def load_pairs(run_dirs: list[Path]) -> list[dict]:
    pairs = []
    for run_dir in run_dirs:
        meta = json.loads((Path(run_dir) / "run.json").read_text())
        records, manifest = load_dataset(meta["dataset"])
        if manifest["sha256"] != meta["dataset_sha256"]:
            raise ValueError(f"{run_dir}: dataset {meta['dataset']} changed since the run")
        by_id = {r["record_id"]: r for r in records}
        for path in sorted(Path(run_dir).glob("*.jsonl")):
            latest = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                latest[(row["record_id"], row["with_explanation"])] = row
            for (rid, with_expl), row in latest.items():
                if not with_expl:
                    continue
                rec = by_id[rid]
                base = latest.get((rid, False))
                pred = parse_answer(row["raw_response"]) if row.get("raw_response") is not None else None
                pred_wo = parse_answer(base["raw_response"]) if base and base.get("raw_response") is not None else None
                pairs.append({
                    "experiment": meta["experiment"], "simulator": row["simulator"], "dataset": rec["dataset"],
                    "variant": rec.get("variant") or "base", "record_id": rid, "item_id": rec["item_id"],
                    "cluster": f"{rec['dataset']}|{rec['cluster_id']}", "rule_id": rec["rule_id"],
                    "explanation_type": rec["explanation_type"], "edit_size": rec.get("edit_size"),
                    "explanation": explanation_label(rec), "level": rec["level"],
                    "evidence_condition": rec["evidence_condition"], "discriminating": rec["discriminating"],
                    **{f"rule_tags.{k}": v for k, v in (rec.get("rule_tags") or {}).items()},
                    "valid": pred is not None and (base is None or pred_wo is not None),
                    "with": pred == rec["label_cx"], "without": (pred_wo == rec["label_cx"]) if base else None,
                    "follows_g": pred == rec["explanation_label_cx"], "label_cx": rec["label_cx"],
                    "error": bool(row.get("error")),
                })
    return pairs


def accuracy(cell: list[dict], field: str = "with") -> float:
    v = [p[field] for p in cell if p["valid"] and p[field] is not None]
    return sum(v) / len(v) if v else float("nan")


def nsg(cell: list[dict]) -> float:
    aw, ao = accuracy(cell, "with"), accuracy(cell, "without")
    if ao != ao:  # no-explanation prompts were not run
        return float("nan")
    return (aw - ao) / (1 - ao) if ao < 1 else 0.0


def reliance(cell: list[dict]) -> float:
    """Share of answers that follow the explanation's rule, on discriminating records."""
    disc = [p for p in cell if p["discriminating"] and p["valid"]]
    return sum(p["follows_g"] for p in disc) / len(disc) if disc else float("nan")


METRICS = {"acc": accuracy, "acc_wo": lambda c: accuracy(c, "without"), "nsg": nsg, "reliance": reliance}


def cluster_ci(cell: list[dict], stat, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for p in cell:
        by[p["cluster"]].append(p)
    keys = list(by)
    vals = sorted(v for v in (stat([p for k in (rng.choice(keys) for _ in keys) for p in by[k]]) for _ in range(n_boot))
                  if v == v)
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def group(pairs: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for p in pairs:
        out[tuple(p.get(k) for k in keys)].append(p)
    return out
