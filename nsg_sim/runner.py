"""Run an experiment: dataset records -> prompts -> simulators -> runs/<experiment>/<simulator>.jsonl.

- Records are filtered and sampled per cell (rule, explanation type, evidence condition); sampling
  picks counterfactual items and keeps all their specificity levels, so levels stay paired.
- Identical prompts (e.g. the no-explanation prompt, shared by every level) are sent once.
- Re-running resumes: rows already answered without an API error are skipped.
- The run directory records the resolved config, dataset hash and git commit. Resuming on a
  dataset whose content changed is refused.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from nsg_sim.backends.base import Request, make_backend
from nsg_sim.config import ExperimentConfig
from nsg_sim.prompts.build import build_prompt
from nsg_sim.prompts.parse import parse_answer
from nsg_sim.storage import REPO_ROOT, git_commit, load_dataset

RUNS_DIR = REPO_ROOT / "runs"


def select_records(records: list[dict], cfg: ExperimentConfig) -> list[dict]:
    flt, smp = cfg.filter, cfg.sample
    rs = [r for r in records
          if (flt.explanation_types is None or r["explanation_type"] in flt.explanation_types)
          and (flt.levels is None or r["level"] in flt.levels)
          and (flt.evidence_conditions is None or r["evidence_condition"] in flt.evidence_conditions)
          and (flt.rule_ids is None or r["rule_id"] in flt.rule_ids)
          and all(_field(r, k) in v for k, v in flt.where.items())]
    if smp.per_cell:
        rng = random.Random(smp.seed)
        cells = defaultdict(set)
        for r in rs:
            cells[(r["rule_id"], r["explanation_type"], r["evidence_condition"])].add(r["item_id"])
        chosen = set()
        for cell in sorted(cells):
            items = sorted(cells[cell])
            chosen.update(rng.sample(items, min(smp.per_cell, len(items))))
        rs = [r for r in rs if r["item_id"] in chosen]
    return rs[: smp.limit] if smp.limit else rs


def _field(record: dict, dotted: str):
    value = record
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", text)


def plan_jobs(cfg: ExperimentConfig, records: list[dict], task: dict) -> list[tuple]:
    """(record, with_explanation, prompt, prompt_sha) for every row to produce."""
    variants = [True, False] if cfg.with_baseline else [True]
    jobs = []
    for r in records:
        for with_expl in variants:
            prompt = build_prompt(r, task, cfg.prompt.template, cfg.prompt.instruction,
                                  cfg.prompt.explanation_format, with_expl)
            jobs.append((r, with_expl, prompt, hashlib.sha256(prompt.encode()).hexdigest()[:16]))
    return jobs


def run_dir(cfg: ExperimentConfig, root: Path | None = None) -> Path:
    return (root or RUNS_DIR) / safe_name(cfg.name)


def run_experiment(cfg: ExperimentConfig, dry_run: bool = False, root: Path | None = None, log=print) -> Path:
    records, manifest = load_dataset(cfg.dataset)
    if cfg.dataset_sha and not manifest["sha256"].startswith(cfg.dataset_sha):
        raise ValueError(f"dataset {cfg.dataset} has sha {manifest['sha256'][:12]}, config pins {cfg.dataset_sha}")
    selected = select_records(records, cfg)
    jobs = plan_jobs(cfg, selected, manifest["domain"]["task"])
    out_dir = run_dir(cfg, root)
    meta_path = out_dir / "run.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous["dataset_sha256"] != manifest["sha256"]:
            raise ValueError(f"{out_dir} was run on another version of {cfg.dataset}; use a new experiment name")

    log(f"{cfg.name}: {len(selected)} records x {2 if cfg.with_baseline else 1} prompt variants = {len(jobs)} rows "
        f"per simulator, {len({j[3] for j in jobs})} unique prompts")
    plans = []
    for sim in cfg.simulators:
        path = out_dir / f"{safe_name(sim.name)}.jsonl"
        done = _done(path)
        todo = [j for j in jobs if (j[0]["record_id"], j[1]) not in done]
        backend_calls = len({j[3] for j in todo}) if sim.backend != "baseline" else 0
        log(f"  {sim.name} [{sim.backend}]: {len(todo)} rows to run ({len(jobs) - len(todo)} done), {backend_calls} API requests")
        if cfg.run.max_requests is not None and backend_calls > cfg.run.max_requests:
            raise RuntimeError(f"{sim.name}: {backend_calls} requests exceed run.max_requests={cfg.run.max_requests}")
        plans.append((sim, path, todo))
    if dry_run:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": cfg.name, "config": cfg.model_dump(mode="json"), "dataset": cfg.dataset,
            "dataset_sha256": manifest["sha256"], "git_commit": git_commit(),
            "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for sim, path, todo in plans:
        if todo:
            asyncio.run(_run_simulator(cfg, sim, path, todo, log))
    return out_dir


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row.get("error"):
            done.add((row["record_id"], row["with_explanation"]))
    return done


async def _run_simulator(cfg: ExperimentConfig, sim, path: Path, todo: list[tuple], log) -> None:
    backend = make_backend(sim, cfg.run)
    semaphore = asyncio.Semaphore(cfg.run.concurrency)
    cache: dict[str, asyncio.Task] = {}
    counts = defaultdict(int)
    every = max(1, len(todo) // 10) if backend.calls_api else len(todo)  # baselines: one summary line

    async def call(prompt, record, with_expl):
        async with semaphore:
            return await backend.complete(Request(prompt, record, with_expl))

    async def one(job, fh):
        record, with_expl, prompt, sha = job
        # baselines answer per record; API responses are shared by identical prompts
        key = sha if backend.calls_api else f"{record['record_id']}|{with_expl}"
        if key not in cache:
            cache[key] = asyncio.ensure_future(call(prompt, record, with_expl))
        resp = await cache[key]
        prediction = parse_answer(resp.text)
        fh.write(json.dumps({
            "record_id": record["record_id"], "item_id": record["item_id"], "cluster_id": record["cluster_id"],
            "simulator": sim.name, "backend": sim.backend, "model": sim.model,
            "reasoning_effort": sim.reasoning_effort, "with_explanation": with_expl, "prompt_sha": sha,
            "prediction": prediction, "raw_response": resp.text, "reasoning": resp.reasoning,
            "finish_reason": resp.finish_reason, "latency_s": resp.latency_s,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "error": resp.error,
        }) + "\n")
        fh.flush()
        counts["error" if resp.error else "unparsed" if prediction is None else "ok"] += 1
        n = sum(counts.values())
        if n % every == 0 or n == len(todo):
            log(f"    {sim.name}: {n}/{len(todo)} {dict(counts)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as fh:
            await asyncio.gather(*(one(j, fh) for j in todo))
    finally:
        await backend.close()
