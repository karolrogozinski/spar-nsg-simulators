"""Elicit a simulator's priors about a domain, independently of any rule or explanation.

Two measures per simulator and domain view (a dataset's domain, e.g. triage or its abstract twin):
  decision   cases sampled from the domain (none of them used in the dataset), shown one at a time
             with no rule, explanation or reference case; the simulator gives P(YES) in 0-100
  direction  per feature: does a higher value (being true, which category) make YES more likely?
Each question is asked in several wordings. Results go to runs/<name>/: inputs.json (the sampled
cases per view), <simulator>.jsonl (one row per answer) and run.json. Runs resume like `nsg run`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path

from nsg_sim.backends.base import Request, make_backend
from nsg_sim.builder import domain_for_manifest
from nsg_sim.config import PriorsConfig
from nsg_sim.prompts.parse import parse_answer
from nsg_sim.runner import RUNS_DIR, safe_name
from nsg_sim.sampling import key
from nsg_sim.storage import git_commit, load_dataset

DECISION = [
    "{decision}\n\nHere is a {subject}:\n{description}\n\nYou know nothing else about how this {actor_short} "
    "decides. Based on what you would expect, how likely is it that the {actor_short} decides YES for this {subject}?",
    "Think of a typical {actor}. {decision}\n\n{Subject}:\n{description}\n\n"
    "What is the probability that the {actor_short} says YES?",
    "Without any further information, predict the decision of a {actor} about the {subject} below. {decision}\n\n"
    "{description}\n\nGive your best estimate of the chance of a YES decision.",
]
DECISION_FORMAT = ("\n\nAnswer in this format exactly:\n[PROBABILITY]\na number from 0 to 100\n\n"
                   "[ANSWER]\nYES or NO")

DIRECTION_LEAD = [
    "{decision}",
    "Think of a typical {actor}. {decision}",
    "Without any further information about how the {actor_short} decides: {decision}",
]
DIRECTION_QUESTION = {
    "numeric": ("Two {subject}s are identical except that one has a higher {label} than the other. For which one is "
                "the {actor_short} more likely to decide YES?\n\n[ANSWER]\nHIGHER, LOWER or SAME"),
    "bool": ("Two {subject}s are identical except that one has {label} and the other does not. For which one is the "
             "{actor_short} more likely to decide YES?\n\n[ANSWER]\nWITH, WITHOUT or SAME"),
    "categorical": ("Two or more {subject}s are identical except for their {label} ({values}). For which value is the "
                    "{actor_short} most likely to decide YES?\n\n[ANSWER]\none of: {values}, or SAME"),
}
DIRECTION_FORMAT = "\n\nAnswer with the [ANSWER] section only."


def _fields(task: dict) -> dict:
    return {**task, "Subject": task["subject"].capitalize()}


def sample_inputs(domain, records: list[dict], n: int, seed: int) -> list[dict]:
    """n distinct cases from the domain, none equal to an input used anywhere in the dataset."""
    used = {key(r["input_x"]) for r in records} | {key(r["input_cx"]) for r in records}
    used |= {key(e["input"]) for r in records for e in r["evidence"]}
    rng = random.Random(f"{seed}|priors")
    out, seen = [], set()
    for _ in range(100 * n):
        x = domain.sample_input(rng)
        k = key(x)
        if k not in used and k not in seen:
            seen.add(k)
            out.append(x)
            if len(out) == n:
                return out
    raise RuntimeError(f"could only sample {len(out)} cases outside the dataset")


def decision_prompt(domain, x: dict, wording: int) -> str:
    return DECISION[wording].format(**_fields(domain.task), description=domain.describe(x)) + DECISION_FORMAT


def direction_prompt(domain, feature: str, wording: int) -> str:
    f = domain.feature(feature)
    question = DIRECTION_QUESTION[f.kind].format(**_fields(domain.task), label=f.label,
                                                 values=", ".join(map(str, f.values)))
    return DIRECTION_LEAD[wording].format(**_fields(domain.task)) + "\n\n" + question + DIRECTION_FORMAT


PROBABILITY = re.compile(r"\[\s*PROBABILITY\s*\]\s*:?\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
ANSWER_TOKEN = re.compile(r"\[\s*ANSWER\s*\]\s*:?\s*\**\s*([A-Za-z_0-9-]+)", re.IGNORECASE)


def parse_probability(text: str | None) -> float | None:
    text = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.DOTALL)
    m = PROBABILITY.findall(text)
    if not m:
        return None
    p = float(m[-1])
    return p / 100 if 0 <= p <= 100 else None


def parse_direction(text: str | None, feature) -> object:
    """Normalised belief: "high"/"low" (numeric), True/False (bool), a value (categorical), "same", or None."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.DOTALL)
    m = ANSWER_TOKEN.findall(text)
    if m:
        word = m[-1].strip().lower()
    else:  # a bare answer on the first line ("HIGHER")
        first = next((line for line in text.splitlines() if line.strip()), "")
        word = first.strip().strip("*.:# ").lower()
    if word == "same":
        return "same"
    if feature.kind == "numeric":
        return {"higher": "high", "lower": "low"}.get(word)
    if feature.kind == "bool":
        return {"with": True, "without": False}.get(word)
    values = {str(v).lower(): v for v in feature.values}
    return values.get(word)


def plan(cfg: PriorsConfig) -> tuple[dict, list[dict]]:
    """Sampled inputs per view and the list of questions (one per prompt)."""
    inputs, jobs = {}, []
    for name in cfg.datasets:
        records, manifest = load_dataset(name)
        domain = domain_for_manifest(manifest)
        view = domain.name
        if view in inputs:
            continue  # e.g. a mirror twin shares its base dataset's domain view
        xs = sample_inputs(domain, records, cfg.n_inputs, cfg.seed)
        inputs[view] = xs
        for i, x in enumerate(xs):
            for w in range(cfg.wordings):
                jobs.append({"view": view, "kind": "decision", "item": i, "wording": w,
                             "prompt": decision_prompt(domain, x, w)})
        if cfg.directions:
            for feature in domain.features:
                for w in range(cfg.wordings):
                    jobs.append({"view": view, "kind": "direction", "item": feature, "wording": w,
                                 "prompt": direction_prompt(domain, feature, w)})
    for j in jobs:
        j["prompt_sha"] = hashlib.sha256(j["prompt"].encode()).hexdigest()[:16]
    return inputs, jobs


def run_priors(cfg: PriorsConfig, dry_run: bool = False, root: Path | None = None, log=print) -> Path:
    inputs, jobs = plan(cfg)
    out = (root or RUNS_DIR) / safe_name(cfg.name)
    log(f"{cfg.name}: {len(jobs)} questions per simulator ({', '.join(f'{v}: {len(xs)} cases' for v, xs in inputs.items())})")
    todo_by_sim = {}
    for sim in cfg.simulators:
        path = out / f"{safe_name(sim.name)}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if not row.get("error"):
                    done.add((row["view"], row["kind"], str(row["item"]), row["wording"]))
        todo = [j for j in jobs if (j["view"], j["kind"], str(j["item"]), j["wording"]) not in done]
        log(f"  {sim.name} [{sim.backend}]: {len(todo)} to ask ({len(jobs) - len(todo)} done)")
        if cfg.run.max_requests is not None and len(todo) > cfg.run.max_requests:
            raise RuntimeError(f"{sim.name}: {len(todo)} requests exceed run.max_requests={cfg.run.max_requests}")
        todo_by_sim[sim.name] = (sim, path, todo)
    if dry_run:
        return out
    out.mkdir(parents=True, exist_ok=True)
    inputs_path = out / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError(f"{out} was made with other inputs; use a new name")
    inputs_path.write_text(json.dumps(inputs) + "\n")
    (out / "run.json").write_text(json.dumps({
        "experiment": cfg.name, "kind": "priors", "config": cfg.model_dump(mode="json"), "git_commit": git_commit(),
        "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=2) + "\n")
    for sim, path, todo in todo_by_sim.values():
        if todo:
            asyncio.run(_ask(cfg, sim, path, todo, log))
    return out


async def _ask(cfg: PriorsConfig, sim, path: Path, todo: list[dict], log) -> None:
    backend = make_backend(sim, cfg.run)
    semaphore = asyncio.Semaphore(cfg.run.concurrency)
    counts = {"ok": 0, "unparsed": 0, "error": 0}
    every = max(1, len(todo) // 10)

    async def one(job, fh):
        async with semaphore:
            resp = await backend.complete(Request(job["prompt"]))
        row = {k: job[k] for k in ("view", "kind", "item", "wording", "prompt_sha")}
        row.update({"simulator": sim.name, "model": sim.model, "tags": sim.tags, "raw_response": resp.text,
                    "reasoning": resp.reasoning, "provider": resp.provider, "error": resp.error,
                    "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens})
        if job["kind"] == "decision":
            row["probability"] = parse_probability(resp.text)
            row["answer"] = parse_answer(resp.text)
            parsed = row["probability"] is not None
        else:
            row["answer"] = None  # parsed at analysis time against the domain's feature
            parsed = bool(resp.text and resp.text.strip())
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        counts["error" if resp.error else "ok" if parsed else "unparsed"] += 1
        n = sum(counts.values())
        if n % every == 0 or n == len(todo):
            log(f"    {sim.name}: {n}/{len(todo)} {counts}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as fh:
            await asyncio.gather(*(one(j, fh) for j in todo))
    finally:
        await backend.close()
