"""`nsg run EXPERIMENT_YAML [--dry-run] [--build-missing]` and `nsg ping BACKEND MODEL`."""

from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    p = sub.add_parser("run", help="run an experiment")
    p.add_argument("config", type=Path)
    p.add_argument("--dry-run", action="store_true", help="only count rows and API requests")
    p.add_argument("--build-missing", action="store_true",
                   help="build the dataset from configs/datasets/ if it is not built yet")
    p.add_argument("--runs-dir", type=Path, help="default: <repo>/runs")
    p.add_argument("--no-report", action="store_true", help="skip writing summary.csv and the figure")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ping", help="send one short request to check a backend connection")
    p.add_argument("backend", help="ollama | openrouter | openai | vllm")
    p.add_argument("model")
    p.add_argument("--reasoning-effort")
    p.add_argument("--base-url")
    p.set_defaults(func=cmd_ping)


def cmd_run(args) -> int:
    from nsg_sim.config import load_experiment_config
    from nsg_sim.runner import run_experiment
    cfg = load_experiment_config(args.config)
    if args.build_missing:
        _build_if_missing(cfg.dataset)
    out = run_experiment(cfg, dry_run=args.dry_run, root=args.runs_dir)
    if args.dry_run:
        print(f"dry run, nothing sent; results would go to {out}")
        return 0
    print(f"results in {out}")
    if not args.no_report:
        from nsg_sim.analysis.report import write_run_report
        for path in write_run_report(out):
            print(f"  wrote {path.name}")
    return 0


def _build_if_missing(name: str) -> None:
    from nsg_sim.builder import build_dataset
    from nsg_sim.config import load_dataset_config, load_yaml
    from nsg_sim.storage import REPO_ROOT, dataset_dir
    if (dataset_dir(name) / "manifest.json").exists():
        return
    recipes = [p for p in (REPO_ROOT / "configs" / "datasets").glob("*.yaml") if load_yaml(p).get("name") == name]
    if not recipes:
        raise FileNotFoundError(f"dataset {name!r} is not built and no recipe in configs/datasets/ has that name")
    cfg = load_dataset_config(recipes[0])
    if cfg.derive_from:
        _build_if_missing(cfg.derive_from)
    build_dataset(cfg)
    print(f"built {name} from {recipes[0]}")


def cmd_ping(args) -> int:
    import asyncio

    from nsg_sim.backends.base import Request, make_backend
    from nsg_sim.config import RunSettings, Simulator
    sim = Simulator(backend=args.backend, model=args.model, reasoning_effort=args.reasoning_effort, base_url=args.base_url)
    try:
        backend = make_backend(sim, RunSettings(retries=0, max_tokens=512, timeout=60))
    except (RuntimeError, ValueError) as exc:
        print(f"FAILED  {sim.name} [{sim.backend}]: {exc}")
        return 1

    async def go():
        try:
            return await backend.complete(Request("Reply with the single word OK."))
        finally:
            await backend.close()

    resp = asyncio.run(go())
    if resp.error:
        print(f"FAILED  {sim.name} [{sim.backend}]: {resp.error}")
        return 1
    print(f"ok      {sim.name} [{sim.backend}]  {resp.latency_s}s  tokens in/out {resp.prompt_tokens}/{resp.completion_tokens}")
    print(f"        answer: {(resp.text or '').strip()[:200]!r}")
    if resp.reasoning:
        print(f"        reasoning: {resp.reasoning.strip()[:200]!r}")
    return 0
