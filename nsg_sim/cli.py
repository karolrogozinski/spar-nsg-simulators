"""Command line: nsg <command> ...

  nsg validate CONFIG...           check dataset / experiment YAML files without running anything
  nsg preview DATASET_YAML         show rules, texts per level and an example record, saves nothing
  nsg build DATASET_YAML...        build datasets into data/<name>/
  nsg list                         built datasets
  nsg run EXPERIMENT_YAML          query simulators (see --dry-run)
  nsg analyze RUN_DIR              tables of accuracy / NSG
  nsg plot RUN_DIR                 figures
  nsg report RUN_DIR               rewrite summary.csv and the figure inside a run folder
  nsg push NAME / nsg pull NAME    share datasets through the Hugging Face hub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nsg_sim import cli_analyze, cli_hub, cli_run
from nsg_sim.config import DatasetConfig, ExperimentConfig, load_yaml


def _config_kind(raw: dict) -> str:
    return "experiment" if "simulators" in raw else "dataset"


def cmd_validate(args) -> int:
    ok = True
    for path in args.configs:
        raw = load_yaml(path)
        kind = _config_kind(raw)
        try:
            (ExperimentConfig if kind == "experiment" else DatasetConfig).model_validate(raw)
            if kind == "dataset" and raw.get("domain"):
                from nsg_sim.domain.tabular import load_domain
                from nsg_sim.rules.sources import rules_from_config
                cfg = DatasetConfig.model_validate(raw)
                rules_from_config(cfg, load_domain(cfg.domain))
            print(f"ok       {kind:<10} {path}")
        except Exception as exc:  # report every file, then fail
            ok = False
            print(f"INVALID  {kind:<10} {path}\n  {exc}")
    return 0 if ok else 1


def cmd_build(args) -> int:
    from nsg_sim.builder import build_dataset
    from nsg_sim.config import load_dataset_config
    llm = None
    for path in args.configs:
        cfg = load_dataset_config(path)
        if any(lvl.renderer == "llm" for lvl in cfg.levels) and llm is None:
            from nsg_sim.render.llm import LLMRenderer
            llm = LLMRenderer()
        records, manifest = build_dataset(cfg, llm=llm)
        print(f"built {cfg.name}: {manifest['n_records']} records, sha {manifest['sha256'][:12]}")
        if args.verbose:
            print(json.dumps(manifest["stats"], indent=2))
    return 0


def cmd_preview(args) -> int:
    from nsg_sim.builder import build_dataset
    from nsg_sim.config import load_dataset_config
    cfg = load_dataset_config(args.config)
    llm = None
    if any(lvl.renderer == "llm" for lvl in cfg.levels):
        from nsg_sim.render.llm import LLMRenderer
        llm = LLMRenderer()
    records, manifest = build_dataset(cfg, llm=llm, save=False)
    rule_ids = list(dict.fromkeys(r["rule_id"] for r in records))
    show = rule_ids if args.rule == "all" else [args.rule] if args.rule else rule_ids[: args.rules]
    for rid in show:
        rs = [r for r in records if r["rule_id"] == rid and r["evidence_condition"] == "none"]
        print(f"\n=== {rid}   {rs[0]['rule_formal']}   tags {rs[0]['rule_tags']}")
        seen = set()
        for r in rs:
            key = (r["explanation_type"], r["level"])
            if key in seen:
                continue
            seen.add(key)
            text = r["explanation_text"].replace("\n", "\n" + " " * 26)
            print(f"  {r['explanation_type']:<16} {r['level']:<4}  {text}")
    r = next((r for r in records if r["rule_id"] == show[0] and r["discriminating"]), records[0])
    print(f"\n=== example record {r['record_id']}")
    print(f"  reference      {r['description_x']}  -> {r['label_x']}")
    print(f"  counterfactual {r['description_cx']}  -> true {r['label_cx']}, "
          f"explanation says {r['explanation_label_cx']}{'  (discriminating)' if r['discriminating'] else ''}")
    stats = manifest["stats"]
    print(f"\n{len(records)} records, {stats['n_rules']} rules, yes_share {stats['yes_share']}, "
          f"discriminating_share {stats['discriminating_share']}  (nothing saved; `nsg build` to save)")
    return 0


def cmd_list(args) -> int:
    from nsg_sim.storage import data_root, list_datasets
    rows = list_datasets()
    if not rows:
        print(f"no datasets in {data_root()}")
    for r in rows:
        print(f"{r['name']:<32} {r['n_records']:>7} records  sha {r['sha256'][:12]}  {r['created']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(prog="nsg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check YAML configs")
    p.add_argument("configs", nargs="+", type=Path)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("preview", help="show rules, texts and an example record without saving")
    p.add_argument("config", type=Path)
    p.add_argument("--rules", type=int, default=3, help="how many rules to show (default 3)")
    p.add_argument("--rule", help="one rule id, or 'all'")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("build", help="build datasets")
    p.add_argument("configs", nargs="+", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("list", help="list built datasets")
    p.set_defaults(func=cmd_list)

    for module in (cli_run, cli_analyze, cli_hub):
        module.register(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
