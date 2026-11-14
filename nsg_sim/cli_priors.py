"""`nsg priors CONFIG` (elicit priors) and `nsg priors-table RUN_DIR --datasets ...` (rule-prior agreement)."""

from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    p = sub.add_parser("priors", help="ask simulators about the domain with no rule or explanation")
    p.add_argument("config", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--runs-dir", type=Path)
    p.set_defaults(func=cmd_priors)

    p = sub.add_parser("priors-table", help="agreement of each dataset rule with each simulator's priors")
    p.add_argument("run", type=Path)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--csv", type=Path)
    p.add_argument("--features", action="store_true",
                   help="per-feature rho of P(YES) toward yes_when and each domain's prior strength instead")
    p.set_defaults(func=cmd_table)


def cmd_priors(args) -> int:
    from nsg_sim.config import PriorsConfig, load_yaml
    from nsg_sim.priors import run_priors
    cfg = PriorsConfig.model_validate(load_yaml(args.config))
    out = run_priors(cfg, dry_run=args.dry_run, root=args.runs_dir)
    print(("dry run, nothing sent; results would go to " if args.dry_run else "results in ") + str(out))
    return 0


def cmd_table(args) -> int:
    from nsg_sim.analysis.priors import feature_table, prior_strength, rule_table
    from nsg_sim.analysis.report import write_csv
    fmt = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731

    def show(rows, cols):
        print("  ".join(cols))
        for r in rows:
            print("  ".join(fmt(r[c]) for c in cols))

    if args.features:
        rows = feature_table(args.run, args.datasets)
        cols = ["simulator", "dataset", "feature", "rule_feature", "rho", "n_cases"]
        show(rows, cols)
        print()
        show(prior_strength(rows), ["simulator", "dataset", "strength", "alignment", "n_features"])
    else:
        rows = rule_table(args.run, args.datasets)
        cols = ["simulator", "dataset", "variant", "rule_id", "auc", "dir_agreement", "n_directions", "consistency",
                "n_cases"]
        show(rows, cols)
    if args.csv:
        write_csv(rows, cols, args.csv)
    return 0
