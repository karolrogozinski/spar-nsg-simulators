"""`nsg analyze RUN_DIR... [--by ...]` and `nsg plot RUN_DIR... [--metric ...]`."""

from __future__ import annotations

from pathlib import Path

DEFAULT_BY = "simulator,explanation,level,evidence_condition"


def register(sub) -> None:
    p = sub.add_parser("analyze", help="accuracy / NSG tables for one or more runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--by", default=DEFAULT_BY,
                   help=f"comma-separated grouping fields (default {DEFAULT_BY}); "
                        "e.g. explanation_type, edit_size, variant, rule_tags.n_conditions, discriminating")
    p.add_argument("--where", action="append", default=[], help="field=value filters, repeatable")
    p.add_argument("--csv", type=Path, help="also write the table as CSV")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="(re)write summary.csv and the figure inside run folders")
    p.add_argument("runs", nargs="+", type=Path)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="line plot of a metric across levels")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--metric", default="nsg", choices=["nsg", "acc", "acc_wo", "reliance"])
    p.add_argument("--x", default="level")
    p.add_argument("--lines", default="explanation")
    p.add_argument("--panels", default="simulator")
    p.add_argument("--where", action="append", default=[], help="field=value filters, repeatable")
    p.add_argument("--title")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_plot)


def _filtered(runs, where):
    from nsg_sim.analysis.metrics import load_pairs
    pairs = load_pairs(runs)
    for cond in where:
        k, v = cond.split("=", 1)
        pairs = [p for p in pairs if str(p.get(k)) == v]
    return pairs


def cmd_analyze(args) -> int:
    from nsg_sim.analysis.report import COLUMNS, summary_rows, write_csv
    pairs = _filtered(args.runs, args.where)
    keys = args.by.split(",")
    rows = summary_rows(pairs, keys)
    cols = keys + COLUMNS
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols} if rows else {}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str("-" if r.get(c) is None else r.get(c, "")).rjust(widths[c]) for c in cols))
    if args.csv:
        write_csv(rows, cols, args.csv)
    return 0


def cmd_report(args) -> int:
    from nsg_sim.analysis.report import write_run_report
    for run in args.runs:
        for path in write_run_report(run):
            print(f"wrote {path}")
    return 0


def cmd_plot(args) -> int:
    from nsg_sim.analysis.plots import plot_metric
    pairs = _filtered(args.runs, args.where)
    levels = sorted({p["level"] for p in pairs})
    fig = plot_metric(pairs, args.metric, args.x, args.lines, args.panels,
                      x_order=levels if args.x == "level" else None, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0
