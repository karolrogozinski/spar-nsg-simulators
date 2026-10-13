"""Summary tables and the per-run report written next to a run's results.

`nsg run` calls `write_run_report` after every run, so each runs/<name>/ folder holds its raw
answers (<simulator>.jsonl), the setup (run.json), a table (summary.csv) and a figure (nsg.png,
or acc.png when the run has no prompts without the explanation). `nsg report` rewrites them.
"""

from __future__ import annotations

import csv
from pathlib import Path

from nsg_sim.analysis.metrics import METRICS, cluster_ci, group, load_pairs

COLUMNS = ["n", "clusters", "invalid", "acc", "acc_ci", "acc_wo", "nsg", "nsg_ci", "reliance"]
REPORT_BY = ["simulator", "explanation_type", "level", "evidence_condition"]


def summary_rows(pairs: list[dict], keys: list[str]) -> list[dict]:
    """One row per group: counts, metrics, and 95% cluster-bootstrap intervals for acc and NSG."""
    rows = []
    for k, cell in sorted(group(pairs, keys).items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(keys, k))
        row.update({"n": len(cell), "invalid": round(1 - sum(p["valid"] for p in cell) / len(cell), 3),
                    "clusters": len({p["cluster"] for p in cell})})
        for name, fn in METRICS.items():
            value = fn(cell)
            row[name] = None if value != value else round(value, 3)
            if name in ("nsg", "acc"):
                lo, hi = cluster_ci(cell, fn)
                if lo == lo:
                    row[f"{name}_ci"] = f"[{lo:+.2f}, {hi:+.2f}]"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], cols: list[str], path: Path) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_run_report(run_dir: Path) -> list[Path]:
    """Write summary.csv and a figure into run_dir; returns the written paths."""
    import matplotlib.pyplot as plt

    from nsg_sim.analysis.plots import plot_metric
    run_dir = Path(run_dir)
    pairs = load_pairs([run_dir])
    if not pairs:
        return []
    out = run_dir / "summary.csv"
    write_csv(summary_rows(pairs, REPORT_BY), REPORT_BY + COLUMNS, out)
    written = [out]
    # the figure: cases without evidence, NSG when the run has the no-explanation prompts
    shown = [p for p in pairs if p["evidence_condition"] == "none"] or pairs
    metric = "nsg" if any(p["without"] is not None for p in shown) else "acc"
    title = f"{run_dir.name}: {'NSG' if metric == 'nsg' else 'accuracy'} by level"
    fig = plot_metric(shown, metric, x="level", lines="explanation_type", panels="simulator",
                      x_order=sorted({p["level"] for p in shown}), title=title)
    fig_path = run_dir / f"{metric}.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(fig_path)
    return written
