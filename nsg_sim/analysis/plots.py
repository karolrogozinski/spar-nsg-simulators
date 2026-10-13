"""Line plots of a metric across specificity levels (or any other ordered field).

One line per value of `lines` (e.g. explanation_type, variant), one panel per value of `panels`
(e.g. simulator), 95% cluster-bootstrap bands, direct labels at the line ends. Colours follow a
fixed categorical order so the same line keeps its colour across figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsg_sim.analysis.metrics import METRICS, cluster_ci, group  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, INK_3, GRID, AXIS = "#1a1a19", "#3d3d3a", "#5f5e5a", "#e6e5e0", "#c3c2b7"
METRIC_LABELS = {"nsg": "NSG", "acc": "accuracy with explanation", "acc_wo": "accuracy without explanation",
                 "reliance": "share following the explanation (discriminating)"}


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_3, labelsize=9)


def _label_ends(ax, ends, last_x, gap_frac=0.06):
    """Labels right of lines reaching the last x (nudged apart), below-right of the end of shorter lines."""
    lo, hi = ax.get_ylim()
    gap = gap_frac * (hi - lo)
    right = sorted(e for e in ends if e[1] == last_x)
    for i in range(1, len(right)):
        right[i][0] = max(right[i][0], right[i - 1][0] + gap)
    if right and right[-1][0] > hi:
        shift = right[-1][0] - hi
        for e in right:
            e[0] -= shift
    for y, x, text in right:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK_2, annotation_clip=False)
    for y, x, text in (e for e in ends if e[1] != last_x):
        ax.annotate(text, (x, y), xytext=(8, -10), textcoords="offset points", va="top", fontsize=9, color=INK_2)


def plot_metric(pairs: list[dict], metric: str = "nsg", x: str = "level", lines: str = "explanation_type",
                panels: str = "simulator", x_order: list | None = None, line_order: list | None = None,
                title: str | None = None):
    stat = METRICS[metric]
    panel_vals = sorted({p[panels] for p in pairs}, key=str)
    x_vals = x_order or sorted({p[x] for p in pairs}, key=str)
    line_vals = line_order or sorted({p[lines] for p in pairs}, key=str)
    colors = {v: SLOTS[i % len(SLOTS)] for i, v in enumerate(line_vals)}
    fig, axes = plt.subplots(1, len(panel_vals), figsize=(5.6 * len(panel_vals) + 1.4, 4.6), sharey=True, squeeze=False)
    all_ends = []
    for ax, pv in zip(axes[0], panel_vals):
        cells = group([p for p in pairs if p[panels] == pv], [lines, x])
        ends = []
        for lv in line_vals:
            pts = [(i, cells[(lv, xv)]) for i, xv in enumerate(x_vals) if (lv, xv) in cells]
            if not pts:
                continue
            xs = [i for i, _ in pts]
            ys = [stat(c) for _, c in pts]
            cis = [cluster_ci(c, stat) for _, c in pts]
            ax.fill_between(xs, [c[0] for c in cis], [c[1] for c in cis], color=colors[lv], alpha=0.12, linewidth=0)
            ax.plot(xs, ys, color=colors[lv], linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                    markeredgewidth=2, zorder=3)
            ends.append([ys[-1], xs[-1], str(lv)])
        if metric == "nsg":
            ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        all_ends.append((ax, ends))
        ax.set_xticks(range(len(x_vals)), [str(v) for v in x_vals])
        ax.set_xlim(-0.3, len(x_vals) - 0.3 + 0.6)
        ax.set_xlabel(x, color=INK_3, fontsize=9)
        ax.set_title(str(pv), loc="left", fontsize=10.5, color=INK, pad=12)
        _style(ax)
    axes[0][0].set_ylabel(METRIC_LABELS.get(metric, metric), color=INK_2)
    for ax, ends in all_ends:  # after every panel is drawn, so the shared y-range is final
        _label_ends(ax, ends, len(x_vals) - 1)
    fig.suptitle(title or f"{METRIC_LABELS.get(metric, metric)} by {x} and {lines}", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.9, "bands: 95% bootstrap intervals over reference inputs", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=4)
    return fig
