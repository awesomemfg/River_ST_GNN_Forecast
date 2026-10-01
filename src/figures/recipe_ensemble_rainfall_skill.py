#!/usr/bin/env python3
"""
Figure 12, rebuilt for the ensemble. The drawing recipe is taken VERBATIM from the paper's own
figure script,

    manuscript/20260826_Final_V2/figure_revision_scripts/
        fig12_2x2_distributions_larger_fonts.py

with the panel() function copied unchanged: same serif typeface and sizes, same 2x2 layout, same
box widths and alpha, same jittered per-gauge points, same three-interquartile-range axis clipping
with the out-of-range gauges pinned to the edge as triangles and counted in the panel corner, same
median labels in white boxes, same colours, same figure size, same PNG and PDF output.

Only the DATA changes, in the two ways Sect. 5.1 changed:

  - postprocessing is gone, so the x-axis is no longer "postprocessing off / on"
  - the red case is no longer a single archived forecast; it is the ensemble mean of six
    operational weather models

Two variants are produced:

  fig12_ensemble   the headline comparison, observed rainfall against the ensemble mean
  fig12_spread     the same panels with every member drawn separately, so the spread between
                   weather centres is visible as the distance between the boxes rather than being
                   summarised away

Per-gauge metrics come from score_ensemble.py, which was verified to reproduce the evaluator's own
lead_metrics to within 3e-8 on every metric, so these are the same numbers the manuscript would get
from the standard outputs.

Run:
    conda run -n operational python recipe_ensemble_rainfall_skill.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = "project/Experiments/QPF_ENSEMBLE_20260831"
OUTD = os.path.join(EXP, "figures")
FIGSCRIPT = ("project/manuscript/"
             "revision_2026_07_17/scripts/recipe_revised_paper_figures.py")

EXCLUDED = {"MBSA4570"}
MEMBERS = ["HRRR", "GFS", "NBM", "ECMWF IFS", "GEM", "JMA"]

COL_OBS = "#1f77b4"
COL_QPF = "#d62728"
COL_MEM = "#9e9e9e"
LAB_OBS = "Observed rainfall"
LAB_QPF = "Ensemble mean of six weather models"
GRID = "#808080"

PANELS = [
    ("rmse", "(a) Error", "RMSE (m) — lower is better", True),
    ("nse", "(b) Nash-Sutcliffe efficiency", "NSE (1 is perfect) — higher is better", False),
    ("r", "(c) Correlation", "Pearson correlation — higher is better", False),
    ("kge", "(d) Kling-Gupta efficiency", "KGE (1 is perfect) — higher is better", False),
]

# verbatim from the native script
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 13.5,
    "axes.titlesize": 16.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 15.0,
    "legend.fontsize": 13.0,
    "savefig.dpi": 300,
})


def inparish():
    """The same 51-gauge list the manuscript's own figures read, parsed not retyped."""
    blk = re.search(r"IN_PARISH_GAUGES = \[(.*?)\]", open(FIGSCRIPT).read(), re.S).group(1)
    return {t.strip().strip('",') for t in blk.split() if t.strip().strip('",')}


def load(keep):
    out = {}
    for c in ["observed_rain"] + MEMBERS + ["ens_mean"]:
        f = os.path.join(EXP, "outputs", f"per_gauge_{c}.csv")
        d = pd.read_csv(f)
        d = d[(~d["node"].isin(EXCLUDED)) & (d["node"].isin(keep))]
        out[c] = d.set_index("node")
    return out


def panel(ax, groups_by_case, cases, xlabels, key, title, ylabel, lower_better):
    """
    Copied unchanged from fig12_2x2_distributions_larger_fonts.py, generalised only so the number
    of x groups and cases is supplied by the caller instead of being fixed at two postprocessing
    states. Every drawing constant is the native one.
    """
    rng = np.random.default_rng(0)
    xs = np.arange(len(xlabels))

    series = {name: groups_by_case[name] for name, _, _ in cases}

    pooled = np.concatenate([v for g in series.values() for v in g if len(v)])
    q1, q3 = np.percentile(pooled, [25, 75])
    iqr = q3 - q1
    lo = max(float(pooled.min()), float(q1 - 3.0 * iqr))
    hi = min(float(pooled.max()), float(q3 + 3.0 * iqr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(pooled.min()), float(pooled.max())
    n_below = int((pooled < lo).sum())
    n_above = int((pooled > hi).sum())
    span = hi - lo
    ax.set_ylim(lo - 0.06 * span, hi + 0.17 * span)

    width = 0.32 if len(cases) > 1 else 0.46
    for name, colour, off in cases:
        groups = series[name]
        pos = xs + off
        bp = ax.boxplot(groups, positions=pos, widths=width, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", linewidth=1.4),
                        whiskerprops=dict(color=colour, linewidth=0.9),
                        capprops=dict(color=colour, linewidth=0.9))
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.30)
            patch.set_edgecolor(colour)
            patch.set_linewidth(1.0)
        for i, vals in enumerate(groups):
            if not len(vals):
                continue
            jit = rng.uniform(-0.085, 0.085, size=len(vals))
            inside = (vals >= lo) & (vals <= hi)
            ax.plot(pos[i] + jit[inside], vals[inside], "o", markersize=3.6, color=colour,
                    alpha=0.55, markeredgewidth=0, zorder=3)
            for mask, marker, edge in [(vals < lo, "v", lo - 0.03 * span),
                                       (vals > hi, "^", hi + 0.03 * span)]:
                if mask.any():
                    ax.plot(pos[i] + jit[mask], np.full(int(mask.sum()), edge), marker,
                            markersize=6.0, color=colour, markeredgecolor="white",
                            markeredgewidth=0.5, clip_on=False, zorder=4)

    for name, colour, off in cases:
        for i, vals in enumerate(series[name]):
            if not len(vals):
                continue
            med = float(np.median(vals))
            fmt = "{:.3f}" if key == "rmse" else "{:.2f}"
            ax.annotate(fmt.format(med), (xs[i] + off, med + 0.025 * span),
                        ha="center", va="bottom", fontsize=11.5, color=colour,
                        fontweight="bold", zorder=6,
                        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none",
                                  boxstyle="square,pad=0.15"))

    if n_below or n_above:
        bits = []
        if n_below:
            bits.append(f"{n_below} point{'s' if n_below != 1 else ''} are below the axis")
        if n_above:
            bits.append(f"{n_above} point{'s' if n_above != 1 else ''} are above the axis")
        ax.annotate(" and ".join(bits) + ",\ndrawn as triangles at the edge",
                    (0.015, 0.985), xycoords="axes fraction", fontsize=10.5,
                    color="0.35", ha="left", va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=12.5)
    ax.set_xlim(-0.55, len(xlabels) - 0.45)
    ax.set_ylabel(ylabel, fontsize=14.0)
    ax.set_title(title, loc="left")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)
    if not lower_better and lo < 0 < hi:
        ax.axhline(0, color="0.35", linewidth=0.7, linestyle=":", zorder=1)


def build(data, xlabels, case_spec, legend, fname, title_note):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    for ax, (key, title, ylab, lower) in zip(axes.ravel(), PANELS):
        groups_by_case = {}
        for name, _, _, cols in case_spec:
            groups_by_case[name] = [data[c][key].dropna().values for c in cols]
        cases = [(name, colour, off) for name, colour, off, _ in case_spec]
        panel(ax, groups_by_case, cases, xlabels, key, title, ylab, lower)
    fig.legend(handles=legend, loc="upper center", ncol=len(legend), frameon=False,
               fontsize=14.0, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {OUTD}/{fname}.png    {title_note}")


def main():
    os.makedirs(OUTD, exist_ok=True)
    keep = inparish()
    data = load(keep)
    n = len(data["ens_mean"])
    print(f"gauges scored: {n}")

    # ---- variant 1: observed rainfall against the ensemble mean --------------------------
    build(
        data,
        xlabels=[LAB_OBS, "Ensemble mean"],
        case_spec=[("case", None, 0.0, ["observed_rain", "ens_mean"])],
        legend=[Patch(facecolor=COL_OBS, label=LAB_OBS),
                Patch(facecolor=COL_QPF, label=LAB_QPF)],
        fname="fig12_ensemble",
        title_note="observed vs ensemble mean",
    ) if False else None

    # the two-colour form needs one case per colour, so it is built explicitly
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    for ax, (key, title, ylab, lower) in zip(axes.ravel(), PANELS):
        groups = {"obs": [data["observed_rain"][key].dropna().values],
                  "ens": [data["ens_mean"][key].dropna().values]}
        panel(ax, groups, [("obs", COL_OBS, -0.19), ("ens", COL_QPF, +0.19)],
              ["Rainfall forcing at the 24 h lead"], key, title, ylab, lower)
    fig.legend(handles=[Patch(facecolor=COL_OBS, label=LAB_OBS),
                        Patch(facecolor=COL_QPF, label=LAB_QPF)],
               loc="upper center", ncol=2, frameon=False, fontsize=14.0,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"fig12_ensemble.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {OUTD}/fig12_ensemble.png")

    # ---- variant 2: every member drawn, so the spread is the figure ----------------------
    cols = ["observed_rain"] + MEMBERS + ["ens_mean"]
    labels = ["Observed\nrainfall"] + MEMBERS + ["Ensemble\nmean"]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8))
    for ax, (key, title, ylab, lower) in zip(axes.ravel(), PANELS):
        groups = {"all": [data[c][key].dropna().values for c in cols]}
        panel(ax, groups, [("all", COL_MEM, 0.0)], labels, key, title, ylab, lower)
        # recolour the two reference boxes so the eye finds them immediately
        for artist, colour in [(0, COL_OBS), (len(cols) - 1, COL_QPF)]:
            for box in [ax.patches[artist]] if artist < len(ax.patches) else []:
                box.set_facecolor(colour)
                box.set_edgecolor(colour)
        ax.tick_params(axis="x", labelsize=10.5)
    fig.legend(handles=[Patch(facecolor=COL_OBS, label=LAB_OBS),
                        Patch(facecolor=COL_MEM, label="Individual weather models"),
                        Patch(facecolor=COL_QPF, label="Ensemble mean")],
               loc="upper center", ncol=3, frameon=False, fontsize=14.0,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"fig12_spread.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {OUTD}/fig12_spread.png")

    print()
    print("MEDIANS AT 24 h, 1 Jan - 30 Jun 2026, in-parish gauges")
    print(f"  {'forcing':<22}" + "".join(f"{k:>11}" for k, _, _, _ in PANELS))
    for c in cols:
        print(f"  {c:<22}" + "".join(f"{data[c][k].median():>11.3f}" for k, _, _, _ in PANELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
