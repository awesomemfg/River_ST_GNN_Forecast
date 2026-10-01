"""Figures 12 and 13 of the QPF ensemble experiment, redrawn for System B.

The drawing functions are copied unchanged from
  src/figures/recipe_ensemble_rainfall_skill.py
      (panel(), itself copied from the paper's fig12_2x2_distributions_larger_fonts.py)
  src/figures/recipe_ensemble_crest_error.py
      (bars(), itself copied from the paper's fig13_flood_safety_larger_fonts.py)
Same typeface and sizes, layout, figure sizes, colours, clipping rule, annotations and PNG plus PDF output.

Only the data change:
  - the model is System B (three seeds). Per-gauge 24 h metrics are averaged over the seeds before the boxes
    are drawn (../outputs/gauge_metrics_24h_mean_over_seeds.csv)
  - Figure 13 strata and missed-crest rates are computed for each seed from the System B scorer's window
    peaks and then averaged over seeds (../outputs/window_summary_by_seed.csv)
  - System B's own issue-time HRRR case is added as one extra box in the spread variant and one extra bar
    in panel (c), because it is the forecast-rain case System B already reports
  - panel (c) of Figure 13 gets a neutral title; the old title stated a result of the old model

Outputs: ../figures/fig12_ensemble.{png,pdf}, ../figures/fig12_spread.{png,pdf}, ../figures/fig13_ensemble.{png,pdf},
         ../outputs/fig13_strata_mean_over_seeds.csv, ../outputs/fig13_missed_gt061_mean_over_seeds.csv
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = "project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914"
OUTD = os.path.join(EXP, "figures")
MEMBERS = ["HRRR", "GFS", "NBM", "ECMWF IFS", "GEM", "JMA"]
ENS = "Ensemble mean, 6"
OBS = "Observed rain"
HRRR_L2 = "HRRR, issue time"

COL_OBS = "#1f77b4"
COL_QPF = "#d62728"
COL_MEM = "#9e9e9e"
COL_L2 = "#ff7f0e"
LAB_OBS = "Observed rainfall"
LAB_QPF = "Ensemble mean of six weather models"
GRID = "#808080"

PANELS = [
    ("rmse", "(a) Error", "RMSE (m) — lower is better", True),
    ("nse", "(b) Nash-Sutcliffe efficiency", "NSE (1 is perfect) — higher is better", False),
    ("r", "(c) Correlation", "Pearson correlation — higher is better", False),
    ("kge", "(d) Kling-Gupta efficiency", "KGE (1 is perfect) — higher is better", False),
]

FIG12_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 13.5,
    "axes.titlesize": 16.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 15.0,
    "legend.fontsize": 13.0,
    "savefig.dpi": 300,
}
FIG13_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 14.0,
    "axes.titlesize": 17.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 16.0,
    "legend.fontsize": 13.0,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "savefig.dpi": 300,
}
UNDERCALL_M = 0.1524
STRATA_LABELS = {"0.15-0.30 m": "0.15–0.30\nm", "0.30-0.61 m": "0.30–0.61\nm", "above 0.61 m": ">0.61\nm"}


def panel(ax, groups_by_case, cases, xlabels, key, title, ylabel, lower_better):
    """Copied unchanged from recipe_ensemble_rainfall_skill.py."""
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


def bars(ax, t, col_a, col_b, lab_a, lab_b, ya, yb, ylabel, title, pct):
    """Copied unchanged from recipe_ensemble_crest_error.py."""
    x = np.arange(len(t))
    ax.bar(x - 0.19, t[ya], 0.36, color=col_a, label=lab_a)
    ax.bar(x + 0.19, t[yb], 0.36, color=col_b, label=lab_b)
    for i in range(len(t)):
        d = t[yb].iloc[i] - t[ya].iloc[i]
        top = max(t[ya].iloc[i], t[yb].iloc[i])
        txt = f"{d:+.1f}%" if pct else f"{d:+.3f} m"
        ax.annotate(txt, (x[i], top), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=13.5, color="0.2", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(t["stratum"], fontsize=12.5)
    ax.set_xlabel("Observed 24 h stage rise")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(t[[ya, yb]].to_numpy().max() * 1.22, 1e-6))
    ax.set_title(title, loc="left")
    ax.grid(color=GRID, axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.93)


def main():
    os.makedirs(OUTD, exist_ok=True)
    gauge = pd.read_csv(os.path.join(EXP, "outputs", "gauge_metrics_24h_mean_over_seeds.csv"))
    data = {name: frame.set_index("node") for name, frame in gauge.groupby("forcing")}
    print("gauges per forcing:", {k: len(v) for k, v in data.items()})

    # ---- Figure 12, variant 1: observed rainfall against the ensemble mean ----
    plt.rcParams.update(FIG12_RC)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    for ax, (key, title, ylab, lower) in zip(axes.ravel(), PANELS):
        groups = {"obs": [data[OBS][key].dropna().values],
                  "ens": [data[ENS][key].dropna().values]}
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

    # ---- Figure 12, variant 2: every member drawn ----
    cols = [OBS] + MEMBERS + [HRRR_L2, ENS]
    labels = ["Observed\nrainfall"] + MEMBERS + ["HRRR\nissue time", "Ensemble\nmean"]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8))
    for ax, (key, title, ylab, lower) in zip(axes.ravel(), PANELS):
        groups = {"all": [data[c][key].dropna().values for c in cols]}
        panel(ax, groups, [("all", COL_MEM, 0.0)], labels, key, title, ylab, lower)
        for artist, colour in [(0, COL_OBS), (len(cols) - 2, COL_L2), (len(cols) - 1, COL_QPF)]:
            for box in [ax.patches[artist]] if artist < len(ax.patches) else []:
                box.set_facecolor(colour)
                box.set_edgecolor(colour)
        ax.tick_params(axis="x", labelsize=10.5)
    fig.legend(handles=[Patch(facecolor=COL_OBS, label=LAB_OBS),
                        Patch(facecolor=COL_MEM, label="Individual weather models, forecast about 24 h ahead"),
                        Patch(facecolor=COL_L2, label="HRRR, newest cycle at each issue time"),
                        Patch(facecolor=COL_QPF, label="Ensemble mean")],
               loc="upper center", ncol=2, frameon=False, fontsize=14.0,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"fig12_spread.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {OUTD}/fig12_spread.png")

    # ---- Figure 13 ----
    plt.rcParams.update(FIG13_RC)
    windows = pd.read_csv(os.path.join(EXP, "outputs", "window_summary_by_seed.csv"))
    rows = []
    for band, pretty in STRATA_LABELS.items():
        a = windows[windows["forcing"] == OBS]
        b = windows[windows["forcing"] == ENS]
        rows.append({"stratum": pretty, "n": int(a["n_band_" + band].mean()),
                     "peak_a": float(a["mean_peak_abs_error_m_band_" + band].mean()),
                     "peak_b": float(b["mean_peak_abs_error_m_band_" + band].mean()),
                     "miss_a": float(a["missed_crest_pct_band_" + band].mean()),
                     "miss_b": float(b["missed_crest_pct_band_" + band].mean())})
    rain = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.9))
    bars(axes[0], rain, COL_OBS, COL_QPF, "Observed rainfall", "Ensemble mean",
         "peak_a", "peak_b", "Mean error in the peak level (m)",
         "(a) Accuracy of the crest", pct=False)
    bars(axes[1], rain, COL_OBS, COL_QPF, "Observed rainfall", "Ensemble mean",
         "miss_a", "miss_b", "Crests missed (%)",
         "(b) How often the crest is missed", pct=True)
    axes[1].set_ylim(0, 112)

    ax = axes[2]
    forcings = MEMBERS + [HRRR_L2, ENS]
    bar_labels = ["HRRR", "GFS", "NBM", "ECMWF", "GEM", "JMA", "HRRR issue time", "Ensemble mean"]
    x = np.arange(len(bar_labels))
    vals = [float(windows[windows["forcing"] == f]["missed_crest_pct_band_above 0.61 m"].mean()) for f in forcings]
    n_top = int(windows[windows["forcing"] == OBS]["n_band_above 0.61 m"].mean())
    colours = [COL_MEM] * len(MEMBERS) + [COL_L2, COL_QPF]
    ax.bar(x, vals, 0.62, color=colours)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}%", (x[i], v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=12.5, color="0.2", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=11.0, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Crests missed (%)")
    ax.set_xlabel(f"Rises above 0.61 m  (n = {n_top:,} per seed)")
    ax.set_ylim(0, 112)
    ax.set_title("(c) Crests missed on rises above 0.61 m", loc="left")
    ax.grid(color=GRID, axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"fig13_ensemble.{ext}"), bbox_inches="tight")
    plt.close(fig)
    rain.to_csv(os.path.join(EXP, "outputs", "fig13_strata_mean_over_seeds.csv"), index=False)
    pd.DataFrame({"forcing": forcings, "missed_pct_gt061": vals}).to_csv(
        os.path.join(EXP, "outputs", "fig13_missed_gt061_mean_over_seeds.csv"), index=False)
    print(rain.round(3).to_string(index=False))
    for f, v in zip(forcings, vals):
        print(f"  {f:<20}{v:6.1f}%")
    print(f"[out] {OUTD}/fig13_ensemble.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
