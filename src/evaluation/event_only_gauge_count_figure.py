"""Figure 16 on event origins: how many in-parish gauges the forecast needs.

The published figure answers that over every origin. This builds the same figure once per event
definition, so the question can be asked during pump operation and at high stage, which is when the
answer carries weight.

The geometry is the published recipe's, copied faithfully from
  project/manuscript/
  20260915_Revision_6/scripts/build_rev6_fig16_gauge_count.py
That script is monolithic rather than function-based, so its plotting block is reproduced here rather
than imported: same typeface and sizes, same colors, same box styling, same per-box median
annotations, the same single 68-gauge training-seed band, and the same two-panel layout with median
NSE, correlation and KGE on the right.

Only the data differ. Each origin set is drawn from ../outputs/event_network_levels_h24.csv, written
by gauge_network_skill.py, which scores every network size on the ten-gauge fixed core of its
draw exactly as the published table does.

Outputs: ../figures/event_fig16_gauge_count_<origin set>.{png,pdf}
         ../outputs/event_fig16_summary.csv

Usage:
  conda run -n operational python -u event_only_gauge_count_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
FIGURE_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "figures")
LEVELS = os.path.join(OUTPUT_DIRECTORY, "event_network_levels_h24.csv")

SIZES = [68, 51, 40, 30, 20, 10]
ORIGIN_SETS = [
    ("all", "Every origin"),
    ("pump_at_issue", "Pumps running at issue"),
    ("stage_p80", "Stage above the 80th percentile"),
    ("stage_p90", "Stage above the 90th percentile"),
]

COLOR_IN_PARISH = "#4f91c7"
COLOR_CONTROL = "#b2182b"
COLOR_PARENT = "#737f8c"
COLOR_NSE = "#1f77b4"
COLOR_CORR = "#1b7837"
COLOR_KGE = "#d95f02"
COLOR_FLOOR = "#f7d794"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 17.0,
    "axes.titlesize": 20.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 18.0,
    "xtick.labelsize": 16.5,
    "ytick.labelsize": 16.5,
    "legend.fontsize": 15.0,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

if not os.path.exists(LEVELS):
    raise SystemExit("[FATAL] run gauge_network_skill.py first; missing " + LEVELS)
levels_all = pd.read_csv(LEVELS)
print("[data] rows", len(levels_all), "origin sets", sorted(levels_all["origin_set"].unique()), flush=True)

summary_rows = []
for origin_set, set_title in ORIGIN_SETS:
    levels = levels_all[levels_all["origin_set"] == origin_set]
    if levels.empty:
        print("[skip] no rows for", origin_set, flush=True)
        continue

    groups = [levels[levels["size"] == size]["RMSE_m"].dropna().to_numpy(float) for size in SIZES]
    if any(len(values) == 0 for values in groups):
        print("[skip] a network size has no scored gauges for", origin_set, flush=True)
        continue
    medians = {size: float(np.median(values)) for size, values in zip(SIZES, groups)}
    skill = levels.groupby("size")[["NSE", "Pearson_r", "KGE"]].median()

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))

    # ---------------- (a) absolute per-gauge h+24 RMSE, one 68-gauge noise band
    axis = axes[0]
    positions_a = np.arange(len(SIZES))
    full_floor = float(
        levels[levels["size"] == 68]
        .groupby(["draw", "node"])["RMSE_m"]
        .agg(lambda values: values.max() - values.min())
        .median()
    )
    control_median = medians[68]
    axis.axhspan(control_median - full_floor, control_median + full_floor,
                 color=COLOR_FLOOR, alpha=0.60, zorder=1,
                 label=f"68-gauge training-seed variation, ±{full_floor:.3f} m")
    axis.axhline(control_median, color="0.25", linewidth=1.2, linestyle="--", zorder=2)
    boxes = axis.boxplot(groups, positions=positions_a, widths=0.62, showfliers=False,
                         patch_artist=True, medianprops={"color": "black", "linewidth": 1.8},
                         whiskerprops={"color": "0.35"}, capprops={"color": "0.35"},
                         boxprops={"edgecolor": "0.25", "linewidth": 0.9})
    box_colors = {68: COLOR_CONTROL, 51: COLOR_PARENT}
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(box_colors.get(SIZES[index], COLOR_IN_PARISH))
        patch.set_alpha(0.55)
    generator = np.random.default_rng(101)
    for position, values in zip(positions_a, groups):
        jitter = generator.uniform(-0.17, 0.17, size=len(values))
        axis.scatter(position + jitter, values, s=7, color="0.25", alpha=0.35, linewidths=0, zorder=4)
    for position, size in zip(positions_a, SIZES):
        axis.annotate(f"{medians[size]:.3f}", (position, medians[size]), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=11.0, fontweight="bold", color="black",
                      zorder=6)
    axis.set_xticks(positions_a)
    axis.set_xticklabels([str(size) for size in SIZES])
    axis.set_xlabel("Gauges used by the model")
    axis.set_ylabel("Per-gauge h+24 RMSE (m)")
    rmse_all = np.concatenate(groups)
    rmse_span = float(rmse_all.max() - rmse_all.min())
    lower_limit = min(float(rmse_all.min()) - rmse_span * 0.04, control_median - full_floor)
    upper_limit = float(rmse_all.max()) + rmse_span * 0.04
    axis.set_ylim(lower_limit, upper_limit)
    axis.set_title("(a) Per-gauge h+24 RMSE", loc="left")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", framealpha=0.93, fontsize=10.8)

    # ---------------- (b) median NSE, correlation and KGE
    axis = axes[1]
    positions = np.arange(len(SIZES))
    for column, color, label, marker in [("NSE", COLOR_NSE, "NSE", "o"),
                                         ("Pearson_r", COLOR_CORR, "Correlation", "s"),
                                         ("KGE", COLOR_KGE, "KGE", "^")]:
        values = [float(skill.loc[size, column]) for size in SIZES]
        axis.plot(positions, values, marker=marker, markersize=9, linewidth=2.0, color=color,
                  label=label)
        for position, value in zip(positions, values):
            axis.annotate(f"{value:.3f}", (position, value), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=9.8, color=color)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(size) for size in SIZES])
    axis.set_xlabel("Gauges used by the model")
    axis.margins(y=0.13)
    axis.set_ylabel("Median value at h+24")
    axis.set_title("(b) Median skill at every network size", loc="left")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", framealpha=0.93, ncol=3)

    figure.suptitle("Gauge network size and forecast error: " + set_title,
                    fontsize=21.0, fontweight="bold", y=0.995)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.115, wspace=0.22)
    stem = os.path.join(FIGURE_DIRECTORY, "event_fig16_gauge_count_" + origin_set)
    for suffix in (".png", ".pdf"):
        figure.savefig(stem + suffix, bbox_inches="tight")
        print("[SAVED]", stem + suffix, flush=True)
    plt.close(figure)

    for size in SIZES:
        summary_rows.append({
            "origin_set": origin_set,
            "size": size,
            "median_rmse_m": medians[size],
            "change_vs_68_m": medians[size] - medians[68],
            "seed_band_m": full_floor,
            "within_seed_band": abs(medians[size] - medians[68]) <= full_floor,
            "median_NSE": float(skill.loc[size, "NSE"]),
            "median_Pearson_r": float(skill.loc[size, "Pearson_r"]),
            "median_KGE": float(skill.loc[size, "KGE"]),
        })

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIRECTORY, "event_fig16_summary.csv")
summary.to_csv(summary_path, index=False)
print("[SAVED]", summary_path, flush=True)
if len(summary):
    print(summary.round(4).to_string(index=False), flush=True)
print("EVENT_FIG16_DONE", flush=True)
