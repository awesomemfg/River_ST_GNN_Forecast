"""Five figures that carry the argument: where the ST-GNN wins, by how much, and why.

Each answers one question a co-author or a reviewer will ask.

  1 skill_by_lead          Does the ST-GNN beat the others, at every lead and in every event set?
  2 what_each_step_buys    Where does its advantage over persistence come from?
  3 graph_effect_vs_quality Why does the advantage shrink with forecast rain?
  4 per_gauge_wins         Is the win broad across gauges, or a few gauges carrying a median?
  5 event_definitions      Are the pump event and the high-stage event the same thing?

Every number is read from files already written by this experiment. No model is run again.

  ../outputs/event_conditioned_summary.csv            four models, five origin sets, four leads
  ../outputs/event_conditioned_per_gauge_metrics.csv  the per-gauge values behind those medians
  ../outputs/graph_and_cell_effect_bootstrap.csv      paired effects with 95 percent intervals
  ../outputs/graph_effect_versus_forcing_quality.csv  the graph effect at eight rainfall qualities
  ../outputs/event_definition_overlaps.csv            how much the two event definitions overlap

Usage:
  conda run -n operational python -u summary_figures.py
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

MODEL_COLORS = {
    "ST-GNN": "#D95F70",
    "GRU": "#3E8E4F",
    "LSTM": "#2C6E9B",
    "persistence": "#7F7F7F",
}
MODEL_ORDER = ["ST-GNN", "GRU", "LSTM", "persistence"]
MODEL_MARKERS = {"ST-GNN": "o", "GRU": "^", "LSTM": "s", "persistence": "D"}
ORIGIN_SETS = [
    ("all", "Every origin"),
    ("pump_at_issue", "Pumps running at issue"),
    ("stage_p80", "Stage above p80"),
    ("stage_p90", "Stage above p90"),
]
LEADS = [3, 6, 12, 24]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 13.0,
    "axes.titlesize": 15.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.5,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[argument]", *parts, flush=True)


def save(figure, stem):
    for suffix in (".png", ".pdf"):
        path = os.path.join(FIGURE_DIRECTORY, stem + suffix)
        figure.savefig(path, bbox_inches="tight")
        log("wrote", path)
    plt.close(figure)


def read(name):
    path = os.path.join(OUTPUT_DIRECTORY, name)
    if not os.path.exists(path):
        raise SystemExit("[FATAL] missing input: " + path)
    return pd.read_csv(path)


# ----------------------------------------------------------------- 1 skill by lead
summary = read("event_conditioned_summary.csv")
figure, axes = plt.subplots(2, 4, figsize=(19.0, 9.0), sharex=True)
for column_index, (origin_set, set_title) in enumerate(ORIGIN_SETS):
    for row_index, (metric, ylabel, lower_better) in enumerate(
        [("RMSE_m", "h+lead RMSE (m), lower is better", True),
         ("NSE", "NSE, higher is better", False)]
    ):
        axis = axes[row_index, column_index]
        for model in MODEL_ORDER:
            block = summary[(summary["model"] == model) & (summary["origin_set"] == origin_set)]
            block = block.set_index("lead_hours").reindex(LEADS)
            axis.plot(LEADS, block[metric].to_numpy(dtype=float), marker=MODEL_MARKERS[model],
                      markersize=8, linewidth=2.0, color=MODEL_COLORS[model], label=model)
        if metric == "NSE":
            axis.axhline(0.0, color="0.4", linewidth=0.9, linestyle=":")
        axis.set_xticks(LEADS)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        if row_index == 0:
            axis.set_title(set_title)
        if row_index == 1:
            axis.set_xlabel("Forecast lead (h)")
        if column_index == 0:
            axis.set_ylabel(ylabel)
handles, labels = axes[0, 0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
figure.suptitle("With observed rainfall the ST-GNN leads at every lead and in every event set",
                fontsize=18.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.975))
save(figure, "argument_1_skill_by_lead")

# ----------------------------------------------------------------- 2 what each step buys
bootstrap = read("graph_and_cell_effect_bootstrap.csv")
observed = bootstrap[bootstrap["arm"] == "observed"]
steps = [
    ("model effect (persistence minus ST-GNN)", "Whole model\nover persistence", "#4C4C4C"),
    ("cell effect (LSTM minus GRU)", "Recurrent cell\nLSTM to GRU", "#2C6E9B"),
    ("graph effect (GRU minus ST-GNN)", "Message passing\nGRU to ST-GNN", "#D95F70"),
]
figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True)
for axis, lead in zip(axes, [12, 24]):
    positions = np.arange(len(ORIGIN_SETS))
    height = 0.26
    for offset, (comparison, label, color) in zip([-height, 0.0, height], steps):
        values = []
        low = []
        high = []
        for origin_set, _ in ORIGIN_SETS:
            row = observed[(observed["comparison"] == comparison)
                           & (observed["origin_set"] == origin_set)
                           & (observed["lead_hours"] == lead)]
            if row.empty:
                values.append(np.nan)
                low.append(np.nan)
                high.append(np.nan)
                continue
            values.append(float(row["median_difference_m"].iloc[0]))
            low.append(float(row["median_difference_m"].iloc[0] - row["ci_low_m"].iloc[0]))
            high.append(float(row["ci_high_m"].iloc[0] - row["median_difference_m"].iloc[0]))
        axis.barh(positions + offset, values, height=height, color=color, alpha=0.85, label=label,
                  xerr=[low, high], error_kw={"ecolor": "0.3", "elinewidth": 1.1, "capsize": 3})
    axis.axvline(0.0, color="0.3", linewidth=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels([label for _, label in ORIGIN_SETS])
    axis.set_xlabel("RMSE improvement (m), paired on the gauge")
    axis.set_title(f"h+{lead}")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
figure.suptitle("Message passing is the largest single source of the advantage over persistence",
                fontsize=17.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.965))
save(figure, "argument_2_what_each_step_buys")

# ----------------------------------------------------------------- 3 graph effect against forcing quality
quality = read("graph_effect_versus_forcing_quality.csv")
figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
for axis, lead in zip(axes, [12, 24]):
    block = quality[quality["lead_hours"] == lead].sort_values("stgnn_median_rmse_m")
    x = block["stgnn_median_rmse_m"].to_numpy(dtype=float)
    y = block["graph_effect_m"].to_numpy(dtype=float)
    lower = y - block["ci_low_m"].to_numpy(dtype=float)
    upper = block["ci_high_m"].to_numpy(dtype=float) - y
    colors = ["#2E7D32" if value > 0 else "#B23A3A" for value in y]
    axis.axhline(0.0, color="0.3", linewidth=1.0, linestyle="--")
    axis.errorbar(x, y, yerr=[lower, upper], fmt="none", ecolor="0.45", elinewidth=1.1, capsize=3,
                  zorder=2)
    axis.scatter(x, y, s=90, c=colors, zorder=3, edgecolors="white", linewidth=0.8)
    # The forcings sit close together on the quality axis, so labels collide if they all go above
    # the point. Alternating above and below keeps every name readable.
    for index, (xi, yi, name) in enumerate(zip(x, y, block["forcing"])):
        above = index % 2 == 0
        axis.annotate(name, (xi, yi), textcoords="offset points",
                      xytext=(0, 13 if above else -20), ha="center",
                      va="bottom" if above else "top", fontsize=10.5, color="0.25")
    if len(block) >= 3:
        correlation = float(np.corrcoef(x, y)[0, 1])
        axis.annotate(f"correlation {correlation:+.2f}", (0.02, 0.05), xycoords="axes fraction",
                      fontsize=13.0, color="0.25", fontweight="bold")
    axis.set_xlabel("ST-GNN error under that rainfall forcing (m)")
    axis.set_ylabel("Graph effect (m), positive means the graph helps")
    axis.set_title(f"h+{lead}")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.margins(y=0.18)
figure.suptitle("The graph pays in proportion to how good the rainfall forecast is",
                fontsize=17.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.965))
save(figure, "argument_3_graph_effect_vs_forcing_quality")

# ----------------------------------------------------------------- 4 per-gauge wins
per_gauge = read("event_conditioned_per_gauge_metrics.csv")
seed_mean = (
    per_gauge.groupby(["model", "origin_set", "lead_hours", "gauge"], as_index=False)["RMSE_m"].mean()
)
figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), sharey=True)
for axis, (origin_set, set_title) in zip(axes, [ORIGIN_SETS[0], ORIGIN_SETS[2]]):
    block = seed_mean[(seed_mean["origin_set"] == origin_set) & (seed_mean["lead_hours"] == 24)]
    wide = block.pivot(index="gauge", columns="model", values="RMSE_m").dropna()
    difference = (wide["LSTM"] - wide["ST-GNN"]).sort_values()
    colors = ["#2E7D32" if value > 0 else "#B23A3A" for value in difference]
    axis.bar(np.arange(len(difference)), difference.to_numpy(), color=colors, width=0.86)
    axis.axhline(0.0, color="0.3", linewidth=1.0)
    wins = int((difference > 0).sum())
    axis.annotate(f"ST-GNN lower at {wins} of {len(difference)} gauges",
                  (0.03, 0.94), xycoords="axes fraction", fontsize=13.5, fontweight="bold",
                  color="#2E7D32", va="top")
    axis.set_title(set_title + ", h+24")
    axis.set_xlabel("Gauges, sorted")
    axis.set_xticks([])
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
axes[0].set_ylabel("LSTM RMSE minus ST-GNN RMSE (m)\npositive means the ST-GNN is better")
figure.suptitle("The advantage is broad across gauges, not a few gauges carrying the median",
                fontsize=17.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.965))
save(figure, "argument_4_per_gauge_wins")

# ----------------------------------------------------------------- 5 event definitions
sizes = read("event_definition_sizes.csv")
overlaps = read("event_definition_overlaps.csv")
figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

axis = axes[0]
order = ["pump_window", "pump_at_issue", "stage_p80", "stage_p90"]
labels = ["Pumps in the\n24 h window", "Pumps at issue", "Stage above p80", "Stage above p90"]
shares = [float(sizes[sizes["definition"] == name]["share_of_pairs"].iloc[0]) * 100 for name in order]
colors = ["#C9C9C9", "#2C6E9B", "#D95F70", "#8E3B46"]
axis.bar(np.arange(len(order)), shares, color=colors, width=0.62)
for position, value in enumerate(shares):
    axis.annotate(f"{value:.1f}%", (position, value), textcoords="offset points", xytext=(0, 5),
                  ha="center", fontsize=12.5, fontweight="bold")
axis.set_xticks(np.arange(len(order)))
axis.set_xticklabels(labels)
axis.set_ylabel("Share of gauge-origin pairs (%)")
axis.set_title("(a) How selective each definition is", loc="left")
axis.grid(axis="y", alpha=0.25)
axis.set_axisbelow(True)
axis.margins(y=0.16)

axis = axes[1]
pairs = [("pump_at_issue", "stage_p80"), ("pump_at_issue", "stage_p90")]
positions = np.arange(len(pairs))
height = 0.34
for offset, key, label, color in [
    (-height / 2, "share_of_a_also_b", "Of pumping hours,\nalso high stage", "#2C6E9B"),
    (height / 2, "share_of_b_also_a", "Of high-stage hours,\nalso pumping", "#D95F70"),
]:
    values = []
    for first, second in pairs:
        row = overlaps[(overlaps["definition_a"] == first) & (overlaps["definition_b"] == second)]
        values.append(float(row[key].iloc[0]) * 100 if not row.empty else np.nan)
    axis.barh(positions + offset, values, height=height, color=color, label=label, alpha=0.9)
    for position, value in zip(positions + offset, values):
        axis.annotate(f"{value:.0f}%", (value, position), textcoords="offset points", xytext=(4, 0),
                      va="center", fontsize=12.0, fontweight="bold")
axis.set_yticks(positions)
axis.set_yticklabels(["pumps vs p80", "pumps vs p90"])
axis.set_xlabel("Overlap (%)")
axis.set_xlim(0, 100)
axis.set_title("(b) The two definitions are not the same thing", loc="left")
axis.legend(loc="lower right", frameon=False, fontsize=11.5)
axis.grid(axis="x", alpha=0.25)
axis.set_axisbelow(True)

figure.suptitle("Pumping usually means high stage, but high stage usually means no pumping",
                fontsize=17.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.955))
save(figure, "argument_5_event_definitions")

# ----------------------------------------------------------------- 6 ST-GNN against the LSTM, by rainfall quality
head_to_head = read("lstm_vs_stgnn_across_forcings.csv")
figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
for axis, lead in zip(axes, [12, 24]):
    block = head_to_head[head_to_head["lead_hours"] == lead].sort_values("stgnn_median_rmse_m")
    x = block["stgnn_median_rmse_m"].to_numpy(dtype=float)
    y = block["stgnn_advantage_m"].to_numpy(dtype=float)
    lower = y - block["ci_low_m"].to_numpy(dtype=float)
    upper = block["ci_high_m"].to_numpy(dtype=float) - y
    excludes = block["excludes_zero"].to_numpy(dtype=bool)
    colors = [
        ("#2E7D32" if value > 0 else "#B23A3A") if solid else "#9A9A9A"
        for value, solid in zip(y, excludes)
    ]
    axis.axhline(0.0, color="0.3", linewidth=1.0, linestyle="--")
    axis.errorbar(x, y, yerr=[lower, upper], fmt="none", ecolor="0.45", elinewidth=1.1, capsize=3,
                  zorder=2)
    axis.scatter(x, y, s=95, c=colors, zorder=3, edgecolors="white", linewidth=0.8)
    for index, (xi, yi, label) in enumerate(zip(x, y, block["forcing"])):
        above = index % 2 == 0
        axis.annotate(label, (xi, yi), textcoords="offset points",
                      xytext=(0, 13 if above else -20), ha="center",
                      va="bottom" if above else "top", fontsize=10.5, color="0.25")
    if len(block) >= 3:
        correlation = float(np.corrcoef(x, y)[0, 1])
        axis.annotate(f"correlation {correlation:+.2f}", (0.02, 0.05), xycoords="axes fraction",
                      fontsize=13.0, color="0.25", fontweight="bold")
    axis.set_xlabel("ST-GNN error under that rainfall forcing (m)")
    axis.set_ylabel("ST-GNN advantage over the LSTM (m)")
    axis.set_title(f"h+{lead}")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.margins(y=0.20)
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="none", markersize=11, color="#2E7D32",
               label="ST-GNN better, interval excludes zero"),
    plt.Line2D([0], [0], marker="o", linestyle="none", markersize=11, color="#B23A3A",
               label="LSTM better, interval excludes zero"),
    plt.Line2D([0], [0], marker="o", linestyle="none", markersize=11, color="#9A9A9A",
               label="interval spans zero"),
]
figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.004))
figure.suptitle("The ST-GNN's lead over the LSTM returns as the rainfall forecast improves",
                fontsize=17.0, fontweight="bold", y=0.997)
figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.965))
save(figure, "argument_6_lstm_vs_stgnn_ladder")

print("ARGUMENT_FIGURES_DONE", flush=True)
