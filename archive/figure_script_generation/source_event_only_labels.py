"""Replot Figure 12 (event-only simulated operational forecast) with value labels on panel (c).

Farid asked on 2026-09-24 for the same changes as Figure 11. The plotting code is copied unchanged from
  20260921_Revision_9/scripts/source_event_only_model_comparison.py (its lines 48-59 and 308-551).
The two tables that script computed and saved are read back instead of being recomputed:
  20260921_Revision_9/work/event_model_comparison/event_replay_fixed_lead_per_gauge.csv
  20260921_Revision_9/work/event_model_comparison/event_replay_rise_strata.csv
Changes: panel (c) bars carry their values in the panel (a) style, with the panel (a) headroom (1.38 times the
largest bar), and the panel titles read "(a) RMSE at fixed lead times", "(b) NSE at fixed lead times", and
"(c) RMSE grouped by observed rise". The panel (b) values are saved to work/fig12_panel_b_nse.csv.
Data, colours, the figure title, and the layout are unchanged.
Outputs: 20260922_Revision_13/figures/fig14_replay_event_model_comparison_jan_aug.png and .pdf
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = "project/manuscript"
SOURCE_WORK = os.path.join(PAPER, "20260921_Revision_9", "work", "event_model_comparison")
REVISION = os.path.join(PAPER, "20260922_Revision_13")
FIGURE_DIRECTORY = os.path.join(REVISION, "figures")
os.makedirs(FIGURE_DIRECTORY, exist_ok=True)

RISE_GROUPS = [
    ("rise < 0.30 ft", -np.inf, 0.30),
    ("0.30-1.00 ft", 0.30, 1.00),
    ("1.00-2.00 ft", 1.00, 2.00),
    ("rise >= 2.00 ft", 2.00, np.inf),
]
RISE_LABELS = [
    "< 0.09 m\nnear-flat",
    "0.09-0.30 m",
    "0.30-0.61 m",
    ">= 0.61 m\nflood rises",
]

fixed_frame = pd.read_csv(os.path.join(SOURCE_WORK, "event_replay_fixed_lead_per_gauge.csv"))
strata_frame = pd.read_csv(os.path.join(SOURCE_WORK, "event_replay_rise_strata.csv"))

model_colors = {
    "LSTM": "#6a51a3",
    "GRU": "#d8741c",
    "ST-GNN": "#1b7837",
    "Persistence": "#777777",
}
model_markers = {
    "LSTM": "^",
    "GRU": "o",
    "ST-GNN": "s",
}
model_order = ["LSTM", "GRU", "ST-GNN", "Persistence"]
lead_order = [3, 6, 12, 24]

panel_a = {}
efficiency_rows = []
for model_name in model_order:
    model_block = fixed_frame[fixed_frame["model"] == model_name]
    seed_mean = model_block.groupby(
        ["gauge", "lead_hours"],
        as_index=False,
    )[["RMSE_m", "NSE"]].mean()
    panel_a[model_name] = []
    for lead_hours in lead_order:
        lead_block = seed_mean[seed_mean["lead_hours"] == lead_hours]
        panel_a[model_name].append(float(lead_block["RMSE_m"].median()))
        if model_name != "Persistence":
            nse_values = lead_block["NSE"].dropna().to_numpy(float)
            efficiency_rows.append(
                {
                    "model": model_name,
                    "lead_hours": lead_hours,
                    "median": float(np.median(nse_values)),
                    "q25": float(np.percentile(nse_values, 25)),
                    "q75": float(np.percentile(nse_values, 75)),
                }
            )
efficiency_frame = pd.DataFrame(efficiency_rows)
# Revision 13: save the panel (b) values, so the text can quote them.
efficiency_frame.to_csv(os.path.join(REVISION, "work", "fig12_panel_b_nse.csv"), index=False)

rise_rows = []
for rise_group, _lower_bound, _upper_bound in RISE_GROUPS:
    row = {"rise_group": rise_group}
    for model_name in model_order:
        model_values = strata_frame[
            (strata_frame["model"] == model_name)
            & (strata_frame["rise_group"] == rise_group)
        ]
        row[model_name + "_rmse"] = float(model_values["median_rmse24_m"].mean())
        if model_name != "Persistence":
            row[model_name + "_wins"] = float(
                model_values["beats_persistence_fraction"].mean()
            )
    rise_rows.append(row)
rise_frame = pd.DataFrame(rise_rows)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 13.2,
        "axes.titlesize": 16.0,
        "axes.titleweight": "bold",
        "axes.titlepad": 8.0,
        "axes.labelsize": 13.5,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 11.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)

figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))
rmse_axis = axes[0, 0]
nse_axis = axes[0, 1]
rise_rmse_axis = axes[1, 0]
win_axis = axes[1, 1]
x_lead = np.arange(len(lead_order))
bar_width = 0.20
bar_offsets = {
    "LSTM": -1.5 * bar_width,
    "GRU": -0.5 * bar_width,
    "ST-GNN": 0.5 * bar_width,
    "Persistence": 1.5 * bar_width,
}
for model_name in model_order:
    values = np.asarray(panel_a[model_name])
    bars = rmse_axis.bar(
        x_lead + bar_offsets[model_name],
        values,
        bar_width,
        color=model_colors[model_name],
        edgecolor="black",
        linewidth=0.5,
        label=model_name,
    )
    for bar, value in zip(bars, values):
        rmse_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.004,
            format(value, ".3f"),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10.5,
            color=model_colors[model_name],
            fontweight="bold",
        )
rmse_axis.set_xticks(x_lead)
rmse_axis.set_xticklabels(["h+3", "h+6", "h+12", "h+24"])
rmse_axis.set_ylabel("Median per-gauge RMSE (m)")
rmse_axis.set_title("(a) RMSE at fixed lead times")
maximum_bar = max(max(values) for values in panel_a.values())
rmse_axis.set_ylim(0.0, maximum_bar * 1.38)
rmse_axis.grid(axis="y", alpha=0.25)
rmse_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

for model_name in ["LSTM", "GRU", "ST-GNN"]:
    model_summary = efficiency_frame[
        efficiency_frame["model"] == model_name
    ].sort_values("lead_hours")
    median_values = model_summary["median"].to_numpy(float)
    lower_values = model_summary["q25"].to_numpy(float)
    upper_values = model_summary["q75"].to_numpy(float)
    nse_axis.fill_between(
        x_lead,
        lower_values,
        upper_values,
        color=model_colors[model_name],
        alpha=0.13,
        zorder=2,
    )
    nse_axis.plot(
        x_lead,
        median_values,
        color=model_colors[model_name],
        linewidth=2.1,
        marker=model_markers[model_name],
        label=model_name,
        zorder=4,
    )
nse_axis.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
nse_axis.set_xticks(x_lead)
nse_axis.set_xticklabels(["h+3", "h+6", "h+12", "h+24"])
nse_axis.set_ylabel("Per-gauge NSE (line = median, band = IQR)")
nse_axis.set_title("(b) NSE at fixed lead times")
nse_axis.grid(alpha=0.25)
nse_axis.legend(loc="lower left", framealpha=0.92)

x_rise = np.arange(len(RISE_GROUPS))
for model_name in model_order:
    values = rise_frame[model_name + "_rmse"].to_numpy(float)
    bars = rise_rmse_axis.bar(
        x_rise + bar_offsets[model_name],
        values,
        bar_width,
        color=model_colors[model_name],
        edgecolor="black",
        linewidth=0.5,
        label=model_name,
    )
    # Revision 13: value labels on panel (c), in the same style as panel (a).
    for bar, value in zip(bars, values):
        rise_rmse_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.008,
            format(value, ".3f"),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10.5,
            color=model_colors[model_name],
            fontweight="bold",
        )
rise_rmse_axis.set_xticks(x_rise)
rise_rmse_axis.set_xticklabels(RISE_LABELS)
rise_rmse_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
rise_rmse_axis.set_ylabel("Median 0-24 h RMSE (m)")
rise_rmse_axis.set_title("(c) RMSE grouped by observed rise")
rise_rmse_axis.set_ylim(0.0, float(np.nanmax(rise_frame[[m + "_rmse" for m in model_order]].to_numpy(float))) * 1.38)
rise_rmse_axis.grid(axis="y", alpha=0.25)
rise_rmse_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

win_width = 0.24
win_offsets = {
    "LSTM": -win_width,
    "GRU": 0.0,
    "ST-GNN": win_width,
}
win_axis.axhline(
    50.0,
    color="#444444",
    linewidth=1.3,
    linestyle="--",
    label="50% reference",
)
for model_name in ["LSTM", "GRU", "ST-GNN"]:
    values = 100.0 * rise_frame[model_name + "_wins"].to_numpy(float)
    bars = win_axis.bar(
        x_rise + win_offsets[model_name],
        values,
        win_width,
        color=model_colors[model_name],
        edgecolor="black",
        linewidth=0.5,
        label=model_name,
    )
    for bar in bars:
        win_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            format(bar.get_height(), ".0f") + "%",
            ha="center",
            va="bottom",
            fontsize=11.0,
            color=model_colors[model_name],
            fontweight="bold",
        )
win_axis.set_xticks(x_rise)
win_axis.set_xticklabels(RISE_LABELS)
win_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
win_axis.set_ylabel("Gauge-cycles beating persistence (%)")
win_axis.set_title("(d) Frequency of improvement over persistence")
win_axis.set_ylim(0.0, 112.0)
win_axis.grid(axis="y", alpha=0.25)
win_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

figure.suptitle(
    "Event-only replay, January-August 2026",
    fontsize=19.5,
    fontweight="bold",
    y=0.998,
)
figure.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.10,
    top=0.895,
    wspace=0.34,
    hspace=0.45,
)
output_png = os.path.join(
    FIGURE_DIRECTORY,
    "fig14_replay_event_model_comparison_jan_aug.png",
)
output_pdf = os.path.join(
    FIGURE_DIRECTORY,
    "fig14_replay_event_model_comparison_jan_aug.pdf",
)
figure.savefig(output_png, bbox_inches="tight")
figure.savefig(output_pdf, bbox_inches="tight")
print("[SAVE]", output_png, flush=True)
print("[SAVE]", output_pdf, flush=True)
plt.close(figure)
print("REV13_FIG12_DONE", flush=True)
