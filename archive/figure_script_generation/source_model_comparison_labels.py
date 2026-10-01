"""Replot Figure 11 of the HESS manuscript with value labels on panel (c).

Farid asked on 2026-09-24 for Figure 11 "verbatim" but with the panel (c) numbers shown, as in panel (a).
This script copies the Figure 11 recipe unchanged from
  20260921_Revision_9/scripts/source_simulated_forecast_figures.py (its lines 14-32 and 187-410)
and changes only panel (c) and three panel titles. Panel (c): each bar gets its value, in the panel (a) label style, and the y axis gets the
panel (a) headroom (1.38 times the largest bar) so the labels fit. Farid then asked (same day) to drop the word "efficiency": the panel titles now read "(a) RMSE at fixed lead
times", "(b) NSE at fixed lead times", and "(c) RMSE grouped by observed rise". The script also saves the panel (b)
values to work/fig11_panel_b_nse.csv. Later the same day Farid renamed the evaluation from "replay" to "simulated
forecast", so the figure title now reads "Simulated operational forecast, January-August 2026" instead of "Replay of the
actual forecast, January-August 2026". Data, colours, and the layout are unchanged.
Outputs: 20260922_Revision_13/figures/fig11_forecast_rain_model_comparison_jan_aug.png and .pdf
"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = "project/manuscript"
EXPERIMENT = ("project/Experiments/"
              "EXTEND_2026_AUG31_EVENTS_20260918")
REVISION = os.path.join(PAPER, "20260922_Revision_13")
FIGURES = os.path.join(REVISION, "figures")
FIXED_LEADS = [3, 6, 12, 24]
FT2M = 0.3048

SUMMARY = os.path.join(EXPERIMENT, "outputs", "summary_hrrr_jan_aug_postproc")
FIXED_LEAD_PER_GAUGE = os.path.join(SUMMARY, "fixed_lead_per_gauge.csv")
RISE_STRATA = os.path.join(SUMMARY, "persistence_rise_strata.csv")

fixed_lead = pd.read_csv(FIXED_LEAD_PER_GAUGE)

model_colors = {
    "LSTM": "#6a51a3",
    "GRU": "#d8741c",
    "ST-GNN": "#1b7837",
    "Persistence": "#777777",
}
model_markers = {"LSTM": "^", "GRU": "o", "ST-GNN": "s"}
panel_a = {}
efficiency_summary_rows = []
for model in ["LSTM", "GRU", "ST-GNN", "Persistence"]:
    model_block = fixed_lead[fixed_lead["model"] == model]
    if len(model_block) == 0:
        raise Exception("Figure 11 has no fixed-lead rows for " + model)
    seed_mean = model_block.groupby(["gauge", "lead_hours"], as_index=False)[
        ["RMSE_m", "NSE"]
    ].mean()
    panel_a[model] = []
    for lead_hours in FIXED_LEADS:
        lead_block = seed_mean[seed_mean["lead_hours"] == lead_hours]
        panel_a[model].append(float(lead_block["RMSE_m"].median()))
        if model != "Persistence":
            nse_values = lead_block["NSE"].dropna().to_numpy(float)
            efficiency_summary_rows.append(
                {
                    "model": model,
                    "lead_hours": lead_hours,
                    "median": float(np.median(nse_values)),
                    "q25": float(np.percentile(nse_values, 25)),
                    "q75": float(np.percentile(nse_values, 75)),
                }
            )
efficiency_summary = pd.DataFrame(efficiency_summary_rows)
# Revision 13: save the panel (b) values, so the text can quote them.
efficiency_summary.to_csv(os.path.join(REVISION, "work", "fig11_panel_b_nse.csv"), index=False)

rise_data = pd.read_csv(RISE_STRATA)
rise_groups = ["rise < 0.30 ft", "0.30-1.00 ft", "1.00-2.00 ft", "rise >= 2.00 ft"]
rise_labels = ["< 0.09 m\nnear-flat", "0.09-0.30 m", "0.30-0.61 m", ">= 0.61 m\nflood rises"]
rise_data = rise_data.set_index("rise_group").loc[rise_groups].reset_index()
for column_name in [
    "median_rmse24_persist",
    "median_rmse24_lstm",
    "median_rmse24_bilstm",
    "median_rmse24_stgnn",
]:
    rise_data[column_name] = rise_data[column_name] * FT2M

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
x_lead = np.arange(len(FIXED_LEADS))
bar_width = 0.20
bar_offsets = {
    "LSTM": -1.5 * bar_width,
    "GRU": -0.5 * bar_width,
    "ST-GNN": 0.5 * bar_width,
    "Persistence": 1.5 * bar_width,
}
for model in ["LSTM", "GRU", "ST-GNN", "Persistence"]:
    values = np.asarray(panel_a[model])
    bars = rmse_axis.bar(
        x_lead + bar_offsets[model],
        values,
        bar_width,
        color=model_colors[model],
        edgecolor="black",
        linewidth=0.5,
        label=model,
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
            color=model_colors[model],
            fontweight="bold",
        )
rmse_axis.set_xticks(x_lead)
rmse_axis.set_xticklabels(["h+3", "h+6", "h+12", "h+24"])
rmse_axis.set_ylabel("Median per-gauge RMSE (m)")
rmse_axis.set_title("(a) RMSE at fixed lead times")
rmse_axis.set_ylim(0.0, max(max(values) for values in panel_a.values()) * 1.38)
rmse_axis.grid(axis="y", alpha=0.25)
rmse_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

for model in ["LSTM", "GRU", "ST-GNN"]:
    model_summary = efficiency_summary[efficiency_summary["model"] == model].sort_values("lead_hours")
    median_values = model_summary["median"].to_numpy(float)
    lower_values = model_summary["q25"].to_numpy(float)
    upper_values = model_summary["q75"].to_numpy(float)
    nse_axis.fill_between(
        x_lead,
        lower_values,
        upper_values,
        color=model_colors[model],
        alpha=0.13,
        zorder=2,
    )
    nse_axis.plot(
        x_lead,
        median_values,
        color=model_colors[model],
        linewidth=2.1,
        marker=model_markers[model],
        label=model,
        zorder=4,
    )
nse_axis.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
nse_axis.set_xticks(x_lead)
nse_axis.set_xticklabels(["h+3", "h+6", "h+12", "h+24"])
nse_axis.set_ylabel("Per-gauge NSE (line = median, band = IQR)")
nse_axis.set_title("(b) NSE at fixed lead times")
nse_axis.grid(alpha=0.25)
nse_axis.legend(loc="lower left", framealpha=0.92)

x_rise = np.arange(len(rise_groups))
rise_columns = {
    "Persistence": "median_rmse24_persist",
    "LSTM": "median_rmse24_lstm",
    "GRU": "median_rmse24_bilstm",
    "ST-GNN": "median_rmse24_stgnn",
}
for model in ["Persistence", "LSTM", "GRU", "ST-GNN"]:
    values = rise_data[rise_columns[model]].to_numpy(float)
    bars = rise_rmse_axis.bar(
        x_rise + bar_offsets[model],
        values,
        bar_width,
        color=model_colors[model],
        edgecolor="black",
        linewidth=0.5,
        label=model,
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
            color=model_colors[model],
            fontweight="bold",
        )
rise_rmse_axis.set_xticks(x_rise)
rise_rmse_axis.set_xticklabels(rise_labels)
rise_rmse_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
rise_rmse_axis.set_ylabel("Median 0-24 h RMSE (m)")
rise_rmse_axis.set_title("(c) RMSE grouped by observed rise")
rise_rmse_axis.set_ylim(0.0, float(np.nanmax(rise_data[list(rise_columns.values())].to_numpy(float))) * 1.38)
rise_rmse_axis.grid(axis="y", alpha=0.25)
rise_rmse_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

win_width = 0.24
win_offsets = {"LSTM": -win_width, "GRU": 0.0, "ST-GNN": win_width}
win_columns = {
    "LSTM": "lstm_beats_persist_frac",
    "GRU": "bilstm_beats_persist_frac",
    "ST-GNN": "stgnn_beats_persist_frac",
}
win_axis.axhline(50.0, color="#444444", linewidth=1.3, linestyle="--", label="50% reference")
for model in ["LSTM", "GRU", "ST-GNN"]:
    values = 100.0 * rise_data[win_columns[model]].to_numpy(float)
    bars = win_axis.bar(
        x_rise + win_offsets[model],
        values,
        win_width,
        color=model_colors[model],
        edgecolor="black",
        linewidth=0.5,
        label=model,
    )
    for bar in bars:
        win_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            format(bar.get_height(), ".0f") + "%",
            ha="center",
            va="bottom",
            fontsize=11.0,
            color=model_colors[model],
            fontweight="bold",
        )
win_axis.set_xticks(x_rise)
win_axis.set_xticklabels(rise_labels)
win_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
win_axis.set_ylabel("Gauge-cycles beating persistence (%)")
win_axis.set_title("(d) Frequency of improvement over persistence")
win_axis.set_ylim(0.0, 112.0)
win_axis.grid(axis="y", alpha=0.25)
win_axis.legend(loc="upper left", framealpha=0.92, ncol=2)

figure.suptitle(
    "Simulated operational forecast, January-August 2026",
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
FIGURE_11_PNG = os.path.join(FIGURES, "fig11_forecast_rain_model_comparison_jan_aug.png")
FIGURE_11_PDF = os.path.join(FIGURES, "fig11_forecast_rain_model_comparison_jan_aug.pdf")
figure.savefig(FIGURE_11_PNG, bbox_inches="tight")
figure.savefig(FIGURE_11_PDF, bbox_inches="tight")
plt.close(figure)
print("[SAVED]", FIGURE_11_PNG, flush=True)
print("[SAVED]", FIGURE_11_PDF, flush=True)
