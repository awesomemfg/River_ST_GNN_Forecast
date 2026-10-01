"""Build the three January-August 2026 figures for Section 3.3.

The three figures retain the native Revision 7 designs:

1. ST-GNN fixed-lead distributions and the h+24 spatial map.
2. Fixed-lead and rising-stage comparisons of the ST-GNN, LSTM, GRU, and persistence.
3. ST-GNN h+6 hydrographs at 16 gauges.

All results use issue-time HRRR rainfall, 5,832 hourly forecast origins from 1 January through
31 August 2026, 51 in-parish gauges, frozen weights, and the Section 2.5.2 postprocessing. The model-comparison
figure averages per-gauge metrics over seeds before taking the median across gauges. The hydrograph
figure uses seed 101 and the P80 stage forecast, matching the manuscript's plotted quantile.
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
REVISION = os.path.join(PAPER, "20260921_Revision_9")
FIGURES = os.path.join(REVISION, "figures")
WORK = os.path.join(REVISION, "work", "section33")
SUMMARY = os.path.join(EXPERIMENT, "outputs", "summary_hrrr_jan_aug_postproc")
FIXED_LEAD_PER_GAUGE = os.path.join(SUMMARY, "fixed_lead_per_gauge.csv")
RISE_STRATA = os.path.join(SUMMARY, "persistence_rise_strata.csv")
STGNN_RUN = os.path.join(EXPERIMENT, "runs", "stgnn_seed101_hrrr_jan_aug_postproc.npz")
PICKLE = ("project/hpc/Experiments/"
          "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
          "global_features_all_stations_feature_engineered_20260911.pkl")
BASIN_SOURCE = os.path.join(
    PAPER,
    "revision_2026_07_17/contact_sheets_all_51/figure6_retrospective_h24/"
    "figure6_retrospective_h24_all51_metrics.csv",
)
SPATIAL_RECIPE = "src/figures/recipe_spatial_skill_map.py"



CONTACT_RECIPE = "src/figures/recipe_contact_sheets.py"



FIXED_LEADS = [3, 6, 12, 24]
FT2M = 0.3048
QUANTILE = 0.8
H6_LEAD_STEP = 24

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def apply_counted(source, substitutions, label):
    """Apply exact substitutions and stop if a native recipe changed unexpectedly."""
    for entry in substitutions:
        old_text = entry[0]
        new_text = entry[1]
        substitution_name = entry[2]
        count = source.count(old_text)
        if count != 1:
            raise Exception(
                label + ": expected one '" + substitution_name + "' target, found " + str(count)
            )
        source = source.replace(old_text, new_text, 1)
    print(
        "[VERIFY]",
        label,
        "applied",
        len(substitutions),
        "counted substitutions:",
        ", ".join(entry[2] for entry in substitutions),
        flush=True,
    )
    return source


def calculate_per_gauge_metrics(frame, basin_table):
    """Build the native contact-sheet metric table from its fixed-lead trace."""
    rows = []
    for _, basin_row in basin_table.iterrows():
        gauge = str(basin_row["gauge"])
        gauge_frame = frame[frame["gauge"] == gauge]
        observed = pd.to_numeric(gauge_frame["observed_stage_m"], errors="coerce").to_numpy(float)
        forecast = pd.to_numeric(gauge_frame["forecast_stage_m"], errors="coerce").to_numpy(float)
        valid = np.isfinite(observed) & np.isfinite(forecast)
        observed_valid = observed[valid]
        forecast_valid = forecast[valid]
        denominator = float(np.sum((observed_valid - observed_valid.mean()) ** 2))
        nse = np.nan
        correlation = np.nan
        if denominator > 1.0e-12:
            nse = 1.0 - float(np.sum((forecast_valid - observed_valid) ** 2)) / denominator
        if len(observed_valid) > 1:
            correlation = float(np.corrcoef(observed_valid, forecast_valid)[0, 1])
        rows.append(
            {
                "dataset": "figure9_prospective_h6",
                "gauge": gauge,
                "basin": basin_row["basin"],
                "basin_name": basin_row["basin_name"],
                "basin_order": basin_row["basin_order"],
                "row_count": len(gauge_frame),
                "n_pairs": int(valid.sum()),
                "missing_observations": int((~np.isfinite(observed)).sum()),
                "missing_predictions": int((~np.isfinite(forecast)).sum()),
                "rmse_m": float(np.sqrt(np.mean((forecast_valid - observed_valid) ** 2))),
                "nse": nse,
                "correlation": correlation,
                "verifying_time_start_utc": gauge_frame["verifying_time_utc"].min(),
                "verifying_time_end_utc": gauge_frame["verifying_time_utc"].max(),
            }
        )
    return pd.DataFrame(rows)


for required_path in (FIXED_LEAD_PER_GAUGE, RISE_STRATA, STGNN_RUN, PICKLE, BASIN_SOURCE):
    if not os.path.isfile(required_path):
        raise Exception("Required Section 3.3 input is missing: " + required_path)

# ----------------------------------------------------------------------------------------------
# Figure 10: the Revision 7 ST-GNN fixed-lead distributions and spatial map.
# ----------------------------------------------------------------------------------------------
fixed_lead = pd.read_csv(FIXED_LEAD_PER_GAUGE)
stgnn_fixed_lead = fixed_lead[
    (fixed_lead["model"] == "ST-GNN") & fixed_lead["lead_hours"].isin([6, 24])
].copy()
stgnn_fixed_lead = (
    stgnn_fixed_lead.groupby(["gauge", "lead_hours"], as_index=False)[
        ["RMSE_m", "Pearson_r", "NSE"]
    ]
    .mean()
    .rename(columns={"RMSE_m": "rmse_m", "Pearson_r": "correlation", "NSE": "nse"})
)
if stgnn_fixed_lead["gauge"].nunique() != 51:
    raise Exception("Figure 10 does not contain 51 gauges")
if sorted(stgnn_fixed_lead["lead_hours"].unique().tolist()) != [6, 24]:
    raise Exception("Figure 10 does not contain h+6 and h+24")
FIGURE_10_METRICS = os.path.join(WORK, "stgnn_hrrr_fixed_lead_metrics_jan_aug.csv")
stgnn_fixed_lead.to_csv(FIGURE_10_METRICS, index=False)
print("[DATA] wrote", FIGURE_10_METRICS, flush=True)

spatial_source = open(SPATIAL_RECIPE, encoding="utf-8").read()
spatial_source = apply_counted(
    spatial_source,
    [
        (
            'METRICS_PATH = os.path.join(\n    PROJECT_DIRECTORY,\n    "figure_library_outputs",\n'
            '    "rolling_but_live",\n    "live_fixed_lead_metrics_inparish51.csv",\n)',
            'METRICS_PATH = "' + FIGURE_10_METRICS + '"',
            "metrics path",
        ),
        (
            'OUTPUT_DIRECTORY = os.path.join(\n    PROJECT_DIRECTORY,\n    "20260826_Final_V2",\n'
            '    "output",\n    "figures",\n)',
            'OUTPUT_DIRECTORY = "' + FIGURES + '"',
            "output directory",
        ),
        ('OUTPUT_PNG = os.path.join(OUTPUT_DIRECTORY, "fig09.png")',
         'OUTPUT_PNG = os.path.join(OUTPUT_DIRECTORY, "fig10_forecast_rain_skill_jan_aug.png")',
         "PNG name"),
        ('OUTPUT_PDF = os.path.join(OUTPUT_DIRECTORY, "fig09.pdf")',
         'OUTPUT_PDF = os.path.join(OUTPUT_DIRECTORY, "fig10_forecast_rain_skill_jan_aug.pdf")',
         "PDF name"),
        (
            '    "Delivered operational ST-GNN performance\\n"\n'
            '    "371 hourly issues, 7-22 June 2026, 51 in-parish gauges",',
            '    "ST-GNN replay with postprocessing and issue-time HRRR rainfall\\n"\n'
            '    "5,832 hourly forecast origins, 1 January-31 August 2026, 51 in-parish gauges",',
            "title and period",
        ),
    ],
    "Section 3.3 Figure 10",
)
exec(compile(spatial_source, SPATIAL_RECIPE, "exec"),
     {"__file__": SPATIAL_RECIPE, "__name__": "__main__"})
plt.close("all")

# ----------------------------------------------------------------------------------------------
# Figure 11: the Revision 7 four-panel comparison, now with the GRU as a distinct model.
# ----------------------------------------------------------------------------------------------
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
rmse_axis.set_title("(a) Error at fixed lead times")
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
nse_axis.set_title("(b) Fixed-lead efficiency")
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
    rise_rmse_axis.bar(
        x_rise + bar_offsets[model],
        rise_data[rise_columns[model]],
        bar_width,
        color=model_colors[model],
        edgecolor="black",
        linewidth=0.5,
        label=model,
    )
rise_rmse_axis.set_xticks(x_rise)
rise_rmse_axis.set_xticklabels(rise_labels)
rise_rmse_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
rise_rmse_axis.set_ylabel("Median 0-24 h RMSE (m)")
rise_rmse_axis.set_title("(c) Error magnitude by observed rise")
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
    "Replay of the actual forecast, January-August 2026",
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

# ----------------------------------------------------------------------------------------------
# Figure 12: the Revision 7 h+6 hydrograph contact sheet, extended to the 5,832 origins.
# ----------------------------------------------------------------------------------------------
archive = np.load(STGNN_RUN, allow_pickle=True)
quantile_values = [float(value) for value in archive["quantiles_saved"]]
if QUANTILE not in quantile_values:
    raise Exception("The ST-GNN archive does not contain the P80 forecast")
quantile_index = quantile_values.index(QUANTILE)
origins = pd.to_datetime(archive["origins_utc"])
gauges = [str(node) for node in archive["nodes"]]
forecast_metres = archive["pred_ft"][:, H6_LEAD_STEP - 1, :, quantile_index].astype(float) * FT2M
verifying_times = origins + pd.Timedelta(hours=H6_LEAD_STEP * 0.25)
stage_frame = pd.read_pickle(PICKLE).sort_index()
if stage_frame.index.tz is not None:
    stage_frame.index = stage_frame.index.tz_convert("UTC").tz_localize(None)
trace_frames = []
for gauge_index, gauge in enumerate(gauges):
    stage_column = gauge + "_stage_ft"
    if stage_column not in stage_frame.columns:
        continue
    observed_metres = stage_frame[stage_column].reindex(verifying_times).to_numpy(float) * FT2M
    trace_frames.append(
        pd.DataFrame(
            {
                "gauge": gauge,
                "verifying_time_utc": verifying_times,
                "observed_stage_m": observed_metres,
                "forecast_stage_m": forecast_metres[:, gauge_index],
            }
        )
    )
h6_trace = pd.concat(trace_frames, ignore_index=True)
H6_TRACE_PATH = os.path.join(WORK, "forecast_rain_h6_all51_traces_jan_aug.csv")
h6_trace.to_csv(H6_TRACE_PATH, index=False)
basin_table = pd.read_csv(BASIN_SOURCE)[["gauge", "basin", "basin_name", "basin_order"]]
basin_table["gauge"] = basin_table["gauge"].astype(str)
h6_metrics = calculate_per_gauge_metrics(h6_trace, basin_table)
H6_METRIC_PATH = os.path.join(WORK, "forecast_rain_h6_all51_metrics_jan_aug.csv")
h6_metrics.to_csv(H6_METRIC_PATH, index=False)
print("[DATA] wrote", H6_TRACE_PATH, len(h6_trace), "rows", flush=True)
print("[DATA] wrote", H6_METRIC_PATH, flush=True)

contact_source = open(CONTACT_RECIPE, encoding="utf-8").read()
contact_source = apply_counted(
    contact_source,
    [
        (
            'REVISION_OUTPUT_DIRECTORY = Path(\n    "project/"\n'
            '    "manuscript/20260826_Final_V2/"\n'
            '    "tmp/contact_sheets_font_revision"\n)',
            'REVISION_OUTPUT_DIRECTORY = Path("' + WORK + '")',
            "work directory",
        ),
        (
            'FINAL_FIGURE_DIRECTORY = Path(\n    "project/"\n'
            '    "manuscript/20260826_Final_V2/"\n'
            '    "output/figures"\n)',
            'FINAL_FIGURE_DIRECTORY = Path("' + FIGURES + '")',
            "figure directory",
        ),
        (
            'FIGURE9_H6_TRACE_PATH = Path(',
            'FIGURE9_H6_TRACE_PATH = Path("' + H6_TRACE_PATH + '")\nUNUSED_H6 = Path(',
            "h+6 trace path",
        ),
        (
            'FIGURE9_H6_METRIC_PATH = (\n    FIGURE9_H6_DIRECTORY\n'
            '    / "figure9_prospective_h6_all51_metrics.csv"\n)',
            'FIGURE9_H6_METRIC_PATH = Path("' + H6_METRIC_PATH + '")',
            "h+6 metric path",
        ),
        (
            '    figure.suptitle(\n        dataset_spec["title"],\n        fontsize=16.0,\n'
            '        fontweight="bold",\n        y=0.994,\n    )',
            '    figure.suptitle(\n        dataset_spec["title"],\n        fontsize=19.0,\n'
            '        fontweight="bold",\n        y=0.997,\n    )',
            "title size",
        ),
        (
            '    figure.text(\n        0.5,\n        0.968,\n        (\n'
            '            dataset_spec["subtitle"]\n            + " | Selection: "\n'
            '            + criterion_spec["description"]\n        ),\n'
            '        horizontalalignment="center",\n        verticalalignment="top",\n'
            '        fontsize=10.2,\n    )\n',
            "",
            "remove subtitle",
        ),
        (
            '        bbox_to_anchor=(0.5, 0.922),\n        ncol=2,\n'
            '        frameon=False,\n        fontsize=10.2,',
            '        bbox_to_anchor=(0.5, 0.974),\n        ncol=2,\n'
            '        frameon=False,\n        fontsize=13.0,',
            "legend position",
        ),
        (
            '        "Stage (m)",\n        rotation=90,\n        horizontalalignment="center",\n'
            '        verticalalignment="center",\n        fontsize=11.0,\n    )',
            '        "Stage (m)",\n        rotation=90,\n        horizontalalignment="center",\n'
            '        verticalalignment="center",\n        fontsize=14.0,\n    )',
            "stage label size",
        ),
        (
            '    figure.text(\n        0.5,\n        0.010,\n        "Verifying time (UTC)",\n'
            '        horizontalalignment="center",\n        verticalalignment="bottom",\n'
            '        fontsize=9.5,\n        color="#444444",\n    )',
            '    figure.text(\n        0.5,\n        0.008,\n        "Verifying time (UTC)",\n'
            '        horizontalalignment="center",\n        verticalalignment="bottom",\n'
            '        fontsize=14.0,\n        fontweight="bold",\n        color="#222222",\n    )',
            "time label size",
        ),
        (
            '    figure.subplots_adjust(\n        left=0.073,\n        right=0.996,\n'
            '        top=0.845,\n        bottom=0.070,\n        hspace=0.68,\n'
            '        wspace=0.18,\n    )',
            '    figure.subplots_adjust(\n        left=0.076,\n        right=0.996,\n'
            '        top=0.872,\n        bottom=0.078,\n        hspace=0.62,\n'
            '        wspace=0.18,\n    )',
            "panel layout",
        ),
        (
            '    panel_title = (\n        gauge\n        + " | "\n        + basin_name\n'
            '        + "\\nRMSE "\n        + rmse_text\n        + " m | NSE "\n'
            '        + nse_text\n        + " | r "\n        + correlation_text\n'
            '        + " | n "\n        + str(pair_count)\n    )',
            '    panel_title = (\n        gauge\n        + " | "\n        + basin_name\n'
            '        + "\\nRMSE "\n        + rmse_text\n        + " m | NSE "\n'
            '        + nse_text\n        + " | r "\n        + correlation_text\n    )',
            "panel title content",
        ),
        ('        panel_title,\n        fontsize=9.2,',
         '        panel_title,\n        fontsize=12.2,',
         "panel title size"),
        (
            '    axis.tick_params(\n        axis="both",\n        labelsize=9.8,\n'
            '        length=2.0,\n        width=0.5,\n        pad=1.2,\n    )',
            '    axis.tick_params(\n        axis="both",\n        labelsize=12.2,\n'
            '        length=2.6,\n        width=0.6,\n        pad=1.6,\n    )',
            "tick label size",
        ),
        (
            '            "Delivered ST-GNN fixed-lead h+6 forecasts "\n'
            '            "across in-parish gauges"',
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+6"',
            "brief title",
        ),
        (
            '        "expected_points_per_gauge": 371,\n'
            '        "title": (\n'
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+6"',
            '        "expected_points_per_gauge": 5832,\n'
            '        "title": (\n'
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+6"',
            "January-August origin count",
        ),
        (
            '        "date_tick_mode": "four_day",\n'
            '        "source_format": "figure9_h6_raw_trace",',
            '        "date_tick_mode": "monthly",\n'
            '        "source_format": "figure9_h6_raw_trace",',
            "monthly date ticks",
        ),
        ('"forecast_label": "Delivered ST-GNN h+6 forecast",',
         '"forecast_label": "ST-GNN h+6 forecast",',
         "forecast label"),
        (
            '        elif dataset == "figure9_prospective_h24" and criterion_spec["variant"] == "golden06":',
            '        elif dataset == "figure9_prospective_h6" and criterion_spec["variant"] == "golden06":',
            "h+6 output alias",
        ),
        (
            '    if dataset_spec["dataset"] in {\n        "figure9_prospective_h24",\n'
            '        "figure6_retrospective_h24",\n    }',
            '    if dataset_spec["dataset"] in {\n        "figure9_prospective_h6",\n    }',
            "one dataset only",
        ),
    ],
    "Section 3.3 Figure 12",
)
exec(compile(contact_source, CONTACT_RECIPE, "exec"),
     {"__file__": CONTACT_RECIPE, "__name__": "__main__"})
plt.close("all")
native_contact_output = os.path.join(FIGURES, "fig11.png")
FIGURE_12_PNG = os.path.join(FIGURES, "fig12_forecast_rain_h6_jan_aug.png")
if not os.path.isfile(native_contact_output):
    raise Exception("The native contact-sheet recipe did not create " + native_contact_output)
shutil.move(native_contact_output, FIGURE_12_PNG)
print("[SAVED]", FIGURE_12_PNG, flush=True)

for required_output in (
    os.path.join(FIGURES, "fig10_forecast_rain_skill_jan_aug.png"),
    FIGURE_11_PNG,
    FIGURE_12_PNG,
):
    if not os.path.isfile(required_output):
        raise Exception("Missing Section 3.3 figure: " + required_output)
plt.show()
print("REV8_SECTION33_FIGURES_DONE", flush=True)
