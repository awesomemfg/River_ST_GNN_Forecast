"""Build the event-only replay comparison for Revision 9.

Event-only is defined in the manuscript as a forecast origin at which at least one parish pump is
already running. The comparison uses the same 927 origins, 51 gauges, issue-time HRRR rainfall,
postprocessing, models, seeds, fixed leads, and aggregation order as the all-time replay figure.
"""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_DIRECTORY = (
    "project/"
    "manuscript/20260921_Revision_9"
)
EXPERIMENT_DIRECTORY = (
    "project/Experiments/"
    "EXTEND_2026_AUG31_EVENTS_20260918"
)
RUN_DIRECTORY = os.path.join(EXPERIMENT_DIRECTORY, "runs")
FIGURE_DIRECTORY = os.path.join(PAPER_DIRECTORY, "figures")
WORK_DIRECTORY = os.path.join(PAPER_DIRECTORY, "work", "event_model_comparison")
MATRIX_PATH = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
INPARISH_PATH = (
    "project/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json"
)

SEEDS = [101, 202, 303]
MODEL_ARCHIVES = {
    "ST-GNN": "stgnn",
    "LSTM": "lstm",
    "GRU": "gru",
}
FIXED_LEADS = {
    3: 11,
    6: 23,
    12: 47,
    24: 95,
}
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
FT2M = 0.3048
LOOK_FORWARD_STEPS = 96
QUANTILE = 0.8

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
os.makedirs(WORK_DIRECTORY, exist_ok=True)

print("[CONFIG] current working directory:", os.getcwd(), flush=True)
print("[CONFIG] matrix:", MATRIX_PATH, flush=True)
print("[CONFIG] run directory:", RUN_DIRECTORY, flush=True)

with open(INPARISH_PATH, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
gauges = sorted(str(value) for value in manifest["nodes"])
print("[LOAD] in-parish gauges:", len(gauges), flush=True)

loaded_predictions = {}
common_origins = None
for model_name, archive_prefix in MODEL_ARCHIVES.items():
    for seed in SEEDS:
        archive_path = os.path.join(
            RUN_DIRECTORY,
            archive_prefix + "_seed" + str(seed) + "_hrrr_jan_aug_postproc.npz",
        )
        if not os.path.isfile(archive_path):
            raise FileNotFoundError("Missing replay archive: " + archive_path)
        with np.load(archive_path, allow_pickle=True) as archive:
            archive_origins = pd.DatetimeIndex(
                pd.to_datetime([str(value) for value in archive["origins_utc"]])
            )
            archive_nodes = [str(value) for value in archive["nodes"]]
            archive_quantiles = [float(value) for value in archive["quantiles_saved"]]
            if QUANTILE not in archive_quantiles:
                raise Exception("Replay archive does not contain P80: " + archive_path)
            gauge_columns = [archive_nodes.index(gauge) for gauge in gauges]
            quantile_index = archive_quantiles.index(QUANTILE)
            prediction = archive["pred_ft"][:, :, gauge_columns, quantile_index].astype(
                np.float32
            )
        loaded_predictions[(model_name, seed)] = (archive_origins, prediction)
        if common_origins is None:
            common_origins = archive_origins
        else:
            common_origins = common_origins.intersection(archive_origins)
        print(
            "[LOAD]",
            model_name,
            "seed",
            seed,
            "shape",
            prediction.shape,
            flush=True,
        )

origins = pd.DatetimeIndex(sorted(common_origins))
if len(origins) != 5832:
    raise Exception("Expected 5,832 common replay origins, found " + str(len(origins)))
print("[VERIFY] common origins:", len(origins), origins[0], origins[-1], flush=True)

matrix = pd.read_pickle(MATRIX_PATH).sort_index()
if matrix.index.tz is not None:
    matrix.index = matrix.index.tz_convert("UTC").tz_localize(None)
row_of = {time_point: position for position, time_point in enumerate(matrix.index)}
origin_positions = np.asarray([row_of[time_point] for time_point in origins], dtype=int)

pump_columns = sorted(
    str(column_name)
    for column_name in matrix.columns
    if re.search(r"PD\d+_status$", str(column_name))
)
if len(pump_columns) == 0:
    raise Exception("No parish pump-status columns were found in the input matrix")
pump_running = np.zeros(len(matrix), dtype=bool)
for pump_column in pump_columns:
    pump_values = pd.to_numeric(matrix[pump_column], errors="coerce").fillna(0.0)
    pump_running = pump_running | (pump_values.to_numpy(float) > 0.5)
event_origin_mask = pump_running[origin_positions]
event_origin_count = int(event_origin_mask.sum())
if event_origin_count != 927:
    raise Exception("Expected 927 event-only origins, found " + str(event_origin_count))
print(
    "[VERIFY] event-only origins:",
    event_origin_count,
    "of",
    len(origins),
    "using",
    len(pump_columns),
    "pump-status columns",
    flush=True,
)

stage = np.full((len(matrix), len(gauges)), np.nan, dtype=np.float32)
stage_raw = np.full_like(stage, np.nan)
for gauge_index, gauge in enumerate(gauges):
    column_name = gauge + "_stage_ft"
    if column_name not in matrix.columns:
        raise Exception("Missing stage column: " + column_name)
    series = pd.to_numeric(matrix[column_name], errors="coerce")
    stage_raw[:, gauge_index] = series.to_numpy(np.float32)
    trailing_standard_deviation = series.rolling(
        window=96,
        center=False,
        min_periods=96,
    ).std()
    stage[:, gauge_index] = series.mask(
        trailing_standard_deviation < 1.0e-6
    ).to_numpy(np.float32)

truth_steps = origin_positions[:, None] + 1 + np.arange(LOOK_FORWARD_STEPS)[None, :]
truth = stage[truth_steps]

predictions = {}
for model_seed, archive_values in loaded_predictions.items():
    archive_origins = archive_values[0]
    archive_prediction = archive_values[1]
    origin_lookup = {
        time_point: origin_index
        for origin_index, time_point in enumerate(archive_origins)
    }
    selected_rows = [origin_lookup[time_point] for time_point in origins]
    predictions[model_seed] = archive_prediction[selected_rows]
predictions[("Persistence", 0)] = np.repeat(
    stage_raw[origin_positions][:, None, :],
    LOOK_FORWARD_STEPS,
    axis=1,
)


def calculate_fixed_metrics(observed_values, forecast_values):
    """Calculate the four manuscript metrics for one gauge and fixed lead."""
    valid = np.isfinite(observed_values) & np.isfinite(forecast_values)
    observed = observed_values[valid].astype(float)
    forecast = forecast_values[valid].astype(float)
    result = {
        "n_pairs": int(valid.sum()),
        "RMSE_m": np.nan,
        "NSE": np.nan,
        "Pearson_r": np.nan,
        "KGE": np.nan,
    }
    if len(observed) == 0:
        return result
    error = forecast - observed
    result["RMSE_m"] = float(np.sqrt(np.mean(error ** 2))) * FT2M
    observed_mean = float(np.mean(observed))
    forecast_mean = float(np.mean(forecast))
    observed_standard_deviation = float(np.std(observed))
    forecast_standard_deviation = float(np.std(forecast))
    denominator = float(np.sum((observed - observed_mean) ** 2))
    if denominator > 1.0e-9:
        result["NSE"] = 1.0 - float(np.sum(error ** 2)) / denominator
    if observed_standard_deviation > 1.0e-9 and forecast_standard_deviation > 1.0e-9:
        result["Pearson_r"] = float(np.corrcoef(observed, forecast)[0, 1])
    if np.isfinite(result["Pearson_r"]) and abs(observed_mean) > 1.0e-9:
        alpha = forecast_standard_deviation / observed_standard_deviation
        beta = forecast_mean / observed_mean
        result["KGE"] = 1.0 - float(
            np.sqrt((result["Pearson_r"] - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        )
    return result


fixed_rows = []
for model_seed, prediction in predictions.items():
    model_name = model_seed[0]
    seed = model_seed[1]
    for lead_hours, lead_step in FIXED_LEADS.items():
        for gauge_index, gauge in enumerate(gauges):
            observed_values = truth[event_origin_mask, lead_step, gauge_index]
            forecast_values = prediction[event_origin_mask, lead_step, gauge_index]
            metric_values = calculate_fixed_metrics(observed_values, forecast_values)
            fixed_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "gauge": gauge,
                    "lead_hours": lead_hours,
                    **metric_values,
                }
            )
fixed_frame = pd.DataFrame(fixed_rows)
fixed_path = os.path.join(WORK_DIRECTORY, "event_replay_fixed_lead_per_gauge.csv")
fixed_frame.to_csv(fixed_path, index=False)
print("[SAVE]", fixed_path, "rows", len(fixed_frame), flush=True)


def calculate_window_rmse(prediction):
    """Return one 0 to 24 h RMSE value for each origin and gauge."""
    error = prediction - truth
    valid = np.isfinite(error)
    valid_count = valid.sum(axis=1)
    squared_error_sum = np.where(valid, error ** 2, 0.0).sum(axis=1)
    return np.where(
        valid_count > 0,
        np.sqrt(squared_error_sum / np.maximum(valid_count, 1)),
        np.nan,
    )


window_rmse = {
    model_seed: calculate_window_rmse(prediction)
    for model_seed, prediction in predictions.items()
}
matched = np.logical_and.reduce(
    [np.isfinite(values) for values in window_rmse.values()]
)
matched = matched & event_origin_mask[:, None]

finite_truth = np.isfinite(truth)
enough_truth = finite_truth.sum(axis=1) >= 8
first_valid_index = np.argmax(finite_truth, axis=1)
first_valid_value = np.take_along_axis(
    truth,
    first_valid_index[:, None, :],
    axis=1,
)[:, 0, :]
maximum_value = np.nanmax(
    np.where(finite_truth, truth, -np.inf),
    axis=1,
)
rise_ft = np.where(enough_truth, maximum_value - first_valid_value, np.nan)

strata_rows = []
for model_seed, model_rmse in window_rmse.items():
    model_name = model_seed[0]
    seed = model_seed[1]
    base_mask = matched & np.isfinite(rise_ft)
    for rise_group, lower_bound, upper_bound in RISE_GROUPS:
        group_mask = base_mask & (rise_ft >= lower_bound) & (rise_ft < upper_bound)
        model_values = model_rmse[group_mask]
        persistence_values = window_rmse[("Persistence", 0)][group_mask]
        strata_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "rise_group": rise_group,
                "n": int(group_mask.sum()),
                "median_rmse24_m": float(np.median(model_values)) * FT2M,
                "beats_persistence_fraction": float(
                    np.mean(model_values < persistence_values)
                ),
            }
        )
strata_frame = pd.DataFrame(strata_rows)
strata_path = os.path.join(WORK_DIRECTORY, "event_replay_rise_strata.csv")
strata_frame.to_csv(strata_path, index=False)
print("[SAVE]", strata_path, "rows", len(strata_frame), flush=True)

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
rmse_axis.set_title("(a) Error at fixed lead times")
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
nse_axis.set_title("(b) Fixed-lead efficiency")
nse_axis.grid(alpha=0.25)
nse_axis.legend(loc="lower left", framealpha=0.92)

x_rise = np.arange(len(RISE_GROUPS))
for model_name in model_order:
    rise_rmse_axis.bar(
        x_rise + bar_offsets[model_name],
        rise_frame[model_name + "_rmse"],
        bar_width,
        color=model_colors[model_name],
        edgecolor="black",
        linewidth=0.5,
        label=model_name,
    )
rise_rmse_axis.set_xticks(x_rise)
rise_rmse_axis.set_xticklabels(RISE_LABELS)
rise_rmse_axis.set_xlabel("Observed stage rise over the 24 h verifying window")
rise_rmse_axis.set_ylabel("Median 0-24 h RMSE (m)")
rise_rmse_axis.set_title("(c) Error magnitude by observed rise")
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
plt.show()
plt.close(figure)
print("REV9_EVENT_MODEL_COMPARISON_DONE", flush=True)
