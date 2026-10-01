"""Build the revised graph-ranking, reforecast, and structure-sensitivity figures."""

import os
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVISION_DIR = "project/manuscript/revision_2026_07_17"
FIGURE_DIR = os.path.join(REVISION_DIR, "figures_rewritten")

GRAPH_EXPERIMENT_DIR = "project/Experiments/REFORECAST_GRAPH_RANKING_2026H1"
GRAPH_RESULTS_DIR = os.path.join(GRAPH_EXPERIMENT_DIR, "results")
GRAPH_EVALUATION_DIR = os.path.join(
    GRAPH_RESULTS_DIR,
    "evaluations",
    "out_obs_pre2026",
)

STRUCTURE_EXPERIMENT_DIR = "project/Experiments/HBSU0398_STRUCT_ASIFLIVE_20260717"
STRUCTURE_OUTPUT_DIR = os.path.join(STRUCTURE_EXPERIMENT_DIR, "outputs")

GRAPH_SOURCE_PNG = os.path.join(
    GRAPH_RESULTS_DIR,
    "figures",
    "fig_graph_ranking_reforecast_2026H1.png",
)
GRAPH_SOURCE_PDF = os.path.join(
    GRAPH_RESULTS_DIR,
    "figures",
    "fig_graph_ranking_reforecast_2026H1.pdf",
)
GRAPH_TARGET_PNG = os.path.join(
    FIGURE_DIR,
    "fig05_graph_ranking_reforecast_2026H1.png",
)
GRAPH_TARGET_PDF = os.path.join(
    FIGURE_DIR,
    "fig05_graph_ranking_reforecast_2026H1.pdf",
)

LEAD_METRICS_PATH = os.path.join(
    GRAPH_EVALUATION_DIR,
    "reforecast_2026H1_lead_metrics.csv",
)
GAUGE_METRICS_PATH = os.path.join(
    GRAPH_EVALUATION_DIR,
    "reforecast_2026H1_metrics.csv",
)
H24_TIMESERIES_PATH = os.path.join(
    GRAPH_EVALUATION_DIR,
    "reforecast_2026H1_h96_timeseries.csv",
)

REFORECAST_SUMMARY_PNG = os.path.join(
    FIGURE_DIR,
    "fig06_reforecast_obs_pre2026_51gauges.png",
)
REFORECAST_SUMMARY_PDF = os.path.join(
    FIGURE_DIR,
    "fig06_reforecast_obs_pre2026_51gauges.pdf",
)
REFORECAST_EXAMPLES_PNG = os.path.join(
    FIGURE_DIR,
    "fig07_reforecast_h24_four_basins_obs_pre2026.png",
)
REFORECAST_EXAMPLES_PDF = os.path.join(
    FIGURE_DIR,
    "fig07_reforecast_h24_four_basins_obs_pre2026.pdf",
)

STRUCTURE_METRICS_PATH = os.path.join(
    STRUCTURE_OUTPUT_DIR,
    "structure_vs_no_structure_metric_deltas_by_gauge.csv",
)
STRUCTURE_FIXED_SERIES_PATH = os.path.join(
    STRUCTURE_OUTPUT_DIR,
    "hbsu0398_fixed_lead_series.csv",
)
STRUCTURE_FIXED_METRICS_PATH = os.path.join(
    STRUCTURE_OUTPUT_DIR,
    "hbsu0398_fixed_lead_metrics.csv",
)
STRUCTURE_FIGURE_PNG = os.path.join(
    FIGURE_DIR,
    "fig11_structure_input_operational_style.png",
)
STRUCTURE_FIGURE_PDF = os.path.join(
    FIGURE_DIR,
    "fig11_structure_input_operational_style.pdf",
)

IN_PARISH_GAUGES = [
    "BCSA1299",
    "BCSA2400",
    "BCSA4409",
    "BCSA4419",
    "BCSA6146",
    "BCSU4350",
    "BCSU4419",
    "BCSU4420",
    "BMSA0347",
    "BMSA1291",
    "BMSA1292",
    "BMSA2200",
    "BMSA2201",
    "BMSA3548",
    "BMSU0581",
    "BMSU0853",
    "BMSU2119",
    "BMSU2210",
    "BMSU2211",
    "BMSU2620",
    "HBSA1384",
    "HBSA2096",
    "HBSU0398",
    "HBSU1384",
    "HBSU1385",
    "HBSU3213",
    "HBSU4751",
    "HBSU6714",
    "MBSA3301",
    "MBSA4314",
    "MBSA4413",
    "MBSA4569",
    "MBSA5275",
    "MBSA5584",
    "MBSA5585",
    "MBSA5715",
    "MBSA5881",
    "MBSA6410",
    "MBSA6512",
    "MBSA6561",
    "MBSA7806",
    "MBSA8344",
    "MBSA9909",
    "MBSA9919",
    "MBSU3561",
    "MBSU4314",
    "MBSU5688",
    "MBSU6557",
    "MBSU8999",
    "MBSU9000",
    "MBSU9909",
]

BASIN_NAMES = {
    "BC": "Bayou Conway",
    "BM": "Bayou Manchac",
    "HB": "Henderson Bayou",
    "MB": "Marvin Braud",
}

BASIN_COLORS = {
    "Bayou Conway": "#0072B2",
    "Bayou Manchac": "#009E73",
    "Henderson Bayou": "#CC79A7",
    "Marvin Braud": "#E69F00",
}

OBSERVED_COLOR = "#111111"
FORECAST_COLOR = "#D1495B"
BASE_COLOR = "#D55E00"
STRUCTURE_COLOR = "#0072B2"
GRID_COLOR = "#808080"


def empirical_cdf(values):
    """Return sorted finite values and their empirical cumulative probabilities."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    sorted_values = np.sort(finite_values)
    cumulative_probability = np.arange(
        1,
        len(sorted_values) + 1,
        dtype=np.float64,
    ) / float(len(sorted_values))
    return sorted_values, cumulative_probability


def shared_square_limits(x_values, y_values, minimum_padding):
    """Return equal x and y limits that contain every finite paired value."""
    combined_values = np.concatenate(
        [
            np.asarray(x_values, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
        ]
    )
    finite_values = combined_values[np.isfinite(combined_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot calculate scatter limits from empty values.")
    lower_value = float(np.min(finite_values))
    upper_value = float(np.max(finite_values))
    value_range = upper_value - lower_value
    padding = max(value_range * 0.07, minimum_padding)
    lower_limit = max(0.0, lower_value - padding)
    upper_limit = upper_value + padding
    return lower_limit, upper_limit


def save_figure(figure, png_path, pdf_path):
    """Save one figure as both PNG and PDF with explicit tracing."""
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    print("[SAVE] PNG:", png_path, flush=True)
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    print("[SAVE] PDF:", pdf_path, flush=True)


required_paths = [
    GRAPH_SOURCE_PNG,
    GRAPH_SOURCE_PDF,
    LEAD_METRICS_PATH,
    GAUGE_METRICS_PATH,
    H24_TIMESERIES_PATH,
    STRUCTURE_METRICS_PATH,
    STRUCTURE_FIXED_SERIES_PATH,
    STRUCTURE_FIXED_METRICS_PATH,
]
for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError(
            "Required figure input is missing: "
            + required_path
        )

os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.0,
        "savefig.dpi": 300,
    }
)

print("[COPY] Graph-ranking figure from the verified experiment.", flush=True)
shutil.copy2(GRAPH_SOURCE_PNG, GRAPH_TARGET_PNG)
shutil.copy2(GRAPH_SOURCE_PDF, GRAPH_TARGET_PDF)
print("[SAVE] PNG:", GRAPH_TARGET_PNG, flush=True)
print("[SAVE] PDF:", GRAPH_TARGET_PDF, flush=True)

print("[LOAD] Reforecast lead metrics:", LEAD_METRICS_PATH, flush=True)
lead_metrics = pd.read_csv(LEAD_METRICS_PATH)
required_lead_columns = {
    "lead_hours",
    "node",
    "RMSE_m",
    "MAE_m",
    "NSE",
    "Pearson_r",
}
missing_lead_columns = required_lead_columns.difference(lead_metrics.columns)
if missing_lead_columns:
    raise KeyError(
        "Lead metric table is missing columns: "
        + str(sorted(missing_lead_columns))
    )

lead_metrics = lead_metrics[
    lead_metrics["node"].isin(IN_PARISH_GAUGES)
].copy()
if lead_metrics["node"].nunique() != 51:
    raise ValueError(
        "Expected 51 in-parish gauges in the lead metric table, found "
        + str(lead_metrics["node"].nunique())
    )

lead_summary = (
    lead_metrics
    .groupby("lead_hours", as_index=False)
    .agg(
        median_RMSE_m=("RMSE_m", "median"),
        median_MAE_m=("MAE_m", "median"),
        median_NSE=("NSE", "median"),
        median_Pearson_r=("Pearson_r", "median"),
    )
    .sort_values("lead_hours")
)

h24_metrics = lead_metrics[
    np.isclose(
        lead_metrics["lead_hours"].to_numpy(dtype=np.float64),
        24.0,
    )
].copy()
if len(h24_metrics) != 51:
    raise ValueError(
        "Expected 51 h+24 gauge rows, found "
        + str(len(h24_metrics))
    )

rmse_values, rmse_ecdf = empirical_cdf(h24_metrics["RMSE_m"])
nse_values, nse_ecdf = empirical_cdf(h24_metrics["NSE"])
median_h24_rmse = float(h24_metrics["RMSE_m"].median())
median_h24_nse = float(h24_metrics["NSE"].median())

fig_reforecast, reforecast_axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12.2, 7.8),
)

ax_error = reforecast_axes[0, 0]
ax_error.plot(
    lead_summary["lead_hours"],
    lead_summary["median_RMSE_m"],
    color="#003F5C",
    linewidth=2.4,
    label="Median RMSE",
)
ax_error.plot(
    lead_summary["lead_hours"],
    lead_summary["median_MAE_m"],
    color=FORECAST_COLOR,
    linewidth=2.4,
    label="Median MAE",
)
ax_error.set_title("(a) Error across forecast lead times", loc="left")
ax_error.set_xlabel("Forecast lead (h)")
ax_error.set_ylabel("Stage error (m)")
ax_error.set_xlim(0.0, 24.0)
ax_error.legend(loc="upper left", framealpha=0.92)
ax_error.grid(color=GRID_COLOR, alpha=0.22)

ax_skill = reforecast_axes[0, 1]
ax_skill.plot(
    lead_summary["lead_hours"],
    lead_summary["median_NSE"],
    color="#007C91",
    linewidth=2.4,
    label="Median NSE",
)
ax_skill.plot(
    lead_summary["lead_hours"],
    lead_summary["median_Pearson_r"],
    color="#F0A830",
    linewidth=2.4,
    label="Median Pearson r",
)
ax_skill.axhline(
    0.0,
    color="black",
    linewidth=0.8,
    alpha=0.55,
)
ax_skill.set_title("(b) Deterministic skill", loc="left")
ax_skill.set_xlabel("Forecast lead (h)")
ax_skill.set_ylabel("Skill")
ax_skill.set_xlim(0.0, 24.0)
ax_skill.set_ylim(-0.10, 1.05)
ax_skill.legend(loc="lower left", framealpha=0.92)
ax_skill.grid(color=GRID_COLOR, alpha=0.22)

ax_rmse_ecdf = reforecast_axes[1, 0]
ax_rmse_ecdf.plot(
    rmse_values,
    rmse_ecdf,
    color="#003F5C",
    linewidth=2.4,
)
ax_rmse_ecdf.axvline(
    median_h24_rmse,
    color=FORECAST_COLOR,
    linestyle="--",
    linewidth=1.4,
    label="Median = " + f"{median_h24_rmse:.3f}" + " m",
)
ax_rmse_ecdf.set_title("(c) h+24 RMSE distribution", loc="left")
ax_rmse_ecdf.set_xlabel("h+24 RMSE (m)")
ax_rmse_ecdf.set_ylabel("Gauge ECDF")
ax_rmse_ecdf.set_ylim(0.0, 1.02)
ax_rmse_ecdf.legend(loc="lower right", framealpha=0.92)
ax_rmse_ecdf.grid(color=GRID_COLOR, alpha=0.22)

ax_nse_ecdf = reforecast_axes[1, 1]
ax_nse_ecdf.plot(
    nse_values,
    nse_ecdf,
    color="#007C91",
    linewidth=2.4,
)
ax_nse_ecdf.axvline(
    median_h24_nse,
    color=FORECAST_COLOR,
    linestyle="--",
    linewidth=1.4,
    label="Median = " + f"{median_h24_nse:.3f}",
)
ax_nse_ecdf.set_title("(d) h+24 NSE distribution", loc="left")
ax_nse_ecdf.set_xlabel("h+24 NSE")
ax_nse_ecdf.set_ylabel("Gauge ECDF")
ax_nse_ecdf.set_ylim(0.0, 1.02)
ax_nse_ecdf.legend(loc="lower right", framealpha=0.92)
ax_nse_ecdf.grid(color=GRID_COLOR, alpha=0.22)

fig_reforecast.suptitle(
    "Retrospective ST-GNN performance, January-June 2026",
    fontsize=15.0,
    fontweight="bold",
)
fig_reforecast.text(
    0.5,
    0.012,
    "Leakage-safe pre-2026 observation lead-lag graph; 4,344 hourly origins; perfect observed-rainfall forcing; 51 in-parish gauges.",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=8.5,
    color="0.30",
)
fig_reforecast.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
save_figure(
    fig_reforecast,
    REFORECAST_SUMMARY_PNG,
    REFORECAST_SUMMARY_PDF,
)
plt.close(fig_reforecast)

print("[LOAD] Reforecast gauge metrics:", GAUGE_METRICS_PATH, flush=True)
gauge_metrics = pd.read_csv(GAUGE_METRICS_PATH)
gauge_metrics = gauge_metrics[
    gauge_metrics["node"].isin(IN_PARISH_GAUGES)
].copy()
gauge_metrics["basin_key"] = gauge_metrics["node"].str[:2]

selected_gauges = []
selection_rows = []
for basin_key in ["BC", "BM", "HB", "MB"]:
    basin_metrics = gauge_metrics[
        gauge_metrics["basin_key"] == basin_key
    ].copy()
    median_rmse = float(basin_metrics["RMSE_h96_m"].median())
    median_correlation = float(
        basin_metrics["Pearson_r_h96"].median()
    )
    eligible_metrics = basin_metrics[
        basin_metrics["Pearson_r_h96"] >= median_correlation
    ].copy()
    eligible_metrics["rmse_distance_from_basin_median"] = (
        eligible_metrics["RMSE_h96_m"] - median_rmse
    ).abs()
    eligible_metrics = eligible_metrics.sort_values(
        [
            "rmse_distance_from_basin_median",
            "node",
        ]
    )
    selected_row = eligible_metrics.iloc[0]
    selected_gauge = str(selected_row["node"])
    selected_gauges.append(selected_gauge)
    selection_rows.append(
        {
            "basin_key": basin_key,
            "basin": BASIN_NAMES[basin_key],
            "gauge": selected_gauge,
            "basin_median_h24_rmse_m": median_rmse,
            "basin_median_h24_correlation": median_correlation,
            "gauge_h24_rmse_m": float(selected_row["RMSE_h96_m"]),
            "gauge_h24_nse": float(selected_row["NSE_h96"]),
            "gauge_h24_correlation": float(selected_row["Pearson_r_h96"]),
        }
    )

selection_path = os.path.join(
    FIGURE_DIR,
    "fig07_reforecast_h24_four_basins_selection.csv",
)
pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
print("[SAVE] Selection audit:", selection_path, flush=True)
print("[SELECT] Representative gauges:", selected_gauges, flush=True)

h24_timeseries = pd.read_csv(H24_TIMESERIES_PATH)
h24_timeseries = h24_timeseries[
    h24_timeseries["node"].isin(selected_gauges)
].copy()
h24_timeseries["timestamp_utc"] = pd.to_datetime(
    h24_timeseries["timestamp_utc"],
    utc=True,
)

fig_examples, example_axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(12.0, 10.0),
    sharex=True,
)

panel_letters = ["(a)", "(b)", "(c)", "(d)"]
for panel_index, selected_gauge in enumerate(selected_gauges):
    ax = example_axes[panel_index]
    gauge_series = h24_timeseries[
        h24_timeseries["node"] == selected_gauge
    ].sort_values("timestamp_utc")
    gauge_metric_row = gauge_metrics[
        gauge_metrics["node"] == selected_gauge
    ].iloc[0]
    basin_name = BASIN_NAMES[selected_gauge[:2]]

    ax.plot(
        gauge_series["timestamp_utc"],
        gauge_series["observed_stage_m"],
        color=OBSERVED_COLOR,
        linewidth=1.7,
        label="Observed stage",
        zorder=4,
    )
    ax.plot(
        gauge_series["timestamp_utc"],
        gauge_series["predicted_stage_m"],
        color=FORECAST_COLOR,
        linewidth=1.25,
        alpha=0.92,
        label="h+24 reforecast",
        zorder=3,
    )
    ax.set_ylabel("Stage (m)")
    ax.set_title(
        panel_letters[panel_index]
        + " "
        + basin_name
        + ": "
        + selected_gauge
        + "   RMSE="
        + f"{float(gauge_metric_row['RMSE_h96_m']):.3f}"
        + " m, NSE="
        + f"{float(gauge_metric_row['NSE_h96']):.3f}"
        + ", r="
        + f"{float(gauge_metric_row['Pearson_r_h96']):.3f}",
        loc="left",
    )
    ax.grid(color=GRID_COLOR, alpha=0.20)
    if panel_index == 0:
        ax.legend(loc="upper left", ncol=2, framealpha=0.92)

example_axes[-1].set_xlabel("Verifying time (UTC), 2026")
example_axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
example_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig_examples.suptitle(
    "Representative fixed-lead h+24 retrospective reforecasts",
    fontsize=15.0,
    fontweight="bold",
)
fig_examples.text(
    0.5,
    0.012,
    "Within each drainage area, the selected gauge had correlation at or above the area median and RMSE closest to the area median.",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=8.5,
    color="0.30",
)
fig_examples.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])
save_figure(
    fig_examples,
    REFORECAST_EXAMPLES_PNG,
    REFORECAST_EXAMPLES_PDF,
)
plt.close(fig_examples)

print("[LOAD] Structure sensitivity metrics:", STRUCTURE_METRICS_PATH, flush=True)
structure_metrics = pd.read_csv(STRUCTURE_METRICS_PATH)
if len(structure_metrics) != 102:
    raise ValueError(
        "Expected 102 gauge-horizon structure rows, found "
        + str(len(structure_metrics))
    )

structure_fixed_series = pd.read_csv(STRUCTURE_FIXED_SERIES_PATH)
structure_fixed_series["verifying_time_utc"] = pd.to_datetime(
    structure_fixed_series["verifying_time_utc"],
    utc=True,
)
structure_fixed_metrics = pd.read_csv(STRUCTURE_FIXED_METRICS_PATH)

fig_structure = plt.figure(figsize=(13.0, 8.7))
structure_grid = fig_structure.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[1.0, 1.12],
    hspace=0.32,
    wspace=0.22,
)

for panel_index, horizon_key in enumerate(["h06", "h24"]):
    ax = fig_structure.add_subplot(structure_grid[0, panel_index])
    horizon_metrics = structure_metrics[
        structure_metrics["horizon_key"] == horizon_key
    ].copy()
    base_values = horizon_metrics["no_structure_rmse_m"].to_numpy(
        dtype=np.float64
    )
    structure_values = horizon_metrics["structure_rmse_m"].to_numpy(
        dtype=np.float64
    )
    lower_limit, upper_limit = shared_square_limits(
        base_values,
        structure_values,
        minimum_padding=0.01,
    )

    for basin_name, basin_group in horizon_metrics.groupby("basin"):
        ax.scatter(
            basin_group["no_structure_rmse_m"],
            basin_group["structure_rmse_m"],
            s=30,
            color=BASIN_COLORS[basin_name],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=basin_name,
            zorder=3,
        )

    hbsu_row = horizon_metrics[
        horizon_metrics["gauge"] == "HBSU0398"
    ]
    if len(hbsu_row) != 1:
        raise ValueError(
            "Expected exactly one HBSU0398 row for "
            + horizon_key
        )
    ax.scatter(
        hbsu_row["no_structure_rmse_m"],
        hbsu_row["structure_rmse_m"],
        s=115,
        marker="*",
        color="#F0E442",
        edgecolors="black",
        linewidths=0.7,
        label="HBSU0398",
        zorder=5,
    )

    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.72,
        zorder=2,
    )
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("No-structure ST-GNN RMSE (m)")
    ax.set_ylabel("Structure-aware ST-GNN RMSE (m)")
    ax.grid(color=GRID_COLOR, alpha=0.20)

    better_count = int(
        (
            horizon_metrics["rmse_improvement_m"]
            > 1.0e-12
        ).sum()
    )
    median_difference = float(
        (
            horizon_metrics["structure_rmse_m"]
            - horizon_metrics["no_structure_rmse_m"]
        ).median()
    )
    horizon_display = "0-6 h"
    if horizon_key == "h24":
        horizon_display = "0-24 h"

    ax.set_title(
        "(" + chr(ord("a") + panel_index) + ") "
        + horizon_display
        + " pooled RMSE"
    )
    ax.text(
        0.04,
        0.96,
        "Structure lower: "
        + str(better_count)
        + "/51\nMedian ΔRMSE: "
        + f"{median_difference:+.3f}"
        + " m",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=8.6,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.92,
        },
    )
    if panel_index == 1:
        ax.legend(
            loc="lower right",
            fontsize=7.5,
            framealpha=0.92,
        )

for panel_offset, lead_hours in enumerate([6, 24]):
    ax = fig_structure.add_subplot(structure_grid[1, panel_offset])
    fixed_series = structure_fixed_series[
        structure_fixed_series["lead_hours"] == lead_hours
    ].sort_values("verifying_time_utc")
    base_metric = structure_fixed_metrics[
        (structure_fixed_metrics["lead_hours"] == lead_hours)
        & (
            structure_fixed_metrics["model"]
            == "Base ST-GNN without structure inputs"
        )
    ]
    structure_metric = structure_fixed_metrics[
        (structure_fixed_metrics["lead_hours"] == lead_hours)
        & (
            structure_fixed_metrics["model"]
            == "Structure-aware ST-GNN"
        )
    ]
    if len(base_metric) != 1:
        raise ValueError(
            "Expected one base fixed-lead metric row for h+"
            + str(lead_hours)
        )
    if len(structure_metric) != 1:
        raise ValueError(
            "Expected one structure fixed-lead metric row for h+"
            + str(lead_hours)
        )

    ax.plot(
        fixed_series["verifying_time_utc"],
        fixed_series["observed_stage_m"],
        color=OBSERVED_COLOR,
        linewidth=2.0,
        label="Observed stage",
        zorder=4,
    )
    ax.plot(
        fixed_series["verifying_time_utc"],
        fixed_series["base_no_structure_stage_m"],
        color=BASE_COLOR,
        linewidth=1.15,
        alpha=0.92,
        label="No structure",
        zorder=3,
    )
    ax.plot(
        fixed_series["verifying_time_utc"],
        fixed_series["structure_aware_stage_m"],
        color=STRUCTURE_COLOR,
        linewidth=1.15,
        linestyle="--",
        alpha=0.95,
        label="Structure aware",
        zorder=3,
    )
    ax.set_title(
        "(" + chr(ord("c") + panel_offset) + ") HBSU0398 stitched h+"
        + str(lead_hours)
        + " series | RMSE "
        + f"{float(base_metric.iloc[0]['rmse_m']):.3f}"
        + " vs "
        + f"{float(structure_metric.iloc[0]['rmse_m']):.3f}"
        + " m"
    )
    ax.set_xlabel("Verifying time (UTC), 2026")
    ax.set_ylabel("Stage (m)")
    ax.grid(color=GRID_COLOR, alpha=0.20)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=25)
    if panel_offset == 0:
        ax.legend(loc="upper left", framealpha=0.92)

fig_structure.suptitle(
    "Operational-style sensitivity to restored gate, pump, and weir inputs",
    fontsize=15.0,
    fontweight="bold",
)
fig_structure.text(
    0.5,
    0.012,
    "Matched frozen P80 checkpoints, 410 hourly origins from 5-22 June 2026, exact archived QPF, and origin-specific safeguards. Scatter panels pool all forecast-window points; stitched panels use one fixed lead per origin. ΔRMSE = structure-aware minus no-structure.",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=8.0,
    color="0.30",
)
fig_structure.tight_layout(rect=[0.0, 0.045, 1.0, 0.95])
save_figure(
    fig_structure,
    STRUCTURE_FIGURE_PNG,
    STRUCTURE_FIGURE_PDF,
)
plt.close(fig_structure)

print("[COMPLETE] Revised paper figures were generated.", flush=True)
plt.show()
