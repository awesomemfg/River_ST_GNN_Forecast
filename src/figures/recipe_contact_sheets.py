"""Build six 4-by-4 golden contact sheets for each forecast dataset."""

import os
from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image


print("[CONFIG] Current working directory:", os.getcwd(), flush=True)

REVISION_DIRECTORY = Path(
    "project/"
    "manuscript/"
    "revision_2026_07_17"
)
CONTACT_SHEET_DIRECTORY = (
    REVISION_DIRECTORY
    / "contact_sheets_all_51"
)
REVISION_OUTPUT_DIRECTORY = Path(
    "project/"
    "manuscript/20260826_Final_V2/"
    "tmp/contact_sheets_font_revision"
)
FINAL_FIGURE_DIRECTORY = Path(
    "project/"
    "manuscript/20260826_Final_V2/"
    "output/figures"
)
REVISION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
FINAL_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

FIGURE9_DIRECTORY = (
    CONTACT_SHEET_DIRECTORY
    / "figure9_prospective_h24"
)
FIGURE9_TRACE_PATH = (
    FIGURE9_DIRECTORY
    / "figure9_prospective_h24_all51_traces.csv"
)
FIGURE9_METRIC_PATH = (
    FIGURE9_DIRECTORY
    / "figure9_prospective_h24_all51_metrics.csv"
)

FIGURE9_H6_DIRECTORY = (
    CONTACT_SHEET_DIRECTORY
    / "figure9_prospective_h6"
)
FIGURE9_H6_TRACE_PATH = Path(
    "project/"
    "manuscript/"
    "New_20260715_FullRevision/"
    "individual_station_diagnostics/"
    "prospective_h6_all_51_traces.csv"
)
FIGURE9_H6_METRIC_PATH = (
    FIGURE9_H6_DIRECTORY
    / "figure9_prospective_h6_all51_metrics.csv"
)

FIGURE6_DIRECTORY = (
    CONTACT_SHEET_DIRECTORY
    / "figure6_retrospective_h24"
)
FIGURE6_TRACE_PATH = Path(
    "project/"
    "Experiments/REFORECAST_GRAPH_RANKING_2026H1/"
    "results/evaluations/out_obs_pre2026/"
    "reforecast_2026H1_h96_timeseries.csv"
)
FIGURE6_METRIC_PATH = (
    FIGURE6_DIRECTORY
    / "figure6_retrospective_h24_all51_metrics.csv"
)

BASIN_NAMES = {
    "BM": "Bayou Manchac",
    "BC": "Bayou Conway",
    "HB": "Henderson Bayou",
    "MB": "Marvin Braud",
}
BASIN_SEQUENCE = [
    "BM",
    "BC",
    "HB",
    "MB",
]
BASIN_ROW_NUMBER = {
    "BM": 1,
    "BC": 2,
    "HB": 3,
    "MB": 4,
}
EXPECTED_BASIN_COUNTS = {
    "BM": 12,
    "BC": 8,
    "HB": 8,
    "MB": 23,
}

OBSERVED_COLOR = "#171717"
FORECAST_COLOR = "#D95F70"
GRID_COLOR = "#9A9A9A"

PAGE_ROWS = 4
PAGE_COLUMNS = 4
EXPECTED_GAUGE_COUNT = 51
EXPECTED_OUTPUT_WIDTH = 3510
EXPECTED_OUTPUT_HEIGHT = 2490

CRITERION_SPECS = [
    {
        "variant": "golden01",
        "description": "lowest RMSE",
        "score_columns": ["rank_rmse"],
    },
    {
        "variant": "golden02",
        "description": "highest NSE",
        "score_columns": ["rank_nse"],
    },
    {
        "variant": "golden03",
        "description": "highest Pearson r",
        "score_columns": ["rank_correlation"],
    },
    {
        "variant": "golden04",
        "description": "mean within-basin rank across RMSE and NSE",
        "score_columns": [
            "rank_rmse",
            "rank_nse",
        ],
    },
    {
        "variant": "golden05",
        "description": "mean within-basin rank across RMSE and Pearson r",
        "score_columns": [
            "rank_rmse",
            "rank_correlation",
        ],
    },
    {
        "variant": "golden06",
        "description": (
            "mean within-basin rank across RMSE, NSE, and Pearson r"
        ),
        "score_columns": [
            "rank_rmse",
            "rank_nse",
            "rank_correlation",
        ],
    },
]

DATASET_SPECS = [
    {
        "dataset": "figure9_prospective_h6",
        "trace_path": FIGURE9_H6_TRACE_PATH,
        "metric_path": FIGURE9_H6_METRIC_PATH,
        "output_directory": REVISION_OUTPUT_DIRECTORY / "figure9_prospective_h6",
        "expected_points_per_gauge": 371,
        "title": (
            "Delivered ST-GNN fixed-lead h+6 forecasts "
            "across in-parish gauges"
        ),
        "subtitle": (
            "Four selected gauges per drainage area | "
            "371 hourly issues from 7-22 June 2026"
        ),
        "forecast_label": "Delivered ST-GNN h+6 forecast",
        "date_tick_mode": "four_day",
        "source_format": "figure9_h6_raw_trace",
    },
    {
        "dataset": "figure9_prospective_h24",
        "trace_path": FIGURE9_TRACE_PATH,
        "metric_path": FIGURE9_METRIC_PATH,
        "output_directory": REVISION_OUTPUT_DIRECTORY / "figure9_prospective_h24",
        "expected_points_per_gauge": 371,
        "title": (
            "Delivered ST-GNN fixed-lead h+24 forecasts "
            "across in-parish gauges"
        ),
        "subtitle": (
            "Four selected gauges per drainage area | "
            "371 hourly issues from 7-22 June 2026"
        ),
        "forecast_label": "Delivered ST-GNN h+24 forecast",
        "date_tick_mode": "four_day",
        "source_format": "figure9_saved_trace",
    },
    {
        "dataset": "figure6_retrospective_h24",
        "trace_path": FIGURE6_TRACE_PATH,
        "metric_path": FIGURE6_METRIC_PATH,
        "output_directory": REVISION_OUTPUT_DIRECTORY / "figure6_retrospective_h24",
        "expected_points_per_gauge": 4344,
        "title": (
            "Fixed-lead h+24 retrospective ST-GNN reforecasts "
            "across in-parish gauges"
        ),
        "subtitle": (
            "Four selected gauges per drainage area | "
            "4,344 hourly origins from 1 January-30 June 2026 | "
            "perfect observed-rainfall forcing"
        ),
        "forecast_label": "ST-GNN h+24 reforecast",
        "date_tick_mode": "monthly",
        "source_format": "figure6_raw_trace",
    },
]

DATASET_SPECS = [
    dataset_spec
    for dataset_spec in DATASET_SPECS
    if dataset_spec["dataset"] in {
        "figure9_prospective_h24",
        "figure6_retrospective_h24",
    }
]
CRITERION_SPECS = [
    criterion_spec
    for criterion_spec in CRITERION_SPECS
    if criterion_spec["variant"] in {"golden04", "golden06"}
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "axes.edgecolor": "#3D3D3D",
        "axes.linewidth": 0.65,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "path.simplify": True,
        "path.simplify_threshold": 0.15,
    }
)


def calculate_metrics(observed_values, predicted_values):
    """Calculate fixed-lead metrics for one gauge."""
    observed_array = np.asarray(
        observed_values,
        dtype=np.float64,
    )
    predicted_array = np.asarray(
        predicted_values,
        dtype=np.float64,
    )
    finite_mask = (
        np.isfinite(observed_array)
        & np.isfinite(predicted_array)
    )
    observed_finite = observed_array[finite_mask]
    predicted_finite = predicted_array[finite_mask]
    pair_count = int(finite_mask.sum())

    if pair_count == 0:
        return {
            "n_pairs": 0,
            "rmse_m": np.nan,
            "nse": np.nan,
            "correlation": np.nan,
        }

    residuals = predicted_finite - observed_finite
    rmse_m = float(
        np.sqrt(
            np.mean(
                np.square(residuals)
            )
        )
    )
    observed_mean = float(
        np.mean(observed_finite)
    )
    nse_denominator = float(
        np.sum(
            np.square(
                observed_finite - observed_mean
            )
        )
    )

    if nse_denominator > 0.0:
        nse = float(
            1.0
            - np.sum(
                np.square(residuals)
            )
            / nse_denominator
        )
    else:
        nse = np.nan

    observed_standard_deviation = float(
        np.std(observed_finite)
    )
    predicted_standard_deviation = float(
        np.std(predicted_finite)
    )
    if (
        pair_count > 1
        and observed_standard_deviation > 0.0
        and predicted_standard_deviation > 0.0
    ):
        correlation = float(
            np.corrcoef(
                observed_finite,
                predicted_finite,
            )[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "n_pairs": pair_count,
        "rmse_m": rmse_m,
        "nse": nse,
        "correlation": correlation,
    }


def format_metric(value, decimal_places):
    """Format a finite metric for a panel title."""
    if not np.isfinite(value):
        return "NA"
    return format(
        float(value),
        "." + str(decimal_places) + "f",
    )


def calculate_y_limits(observed_values, predicted_values):
    """Return padded station-specific limits containing both traces."""
    combined_values = np.concatenate(
        [
            np.asarray(
                observed_values,
                dtype=np.float64,
            ),
            np.asarray(
                predicted_values,
                dtype=np.float64,
            ),
        ]
    )
    finite_values = combined_values[
        np.isfinite(combined_values)
    ]
    if len(finite_values) == 0:
        return -0.1, 0.1

    minimum_value = float(
        np.min(finite_values)
    )
    maximum_value = float(
        np.max(finite_values)
    )
    value_range = maximum_value - minimum_value
    if value_range <= 0.0:
        padding = max(
            abs(minimum_value) * 0.08,
            0.05,
        )
    else:
        padding = max(
            value_range * 0.08,
            0.025,
        )

    y_minimum = minimum_value - padding
    y_maximum = maximum_value + padding
    if not y_minimum < minimum_value:
        raise ValueError(
            "Calculated lower y limit does not contain the data."
        )
    if not y_maximum > maximum_value:
        raise ValueError(
            "Calculated upper y limit does not contain the data."
        )
    return y_minimum, y_maximum


def configure_time_axis(axis, date_tick_mode):
    """Configure the date ticks for one dataset."""
    if date_tick_mode == "four_day":
        axis.xaxis.set_major_locator(
            mdates.DayLocator(interval=4)
        )
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b")
        )
    elif date_tick_mode == "monthly":
        axis.xaxis.set_major_locator(
            mdates.MonthLocator(interval=1)
        )
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%b")
        )
    else:
        raise ValueError(
            "Unsupported date tick mode: "
            + str(date_tick_mode)
        )


def load_dataset(dataset_spec):
    """Load, standardize, and independently verify one h+24 dataset."""
    dataset = dataset_spec["dataset"]
    trace_path = Path(
        dataset_spec["trace_path"]
    )
    metric_path = Path(
        dataset_spec["metric_path"]
    )
    expected_points_per_gauge = int(
        dataset_spec["expected_points_per_gauge"]
    )

    print("[LOAD]", dataset, "trace:", trace_path, flush=True)
    print("[LOAD]", dataset, "metrics:", metric_path, flush=True)
    if not trace_path.is_file():
        raise FileNotFoundError(
            "Missing trace source: "
            + str(trace_path)
        )
    if not metric_path.is_file():
        raise FileNotFoundError(
            "Missing metric source: "
            + str(metric_path)
        )

    metric_frame = pd.read_csv(
        metric_path
    )
    if len(metric_frame) != EXPECTED_GAUGE_COUNT:
        raise ValueError(
            dataset
            + " metrics must contain exactly 51 rows."
        )
    if metric_frame["gauge"].nunique() != EXPECTED_GAUGE_COUNT:
        raise ValueError(
            dataset
            + " metrics must contain 51 unique gauges."
        )

    metric_frame["gauge"] = (
        metric_frame["gauge"]
        .astype(str)
    )
    metric_frame["basin"] = (
        metric_frame["basin"]
        .astype(str)
    )
    for metric_column in [
        "rmse_m",
        "nse",
        "correlation",
    ]:
        metric_frame[metric_column] = pd.to_numeric(
            metric_frame[metric_column],
            errors="coerce",
        )
        if metric_frame[metric_column].isna().any():
            raise ValueError(
                dataset
                + " has missing values in "
                + metric_column
            )

    actual_basin_counts = (
        metric_frame.groupby("basin")
        .size()
        .to_dict()
    )
    if actual_basin_counts != EXPECTED_BASIN_COUNTS:
        raise ValueError(
            dataset
            + " basin counts do not match the paper population: "
            + str(actual_basin_counts)
        )

    raw_trace_frame = pd.read_csv(
        trace_path
    )
    if dataset_spec["source_format"] == "figure9_saved_trace":
        required_trace_columns = [
            "gauge",
            "time_utc",
            "observed_stage_m",
            "predicted_stage_m",
        ]
        missing_trace_columns = [
            column
            for column in required_trace_columns
            if column not in raw_trace_frame.columns
        ]
        if missing_trace_columns:
            raise ValueError(
                dataset
                + " trace source is missing columns: "
                + str(missing_trace_columns)
            )
        trace_frame = raw_trace_frame[
            required_trace_columns
        ].copy()
    elif dataset_spec["source_format"] == "figure9_h6_raw_trace":
        required_trace_columns = [
            "gauge",
            "verifying_time_utc",
            "observed_stage_m",
            "forecast_stage_m",
        ]
        missing_trace_columns = [
            column
            for column in required_trace_columns
            if column not in raw_trace_frame.columns
        ]
        if missing_trace_columns:
            raise ValueError(
                dataset
                + " trace source is missing columns: "
                + str(missing_trace_columns)
            )
        trace_frame = raw_trace_frame[
            required_trace_columns
        ].copy()
        trace_frame = trace_frame.rename(
            columns={
                "verifying_time_utc": "time_utc",
                "forecast_stage_m": "predicted_stage_m",
            }
        )
    elif dataset_spec["source_format"] == "figure6_raw_trace":
        required_trace_columns = [
            "node",
            "timestamp_utc",
            "observed_stage_m",
            "predicted_stage_m",
        ]
        missing_trace_columns = [
            column
            for column in required_trace_columns
            if column not in raw_trace_frame.columns
        ]
        if missing_trace_columns:
            raise ValueError(
                dataset
                + " trace source is missing columns: "
                + str(missing_trace_columns)
            )
        trace_frame = raw_trace_frame[
            required_trace_columns
        ].copy()
        trace_frame = trace_frame.rename(
            columns={
                "node": "gauge",
                "timestamp_utc": "time_utc",
            }
        )
    else:
        raise ValueError(
            "Unsupported source format: "
            + str(dataset_spec["source_format"])
        )

    trace_frame["gauge"] = (
        trace_frame["gauge"]
        .astype(str)
    )
    paper_gauges = set(
        metric_frame["gauge"].tolist()
    )
    trace_frame = trace_frame[
        trace_frame["gauge"].isin(paper_gauges)
    ].copy()
    trace_frame["time_utc"] = pd.to_datetime(
        trace_frame["time_utc"],
        utc=True,
        errors="coerce",
    )
    trace_frame["observed_stage_m"] = pd.to_numeric(
        trace_frame["observed_stage_m"],
        errors="coerce",
    )
    trace_frame["predicted_stage_m"] = pd.to_numeric(
        trace_frame["predicted_stage_m"],
        errors="coerce",
    )

    if trace_frame["time_utc"].isna().any():
        raise ValueError(
            dataset
            + " contains invalid verifying times."
        )
    trace_gauges = set(
        trace_frame["gauge"].unique().tolist()
    )
    if trace_gauges != paper_gauges:
        raise ValueError(
            dataset
            + " trace gauges do not match metric gauges."
        )

    rows_per_gauge = (
        trace_frame.groupby("gauge")
        .size()
    )
    if int(rows_per_gauge.min()) != expected_points_per_gauge:
        raise ValueError(
            dataset
            + " has too few trace rows for at least one gauge."
        )
    if int(rows_per_gauge.max()) != expected_points_per_gauge:
        raise ValueError(
            dataset
            + " has too many trace rows for at least one gauge."
        )

    maximum_metric_difference = {
        "rmse_m": 0.0,
        "nse": 0.0,
        "correlation": 0.0,
    }
    for _, metric_row in metric_frame.iterrows():
        gauge = str(
            metric_row["gauge"]
        )
        gauge_frame = trace_frame[
            trace_frame["gauge"] == gauge
        ].sort_values("time_utc")
        recalculated_metrics = calculate_metrics(
            gauge_frame["observed_stage_m"],
            gauge_frame["predicted_stage_m"],
        )
        saved_pair_count = int(
            metric_row["n_pairs"]
        )
        saved_missing_observations = int(
            metric_row["missing_observations"]
        )
        saved_missing_predictions = int(
            metric_row["missing_predictions"]
        )
        recalculated_missing_observations = int(
            gauge_frame["observed_stage_m"]
            .isna()
            .sum()
        )
        recalculated_missing_predictions = int(
            gauge_frame["predicted_stage_m"]
            .isna()
            .sum()
        )
        if (
            int(recalculated_metrics["n_pairs"])
            != saved_pair_count
        ):
            raise ValueError(
                dataset
                + " recalculated pair count does not match metrics for "
                + gauge
            )
        if (
            recalculated_missing_observations
            != saved_missing_observations
        ):
            raise ValueError(
                dataset
                + " missing-observation count does not match metrics for "
                + gauge
            )
        if (
            recalculated_missing_predictions
            != saved_missing_predictions
        ):
            raise ValueError(
                dataset
                + " missing-prediction count does not match metrics for "
                + gauge
            )
        for metric_column in [
            "rmse_m",
            "nse",
            "correlation",
        ]:
            metric_difference = abs(
                float(metric_row[metric_column])
                - float(recalculated_metrics[metric_column])
            )
            maximum_metric_difference[metric_column] = max(
                maximum_metric_difference[metric_column],
                metric_difference,
            )
            if metric_difference > 1.0e-10:
                raise ValueError(
                    dataset
                    + " saved metric does not match traces for "
                    + gauge
                    + " and "
                    + metric_column
                    + ". Difference="
                    + str(metric_difference)
                )

    print(
        "[VERIFY]",
        dataset,
        "rows=",
        len(trace_frame),
        "gauges=",
        trace_frame["gauge"].nunique(),
        "points per gauge=",
        int(rows_per_gauge.min()),
        "finite pairs=",
        str(int(metric_frame["n_pairs"].min()))
        + "-"
        + str(int(metric_frame["n_pairs"].max())),
        "missing observations=",
        int(trace_frame["observed_stage_m"].isna().sum()),
        "missing predictions=",
        int(trace_frame["predicted_stage_m"].isna().sum()),
        "metric differences=",
        maximum_metric_difference,
        flush=True,
    )
    return trace_frame, metric_frame


def add_within_basin_ranks(metric_frame):
    """Add comparable ranks where rank 1 is always better."""
    ranked_frames = []
    for basin in BASIN_SEQUENCE:
        basin_frame = metric_frame[
            metric_frame["basin"] == basin
        ].copy()
        if len(basin_frame) < PAGE_COLUMNS:
            raise ValueError(
                "Basin "
                + basin
                + " does not contain at least four gauges."
            )

        basin_frame["rank_rmse"] = basin_frame[
            "rmse_m"
        ].rank(
            method="min",
            ascending=True,
        )
        basin_frame["rank_nse"] = basin_frame[
            "nse"
        ].rank(
            method="min",
            ascending=False,
        )
        basin_frame["rank_correlation"] = basin_frame[
            "correlation"
        ].rank(
            method="min",
            ascending=False,
        )
        ranked_frames.append(
            basin_frame
        )

    ranked_metric_frame = pd.concat(
        ranked_frames,
        ignore_index=True,
    )
    if len(ranked_metric_frame) != EXPECTED_GAUGE_COUNT:
        raise ValueError(
            "Ranked metric table does not contain 51 gauges."
        )
    return ranked_metric_frame


def select_golden_gauges(ranked_metric_frame, criterion_spec):
    """Select four gauges per basin for one ranking criterion."""
    selection_frames = []
    for basin in BASIN_SEQUENCE:
        basin_frame = ranked_metric_frame[
            ranked_metric_frame["basin"] == basin
        ].copy()
        score_columns = criterion_spec[
            "score_columns"
        ]
        basin_frame["criterion_score"] = (
            basin_frame[score_columns]
            .mean(axis=1)
        )
        basin_frame = basin_frame.sort_values(
            [
                "criterion_score",
                "rank_rmse",
                "rank_nse",
                "rank_correlation",
                "gauge",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
            ],
        ).reset_index(drop=True)
        selected_basin_frame = basin_frame.iloc[
            :PAGE_COLUMNS
        ].copy()
        selected_basin_frame["variant"] = criterion_spec[
            "variant"
        ]
        selected_basin_frame[
            "selection_description"
        ] = criterion_spec["description"]
        selected_basin_frame["basin_row"] = (
            BASIN_ROW_NUMBER[basin]
        )
        selected_basin_frame["column"] = np.arange(
            1,
            PAGE_COLUMNS + 1,
            dtype=int,
        )
        selection_frames.append(
            selected_basin_frame
        )

    selection_frame = pd.concat(
        selection_frames,
        ignore_index=True,
    )
    if len(selection_frame) != PAGE_ROWS * PAGE_COLUMNS:
        raise ValueError(
            criterion_spec["variant"]
            + " did not select exactly 16 gauges."
        )
    if selection_frame["gauge"].duplicated().any():
        raise ValueError(
            criterion_spec["variant"]
            + " selected a gauge more than once."
        )
    selected_basin_counts = (
        selection_frame.groupby("basin")
        .size()
        .to_dict()
    )
    expected_selected_counts = {
        basin: PAGE_COLUMNS
        for basin in BASIN_SEQUENCE
    }
    if selected_basin_counts != expected_selected_counts:
        raise ValueError(
            criterion_spec["variant"]
            + " does not contain four gauges per basin."
        )
    return selection_frame


def draw_golden_panel(
    axis,
    gauge_frame,
    metric_row,
    expected_points_per_gauge,
):
    """Draw one unclipped panel matching the established 4-by-4 style."""
    gauge = str(
        metric_row["gauge"]
    )
    basin_name = str(
        metric_row["basin_name"]
    )
    pair_count = int(
        metric_row["n_pairs"]
    )

    axis.plot(
        gauge_frame["time_utc"],
        gauge_frame["observed_stage_m"],
        color=OBSERVED_COLOR,
        linewidth=1.10,
        alpha=0.96,
        label="Observed stage",
        zorder=3,
    )
    axis.plot(
        gauge_frame["time_utc"],
        gauge_frame["predicted_stage_m"],
        color=FORECAST_COLOR,
        linewidth=1.00,
        alpha=0.90,
        label="ST-GNN h+24",
        zorder=2,
    )

    y_minimum, y_maximum = calculate_y_limits(
        gauge_frame["observed_stage_m"],
        gauge_frame["predicted_stage_m"],
    )
    axis.set_ylim(
        y_minimum,
        y_maximum,
    )
    actual_y_minimum, actual_y_maximum = axis.get_ylim()
    finite_values = pd.concat(
        [
            gauge_frame["observed_stage_m"],
            gauge_frame["predicted_stage_m"],
        ],
        ignore_index=True,
    ).dropna()
    if float(finite_values.min()) <= actual_y_minimum:
        raise ValueError(
            "Lower y limit clips gauge "
            + gauge
        )
    if float(finite_values.max()) >= actual_y_maximum:
        raise ValueError(
            "Upper y limit clips gauge "
            + gauge
        )

    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=3)
    )
    axis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.50,
        alpha=0.24,
        zorder=0,
    )
    axis.tick_params(
        axis="both",
        labelsize=9.8,
        length=2.0,
        width=0.5,
        pad=1.2,
    )

    rmse_text = format_metric(
        metric_row["rmse_m"],
        3,
    )
    nse_text = format_metric(
        metric_row["nse"],
        2,
    )
    correlation_text = format_metric(
        metric_row["correlation"],
        2,
    )
    if pair_count <= 0:
        raise ValueError(
            gauge
            + " has no finite metric pairs."
        )
    if pair_count > expected_points_per_gauge:
        raise ValueError(
            gauge
            + " has more finite pairs than trace rows: "
            + str(pair_count)
        )

    panel_title = (
        gauge
        + " | "
        + basin_name
        + "\nRMSE "
        + rmse_text
        + " m | NSE "
        + nse_text
        + " | r "
        + correlation_text
        + " | n "
        + str(pair_count)
    )
    axis.set_title(
        panel_title,
        fontsize=9.2,
        fontweight="bold",
        pad=1.8,
    )


def build_golden_page(
    dataset_spec,
    trace_frame,
    selection_frame,
    criterion_spec,
):
    """Render one 4-by-4 golden PNG."""
    dataset = dataset_spec["dataset"]
    output_directory = Path(
        dataset_spec["output_directory"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_directory
        / (
            dataset
            + "_contact_4x4_"
            + criterion_spec["variant"]
            + ".png"
        )
    )
    time_minimum = trace_frame[
        "time_utc"
    ].min()
    time_maximum = trace_frame[
        "time_utc"
    ].max()

    figure, axes = plt.subplots(
        nrows=PAGE_ROWS,
        ncols=PAGE_COLUMNS,
        figsize=(11.7, 8.3),
        sharex=True,
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.tick_params(
            labelbottom=False
        )

    for basin_row_index, basin in enumerate(
        BASIN_SEQUENCE
    ):
        basin_selection = selection_frame[
            selection_frame["basin"] == basin
        ].sort_values("column")
        if len(basin_selection) != PAGE_COLUMNS:
            raise ValueError(
                criterion_spec["variant"]
                + " does not have four selections for "
                + basin
            )

        for column_index in range(PAGE_COLUMNS):
            axis = axes[
                basin_row_index,
                column_index,
            ]
            metric_row = basin_selection.iloc[
                column_index
            ]
            gauge = str(
                metric_row["gauge"]
            )
            gauge_frame = trace_frame[
                trace_frame["gauge"] == gauge
            ].sort_values("time_utc")
            draw_golden_panel(
                axis,
                gauge_frame,
                metric_row,
                int(
                    dataset_spec[
                        "expected_points_per_gauge"
                    ]
                ),
            )
            axis.set_xlim(
                time_minimum,
                time_maximum,
            )
            configure_time_axis(
                axis,
                dataset_spec["date_tick_mode"],
            )

    for column_index in range(PAGE_COLUMNS):
        bottom_axis = axes[
            PAGE_ROWS - 1,
            column_index,
        ]
        bottom_axis.tick_params(
            labelbottom=True
        )
        plt.setp(
            bottom_axis.get_xticklabels(),
            rotation=28,
            horizontalalignment="right",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OBSERVED_COLOR,
            linewidth=1.5,
            label="Observed stage",
        ),
        Line2D(
            [0],
            [0],
            color=FORECAST_COLOR,
            linewidth=1.4,
            label=dataset_spec["forecast_label"],
        ),
    ]
    figure.suptitle(
        dataset_spec["title"],
        fontsize=16.0,
        fontweight="bold",
        y=0.994,
    )
    figure.text(
        0.5,
        0.968,
        (
            dataset_spec["subtitle"]
            + " | Selection: "
            + criterion_spec["description"]
        ),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10.2,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=2,
        frameon=False,
        fontsize=10.2,
    )
    figure.text(
        0.012,
        0.50,
        "Stage (m)",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=11.0,
    )
    figure.text(
        0.5,
        0.010,
        "Verifying time (UTC)",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9.5,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.073,
        right=0.996,
        top=0.845,
        bottom=0.070,
        hspace=0.68,
        wspace=0.18,
    )

    figure.canvas.draw()
    for basin_row_index, basin in enumerate(
        BASIN_SEQUENCE
    ):
        first_axis_position = axes[
            basin_row_index,
            0,
        ].get_position()
        row_center = (
            first_axis_position.y0
            + first_axis_position.y1
        ) / 2.0
        figure.text(
            0.047,
            row_center,
            (
                basin
                + "\n"
                + BASIN_NAMES[basin]
            ),
            rotation=90,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9.0,
            fontweight="bold",
        )

    figure.savefig(
        output_path,
        dpi=300,
    )
    plt.close(
        figure
    )

    if not output_path.is_file():
        raise FileNotFoundError(
            "Golden PNG was not created: "
            + str(output_path)
        )
    if output_path.stat().st_size <= 0:
        raise ValueError(
            "Golden PNG is empty: "
            + str(output_path)
        )
    with Image.open(output_path) as output_image:
        output_width, output_height = output_image.size
    if output_width != EXPECTED_OUTPUT_WIDTH:
        raise ValueError(
            "Golden PNG has unexpected width: "
            + str(output_path)
            + " width="
            + str(output_width)
        )
    if output_height != EXPECTED_OUTPUT_HEIGHT:
        raise ValueError(
            "Golden PNG has unexpected height: "
            + str(output_path)
            + " height="
            + str(output_height)
        )

    print(
        "[SAVE]",
        output_path,
        "bytes=",
        output_path.stat().st_size,
        "dimensions=",
        str(output_width)
        + "x"
        + str(output_height),
        flush=True,
    )
    return output_path


all_output_rows = []
all_selection_rows = []

for dataset_spec in DATASET_SPECS:
    dataset = dataset_spec["dataset"]
    trace_frame, metric_frame = load_dataset(
        dataset_spec
    )
    ranked_metric_frame = add_within_basin_ranks(
        metric_frame
    )

    dataset_output_paths = []
    dataset_selection_frames = []
    for criterion_spec in CRITERION_SPECS:
        selection_frame = select_golden_gauges(
            ranked_metric_frame,
            criterion_spec,
        )
        selection_frame["dataset"] = dataset
        selection_frame["selection_order"] = np.arange(
            1,
            len(selection_frame) + 1,
            dtype=int,
        )
        output_path = build_golden_page(
            dataset_spec,
            trace_frame,
            selection_frame,
            criterion_spec,
        )
        alias_path = None
        if dataset == "figure6_retrospective_h24" and criterion_spec["variant"] == "golden04":
            alias_path = FINAL_FIGURE_DIRECTORY / "fig08.png"
        elif dataset == "figure9_prospective_h24" and criterion_spec["variant"] == "golden06":
            alias_path = FINAL_FIGURE_DIRECTORY / "fig11.png"
        if alias_path is not None:
            shutil.copy2(output_path, alias_path)
            print("[SAVE] Manuscript alias:", alias_path, flush=True)
        selection_frame["output_png"] = str(
            output_path
        )
        dataset_selection_frames.append(
            selection_frame
        )
        all_selection_rows.extend(
            selection_frame.to_dict(
                orient="records"
            )
        )
        dataset_output_paths.append(
            output_path
        )
        all_output_rows.append(
            {
                "dataset": dataset,
                "variant": criterion_spec["variant"],
                "selection_description": (
                    criterion_spec["description"]
                ),
                "output_png": str(output_path),
                "selected_gauges": ",".join(
                    selection_frame[
                        "gauge"
                    ].tolist()
                ),
            }
        )

        print(
            "[SELECTION]",
            dataset,
            criterion_spec["variant"],
            flush=True,
        )
        for basin in BASIN_SEQUENCE:
            basin_gauges = selection_frame[
                selection_frame["basin"] == basin
            ].sort_values("column")["gauge"].tolist()
            print(
                "  ",
                basin,
                basin_gauges,
                flush=True,
            )

    if len(dataset_output_paths) != len(CRITERION_SPECS):
        raise ValueError(
            dataset
            + " did not produce six golden PNGs."
        )
    if len(set(dataset_output_paths)) != len(CRITERION_SPECS):
        raise ValueError(
            dataset
            + " produced duplicate output paths."
        )

    dataset_selection_manifest = pd.concat(
        dataset_selection_frames,
        ignore_index=True,
    )
    expected_manifest_rows = (
        len(CRITERION_SPECS)
        * PAGE_ROWS
        * PAGE_COLUMNS
    )
    if len(dataset_selection_manifest) != expected_manifest_rows:
        raise ValueError(
            dataset
            + " selection manifest does not contain 96 rows."
        )
    dataset_manifest_path = (
        Path(dataset_spec["output_directory"])
        / (
            dataset
            + "_golden_selection_manifest.csv"
        )
    )
    dataset_selection_manifest.to_csv(
        dataset_manifest_path,
        index=False,
    )
    print(
        "[SAVE]",
        dataset_manifest_path,
        "rows=",
        len(dataset_selection_manifest),
        flush=True,
    )

combined_output_index_path = (
    REVISION_OUTPUT_DIRECTORY
    / "golden_contact_sheet_output_index.csv"
)
combined_selection_manifest_path = (
    REVISION_OUTPUT_DIRECTORY
    / "golden_selection_manifest.csv"
)
pd.DataFrame(
    all_output_rows
).to_csv(
    combined_output_index_path,
    index=False,
)
pd.DataFrame(
    all_selection_rows
).to_csv(
    combined_selection_manifest_path,
    index=False,
)
print(
    "[SAVE]",
    combined_output_index_path,
    "rows=",
    len(all_output_rows),
    flush=True,
)
print(
    "[SAVE]",
    combined_selection_manifest_path,
    "rows=",
    len(all_selection_rows),
    flush=True,
)

expected_output_count = (
    len(DATASET_SPECS)
    * len(CRITERION_SPECS)
)
expected_selection_count = (
    len(DATASET_SPECS)
    * len(CRITERION_SPECS)
    * PAGE_ROWS
    * PAGE_COLUMNS
)
if len(all_output_rows) != expected_output_count:
    raise ValueError(
        "Expected exactly "
        + str(expected_output_count)
        + " golden PNG outputs."
    )
if len(all_selection_rows) != expected_selection_count:
    raise ValueError(
        "Expected exactly "
        + str(expected_selection_count)
        + " golden selection rows."
    )

print(
    "[COMPLETE] Generated and verified "
    + str(expected_output_count)
    + " golden PNGs across "
    + str(len(DATASET_SPECS))
    + " datasets.",
    flush=True,
)

plt.show()
