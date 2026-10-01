"""Build paper-grade all-station contact sheets for Figures 6 and 9."""

import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


print("[CONFIG] Current working directory:", os.getcwd(), flush=True)

revision_directory = "project/manuscript/revision_2026_07_17"
output_directory = os.path.join(
    revision_directory,
    "contact_sheets_all_51",
)

prospective_trace_path = "project/manuscript/New_20260715_FullRevision/individual_station_diagnostics/prospective_h6_all_51_traces.csv"
prospective_population_path = "project/manuscript/New_20260715_FullRevision/individual_station_diagnostics/prospective_h6_all_51_metrics.csv"
retrospective_trace_path = "project/Experiments/REFORECAST_GRAPH_RANKING_2026H1/results/evaluations/out_obs_pre2026/reforecast_2026H1_h96_timeseries.csv"

required_paths = [
    prospective_trace_path,
    prospective_population_path,
    retrospective_trace_path,
]
for required_path in required_paths:
    print("[VERIFY]", required_path, flush=True)
    if not os.path.isfile(required_path):
        raise FileNotFoundError("Required contact-sheet input is missing: " + required_path)

os.makedirs(output_directory, exist_ok=True)

basin_names = {
    "BC": "Bayou Conway",
    "BM": "Bayou Manchac",
    "HB": "Henderson Bayou",
    "MB": "Marvin Braud",
}
basin_order = {
    "BC": 0,
    "BM": 1,
    "HB": 2,
    "MB": 3,
}

observed_color = "#171717"
forecast_color = "#D95F70"
grid_color = "#9A9A9A"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
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
    observed_array = np.asarray(observed_values, dtype=np.float64)
    predicted_array = np.asarray(predicted_values, dtype=np.float64)
    finite_mask = np.isfinite(observed_array) & np.isfinite(predicted_array)
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
    rmse_m = float(np.sqrt(np.mean(np.square(residuals))))
    observed_mean = float(np.mean(observed_finite))
    nse_denominator = float(
        np.sum(np.square(observed_finite - observed_mean))
    )

    if nse_denominator > 0.0:
        nse = float(
            1.0
            - np.sum(np.square(residuals))
            / nse_denominator
        )
    else:
        nse = np.nan

    observed_standard_deviation = float(np.std(observed_finite))
    predicted_standard_deviation = float(np.std(predicted_finite))
    if (
        pair_count > 1
        and observed_standard_deviation > 0.0
        and predicted_standard_deviation > 0.0
    ):
        correlation = float(
            np.corrcoef(observed_finite, predicted_finite)[0, 1]
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
    """Format a finite metric or return NA."""
    if not np.isfinite(value):
        return "NA"
    return format(float(value), "." + str(decimal_places) + "f")


def calculate_y_limits(observed_values, predicted_values):
    """Return unclipped station-specific limits with modest padding."""
    combined_values = np.concatenate(
        [
            np.asarray(observed_values, dtype=np.float64),
            np.asarray(predicted_values, dtype=np.float64),
        ]
    )
    finite_values = combined_values[np.isfinite(combined_values)]
    if len(finite_values) == 0:
        return -0.1, 0.1

    minimum_value = float(np.min(finite_values))
    maximum_value = float(np.max(finite_values))
    value_range = maximum_value - minimum_value
    if value_range <= 0.0:
        padding = max(abs(minimum_value) * 0.08, 0.05)
    else:
        padding = max(value_range * 0.08, 0.025)
    return minimum_value - padding, maximum_value + padding


def draw_panel(
    axis,
    gauge_frame,
    metric_row,
    compact,
    expected_count,
    forecast_label,
):
    """Draw one station panel using the common paper style."""
    gauge = str(metric_row["gauge"])
    basin_name = str(metric_row["basin_name"])
    pair_count = int(metric_row["n_pairs"])

    axis.plot(
        gauge_frame["time_utc"],
        gauge_frame["observed_stage_m"],
        color=observed_color,
        linewidth=0.86 if compact else 1.10,
        alpha=0.96,
        label="Observed stage",
        zorder=3,
    )
    axis.plot(
        gauge_frame["time_utc"],
        gauge_frame["predicted_stage_m"],
        color=forecast_color,
        linewidth=0.78 if compact else 1.00,
        alpha=0.90,
        label=forecast_label,
        zorder=2,
    )

    y_minimum, y_maximum = calculate_y_limits(
        gauge_frame["observed_stage_m"],
        gauge_frame["predicted_stage_m"],
    )
    axis.set_ylim(y_minimum, y_maximum)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axis.grid(
        True,
        color=grid_color,
        linewidth=0.42 if compact else 0.50,
        alpha=0.24,
        zorder=0,
    )
    axis.tick_params(
        axis="both",
        labelsize=5.2 if compact else 7.3,
        length=2.0,
        width=0.5,
        pad=1.2,
    )

    rmse_text = format_metric(metric_row["rmse_m"], 3)
    nse_text = format_metric(metric_row["nse"], 2)
    correlation_text = format_metric(metric_row["correlation"], 2)
    count_suffix = ""
    if pair_count < expected_count:
        count_suffix = " | n=" + str(pair_count)

    if compact:
        panel_title = (
            gauge
            + " | RMSE "
            + rmse_text
            + " | NSE "
            + nse_text
            + " | r "
            + correlation_text
            + count_suffix
        )
        axis.set_title(
            panel_title,
            fontsize=6.15,
            fontweight="bold",
            pad=2.2,
        )
    else:
        panel_title = gauge + " | " + basin_name
        metric_title = (
            "RMSE "
            + rmse_text
            + " m | NSE "
            + nse_text
            + " | r "
            + correlation_text
            + " | n "
            + str(pair_count)
        )
        axis.set_title(
            panel_title + "\n" + metric_title,
            fontsize=7.6,
            fontweight="bold",
            pad=1.8,
        )


def configure_time_axis(axis, date_tick_mode):
    """Attach a fresh date locator and formatter to one shared axis group."""
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
            "Unsupported contact-sheet date tick mode: "
            + str(date_tick_mode)
        )


def build_contact_set(
    trace_frame,
    metric_frame,
    dataset_key,
    contact_title,
    contact_subtitle,
    forecast_label,
    date_tick_mode,
    expected_count,
):
    """Build one 51-panel sheet and basin-locked 4-by-4 page sheets."""
    print("[BUILD] Dataset:", dataset_key, flush=True)
    dataset_output_directory = os.path.join(output_directory, dataset_key)
    page_output_directory = os.path.join(
        dataset_output_directory,
        "pages_4x4",
    )
    os.makedirs(page_output_directory, exist_ok=True)

    ordered_metrics = metric_frame.sort_values(
        ["basin_order", "gauge"]
    ).reset_index(drop=True)
    ordered_gauges = ordered_metrics["gauge"].tolist()
    if len(ordered_gauges) != 51:
        raise ValueError(
            "Expected 51 ordered gauges for "
            + dataset_key
            + ", found "
            + str(len(ordered_gauges))
        )

    time_minimum = trace_frame["time_utc"].min()
    time_maximum = trace_frame["time_utc"].max()
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=observed_color,
            linewidth=1.5,
            label="Observed stage",
        ),
        Line2D(
            [0],
            [0],
            color=forecast_color,
            linewidth=1.4,
            label=forecast_label,
        ),
    ]

    contact_rows = 8
    contact_columns = 7
    contact_figure, contact_axes = plt.subplots(
        nrows=contact_rows,
        ncols=contact_columns,
        figsize=(18.4, 13.0),
        sharex=True,
    )
    contact_axes_flat = contact_axes.ravel()
    for axis in contact_axes_flat:
        axis.tick_params(labelbottom=False)

    for gauge_index, gauge in enumerate(ordered_gauges):
        axis = contact_axes_flat[gauge_index]
        gauge_frame = trace_frame[
            trace_frame["gauge"] == gauge
        ].sort_values("time_utc")
        metric_row = ordered_metrics.iloc[gauge_index]
        draw_panel(
            axis,
            gauge_frame,
            metric_row,
            True,
            expected_count,
            forecast_label,
        )
        axis.set_xlim(time_minimum, time_maximum)
        configure_time_axis(axis, date_tick_mode)

    for blank_index in range(len(ordered_gauges), len(contact_axes_flat)):
        contact_axes_flat[blank_index].set_visible(False)

    for column_index in range(contact_columns):
        active_indices = []
        for gauge_index in range(len(ordered_gauges)):
            if gauge_index % contact_columns == column_index:
                active_indices.append(gauge_index)
        if active_indices:
            bottom_index = max(active_indices)
            bottom_axis = contact_axes_flat[bottom_index]
            bottom_axis.tick_params(labelbottom=True)
            plt.setp(
                bottom_axis.get_xticklabels(),
                rotation=28,
                horizontalalignment="right",
            )

    contact_figure.suptitle(
        contact_title,
        fontsize=16.0,
        fontweight="bold",
        y=0.995,
    )
    contact_figure.text(
        0.5,
        0.975,
        contact_subtitle,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9.6,
    )
    contact_figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=2,
        frameon=False,
        fontsize=9.0,
    )
    contact_figure.text(
        0.012,
        0.50,
        "Stage (m)",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10.0,
    )
    contact_figure.text(
        0.5,
        0.008,
        "Verifying time (UTC)",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8.0,
        color="#444444",
    )
    contact_figure.subplots_adjust(
        left=0.040,
        right=0.996,
        top=0.925,
        bottom=0.045,
        hspace=0.50,
        wspace=0.20,
    )

    contact_png_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_contact_7x8.png",
    )
    contact_pdf_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_contact_7x8.pdf",
    )
    contact_figure.savefig(
        contact_png_path,
        dpi=300,
    )
    contact_figure.savefig(
        contact_pdf_path,
    )
    print("[SAVE]", contact_png_path, flush=True)
    print("[SAVE]", contact_pdf_path, flush=True)
    plt.close(contact_figure)

    basin_page_sequence = ["BM", "BC", "HB", "MB"]
    page_rows = 4
    page_columns = 4
    gauges_per_basin_per_page = 4
    basin_gauge_lists = {}
    for basin in basin_page_sequence:
        basin_gauges = ordered_metrics[
            ordered_metrics["basin"] == basin
        ].sort_values("gauge")["gauge"].tolist()
        basin_gauge_lists[basin] = basin_gauges
        print(
            "[PAGINATION]",
            dataset_key,
            basin,
            "gauges=",
            len(basin_gauges),
            flush=True,
        )

    page_count = max(
        int(
            math.ceil(
                len(basin_gauge_lists[basin])
                / gauges_per_basin_per_page
            )
        )
        for basin in basin_page_sequence
    )

    existing_page_prefix = dataset_key + "_contact_4x4_page_"
    for existing_page_name in os.listdir(page_output_directory):
        if existing_page_name.startswith(existing_page_prefix):
            existing_page_path = os.path.join(
                page_output_directory,
                existing_page_name,
            )
            os.remove(existing_page_path)
            print(
                "[REMOVE] Obsolete page output:",
                existing_page_path,
                flush=True,
            )

    multipage_pdf_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_contact_4x4_multipage.pdf",
    )
    page_index_rows = []
    with PdfPages(multipage_pdf_path) as multipage_pdf:
        for page_number in range(1, page_count + 1):
            page_height = 8.3
            page_title_fontsize = 14.0
            page_title_y = 0.994
            page_subtitle_fontsize = 8.8
            page_subtitle_y = 0.968
            page_legend_fontsize = 8.4
            page_legend_y = 0.946
            page_figure, page_axes = plt.subplots(
                nrows=page_rows,
                ncols=page_columns,
                figsize=(11.7, page_height),
                sharex=True,
                squeeze=False,
            )
            page_axes_flat = page_axes.ravel()
            for axis in page_axes_flat:
                axis.tick_params(labelbottom=False)

            visible_panel_positions = []
            page_gauge_count = 0
            page_slice_start = (
                (page_number - 1) * gauges_per_basin_per_page
            )
            page_slice_end = page_slice_start + gauges_per_basin_per_page

            for basin_row_index, basin in enumerate(basin_page_sequence):
                row_gauges = basin_gauge_lists[basin][
                    page_slice_start:page_slice_end
                ]
                print(
                    "[PAGE]",
                    dataset_key,
                    "page=",
                    page_number,
                    "row=",
                    basin_row_index + 1,
                    "basin=",
                    basin,
                    "gauges=",
                    row_gauges,
                    flush=True,
                )

                for basin_column_index in range(page_columns):
                    axis = page_axes[basin_row_index, basin_column_index]
                    page_panel_index = (
                        basin_row_index * page_columns
                        + basin_column_index
                    )
                    if basin_column_index >= len(row_gauges):
                        axis.set_visible(False)
                        continue

                    gauge = row_gauges[basin_column_index]
                    gauge_frame = trace_frame[
                        trace_frame["gauge"] == gauge
                    ].sort_values("time_utc")
                    matching_metrics = ordered_metrics[
                        ordered_metrics["gauge"] == gauge
                    ]
                    if len(matching_metrics) != 1:
                        raise ValueError(
                            "Expected one metric row for gauge "
                            + gauge
                            + ", found "
                            + str(len(matching_metrics))
                        )
                    metric_row = matching_metrics.iloc[0]
                    draw_panel(
                        axis,
                        gauge_frame,
                        metric_row,
                        False,
                        expected_count,
                        forecast_label,
                    )
                    axis.set_xlim(time_minimum, time_maximum)
                    configure_time_axis(axis, date_tick_mode)
                    visible_panel_positions.append(
                        (basin_row_index, basin_column_index)
                    )
                    page_gauge_count += 1
                    page_index_rows.append(
                        {
                            "dataset": dataset_key,
                            "page": page_number,
                            "row": basin_row_index + 1,
                            "column": basin_column_index + 1,
                            "panel": page_panel_index + 1,
                            "gauge": gauge,
                            "basin": metric_row["basin"],
                            "basin_name": metric_row["basin_name"],
                            "rmse_m": metric_row["rmse_m"],
                            "nse": metric_row["nse"],
                            "correlation": metric_row["correlation"],
                            "n_pairs": metric_row["n_pairs"],
                        }
                    )

            if page_gauge_count == 0:
                raise ValueError(
                    "Generated an empty 4-by-4 page for "
                    + dataset_key
                    + ", page "
                    + str(page_number)
                )

            for column_index in range(page_columns):
                active_rows = []
                for basin_row_index, basin_column_index in visible_panel_positions:
                    if basin_column_index == column_index:
                        active_rows.append(basin_row_index)
                if active_rows:
                    bottom_row_index = max(active_rows)
                    bottom_axis = page_axes[
                        bottom_row_index,
                        column_index,
                    ]
                    bottom_axis.tick_params(labelbottom=True)
                    plt.setp(
                        bottom_axis.get_xticklabels(),
                        rotation=28,
                        horizontalalignment="right",
                    )

            page_figure.suptitle(
                contact_title,
                fontsize=page_title_fontsize,
                fontweight="bold",
                y=page_title_y,
            )
            page_figure.text(
                0.5,
                page_subtitle_y,
                contact_subtitle,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=page_subtitle_fontsize,
            )
            page_figure.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, page_legend_y),
                ncol=2,
                frameon=False,
                fontsize=page_legend_fontsize,
            )
            page_figure.text(
                0.018,
                0.50,
                "Stage (m)",
                rotation=90,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=9.3,
            )
            page_figure.text(
                0.5,
                0.010,
                "Verifying time (UTC)",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=7.5,
                color="#444444",
            )
            page_top = 0.890
            page_bottom = 0.070
            page_figure.subplots_adjust(
                left=0.073,
                right=0.996,
                top=page_top,
                bottom=page_bottom,
                hspace=0.50,
                wspace=0.18,
            )

            page_figure.canvas.draw()
            for basin_row_index, basin in enumerate(basin_page_sequence):
                row_position = page_axes[
                    basin_row_index,
                    0,
                ].get_position()
                row_center = (
                    row_position.y0 + row_position.y1
                ) / 2.0
                page_figure.text(
                    0.030,
                    row_center,
                    basin + "\n" + basin_names[basin],
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=7.8,
                    fontweight="bold",
                )

            page_png_path = os.path.join(
                page_output_directory,
                dataset_key
                + "_contact_4x4_page_"
                + format(page_number, "02d")
                + ".png",
            )
            page_pdf_path = os.path.join(
                page_output_directory,
                dataset_key
                + "_contact_4x4_page_"
                + format(page_number, "02d")
                + ".pdf",
            )
            page_figure.savefig(
                page_png_path,
                dpi=300,
            )
            page_figure.savefig(
                page_pdf_path,
            )
            multipage_pdf.savefig(page_figure)
            print("[SAVE]", page_png_path, flush=True)
            print("[SAVE]", page_pdf_path, flush=True)
            plt.close(page_figure)

    print("[SAVE]", multipage_pdf_path, flush=True)

    metrics_csv_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_metrics.csv",
    )
    page_index_csv_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_4x4_page_index.csv",
    )
    ordered_metrics.to_csv(metrics_csv_path, index=False)
    pd.DataFrame(page_index_rows).to_csv(
        page_index_csv_path,
        index=False,
    )
    print("[SAVE]", metrics_csv_path, flush=True)
    print("[SAVE]", page_index_csv_path, flush=True)

    return {
        "dataset": dataset_key,
        "contact_png": contact_png_path,
        "contact_pdf": contact_pdf_path,
        "multipage_pdf": multipage_pdf_path,
        "metrics_csv": metrics_csv_path,
        "page_index_csv": page_index_csv_path,
        "page_count": page_count,
    }


def build_basin_row_contact_sheet(
    trace_frame,
    metric_frame,
    dataset_key,
    contact_title,
    contact_subtitle,
    forecast_label,
    date_tick_mode,
    expected_count,
):
    """Build one tightly packed sheet with one complete basin per row."""
    print("[BUILD] Basin-row sheet:", dataset_key, flush=True)
    dataset_output_directory = os.path.join(
        output_directory,
        dataset_key,
    )
    os.makedirs(dataset_output_directory, exist_ok=True)

    basin_row_sequence = ["BM", "BC", "HB", "MB"]
    expected_basin_counts = {
        "BM": 12,
        "BC": 8,
        "HB": 8,
        "MB": 23,
    }
    actual_basin_counts = (
        metric_frame.groupby("basin")["gauge"].nunique().to_dict()
    )
    if actual_basin_counts != expected_basin_counts:
        raise ValueError(
            "Unexpected basin counts for basin-row contact sheet: "
            + str(actual_basin_counts)
        )

    time_minimum = trace_frame["time_utc"].min()
    time_maximum = trace_frame["time_utc"].max()
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=observed_color,
            linewidth=1.5,
            label="Observed stage",
        ),
        Line2D(
            [0],
            [0],
            color=forecast_color,
            linewidth=1.4,
            label=forecast_label,
        ),
    ]

    basin_row_figure = plt.figure(
        figsize=(27.5, 10.4),
    )
    outer_grid = basin_row_figure.add_gridspec(
        nrows=4,
        ncols=1,
        left=0.034,
        right=0.998,
        top=0.905,
        bottom=0.060,
        hspace=0.62,
    )
    basin_row_axes = []
    basin_index_rows = []

    for basin_row_index, basin in enumerate(basin_row_sequence):
        basin_metrics = metric_frame[
            metric_frame["basin"] == basin
        ].sort_values("gauge").reset_index(drop=True)
        basin_gauges = basin_metrics["gauge"].tolist()
        basin_grid = outer_grid[basin_row_index].subgridspec(
            nrows=1,
            ncols=len(basin_gauges),
            wspace=0.20,
        )
        current_row_axes = []

        for basin_panel_index, gauge in enumerate(basin_gauges):
            axis = basin_row_figure.add_subplot(
                basin_grid[0, basin_panel_index]
            )
            current_row_axes.append(axis)
            gauge_frame = trace_frame[
                trace_frame["gauge"] == gauge
            ].sort_values("time_utc")
            metric_row = basin_metrics.iloc[basin_panel_index]

            draw_panel(
                axis,
                gauge_frame,
                metric_row,
                True,
                expected_count,
                forecast_label,
            )
            axis.set_xlim(time_minimum, time_maximum)
            configure_time_axis(axis, date_tick_mode)

            rmse_text = format_metric(metric_row["rmse_m"], 3)
            nse_text = format_metric(metric_row["nse"], 2)
            correlation_text = format_metric(
                metric_row["correlation"],
                2,
            )
            count_text = ""
            if int(metric_row["n_pairs"]) < expected_count:
                count_text = " | n" + str(int(metric_row["n_pairs"]))
            axis.set_title(
                gauge
                + "\nR "
                + rmse_text
                + " | N "
                + nse_text
                + " | r "
                + correlation_text
                + count_text,
                fontsize=4.85,
                fontweight="bold",
                pad=1.3,
            )
            axis.tick_params(
                axis="both",
                labelsize=4.4,
                length=1.5,
                width=0.45,
                pad=0.8,
            )

            if basin_row_index < len(basin_row_sequence) - 1:
                axis.tick_params(labelbottom=False)
            else:
                plt.setp(
                    axis.get_xticklabels(),
                    rotation=32,
                    horizontalalignment="right",
                )

            basin_index_rows.append(
                {
                    "dataset": dataset_key,
                    "basin_row": basin_row_index + 1,
                    "basin": basin,
                    "basin_name": basin_names[basin],
                    "panel": basin_panel_index + 1,
                    "gauge": gauge,
                    "rmse_m": metric_row["rmse_m"],
                    "nse": metric_row["nse"],
                    "correlation": metric_row["correlation"],
                    "n_pairs": metric_row["n_pairs"],
                }
            )

        basin_row_axes.append(current_row_axes)

    basin_row_figure.suptitle(
        contact_title,
        fontsize=15.5,
        fontweight="bold",
        y=0.995,
    )
    basin_row_figure.text(
        0.5,
        0.969,
        contact_subtitle,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9.2,
    )
    basin_row_figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )
    basin_row_figure.text(
        0.006,
        0.50,
        "Stage (m)",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=9.2,
    )
    basin_row_figure.text(
        0.5,
        0.008,
        "Verifying time (UTC)",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8.0,
        color="#444444",
    )

    basin_row_figure.canvas.draw()
    for basin_row_index, basin in enumerate(basin_row_sequence):
        first_axis_position = basin_row_axes[basin_row_index][0].get_position()
        row_center = (
            first_axis_position.y0 + first_axis_position.y1
        ) / 2.0
        basin_row_figure.text(
            0.019,
            row_center,
            basin
            + "\n"
            + basin_names[basin]
            + "\n(n="
            + str(expected_basin_counts[basin])
            + ")",
            rotation=90,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=6.8,
            fontweight="bold",
        )

    basin_row_png_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_basin_rows.png",
    )
    basin_row_pdf_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_basin_rows.pdf",
    )
    basin_row_index_path = os.path.join(
        dataset_output_directory,
        dataset_key + "_all51_basin_rows_index.csv",
    )
    basin_row_figure.savefig(
        basin_row_png_path,
        dpi=300,
    )
    basin_row_figure.savefig(
        basin_row_pdf_path,
    )
    pd.DataFrame(basin_index_rows).to_csv(
        basin_row_index_path,
        index=False,
    )
    print("[SAVE]", basin_row_png_path, flush=True)
    print("[SAVE]", basin_row_pdf_path, flush=True)
    print("[SAVE]", basin_row_index_path, flush=True)
    plt.close(basin_row_figure)

    return {
        "basin_rows_png": basin_row_png_path,
        "basin_rows_pdf": basin_row_pdf_path,
        "basin_rows_index_csv": basin_row_index_path,
    }


print("[LOAD] Prospective paper population:", prospective_population_path, flush=True)
prospective_population = pd.read_csv(prospective_population_path)
paper_gauges = sorted(
    prospective_population["gauge"].dropna().astype(str).unique().tolist()
)
if len(paper_gauges) != 51:
    raise ValueError(
        "Expected 51 paper gauges, found " + str(len(paper_gauges))
    )

print("[LOAD] Figure 9 prospective h+6 traces:", prospective_trace_path, flush=True)
prospective_traces = pd.read_csv(prospective_trace_path)
prospective_traces["time_utc"] = pd.to_datetime(
    prospective_traces["verifying_time_utc"],
    utc=True,
    errors="coerce",
)
prospective_traces["predicted_stage_m"] = pd.to_numeric(
    prospective_traces["forecast_stage_m"],
    errors="coerce",
)
prospective_traces["observed_stage_m"] = pd.to_numeric(
    prospective_traces["observed_stage_m"],
    errors="coerce",
)
prospective_traces = prospective_traces[
    prospective_traces["gauge"].isin(paper_gauges)
].copy()

print("[LOAD] Figure 6 retrospective h+24 traces:", retrospective_trace_path, flush=True)
retrospective_traces = pd.read_csv(retrospective_trace_path)
retrospective_traces = retrospective_traces[
    retrospective_traces["node"].isin(paper_gauges)
].copy()
retrospective_traces["gauge"] = retrospective_traces["node"].astype(str)
retrospective_traces["time_utc"] = pd.to_datetime(
    retrospective_traces["timestamp_utc"],
    utc=True,
    errors="coerce",
)
retrospective_traces["predicted_stage_m"] = pd.to_numeric(
    retrospective_traces["predicted_stage_m"],
    errors="coerce",
)
retrospective_traces["observed_stage_m"] = pd.to_numeric(
    retrospective_traces["observed_stage_m"],
    errors="coerce",
)
retrospective_traces["basin"] = retrospective_traces["gauge"].str[:2]
retrospective_traces["basin_name"] = retrospective_traces["basin"].map(
    basin_names
)

for dataset_name, trace_frame, expected_rows_per_gauge in [
    ("Figure 9 prospective h+6", prospective_traces, 371),
    ("Figure 6 retrospective h+24", retrospective_traces, 4344),
]:
    dataset_gauges = sorted(trace_frame["gauge"].unique().tolist())
    if dataset_gauges != paper_gauges:
        missing_gauges = sorted(set(paper_gauges) - set(dataset_gauges))
        extra_gauges = sorted(set(dataset_gauges) - set(paper_gauges))
        raise ValueError(
            dataset_name
            + " gauge mismatch. Missing="
            + str(missing_gauges)
            + ", extra="
            + str(extra_gauges)
        )
    rows_per_gauge = trace_frame.groupby("gauge").size()
    if not bool((rows_per_gauge == expected_rows_per_gauge).all()):
        raise ValueError(
            dataset_name
            + " does not contain exactly "
            + str(expected_rows_per_gauge)
            + " rows for every gauge"
        )
    print(
        "[VERIFY]",
        dataset_name,
        "contains",
        len(trace_frame),
        "rows across 51 gauges",
        flush=True,
    )

metric_frames = {}
for dataset_key, trace_frame in [
    ("figure9_prospective_h6", prospective_traces),
    ("figure6_retrospective_h24", retrospective_traces),
]:
    metric_rows = []
    for gauge in paper_gauges:
        gauge_frame = trace_frame[
            trace_frame["gauge"] == gauge
        ].sort_values("time_utc")
        gauge_metrics = calculate_metrics(
            gauge_frame["observed_stage_m"],
            gauge_frame["predicted_stage_m"],
        )
        basin = gauge[:2]
        metric_rows.append(
            {
                "dataset": dataset_key,
                "gauge": gauge,
                "basin": basin,
                "basin_name": basin_names[basin],
                "basin_order": basin_order[basin],
                "row_count": len(gauge_frame),
                "n_pairs": gauge_metrics["n_pairs"],
                "missing_observations": int(
                    gauge_frame["observed_stage_m"].isna().sum()
                ),
                "missing_predictions": int(
                    gauge_frame["predicted_stage_m"].isna().sum()
                ),
                "rmse_m": gauge_metrics["rmse_m"],
                "nse": gauge_metrics["nse"],
                "correlation": gauge_metrics["correlation"],
                "verifying_time_start_utc": gauge_frame["time_utc"].min(),
                "verifying_time_end_utc": gauge_frame["time_utc"].max(),
            }
        )
    metric_frames[dataset_key] = pd.DataFrame(metric_rows)
    print(
        "[METRICS]",
        dataset_key,
        "rows=",
        len(metric_frames[dataset_key]),
        "finite RMSE=",
        int(metric_frames[dataset_key]["rmse_m"].notna().sum()),
        flush=True,
    )

output_rows = []
prospective_output = build_contact_set(
    prospective_traces,
    metric_frames["figure9_prospective_h6"],
    "figure9_prospective_h6",
    "Delivered ST-GNN fixed-lead h+6 forecasts across in-parish gauges",
    "All 51 in-parish gauges | 371 hourly issues from 7-22 June 2026",
    "Delivered ST-GNN h+6 forecast",
    "four_day",
    371,
)
prospective_basin_rows = build_basin_row_contact_sheet(
    prospective_traces,
    metric_frames["figure9_prospective_h6"],
    "figure9_prospective_h6",
    "Delivered ST-GNN fixed-lead h+6 forecasts across in-parish gauges",
    "All 51 in-parish gauges | 371 hourly issues from 7-22 June 2026",
    "Delivered ST-GNN h+6 forecast",
    "four_day",
    371,
)
prospective_output.update(prospective_basin_rows)
output_rows.append(prospective_output)

retrospective_output = build_contact_set(
    retrospective_traces,
    metric_frames["figure6_retrospective_h24"],
    "figure6_retrospective_h24",
    "Fixed-lead h+24 retrospective ST-GNN reforecasts across in-parish gauges",
    "All 51 in-parish gauges | 4,344 hourly origins from 1 January-30 June 2026 | perfect observed-rainfall forcing",
    "ST-GNN h+24 reforecast",
    "monthly",
    4344,
)
retrospective_basin_rows = build_basin_row_contact_sheet(
    retrospective_traces,
    metric_frames["figure6_retrospective_h24"],
    "figure6_retrospective_h24",
    "Fixed-lead h+24 retrospective ST-GNN reforecasts across in-parish gauges",
    "All 51 in-parish gauges | 4,344 hourly origins from 1 January-30 June 2026 | perfect observed-rainfall forcing",
    "ST-GNN h+24 reforecast",
    "monthly",
    4344,
)
retrospective_output.update(retrospective_basin_rows)
output_rows.append(retrospective_output)

output_index_path = os.path.join(
    output_directory,
    "contact_sheet_output_index.csv",
)
pd.DataFrame(output_rows).to_csv(output_index_path, index=False)
print("[SAVE]", output_index_path, flush=True)
print(
    "[COMPLETE] All requested all-station, 4-by-4, and basin-row contact sheets were generated.",
    flush=True,
)

plt.show()
