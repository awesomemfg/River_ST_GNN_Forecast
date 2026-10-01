"""Combine prospective fixed-lead distributions with a spatial h+24 RMSE map.

Figure contract
---------------
Analytical question: How did delivered fixed-lead performance change from h+6
to h+24, and where was long-lead error concentrated across Ascension Parish?

Takeaway: Error and station heterogeneity increase at h+24, with the spatial
map showing that the degradation is not uniform across the 51 in-parish gauges.

Chart family and variant: Three paired box-and-point distributions stacked
beside one large spatial map.

Data sufficiency: 371 hourly delivered forecast cycles at each of 51 gauges,
with one fixed-lead observation-forecast pair per cycle at h+6 and h+24.

Renderer and output: Static Matplotlib with the manuscript basemap, saved as
PNG and PDF.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


print("[CONFIG] Current working directory:", os.getcwd())

PROJECT_DIRECTORY = "project/manuscript"
SCRIPT_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "figure_library")
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from _mapbase import add_cities, load_xy, make_basemap


METRICS_PATH = os.path.join(
    PROJECT_DIRECTORY,
    "figure_library_outputs",
    "rolling_but_live",
    "live_fixed_lead_metrics_inparish51.csv",
)
OUTPUT_DIRECTORY = os.path.join(
    PROJECT_DIRECTORY,
    "20260826_Final_V2",
    "output",
    "figures",
)
OUTPUT_PNG = os.path.join(OUTPUT_DIRECTORY, "fig09.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIRECTORY, "fig09.pdf")

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("Required input is missing: " + METRICS_PATH)

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

metrics_table = pd.read_csv(METRICS_PATH)
if metrics_table["gauge"].nunique() != 51:
    raise ValueError(
        "Expected 51 in-parish gauges, found "
        + str(metrics_table["gauge"].nunique())
    )

lead_values = sorted(metrics_table["lead_hours"].dropna().unique().tolist())
if lead_values != [6, 24]:
    raise ValueError("Expected lead times [6, 24], found " + str(lead_values))

coordinate_lookup = load_xy()
missing_coordinates = sorted(
    set(metrics_table["gauge"].astype(str).unique()) - set(coordinate_lookup)
)
if missing_coordinates:
    raise ValueError(
        "Coordinates are missing for gauges: " + ", ".join(missing_coordinates)
    )

metrics_table["E"] = metrics_table["gauge"].map(
    lambda gauge: coordinate_lookup[str(gauge)][0]
)
metrics_table["N"] = metrics_table["gauge"].map(
    lambda gauge: coordinate_lookup[str(gauge)][1]
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 12.5,
        "axes.titlesize": 14.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)

H6_COLOR = "#2b6cb0"
H24_COLOR = "#d97706"

figure = plt.figure(figsize=(12.0, 7.4), facecolor="white")
outer_grid = figure.add_gridspec(
    1,
    2,
    width_ratios=[0.88, 1.62],
    wspace=0.20,
    left=0.070,
    right=0.955,
    top=0.845,
    bottom=0.095,
)
distribution_grid = outer_grid[0, 0].subgridspec(
    3,
    1,
    hspace=0.48,
)
distribution_axes = [
    figure.add_subplot(distribution_grid[0, 0]),
    figure.add_subplot(distribution_grid[1, 0]),
    figure.add_subplot(distribution_grid[2, 0]),
]
map_axis = figure.add_subplot(outer_grid[0, 1])

metric_configurations = [
    {
        "column": "rmse_m",
        "title": "(a) Per-gauge RMSE (m)",
        "zero_line": None,
        "ylim": None,
    },
    {
        "column": "correlation",
        "title": "(b) Pearson correlation",
        "zero_line": None,
        "ylim": None,
    },
    {
        "column": "nse",
        "title": "(c) Nash-Sutcliffe efficiency",
        "zero_line": 0.0,
        "ylim": (-4.5, 1.1),
    },
]

for axis, metric_configuration in zip(
    distribution_axes,
    metric_configurations,
):
    h6_values = metrics_table[
        metrics_table["lead_hours"].eq(6)
    ][metric_configuration["column"]].dropna().to_numpy(dtype=float)
    h24_values = metrics_table[
        metrics_table["lead_hours"].eq(24)
    ][metric_configuration["column"]].dropna().to_numpy(dtype=float)

    boxplot = axis.boxplot(
        [h6_values, h24_values],
        positions=[1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
        zorder=3,
    )
    boxplot["boxes"][0].set_facecolor(H6_COLOR)
    boxplot["boxes"][0].set_alpha(0.35)
    boxplot["boxes"][0].set_edgecolor(H6_COLOR)
    boxplot["boxes"][1].set_facecolor(H24_COLOR)
    boxplot["boxes"][1].set_alpha(0.35)
    boxplot["boxes"][1].set_edgecolor(H24_COLOR)

    deterministic_offsets = np.linspace(-0.11, 0.11, 51)
    axis.scatter(
        np.full(51, 1.0) + deterministic_offsets,
        h6_values,
        s=18,
        color=H6_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
        zorder=2,
    )
    axis.scatter(
        np.full(51, 2.0) + deterministic_offsets,
        h24_values,
        s=18,
        color=H24_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
        zorder=2,
    )

    if metric_configuration["zero_line"] is not None:
        axis.axhline(
            metric_configuration["zero_line"],
            color="#555555",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

    if metric_configuration["ylim"] is not None:
        axis.set_ylim(metric_configuration["ylim"])

    axis.set_xticks([1, 2])
    axis.set_xticklabels(["h+6", "h+24"])
    axis.set_title(metric_configuration["title"], loc="left", pad=4.0)
    axis.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.55)

h24_map_table = metrics_table[
    metrics_table["lead_hours"].eq(24)
].copy()
make_basemap(
    None,
    h24_map_table["gauge"].tolist(),
    ax=map_axis,
    river_alpha=0.30,
    river_lw=0.38,
    axis_labels=True,
)

map_upper_limit = float(np.percentile(h24_map_table["rmse_m"], 95))
map_scatter = map_axis.scatter(
    h24_map_table["E"],
    h24_map_table["N"],
    c=h24_map_table["rmse_m"],
    cmap="YlOrBr",
    vmin=0.0,
    vmax=map_upper_limit,
    s=88,
    edgecolor="black",
    linewidth=0.65,
    zorder=8,
)
add_cities(map_axis, ["Donaldsonville", "Gonzales"], dot=32, fs=10.5, dy=-1.0)


def format_million_feet(value, _position):
    return f"{value / 1_000_000.0:.2f}"


map_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
map_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
map_axis.xaxis.set_major_formatter(FuncFormatter(format_million_feet))
map_axis.yaxis.set_major_formatter(FuncFormatter(format_million_feet))
map_axis.set_xlabel("Easting (million US ft)", fontsize=12.5)
map_axis.set_ylabel("Northing (million US ft)", fontsize=12.5)
map_axis.tick_params(labelsize=11.5)
map_axis.set_title("(d) Spatial distribution of h+24 RMSE", loc="left", pad=5.0)
colorbar = figure.colorbar(
    map_scatter,
    ax=map_axis,
    orientation="vertical",
    shrink=0.72,
    pad=0.020,
)
colorbar.set_label("h+24 RMSE (m; 95th-percentile cap)", fontsize=11.5)
colorbar.ax.tick_params(labelsize=10.5)
map_axis.text(
    0.985,
    0.985,
    "Median = "
    + f"{h24_map_table['rmse_m'].median():.3f}"
    + " m\nN = 51 gauges",
    transform=map_axis.transAxes,
    ha="right",
    va="top",
    fontsize=10.5,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "#888888",
        "alpha": 0.92,
    },
    zorder=20,
)

figure.suptitle(
    "Delivered operational ST-GNN performance\n"
    "371 hourly issues, 7-22 June 2026, 51 in-parish gauges",
    fontsize=18.0,
    fontweight="bold",
    y=0.985,
)
figure.savefig(OUTPUT_PNG)
figure.savefig(OUTPUT_PDF)
print("[SAVED]", OUTPUT_PNG)
print("[SAVED]", OUTPUT_PDF)

plt.show()
