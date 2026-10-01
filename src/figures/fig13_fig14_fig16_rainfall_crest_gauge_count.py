# Revision 14 copy made by archive/figure_script_generation/make_figure_scripts.py on 2026-09-24.
# Original: archive/figure_script_generation/source_rainfall_crest_gauge_count.py
"""Build Revision 9 figures that compare all times with the event-only subset.

The event-only subset contains the 927 forecast origins at which at least one parish pump is already
running. The script reads only January-August source tables and writes three manuscript figures:

1. Rainfall-forcing skill at h+24 for all times and event-only.
2. Crest error for all times and event-only.
3. Gauge-count sensitivity for all times and event-only.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REVISION_DIRECTORY = (
    "project/"
    "manuscript/20260924_Revision_14"
)
EXPERIMENT_DIRECTORY = (
    "project/Experiments/"
    "EXTEND_2026_AUG31_ANCHORFIX_20260924"
)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_DIRECTORY, "outputs")
FIGURE_DIRECTORY = os.path.join(REVISION_DIRECTORY, "figures")
NATIVE_RECIPE = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/scripts/recipe_rainfall_skill_and_crest.py"
)
RAINFALL_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "event_fig13_per_gauge_h24_jan_aug.csv",
)
CREST_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "event_fig14_crest_by_forcing_jan_aug.csv",
)
NETWORK_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "event_network_levels_h24.csv",
)

ORIGIN_SCOPES = [
    ("all", "All times"),
    ("pump_at_issue", "Event-only"),
]
FORCING_CASES = [
    ("Observed rain", "Observed rainfall"),
    ("Ensemble mean, 6", "Ensemble mean of six weather models"),
    ("HRRR, issue time", "HRRR at issue time"),
]
METRIC_COLUMNS = {
    "rmse": "RMSE_m",
    "nse": "NSE",
    "r": "Pearson_r",
    "kge": "KGE",
}
RISE_BANDS = [
    "0.15-0.30 m",
    "0.30-0.61 m",
    "above 0.61 m",
]
NETWORK_SIZES = [68, 51, 40, 30, 20, 10]

COLOR_IN_PARISH = "#4f91c7"
COLOR_CONTROL = "#b2182b"
COLOR_PARENT = "#737f8c"
COLOR_NSE = "#1f77b4"
COLOR_CORRELATION = "#1b7837"
COLOR_KGE = "#d95f02"
COLOR_SEED_BAND = "#f7d794"

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
print("[CONFIG] current working directory:", os.getcwd(), flush=True)

for required_path in [RAINFALL_PATH, CREST_PATH, NETWORK_PATH, NATIVE_RECIPE]:
    if not os.path.isfile(required_path):
        raise FileNotFoundError("Missing required figure input: " + required_path)


def load_native_recipe():
    """Load the original paper drawing functions without executing its main block."""
    with open(NATIVE_RECIPE, "r", encoding="utf-8") as handle:
        source = handle.read()
    marker = "\ndef main("
    if marker not in source:
        raise Exception("The native recipe no longer defines main(): " + NATIVE_RECIPE)
    namespace = {"__name__": "native_recipe"}
    exec(compile(source.split(marker)[0], NATIVE_RECIPE, "exec"), namespace)
    required_names = [
        "panel",
        "bars",
        "PANELS",
        "FIG12_RC",
        "FIG13_RC",
        "COL_OBS",
        "COL_QPF",
        "COL_L2",
        "STRATA_LABELS",
    ]
    for required_name in required_names:
        if required_name not in namespace:
            raise Exception("Native recipe does not define " + required_name)
    return namespace


native = load_native_recipe()


def rainfall_panel(
    axis,
    groups_by_case,
    cases,
    xlabels,
    metric_key,
    title,
    ylabel,
    lower_is_better,
):
    """Draw the native boxplot recipe with a modest gap between the two origin groups."""
    random_generator = np.random.default_rng(0)
    group_positions = np.asarray([0.0, 1.15], dtype=float)
    series = {name: groups_by_case[name] for name, _color, _offset in cases}
    pooled = np.concatenate(
        [values for groups in series.values() for values in groups if len(values)]
    )
    first_quartile, third_quartile = np.percentile(pooled, [25, 75])
    interquartile_range = third_quartile - first_quartile
    lower_limit = max(
        float(pooled.min()),
        float(first_quartile - 3.0 * interquartile_range),
    )
    upper_limit = min(
        float(pooled.max()),
        float(third_quartile + 3.0 * interquartile_range),
    )
    if not np.isfinite(lower_limit) or not np.isfinite(upper_limit) or upper_limit <= lower_limit:
        lower_limit = float(pooled.min())
        upper_limit = float(pooled.max())
    number_below = int((pooled < lower_limit).sum())
    number_above = int((pooled > upper_limit).sum())
    span = upper_limit - lower_limit
    axis.set_ylim(lower_limit - 0.06 * span, upper_limit + 0.17 * span)

    box_width = 0.32
    for case_name, case_color, case_offset in cases:
        groups = series[case_name]
        positions = group_positions + case_offset
        boxplot = axis.boxplot(
            groups,
            positions=positions,
            widths=box_width,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": case_color, "linewidth": 0.9},
            capprops={"color": case_color, "linewidth": 0.9},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(case_color)
            patch.set_alpha(0.30)
            patch.set_edgecolor(case_color)
            patch.set_linewidth(1.0)
        for group_index, values in enumerate(groups):
            if not len(values):
                continue
            jitter = random_generator.uniform(-0.085, 0.085, size=len(values))
            inside_axis = (values >= lower_limit) & (values <= upper_limit)
            axis.plot(
                positions[group_index] + jitter[inside_axis],
                values[inside_axis],
                "o",
                markersize=3.6,
                color=case_color,
                alpha=0.55,
                markeredgewidth=0,
                zorder=3,
            )
            clipped_groups = [
                (values < lower_limit, "v", lower_limit - 0.03 * span),
                (values > upper_limit, "^", upper_limit + 0.03 * span),
            ]
            for mask, marker, edge in clipped_groups:
                if mask.any():
                    axis.plot(
                        positions[group_index] + jitter[mask],
                        np.full(int(mask.sum()), edge),
                        marker,
                        markersize=6.0,
                        color=case_color,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        clip_on=False,
                        zorder=4,
                    )

    for case_name, case_color, case_offset in cases:
        for group_index, values in enumerate(series[case_name]):
            if not len(values):
                continue
            median = float(np.median(values))
            number_format = "{:.3f}" if metric_key == "rmse" else "{:.2f}"
            axis.annotate(
                number_format.format(median),
                (group_positions[group_index] + case_offset, median + 0.025 * span),
                ha="center",
                va="bottom",
                fontsize=11.5,
                color=case_color,
                fontweight="bold",
                zorder=6,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                    "boxstyle": "square,pad=0.15",
                },
            )

    if number_below or number_above:
        annotation_parts = []
        if number_below:
            noun = "point" if number_below == 1 else "points"
            annotation_parts.append(f"{number_below} {noun} are below the axis")
        if number_above:
            noun = "point" if number_above == 1 else "points"
            annotation_parts.append(f"{number_above} {noun} are above the axis")
        axis.annotate(
            " and ".join(annotation_parts) + ",\ndrawn as triangles at the edge",
            (0.015, 0.985),
            xycoords="axes fraction",
            fontsize=10.5,
            color="0.35",
            ha="left",
            va="top",
        )

    axis.set_xticks(group_positions)
    axis.set_xticklabels(xlabels, fontsize=12.5)
    axis.set_xlim(-0.55, 1.70)
    axis.set_ylabel(ylabel, fontsize=14.0)
    axis.set_title(title, loc="left")
    axis.grid(True, axis="y", alpha=0.3, linewidth=0.4, color=native["GRID"])
    axis.set_axisbelow(True)
    if not lower_is_better and lower_limit < 0 < upper_limit:
        axis.axhline(0, color="0.35", linewidth=0.7, linestyle=":", zorder=1)

# Rainfall-forcing skill, with only all times and event-only.
rainfall = pd.read_csv(RAINFALL_PATH)
rainfall_seed_mean = rainfall.groupby(
    ["forcing", "origin_set", "gauge"],
    as_index=False,
)[["RMSE_m", "Pearson_r", "NSE", "KGE"]].mean()

case_colors = [native["COL_OBS"], native["COL_QPF"], native["COL_L2"]]
# The native recipe uses 0.32-wide boxes. These offsets keep the three
# forcing cases visually separate instead of allowing the translucent boxes
# to overlap.
case_offsets = [-0.34, 0.0, 0.34]
cases = []
for forcing_case, color, offset in zip(FORCING_CASES, case_colors, case_offsets):
    cases.append((forcing_case[1], color, offset))

plt.rcParams.update(native["FIG12_RC"])
rainfall_figure, rainfall_axes = plt.subplots(2, 2, figsize=(15.2, 10.4))
for axis, panel_specification in zip(rainfall_axes.ravel(), native["PANELS"]):
    metric_key = panel_specification[0]
    title = panel_specification[1]
    ylabel = panel_specification[2]
    lower_is_better = panel_specification[3]
    metric_column = METRIC_COLUMNS[metric_key]
    groups_by_case = {}
    for forcing_key, forcing_label in FORCING_CASES:
        groups = []
        for origin_scope, _scope_label in ORIGIN_SCOPES:
            block = rainfall_seed_mean[
                (rainfall_seed_mean["forcing"] == forcing_key)
                & (rainfall_seed_mean["origin_set"] == origin_scope)
            ]
            values = block[metric_column].to_numpy(float)
            groups.append(values[np.isfinite(values)])
        groups_by_case[forcing_label] = groups
    rainfall_panel(
        axis,
        groups_by_case,
        cases,
        [scope_label for _scope_key, scope_label in ORIGIN_SCOPES],
        metric_key,
        title,
        ylabel,
        lower_is_better,
    )

rainfall_handles = []
for forcing_case, color in zip(FORCING_CASES, case_colors):
    rainfall_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=7.0,
            alpha=0.45,
            label=forcing_case[1],
        )
    )
rainfall_figure.legend(
    handles=rainfall_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.002),
)
rainfall_figure.suptitle(
    "ST-GNN skill at h+24: all times and event-only",
    fontsize=19.0,
    fontweight="bold",
    y=0.995,
)
rainfall_figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.972))
rainfall_png = os.path.join(
    FIGURE_DIRECTORY,
    "fig16_rainfall_skill_all_event.png",
)
rainfall_pdf = os.path.join(
    FIGURE_DIRECTORY,
    "fig16_rainfall_skill_all_event.pdf",
)
rainfall_figure.savefig(rainfall_png)
rainfall_figure.savefig(rainfall_pdf)
print("[SAVE]", rainfall_png, flush=True)
print("[SAVE]", rainfall_pdf, flush=True)
plt.show()
plt.close(rainfall_figure)

# Crest error, with only all times and event-only.
crest = pd.read_csv(CREST_PATH)
required_forcings = {
    "Observed rain",
    "Ensemble mean, 6",
    "HRRR, issue time",
}
missing_forcings = required_forcings - set(crest["forcing"].unique())
if missing_forcings:
    raise Exception("Crest table is missing: " + ", ".join(sorted(missing_forcings)))

plt.rcParams.update(native["FIG13_RC"])
crest_figure, crest_axes = plt.subplots(1, 2, figsize=(14.6, 4.8))
for axis, origin_scope in zip(crest_axes.ravel(), ORIGIN_SCOPES):
    origin_key = origin_scope[0]
    origin_label = origin_scope[1]
    rows = []
    origin_block = crest[crest["origin_set"] == origin_key]
    for rise_band in RISE_BANDS:
        observed_value = float(
            origin_block[origin_block["forcing"] == "Observed rain"][
                "mean_peak_abs_error_m_band_" + rise_band
            ].iloc[0]
        )
        ensemble_value = float(
            origin_block[origin_block["forcing"] == "Ensemble mean, 6"][
                "mean_peak_abs_error_m_band_" + rise_band
            ].iloc[0]
        )
        forecast_value = float(
            origin_block[origin_block["forcing"] == "HRRR, issue time"][
                "mean_peak_abs_error_m_band_" + rise_band
            ].iloc[0]
        )
        rows.append(
            {
                "stratum": native["STRATA_LABELS"].get(rise_band, rise_band),
                "observed": observed_value,
                "ensemble": ensemble_value,
                "forecast": forecast_value,
            }
        )
    crest_table = pd.DataFrame(rows)
    x_positions = np.arange(len(crest_table))
    bar_width = 0.25
    crest_cases = [
        ("observed", "Observed rainfall", native["COL_OBS"], -0.27),
        ("ensemble", "Ensemble mean", native["COL_QPF"], 0.0),
        ("forecast", "HRRR at issue time", native["COL_L2"], 0.27),
    ]
    for value_column, case_label, case_color, offset in crest_cases:
        axis.bar(
            x_positions + offset,
            crest_table[value_column],
            bar_width,
            color=case_color,
            label=case_label,
        )
        for row_index, value in enumerate(crest_table[value_column].to_numpy(float)):
            axis.annotate(
                f"{value:.3f}",
                (x_positions[row_index] + offset, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="0.20",
                fontweight="bold",
            )
    maximum_crest_error = float(
        crest_table[["observed", "ensemble", "forecast"]].to_numpy().max()
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(crest_table["stratum"], fontsize=12.5)
    axis.set_xlabel("Observed 24 h stage rise")
    axis.set_ylabel("Mean peak absolute error (m)")
    axis.set_ylim(0.0, maximum_crest_error * 1.24)
    axis.set_title(
        "(a) " + origin_label if origin_key == "all" else "(b) " + origin_label,
        loc="left",
    )
    axis.grid(color=native["GRID"], axis="y", alpha=0.22)
    axis.set_axisbelow(True)
crest_handles, crest_labels = crest_axes.ravel()[0].get_legend_handles_labels()
crest_figure.legend(
    crest_handles,
    crest_labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.004),
)
crest_figure.suptitle(
    "Crest errors during rising stage: all times and event-only",
    fontsize=19.0,
    fontweight="bold",
    y=0.975,
)
crest_figure.tight_layout(rect=(0.0, 0.085, 1.0, 0.965))
crest_png = os.path.join(
    FIGURE_DIRECTORY,
    "fig17_crest_error_all_event.png",
)
crest_pdf = os.path.join(
    FIGURE_DIRECTORY,
    "fig17_crest_error_all_event.pdf",
)
crest_figure.savefig(crest_png)
crest_figure.savefig(crest_pdf)
print("[SAVE]", crest_png, flush=True)
print("[SAVE]", crest_pdf, flush=True)
plt.show()
plt.close(crest_figure)

# Gauge-count sensitivity, with one row for all times and one row for event-only.
network = pd.read_csv(NETWORK_PATH)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 15.0,
        "axes.titlesize": 17.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 16.0,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "legend.fontsize": 12.5,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)

network_figure, network_axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
panel_letters = ["(a)", "(b)", "(c)", "(d)"]
panel_index = 0
for row_index, origin_scope in enumerate(ORIGIN_SCOPES):
    origin_key = origin_scope[0]
    origin_label = origin_scope[1]
    scope_data = network[network["origin_set"] == origin_key]
    if scope_data.empty:
        raise Exception("Gauge-count table has no rows for " + origin_key)
    rmse_groups = []
    for network_size in NETWORK_SIZES:
        values = scope_data[scope_data["size"] == network_size]["RMSE_m"].dropna().to_numpy(float)
        if len(values) == 0:
            raise Exception(
                "Gauge-count table has no values for " + origin_key + " size " + str(network_size)
            )
        rmse_groups.append(values)
    medians = {
        network_size: float(np.median(values))
        for network_size, values in zip(NETWORK_SIZES, rmse_groups)
    }
    skill = scope_data.groupby("size")[["NSE", "Pearson_r", "KGE"]].median()

    rmse_axis = network_axes[row_index, 0]
    positions = np.arange(len(NETWORK_SIZES))
    full_floor = float(
        scope_data[scope_data["size"] == 68]
        .groupby(["draw", "node"])["RMSE_m"]
        .agg(lambda values: values.max() - values.min())
        .median()
    )
    control_median = medians[68]
    rmse_axis.axhspan(
        control_median - full_floor,
        control_median + full_floor,
        color=COLOR_SEED_BAND,
        alpha=0.60,
        zorder=1,
        label="68-gauge seed variation, +/-" + format(full_floor, ".3f") + " m",
    )
    rmse_axis.axhline(
        control_median,
        color="0.25",
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )
    boxes = rmse_axis.boxplot(
        rmse_groups,
        positions=positions,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
        boxprops={"edgecolor": "0.25", "linewidth": 0.9},
    )
    box_colors = {
        68: COLOR_CONTROL,
        51: COLOR_PARENT,
    }
    for box_index, box_patch in enumerate(boxes["boxes"]):
        box_patch.set_facecolor(
            box_colors.get(NETWORK_SIZES[box_index], COLOR_IN_PARISH)
        )
        box_patch.set_alpha(0.55)
    random_generator = np.random.default_rng(101 + row_index)
    for position, values in zip(positions, rmse_groups):
        jitter = random_generator.uniform(-0.17, 0.17, size=len(values))
        rmse_axis.scatter(
            position + jitter,
            values,
            s=7,
            color="0.25",
            alpha=0.35,
            linewidths=0,
            zorder=4,
        )
    for position, network_size in zip(positions, NETWORK_SIZES):
        rmse_axis.annotate(
            format(medians[network_size], ".3f"),
            (position, medians[network_size]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="black",
            zorder=6,
        )
    rmse_axis.set_xticks(positions)
    rmse_axis.set_xticklabels([str(value) for value in NETWORK_SIZES])
    rmse_axis.set_xlabel("Gauges used by the model")
    rmse_axis.set_ylabel("Per-gauge h+24 RMSE (m)")
    rmse_axis.set_title(
        panel_letters[panel_index] + " " + origin_label + ": h+24 RMSE",
        loc="left",
    )
    panel_index += 1
    rmse_axis.grid(axis="y", alpha=0.25)
    rmse_axis.set_axisbelow(True)
    rmse_axis.legend(loc="upper right", framealpha=0.93, fontsize=10.2)

    skill_axis = network_axes[row_index, 1]
    metric_lines = [
        ("NSE", COLOR_NSE, "NSE", "o"),
        ("Pearson_r", COLOR_CORRELATION, "Correlation", "s"),
        ("KGE", COLOR_KGE, "KGE", "^"),
    ]
    for metric_column, color, label, marker in metric_lines:
        values = [float(skill.loc[network_size, metric_column]) for network_size in NETWORK_SIZES]
        skill_axis.plot(
            positions,
            values,
            marker=marker,
            markersize=8,
            linewidth=2.0,
            color=color,
            label=label,
        )
        for position, value in zip(positions, values):
            skill_axis.annotate(
                format(value, ".3f"),
                (position, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9.2,
                color=color,
            )
    skill_axis.set_xticks(positions)
    skill_axis.set_xticklabels([str(value) for value in NETWORK_SIZES])
    skill_axis.set_xlabel("Gauges used by the model")
    skill_axis.set_ylabel("Median value at h+24")
    skill_axis.set_title(
        panel_letters[panel_index] + " " + origin_label + ": median skill",
        loc="left",
    )
    panel_index += 1
    skill_axis.margins(y=0.12)
    skill_axis.grid(alpha=0.25)
    skill_axis.set_axisbelow(True)
    if row_index == 0:
        skill_axis.legend(loc="lower left", framealpha=0.93, ncol=3)

network_figure.suptitle(
    "Gauge network size: all times and event-only",
    fontsize=21.0,
    fontweight="bold",
    y=0.995,
)
network_figure.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.91,
    bottom=0.075,
    wspace=0.24,
    hspace=0.36,
)
network_png = os.path.join(
    FIGURE_DIRECTORY,
    "fig19_gauge_count_all_event.png",
)
network_pdf = os.path.join(
    FIGURE_DIRECTORY,
    "fig19_gauge_count_all_event.pdf",
)
network_figure.savefig(network_png, bbox_inches="tight")
network_figure.savefig(network_pdf, bbox_inches="tight")
print("[SAVE]", network_png, flush=True)
print("[SAVE]", network_pdf, flush=True)
plt.show()
plt.close(network_figure)

print("REV9_ALL_EVENT_FIGURES_DONE", flush=True)
