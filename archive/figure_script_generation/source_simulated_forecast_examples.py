"""Plot Figure 13 at h+24: simulated operational ST-GNN forecasts at 16 in-parish gauges.

Farid asked on 2026-09-24 for the h+6 Figure 13 as an h+24 version. The code is copied from
  20260921_Revision_9/scripts/source_simulated_forecast_figures.py (its lines 14-124 and 415-598),
which built the h+6 figure, with only these changes:
  lead step 24 -> 96 (96 steps of 15 min = h+24), so the forecast verifies 24 h after its origin;
  trace and metric file names, the figure title, and the legend label say h+24;
  outputs go to 20260922_Revision_13 (figures/fig13_forecast_rain_h24_jan_aug.png, work/fig13_h24/);
  observed stage comes from Figure 9's quality-controlled hindcast trace, so both figures blank the same hours.
Everything else is the native contact-sheet recipe. Farid chose to show the 16 gauges of Figure 9 (the h+24
hindcast), so one counted substitution makes the recipe's selection step keep only those gauges, in Figure 9's order. Inputs: the seed-101 ST-GNN
simulated-operational archive (issue-time HRRR rainfall, operational postprocessing, P80) and the frozen stage
matrix on the MIKE SSHFS mount (read as a file, no SSH connection).
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
WORK = os.path.join(REVISION, "work", "fig13_h24")
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
H6_LEAD_STEP = 96  # Revision 13: 96 steps of 15 min is h+24 (the h+6 figure used 24)

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
# Revision 13: use the quality-controlled observed stage of Figure 9. The frozen stage matrix keeps readings that the
# hindcast scoring rejected (for example a flat -1.8 m stretch at BCSA1299 from February to March). Where both hold
# a value they are identical, so taking Figure 9's trace blanks exactly the rejected hours and nothing else.
HINDCAST_TRACE = ("project/Experiments/EXTEND_2026_AUG31_EVENTS_20260918/"
                  "scored_jan_aug/observed_seed101/reforecast_2026H1_h96_timeseries.csv")
hindcast_observed = pd.read_csv(HINDCAST_TRACE, usecols=["node", "timestamp_utc", "observed_stage_m"])
hindcast_times = pd.to_datetime(hindcast_observed["timestamp_utc"])
if hindcast_times.dt.tz is not None:
    hindcast_times = hindcast_times.dt.tz_convert("UTC").dt.tz_localize(None)
hindcast_observed = pd.DataFrame(
    {
        "gauge": hindcast_observed["node"].astype(str),
        "verifying_time_utc": hindcast_times,
        "quality_controlled_stage_m": hindcast_observed["observed_stage_m"],
    }
)
h6_trace["verifying_time_utc"] = pd.to_datetime(h6_trace["verifying_time_utc"])
h6_trace = h6_trace.merge(hindcast_observed, on=["gauge", "verifying_time_utc"], how="left")
both = h6_trace["observed_stage_m"].notna() & h6_trace["quality_controlled_stage_m"].notna()
largest_difference = float((h6_trace.loc[both, "observed_stage_m"] - h6_trace.loc[both, "quality_controlled_stage_m"]).abs().max())
if largest_difference > 1.0e-6:
    raise Exception("The two observed-stage sources disagree by " + str(largest_difference) + " m")
blanked = int((h6_trace["observed_stage_m"].notna() & h6_trace["quality_controlled_stage_m"].isna()).sum())
print("[QC] observed stage from the Figure 9 trace; hours blanked as quality-rejected:", blanked, flush=True)
h6_trace["observed_stage_m"] = h6_trace["quality_controlled_stage_m"]
h6_trace = h6_trace.drop(columns=["quality_controlled_stage_m"])
H6_TRACE_PATH = os.path.join(WORK, "forecast_rain_h24_all51_traces_jan_aug.csv")
h6_trace.to_csv(H6_TRACE_PATH, index=False)
basin_table = pd.read_csv(BASIN_SOURCE)[["gauge", "basin", "basin_name", "basin_order"]]
basin_table["gauge"] = basin_table["gauge"].astype(str)
h6_metrics = calculate_per_gauge_metrics(h6_trace, basin_table)
# Revision 13: Farid chose the 16 gauges of Figure 9 (the h+24 hindcast), in Figure 9's panel order, so the two
# figures differ only in the rainfall and the postprocessing. The native recipe needs all 51 gauges in its metric
# table, so the selection step is overridden instead: it keeps only these gauges, in this order.
H6_FIGURE_GAUGES = [
    "BMSU2119", "BMSU2211", "BMSA1291", "BMSA2201",
    "BCSA1299", "BCSA2400", "BCSU4350", "BCSA4409",
    "HBSU1384", "HBSU1385", "HBSU3213", "HBSU4751",
    "MBSU9909", "MBSA6512", "MBSU4314", "MBSA4314",
]

H6_METRIC_PATH = os.path.join(WORK, "forecast_rain_h24_all51_metrics_jan_aug.csv")
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
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+24"',
            "brief title",
        ),
        (
            '        "expected_points_per_gauge": 371,\n'
            '        "title": (\n'
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+24"',
            '        "expected_points_per_gauge": 5832,\n'
            '        "title": (\n'
            '            "ST-GNN forecasts with issue-time HRRR rainfall at h+24"',
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
         '"forecast_label": "ST-GNN h+24 forecast",',
         "forecast label"),
        (
            '        elif dataset == "figure9_prospective_h24" and criterion_spec["variant"] == "golden06":',
            '        elif dataset == "figure9_prospective_h6" and criterion_spec["variant"] == "golden06":',
            "h+6 output alias",
        ),
        (
            '        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
            '        basin_frame = basin_frame[basin_frame["gauge"].isin(H6_FIGURE_GAUGES)].copy()\n'
            '        basin_frame["criterion_score"] = basin_frame["gauge"].map(\n'
            '            {gauge: position for position, gauge in enumerate(H6_FIGURE_GAUGES)}\n'
            '        )\n'
            '        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
            "same gauges as Figure 9",
        ),
        (
            '    if dataset_spec["dataset"] in {\n        "figure9_prospective_h24",\n'
            '        "figure6_retrospective_h24",\n    }',
            '    if dataset_spec["dataset"] in {\n        "figure9_prospective_h6",\n    }',
            "one dataset only",
        ),
    ],
    "Revision 13 Figure 13 at h+24",
)
exec(compile(contact_source, CONTACT_RECIPE, "exec"),
     {"__file__": CONTACT_RECIPE, "__name__": "__main__", "H6_FIGURE_GAUGES": H6_FIGURE_GAUGES})
plt.close("all")
native_contact_output = os.path.join(FIGURES, "fig11.png")
FIGURE_12_PNG = os.path.join(FIGURES, "fig13_forecast_rain_h24_jan_aug.png")
if not os.path.isfile(native_contact_output):
    raise Exception("The native contact-sheet recipe did not create " + native_contact_output)
shutil.move(native_contact_output, FIGURE_12_PNG)
print("[SAVED]", FIGURE_12_PNG, flush=True)
