# Revision 14 copy made by archive/figure_script_generation/make_figure_scripts.py on 2026-09-24.
# Original: archive/figure_script_generation/source_hindcast_examples.py
"""Figure 9 for revision 13: the revision 8 hindcast contact sheet with "hindcast" in its legend.

Revision 13 change (Farid, 2026-09-24): the legend read "ST-GNN h+24 reforecast", a word Farid replaced with
"hindcast" throughout the paper. This script is build_rev8_fig09.py with one extra counted substitution for that
label and outputs in 20260924_Revision_14. Data, selection rule, gauges, and layout are unchanged.

Revision 8 description follows.

The sheet is not redrawn. Revision 6 built three contact sheets from one native recipe with counted
substitutions, and this repeats the hindcast one exactly, changing only the trace it reads and where
the output goes. Panels, colours, the selection rule, fonts and the layout are the recipe's own.

  Recipe   .../20260826_Final_V2/figure_revision_scripts/recipe_contact_sheets.py
  Pattern  20260915_Revision_6/scripts/build_rev6_contact_sheets.py

Two of revision 6's three sheets showed the June real-time simulation. That experiment is no longer
in the paper, so only the hindcast sheet is built here. The June h+6 trace that revision 6 assembled
from the June archive is not carried over either: it was built at module level, before the sheets
were drawn, so skipping the sheet alone would still read an archive the paper no longer uses.

The trace is the scored January to August hindcast, which has the columns the recipe reads:
node, observed_stage_m, predicted_stage_m and timestamp_utc.

Output: 20260924_Revision_14/figures/fig09_hindcast_contact_sheet_rev13.png
Run with: conda run -n operational python build_rev8_fig09.py
"""
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = "project/manuscript/"
RECIPE = "src/figures/recipe_contact_sheets.py"
EXTEND = ("project/Experiments/"
          "EXTEND_2026_AUG31_ANCHORFIX_20260924/")
SEED = 101
TRACE_PATH = os.path.join(EXTEND, "scored_jan_aug", f"observed_seed{SEED}",
                          "reforecast_2026H1_h96_timeseries.csv")
BASIN_SOURCE = PAPER + ("revision_2026_07_17/contact_sheets_all_51/figure6_retrospective_h24/"
                        "figure6_retrospective_h24_all51_metrics.csv")
REV13 = PAPER + "20260924_Revision_14"
FIGURES = os.path.join(REV13, "figures")
WORK = os.path.join(REV13, "work", "fig09_contact_sheet")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
plt.show = lambda *args, **kwargs: None

for required in (RECIPE, TRACE_PATH, BASIN_SOURCE):
    if not os.path.isfile(required):
        raise SystemExit("[FATAL] missing " + required)

basins = pd.read_csv(BASIN_SOURCE)[["gauge", "basin", "basin_name", "basin_order"]]


def per_gauge_metrics(frame, gauge_column, observed_column, predicted_column, dataset, time_column):
    """The per-gauge table the recipe uses to pick and title its panels, built as revision 6 built it."""
    rows = []
    for _, basin_row in basins.iterrows():
        gauge_frame = frame[frame[gauge_column] == basin_row["gauge"]]
        observed = pd.to_numeric(gauge_frame[observed_column], errors="coerce").to_numpy(float)
        predicted = pd.to_numeric(gauge_frame[predicted_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed_valid = observed[valid]
        predicted_valid = predicted[valid]
        denominator = float(np.sum((observed_valid - observed_valid.mean()) ** 2))
        rows.append({"dataset": dataset, "gauge": basin_row["gauge"], "basin": basin_row["basin"],
                     "basin_name": basin_row["basin_name"], "basin_order": basin_row["basin_order"],
                     "row_count": len(gauge_frame), "n_pairs": int(valid.sum()),
                     "missing_observations": int((~np.isfinite(observed)).sum()),
                     "missing_predictions": int((~np.isfinite(predicted)).sum()),
                     "rmse_m": float(np.sqrt(np.mean((predicted_valid - observed_valid) ** 2))),
                     "nse": 1.0 - float(np.sum((predicted_valid - observed_valid) ** 2)) / denominator,
                     "correlation": float(np.corrcoef(observed_valid, predicted_valid)[0, 1]),
                     "verifying_time_start_utc": gauge_frame[time_column].min(),
                     "verifying_time_end_utc": gauge_frame[time_column].max()})
    return pd.DataFrame(rows)


print("[trace]", TRACE_PATH, flush=True)
trace = pd.read_csv(TRACE_PATH)
METRIC_PATH = os.path.join(WORK, f"figure6_retrospective_h24_seed{SEED}_all51_metrics.csv")
metrics = per_gauge_metrics(trace, "node", "observed_stage_m", "predicted_stage_m",
                            "figure6_retrospective_h24", "timestamp_utc")
metrics.to_csv(METRIC_PATH, index=False)
print("[metrics]", METRIC_PATH, len(metrics), "gauges", flush=True)
print("[metrics] median RMSE", round(float(metrics["rmse_m"].median()), 4), "m | median NSE",
      round(float(metrics["nse"].median()), 4), flush=True)

COMMON = [
    # Revision 14 (Farid, 2026-09-25): BCSA1299 is left out for stale telemetry. The recipe's own ranking then puts
    # BCSA4419 in the fourth Bayou Conway slot.
    ('        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
     '        basin_frame = basin_frame[basin_frame["gauge"] != "BCSA1299"].copy()\n'
     '        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
     "BCSA1299 left out"),
    ('REVISION_OUTPUT_DIRECTORY = Path(\n    "project/"\n'
     '    "manuscript/20260826_Final_V2/"\n    "tmp/contact_sheets_font_revision"\n)',
     'REVISION_OUTPUT_DIRECTORY = Path("' + WORK + '")', "work directory"),
    ('FINAL_FIGURE_DIRECTORY = Path(\n    "project/"\n'
     '    "manuscript/20260826_Final_V2/"\n    "output/figures"\n)',
     'FINAL_FIGURE_DIRECTORY = Path("' + FIGURES + '")', "figure directory"),
    ('    figure.suptitle(\n        dataset_spec["title"],\n        fontsize=16.0,\n        fontweight="bold",\n'
     '        y=0.994,\n    )',
     '    figure.suptitle(\n        dataset_spec["title"],\n        fontsize=19.0,\n        fontweight="bold",\n'
     '        y=0.997,\n    )', "brief title, larger"),
    ('    figure.text(\n        0.5,\n        0.968,\n        (\n            dataset_spec["subtitle"]\n'
     '            + " | Selection: "\n            + criterion_spec["description"]\n        ),\n'
     '        horizontalalignment="center",\n        verticalalignment="top",\n        fontsize=10.2,\n    )\n', "",
     "subtitle removed"),
    ('        bbox_to_anchor=(0.5, 0.922),\n        ncol=2,\n        frameon=False,\n        fontsize=10.2,',
     '        bbox_to_anchor=(0.5, 0.974),\n        ncol=2,\n        frameon=False,\n        fontsize=13.0,',
     "legend below the title, enlarged"),
    ('        "Stage (m)",\n        rotation=90,\n        horizontalalignment="center",\n'
     '        verticalalignment="center",\n        fontsize=11.0,\n    )',
     '        "Stage (m)",\n        rotation=90,\n        horizontalalignment="center",\n'
     '        verticalalignment="center",\n        fontsize=14.0,\n    )', "stage label larger"),
    ('    figure.text(\n        0.5,\n        0.010,\n        "Verifying time (UTC)",\n'
     '        horizontalalignment="center",\n        verticalalignment="bottom",\n        fontsize=9.5,\n'
     '        color="#444444",\n    )',
     '    figure.text(\n        0.5,\n        0.008,\n        "Verifying time (UTC)",\n'
     '        horizontalalignment="center",\n        verticalalignment="bottom",\n        fontsize=14.0,\n'
     '        fontweight="bold",\n        color="#222222",\n    )', "verifying time clear and larger"),
    ('    figure.subplots_adjust(\n        left=0.073,\n        right=0.996,\n        top=0.845,\n'
     '        bottom=0.070,\n        hspace=0.68,\n        wspace=0.18,\n    )',
     '    figure.subplots_adjust(\n        left=0.076,\n        right=0.996,\n        top=0.872,\n'
     '        bottom=0.078,\n        hspace=0.62,\n        wspace=0.18,\n    )', "panels clear of the legend"),
    ('    panel_title = (\n        gauge\n        + " | "\n        + basin_name\n        + "\\nRMSE "\n'
     '        + rmse_text\n        + " m | NSE "\n        + nse_text\n        + " | r "\n'
     '        + correlation_text\n        + " | n "\n        + str(pair_count)\n    )',
     '    panel_title = (\n        gauge\n        + " | "\n        + basin_name\n        + "\\nRMSE "\n'
     '        + rmse_text\n        + " m | NSE "\n        + nse_text\n        + " | r "\n'
     '        + correlation_text\n    )', "panel title without the repeated count"),
    ('        panel_title,\n        fontsize=9.2,', '        panel_title,\n        fontsize=12.2,',
     "panel titles larger"),
    ('    axis.tick_params(\n        axis="both",\n        labelsize=9.8,\n        length=2.0,\n'
     '        width=0.5,\n        pad=1.2,\n    )',
     '    axis.tick_params(\n        axis="both",\n        labelsize=12.2,\n        length=2.6,\n'
     '        width=0.6,\n        pad=1.6,\n    )', "tick labels larger"),
]

SHEET = {
    "name": "fig09_hindcast_contact_sheet_rev13",
    "alias": "fig08.png",
    "dataset": "figure6_retrospective_h24",
    "extra": [
        ('FIGURE6_TRACE_PATH = Path(\n    "project/"\n'
         '    "Experiments/REFORECAST_GRAPH_RANKING_2026H1/"\n    "results/evaluations/out_obs_pre2026/"\n'
         '    "reforecast_2026H1_h96_timeseries.csv"\n)',
         'FIGURE6_TRACE_PATH = Path("' + TRACE_PATH + '")', "hindcast traces"),
        ('FIGURE6_METRIC_PATH = (\n    FIGURE6_DIRECTORY\n    / "figure6_retrospective_h24_all51_metrics.csv"\n)',
         'FIGURE6_METRIC_PATH = Path("' + METRIC_PATH + '")', "hindcast metrics"),
        ('            "Fixed-lead h+24 retrospective ST-GNN reforecasts "\n'
         '            "across in-parish gauges"', '            "ST-GNN hindcasts at h+24"', "brief title"),
        # The recipe checks that every gauge has exactly this many rows, and the constant is the
        # length of the old period. The trace now holds 5,832 origins per gauge, so the check fails
        # with "too few trace rows" even though the trace is longer than expected. This is a period
        # constant, like the dates in the captions, and the check itself is left intact.
        ('        "expected_points_per_gauge": 4344,',
         '        "expected_points_per_gauge": 5832,', "origins per gauge"),
        # Revision 13: the legend says hindcast, the paper's term, instead of reforecast.
        ('"forecast_label": "ST-GNN h+24 reforecast",',
         '"forecast_label": "ST-GNN h+24 hindcast",', "hindcast legend label"),
    ],
}

source = open(RECIPE, encoding="utf-8").read()
substitutions = COMMON + SHEET["extra"] + [
    ('    if dataset_spec["dataset"] in {\n        "figure9_prospective_h24",\n'
     '        "figure6_retrospective_h24",\n    }',
     '    if dataset_spec["dataset"] in {\n        "' + SHEET["dataset"] + '",\n    }', "one dataset only"),
]
for old, new, name in substitutions:
    count = source.count(old)
    if count != 1:
        raise Exception(f"expected exactly one '{name}' target, found {count}")
    source = source.replace(old, new, 1)
print("[VERIFY] contact-sheet recipe applied", len(substitutions), "counted substitutions", flush=True)

try:
    exec(compile(source, RECIPE, "exec"), {"__file__": RECIPE, "__name__": "__main__"})
except SystemExit as stop:
    if stop.code not in (0, None):
        raise
plt.close("all")

produced = os.path.join(FIGURES, SHEET["alias"])
if not os.path.isfile(produced):
    raise Exception("missing output " + produced)
shutil.move(produced, os.path.join(FIGURES, SHEET["name"] + ".png"))
print("[VERIFY] created", os.path.join(FIGURES, SHEET["name"] + ".png"), flush=True)
print("REV13_FIG09_DONE", flush=True)
