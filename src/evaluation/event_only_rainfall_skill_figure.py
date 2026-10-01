"""Figure 13 on event origins: ST-GNN skill at h+24 with three rainfall forcings.

The published figure puts the rainfall forcings on the x axis over all origins. This version keeps the
three forcings as the grouped boxes and puts the event definitions on the x axis, so the reader sees
directly how each forcing holds up as the conditions narrow from every hour to the events that matter.

The drawing function is the paper's own. It is taken verbatim from
  project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914/
  scripts/recipe_rainfall_skill_and_crest.py
by executing that file's source down to its main(), which defines panel(), the rc settings and the
colors, and then calling panel() with event data. Nothing about the geometry, typeface, colors,
clipping rule or annotations is reinvented here.

Input: ../outputs/event_fig13_per_gauge_h24.csv, written by rainfall_forcing_skill.py.
Output: ../figures/event_fig13_rainfall_skill.{png,pdf}

Usage:
  conda run -n operational python -u event_only_rainfall_skill_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
FIGURE_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "figures")

RECIPE = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/scripts/recipe_rainfall_skill_and_crest.py"
)
PER_GAUGE = os.path.join(OUTPUT_DIRECTORY, "event_fig13_per_gauge_h24.csv")
STEM = os.path.join(FIGURE_DIRECTORY, "event_fig13_rainfall_skill")

ORIGIN_SETS = [
    ("all", "Every origin"),
    ("pump_at_issue", "Pumps running\nat issue"),
    ("stage_p80", "Stage above\n80th percentile"),
    ("stage_p90", "Stage above\n90th percentile"),
]
FORCING_CASES = [
    ("Observed rain", "Observed rainfall"),
    ("Ensemble mean, 6", "Ensemble mean of six weather models"),
    ("HRRR, issue time", "HRRR at issue time"),
]
METRIC_COLUMNS = {"rmse": "RMSE_m", "nse": "NSE", "r": "Pearson_r", "kge": "KGE"}

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


def load_recipe():
    """Execute the native recipe's definitions, stopping before its main()."""
    with open(RECIPE, "r", encoding="utf-8") as handle:
        source = handle.read()
    marker = "\ndef main("
    if marker not in source:
        raise SystemExit("[FATAL] the native recipe no longer defines main(): " + RECIPE)
    header = source.split(marker)[0]
    namespace = {"__name__": "native_recipe"}
    exec(compile(header, RECIPE, "exec"), namespace)
    for required in ("panel", "PANELS", "FIG12_RC", "COL_OBS", "COL_QPF", "COL_L2"):
        if required not in namespace:
            raise SystemExit("[FATAL] the native recipe does not define " + required)
    return namespace


native = load_recipe()
panel = native["panel"]
PANELS = native["PANELS"]

frame = pd.read_csv(PER_GAUGE)
seed_mean = (
    frame.groupby(["forcing", "origin_set", "gauge"], as_index=False)[
        ["RMSE_m", "Pearson_r", "NSE", "KGE"]
    ].mean()
)
print("[data] forcings:", sorted(seed_mean["forcing"].unique()), flush=True)

case_colors = [native["COL_OBS"], native["COL_QPF"], native["COL_L2"]]
offsets = [-0.26, 0.0, 0.26]
cases = [
    (label, color, offset)
    for (_, label), color, offset in zip(FORCING_CASES, case_colors, offsets)
]

plt.rcParams.update(native["FIG12_RC"])
figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.4))

for axis, (key, title, ylabel, lower_better) in zip(axes.ravel(), PANELS):
    column = METRIC_COLUMNS[key]
    groups_by_case = {}
    for forcing_key, label in FORCING_CASES:
        groups = []
        for origin_set, _ in ORIGIN_SETS:
            block = seed_mean[
                (seed_mean["forcing"] == forcing_key) & (seed_mean["origin_set"] == origin_set)
            ]
            values = block[column].to_numpy(dtype=float)
            groups.append(values[np.isfinite(values)])
        groups_by_case[label] = groups
    panel(
        axis,
        groups_by_case,
        cases,
        [label for _, label in ORIGIN_SETS],
        key,
        title,
        ylabel,
        lower_better,
    )

handles = [
    plt.Line2D([0], [0], color=color, linewidth=7.0, alpha=0.45, label=label)
    for (_, label), color in zip(FORCING_CASES, case_colors)
]
figure.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.002),
)
figure.suptitle(
    "ST-GNN skill at h+24 as conditions narrow to events",
    fontsize=19.0,
    fontweight="bold",
    y=0.995,
)
figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.972))
for suffix in (".png", ".pdf"):
    figure.savefig(STEM + suffix)
    print("[SAVED]", STEM + suffix, flush=True)
plt.close(figure)
print("EVENT_FIG13_DONE", flush=True)
