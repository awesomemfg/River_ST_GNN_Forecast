"""Figure 14 on event origins: ST-GNN crest errors during rising stage.

The published figure compares observed rainfall against the rainfall forecast across three bands of
observed 24 h stage rise. This version repeats that comparison inside each event definition, so the
reader can see whether the rainfall forecast costs more during pump operation and high stage than it
does on an average day.

Two figures are produced, each a 2-by-2 grid with one panel per origin set:
  event_fig14_crest_error   mean peak absolute error, in meters
  event_fig14_missed_crests percentage of rises whose crest is under-forecast by more than 0.1524 m

The drawing function is the paper's own bars(), taken verbatim from
  project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914/
  scripts/recipe_rainfall_skill_and_crest.py
by executing that file's source down to its main(). Bar widths, offsets, the difference annotation,
the grid and the axis handling are unchanged.

Input: ../outputs/event_fig14_crest_by_forcing.csv, written by rainfall_forcing_skill.py.

Usage:
  conda run -n operational python -u event_only_crest_error_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
FIGURE_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "figures")

RECIPE = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/scripts/recipe_rainfall_skill_and_crest.py"
)
CREST = os.path.join(OUTPUT_DIRECTORY, "event_fig14_crest_by_forcing.csv")

BANDS = ["0.15-0.30 m", "0.30-0.61 m", "above 0.61 m"]
ORIGIN_SETS = [
    ("all", "(a) Every origin"),
    ("pump_at_issue", "(b) Pumps running at issue"),
    ("stage_p80", "(c) Stage above the 80th percentile"),
    ("stage_p90", "(d) Stage above the 90th percentile"),
]
CASE_A = "Observed rain"
CASE_B = "HRRR, issue time"
LABEL_A = "Observed rainfall"
LABEL_B = "HRRR at issue time"

FIGURES = [
    (
        "mean_peak_abs_error_m_band_",
        "event_fig14_crest_error",
        "Mean peak absolute error (m)",
        "Crest errors during rising stage, by event",
        False,
    ),
    (
        "missed_crest_pct_band_",
        "event_fig14_missed_crests",
        "Crests under-forecast by >0.15 m (%)",
        "Missed crests during rising stage, by event",
        True,
    ),
]

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)


def load_recipe():
    """Execute the native recipe's definitions, stopping before its main()."""
    with open(RECIPE, "r", encoding="utf-8") as handle:
        source = handle.read()
    marker = "\ndef main("
    if marker not in source:
        raise SystemExit("[FATAL] the native recipe no longer defines main(): " + RECIPE)
    namespace = {"__name__": "native_recipe"}
    exec(compile(source.split(marker)[0], RECIPE, "exec"), namespace)
    for required in ("bars", "FIG13_RC", "COL_OBS", "COL_L2", "STRATA_LABELS"):
        if required not in namespace:
            raise SystemExit("[FATAL] the native recipe does not define " + required)
    return namespace


native = load_recipe()
bars = native["bars"]
STRATA_LABELS = native["STRATA_LABELS"]

crest = pd.read_csv(CREST)
missing = {CASE_A, CASE_B} - set(crest["forcing"].unique())
if missing:
    raise SystemExit("[FATAL] the crest table is missing forcings: " + ", ".join(sorted(missing)))

for column_prefix, stem, ylabel, suptitle, is_percentage in FIGURES:
    plt.rcParams.update(native["FIG13_RC"])
    figure, axes = plt.subplots(2, 2, figsize=(14.6, 9.6))
    for axis, (origin_set, title) in zip(axes.ravel(), ORIGIN_SETS):
        rows = []
        for band in BANDS:
            block = crest[crest["origin_set"] == origin_set]
            value_a = float(block[block["forcing"] == CASE_A][column_prefix + band].iloc[0])
            value_b = float(block[block["forcing"] == CASE_B][column_prefix + band].iloc[0])
            rows.append(
                {
                    "stratum": STRATA_LABELS.get(band, band),
                    "observed": value_a,
                    "forecast": value_b,
                }
            )
        table = pd.DataFrame(rows)
        bars(
            axis,
            table,
            native["COL_OBS"],
            native["COL_L2"],
            LABEL_A,
            LABEL_B,
            "observed",
            "forecast",
            ylabel,
            title,
            is_percentage,
        )
        counts = crest[crest["origin_set"] == origin_set]
        rise_total = float(counts[counts["forcing"] == CASE_A]["n_rises"].iloc[0])
        axis.annotate(
            f"{rise_total:,.0f} rises",
            (0.985, 0.965),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=12.0,
            color="0.35",
        )

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.004))
    figure.suptitle(suptitle, fontsize=19.0, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.972))
    for suffix in (".png", ".pdf"):
        path = os.path.join(FIGURE_DIRECTORY, stem + suffix)
        figure.savefig(path)
        print("[SAVED]", path, flush=True)
    plt.close(figure)

print("EVENT_FIG14_DONE", flush=True)
