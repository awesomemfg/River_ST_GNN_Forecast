# Revision 14 copy made by archive/figure_script_generation/make_figure_scripts.py on 2026-09-24.
# Original: archive/figure_script_generation/source_appendix_figures.py
"""Appendix figures for revision 8, each from its native recipe with counted substitutions.

Revision 7's appendix held seven figures. Three of them were built from the June real-time
simulation, which the paper no longer reports, so they go:

  A1  digital elevation model and HEC-RAS mesh   kept, unchanged, not built here
  B1  map of h+24 hindcast skill                 rebuilt here on January to August
  B2  ST-GNN minus LSTM, June                    removed with the real-time simulation
  B3  map of the rainfall-forecast penalty       rebuilt here on January to August
  C1  sixteen large-rise cases, June             removed with the real-time simulation
  C2  all-51 montage, hindcast                   rebuilt here on January to August
  C3  all-51 montage, June                       removed with the real-time simulation

The recipes are not rewritten. Each is read, given counted substitutions, and executed, exactly as
revisions 6 and 7 did. Only the inputs and two period constants change.

  B1  figure_library/recipe_spatial_skill.py
      fed the seed-averaged per-gauge table written by appendix_figure_inputs.py, whose h+24
      row reproduces Table 4's ST-GNN column: 0.1459 m and 0.7207.
  B3  figure_library/recipe_spatial_difference.py, its four-panel block only
      fed the by-seed forcing table written by rainfall_and_crest_figure_inputs.py. Its own printed
      check should say the forecast rainfall raises the h+24 RMSE at 50 of the 51 gauges with a
      median of about +0.034 m, which is what Section 4.1 states. If it does not, the figure and the
      text disagree.
  C2  revision_2026_07_17/scripts/recipe_contact_sheets_all_gauges.py
      fed the scored January to August hindcast trace. This recipe reads both its traces before it
      loops over datasets, so the June trace is still read and its montage is produced and then
      discarded. Restructuring the recipe to skip it would be a rewrite, which is not allowed here.
      Its row-count constant is raised from 4,344 to 5,832, or it rejects the longer trace.

Outputs: 20260924_Revision_14/figures/appendix/
Run with: conda run -n operational python source_appendix_figures.py
"""
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = "project/manuscript/"
EXTEND = ("project/Experiments/"
          "EXTEND_2026_AUG31_ANCHORFIX_20260924/")
INPUTS = os.path.join(EXTEND, "outputs", "figure_inputs_jan_aug")
REV8 = PAPER + "20260924_Revision_14"
FIGURES = os.path.join(REV8, "figures", "appendix")
WORK = os.path.join(REV8, "work", "appendix_figures")

B1_METRICS = os.path.join(INPUTS, "appendixB1_live_fixed_lead_metrics_inparish51_jan_aug.csv")
FORCING_PER_GAUGE = os.path.join(INPUTS, "fig13_gauge_metrics_24h_by_seed_jan_aug.csv")
HINDCAST_TRACES = os.path.join(EXTEND, "scored_jan_aug", "observed_seed101",
                               "reforecast_2026H1_h96_timeseries.csv")
# Read by the montage recipe before it loops. Its montage is produced and discarded.
JUNE_TRACES = ("project/Experiments/"
               "SYSTEM_B_471_CHRONOLOGICAL_20260913/paper_rev5/scored/"
               "stgnn_s101_realtime_june371/reforecast_2026H1_h96_timeseries.csv")

SPAN = "1 January to 31 August 2026"
ORIGINS = 5832

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
plt.show = lambda *args, **kwargs: None
sys.path.insert(0, "src/figures")

for required in (B1_METRICS, FORCING_PER_GAUGE, HINDCAST_TRACES, JUNE_TRACES):
    if not os.path.isfile(required):
        raise SystemExit("[FATAL] missing " + required)


def apply_counted(source, substitutions, label):
    for entry in substitutions:
        old, new, name = entry[0], entry[1], entry[-1]
        expected = entry[2] if len(entry) == 4 else 1
        count = source.count(old)
        if count != expected:
            raise Exception(f"{label}: expected {expected} '{name}' target(s), found {count}")
        source = source.replace(old, new)
    print("[VERIFY]", label, "applied", len(substitutions), "counted substitutions:",
          ", ".join(e[-1] for e in substitutions), flush=True)
    return source


def run_source(source, source_path, namespace, outputs):
    namespace.update({"__file__": source_path, "__name__": "__main__"})
    exec(compile(source, source_path, "exec"), namespace)
    plt.close("all")
    for path in outputs:
        if not os.path.isfile(path):
            raise Exception("missing output " + path)
        print("[VERIFY] created", path, flush=True)


from _mapbase import load_xy  # noqa: E402

COORDINATES = load_xy()
print("[LOAD] map coordinates for", len(COORDINATES), "gauges", flush=True)


# ------------------------------------------------------------------------------------------------
# Figure B1: the map of h+24 hindcast skill
# ------------------------------------------------------------------------------------------------
def hindcast_skill():
    """Per-gauge h+24 hindcast skill, already averaged over the three seeds, with map coordinates."""
    table = pd.read_csv(B1_METRICS)
    table = table[table["lead_hours"] == 24]
    rows = {}
    for _, row in table.iterrows():
        gauge = str(row["gauge"])
        if gauge not in COORDINATES:
            continue
        rows[gauge] = {"rmse": float(row["rmse_m"]), "corr": float(row["correlation"]),
                       "kge": float(row["kge"]), "nse": float(row["nse"]),
                       "E": COORDINATES[gauge][0], "N": COORDINATES[gauge][1]}
    frame = pd.DataFrame(rows).T
    print("[B1] gauges:", len(frame), "| median RMSE:", round(float(np.median(frame["rmse"])), 4),
          "| median NSE:", round(float(np.median(frame["nse"])), 4), flush=True)
    return frame


B1_RECIPE = "src/figures/recipe_spatial_skill.py"
source = apply_counted(open(B1_RECIPE, encoding="utf-8").read(), [
    ("from _spatialmap import load_stgnn_skill, archive_span, _scatter_metric, annotate_stats, add_reference_cities, DST",
     "from _spatialmap import _scatter_metric, annotate_stats, add_reference_cities\nDST = \"" + WORK + "\"",
     "loader import and output folder"),
    ("df = load_stgnn_skill()", "df = REV8_HINDCAST_SKILL()", "per-gauge skill"),
    ("span, ncyc = archive_span()", f'span, ncyc = "{SPAN}", {ORIGINS}', "period"),
    ('ax.set_title(f"Delivered ST-GNN forecast skill across Ascension Parish: {PRETTY[key]}\\n"\n'
     '                 f"hourly forecasts, {span}  ({len(gauges)} in-parish gauges)")',
     'ax.set_title(f"ST-GNN hindcast skill across Ascension Parish: {PRETTY[key]}\\n"\n'
     '                 f"{ncyc:,} hourly forecast origins, {span}  ({len(gauges)} in-parish gauges)")',
     "single-metric titles"),
    ('fig.suptitle(f"Delivered ST-GNN per-gauge skill across Ascension Parish - hourly forecasts, {span}",',
     'fig.suptitle(f"ST-GNN per-gauge h+24 hindcast skill across Ascension Parish, {span}",', "panel title"),
    ('"mean 0-24 h RMSE (m)"', '"h+24 RMSE (m)"', "RMSE label"),
    ('"median 0-24 h correlation"', '"h+24 Pearson correlation"', 2, "correlation label"),
    ('"0-24 h KGE"', '"h+24 KGE"', 2, "KGE label"),
    ('"0-24 h NSE"', '"h+24 NSE"', 2, "NSE label"),
    ('"mean 0-24 h RMSE"', '"h+24 RMSE"', "RMSE name"),
], "Figure B1")
run_source(source, B1_RECIPE, {"REV8_HINDCAST_SKILL": hindcast_skill},
           [os.path.join(WORK, "fig11_spatial_panel.png")])
for suffix in (".png", ".pdf"):
    shutil.copyfile(os.path.join(WORK, "fig11_spatial_panel" + suffix),
                    os.path.join(FIGURES, "figB1_hindcast_spatial_skill" + suffix))
print("[SAVED]", os.path.join(FIGURES, "figB1_hindcast_spatial_skill.png"), flush=True)


# ------------------------------------------------------------------------------------------------
# Figure B3: the map of what the rainfall forecast costs
# ------------------------------------------------------------------------------------------------
def rainfall_penalty_difference():
    """Per-gauge h+24 difference, issue-time HRRR minus observed rainfall, seeds averaged first."""
    table = pd.read_csv(FORCING_PER_GAUGE)
    table = table.rename(columns={"node": "gauge", "rmse": "RMSE_m", "r": "Pearson_r",
                                  "nse": "NSE", "kge": "KGE"})
    seed_mean = table.groupby(["forcing", "gauge"], as_index=False)[
        ["RMSE_m", "Pearson_r", "NSE", "KGE"]].mean()
    hrrr = seed_mean[seed_mean["forcing"] == "HRRR, issue time"].set_index("gauge")
    observed = seed_mean[seed_mean["forcing"] == "Observed rain"].set_index("gauge")
    if len(hrrr) == 0 or len(observed) == 0:
        raise Exception("the forcing table has no issue-time HRRR or observed-rainfall rows")
    shared = [gauge for gauge in hrrr.index if gauge in observed.index and gauge in COORDINATES]
    rows = {}
    for gauge in shared:
        rows[gauge] = {
            "rmse": float(hrrr.loc[gauge, "RMSE_m"]) - float(observed.loc[gauge, "RMSE_m"]),
            "corr": float(hrrr.loc[gauge, "Pearson_r"]) - float(observed.loc[gauge, "Pearson_r"]),
            "kge": float(hrrr.loc[gauge, "KGE"]) - float(observed.loc[gauge, "KGE"]),
            "nse": float(hrrr.loc[gauge, "NSE"]) - float(observed.loc[gauge, "NSE"]),
            "E": COORDINATES[gauge][0],
            "N": COORDINATES[gauge][1],
        }
    frame = pd.DataFrame(rows).T
    print("[B3] gauges:", len(frame),
          "| forecast rainfall raises the h+24 RMSE at", int((frame["rmse"] > 0).sum()),
          "| median RMSE difference {:+.4f} m".format(float(np.median(frame["rmse"]))), flush=True)
    print("[B3] Section 4.1 says 50 of the 51 gauges and a median increase of 0.034 m", flush=True)
    return frame


DIFF_RECIPE = "src/figures/recipe_spatial_difference.py"
native = open(DIFF_RECIPE, encoding="utf-8").read()
head_marker = "# ---- Fig 13 (main): Delta RMSE"
panel_start = "fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.8))"
panel_end = 'print("[SAVED] fig15_spatial_panel_4.png")'
for marker in (head_marker, panel_start, panel_end):
    if native.count(marker) != 1:
        raise Exception("difference-map marker is not unique: " + marker[:40])
composed = native[: native.index(head_marker)] + native[native.index(panel_start): native.index(panel_end) + len(panel_end)]
print("[VERIFY] difference-map recipe trimmed to its four-panel block,",
      len(composed.splitlines()), "lines kept of", len(native.splitlines()), flush=True)

B3_NAME = "figB3_rainfall_penalty_spatial"
source = apply_counted(composed, [
    ("from _spatialmap import load_diff, load_model_comparison, archive_span, _scatter_metric, "
     "add_reference_cities, DST",
     'from _spatialmap import _scatter_metric, add_reference_cities\nDST = "' + WORK + '"',
     "loader import and output folder"),
    ("df = load_diff()", "df = REV8_DIFFERENCE()", "per-gauge difference table"),
    ("cmp = load_model_comparison()", "cmp = None   # only the four-panel block runs here",
     "side-by-side table not needed"),
    ("span, _ = archive_span()", "span = REV8_SPAN", "period from the caller"),
    ('"rmse": ("0-24 h RMSE", False, "RdYlGn_r", " m"),\n'
     '    "corr": ("0-24 h correlation", True, "RdYlGn", ""),\n'
     '    "kge": ("0-24 h KGE", True, "RdYlGn", ""),\n'
     '    "nse": ("0-24 h NSE", True, "RdYlGn", ""),',
     '"rmse": ("h+24 RMSE", False, "RdYlGn_r", " m"),\n'
     '    "corr": ("h+24 correlation", True, "RdYlGn", ""),\n'
     '    "kge": ("h+24 KGE", True, "RdYlGn", ""),\n'
     '    "nse": ("h+24 NSE", True, "RdYlGn", ""),', "fixed-lead metric names"),
    ('"ST-GNN better at %d of %d gauges" % (better, n)',
     '"forecast rainfall better at %d of %d gauges" % (better, n)', 2, "annotation wording"),
    ('cb_.set_label(f"ST-GNN $-$ Bi-LSTM {pretty}{unit}; green = ST-GNN better", fontsize=11)',
     'cb_.set_label(f"HRRR $-$ observed rainfall {pretty}{unit}; red = the rainfall forecast costs skill", '
     'fontsize=11)', "colour-bar label"),
    ('fig.suptitle(f"ST-GNN minus Bi-LSTM per-gauge skill across Ascension Parish - hourly forecasts, {span}",',
     'fig.suptitle(f"Penalty of the issue-time HRRR rainfall forecast on per-gauge h+24 ST-GNN skill, {span}",',
     "panel title"),
    ('fig.savefig(f"{DST}/fig15_spatial_panel_4.png", dpi=240, bbox_inches="tight")\n'
     'fig.savefig(f"{DST}/fig15_spatial_panel_4.pdf", bbox_inches="tight")\n'
     'print("[SAVED] fig15_spatial_panel_4.png")',
     'fig.savefig(f"{DST}/' + B3_NAME + '.png", dpi=240, bbox_inches="tight")\n'
     'fig.savefig(f"{DST}/' + B3_NAME + '.pdf", bbox_inches="tight")\n'
     'print("[SAVED] ' + B3_NAME + '.png")', "output name"),
], "Figure B3")
run_source(source, DIFF_RECIPE,
           {"REV8_DIFFERENCE": rainfall_penalty_difference,
            "REV8_SPAN": f"{ORIGINS:,} hourly origins, {SPAN}"},
           [os.path.join(WORK, B3_NAME + ".png")])
for suffix in (".png", ".pdf"):
    shutil.copyfile(os.path.join(WORK, B3_NAME + suffix), os.path.join(FIGURES, B3_NAME + suffix))
print("[SAVED]", os.path.join(FIGURES, B3_NAME + ".png"), flush=True)


# ------------------------------------------------------------------------------------------------
# Figure C2: the all-51 montage of the hindcast
# ------------------------------------------------------------------------------------------------
C_RECIPE = "src/figures/recipe_contact_sheets_all_gauges.py"
source = apply_counted(open(C_RECIPE, encoding="utf-8").read(), [
    ('output_directory = os.path.join(\n    revision_directory,\n    "contact_sheets_all_51",\n)',
     'output_directory = "' + WORK + '"', "output folder"),
    ('prospective_trace_path = "project/manuscript/'
     'New_20260715_FullRevision/individual_station_diagnostics/prospective_h6_all_51_traces.csv"',
     'prospective_trace_path = "' + JUNE_TRACES + '"', "June trace file"),
    ('retrospective_trace_path = "project/Experiments/'
     'REFORECAST_GRAPH_RANKING_2026H1/results/evaluations/out_obs_pre2026/reforecast_2026H1_h96_timeseries.csv"',
     'retrospective_trace_path = "' + HINDCAST_TRACES + '"', "hindcast trace file"),
    ('prospective_traces["time_utc"] = pd.to_datetime(\n    prospective_traces["verifying_time_utc"],\n'
     '    utc=True,\n    errors="coerce",\n)\nprospective_traces["predicted_stage_m"] = pd.to_numeric(\n'
     '    prospective_traces["forecast_stage_m"],\n    errors="coerce",\n)',
     'prospective_traces["gauge"] = prospective_traces["node"].astype(str)\n'
     'prospective_traces["time_utc"] = pd.to_datetime(\n    prospective_traces["timestamp_utc"],\n'
     '    utc=True,\n    errors="coerce",\n)\nprospective_traces["predicted_stage_m"] = pd.to_numeric(\n'
     '    prospective_traces["predicted_stage_m"],\n    errors="coerce",\n)',
     "June trace columns"),
    # The hindcast now holds 5,832 origins per gauge. The recipe checks the count and rejects a
    # trace that does not match, which is a period constant and not a loosening of the check.
    ('("Figure 6 retrospective h+24", retrospective_traces, 4344),',
     f'("Hindcast h+24", retrospective_traces, {ORIGINS}),', "hindcast row check"),
    ('("Figure 9 prospective h+6", prospective_traces, 371),',
     '("June real-time h+24", prospective_traces, 371),', "June row check"),
    ('    "Fixed-lead h+24 retrospective ST-GNN reforecasts across in-parish gauges",\n'
     '    "All 51 in-parish gauges | 4,344 hourly origins from 1 January-30 June 2026 | perfect observed-rainfall forcing",\n'
     '    "ST-GNN h+24 reforecast",',
     '    "Fixed-lead h+24 ST-GNN hindcasts across in-parish gauges",\n'
     f'    "All 51 in-parish gauges | {ORIGINS:,} hourly origins from 1 January-31 August 2026 | observed rainfall",\n'
     '    "ST-GNN h+24 hindcast",', 2, "hindcast titles"),
    ('    "Delivered ST-GNN fixed-lead h+6 forecasts across in-parish gauges",\n'
     '    "All 51 in-parish gauges | 371 hourly issues from 7-22 June 2026",\n'
     '    "Delivered ST-GNN h+6 forecast",',
     '    "ST-GNN fixed-lead h+24 forecasts, real-time simulation, across in-parish gauges",\n'
     '    "All 51 in-parish gauges | 371 hourly issues from 7-22 June 2026",\n'
     '    "ST-GNN h+24 forecast",', 2, "June titles"),
], "Figure C2")
# Both montages are produced because the recipe reads both traces before it loops. Only the hindcast
# one is kept; the June montage is left in the work folder and never copied into the appendix.
run_source(source, C_RECIPE, {},
           [os.path.join(WORK, "figure6_retrospective_h24",
                         "figure6_retrospective_h24_all51_contact_7x8.png")])
for suffix in (".png", ".pdf"):
    shutil.copyfile(os.path.join(WORK, "figure6_retrospective_h24",
                                 "figure6_retrospective_h24_all51_contact_7x8" + suffix),
                    os.path.join(FIGURES, "figC2_hindcast_all51" + suffix))
print("[SAVED]", os.path.join(FIGURES, "figC2_hindcast_all51.png"), flush=True)
print("REV8_APPENDIX_FIGURES_DONE", flush=True)
