"""Figure 8 for revision 8: the hindcast skill figure, on January to August 2026.

The figure is not redrawn. Revision 6's version of this figure is reproduced from its own native
recipe with counted substitutions, and this repeats those substitutions exactly, changing only the
input file and the output names. Panels, colours, limits, fonts and the four metrics are the
recipe's own.

  Recipe   project/home/.../20260811_Revision/Work/scripts/recipe_graph_figure_and_table.py
  Pattern  20260915_Revision_6/scripts/build_rev6_fig08.py

The input is the only real change. Revision 6 read a mean-over-seeds scored directory covering
January to June. The scored directories for this period are per seed, so the three are averaged on
(lead, gauge) first by figure_inputs.py, which is the order the caption states: each
per-gauge metric is averaged over the three trained models before the median is taken.

That file reports a median h+24 RMSE of 0.1459 m and a median NSE of 0.7207 over the 51 in-parish
gauges, which are Table 4's values, so the figure and the table cannot disagree.

Output: 20260920_Revision_8/figures/fig08_hindcast_rev8.png
Run with: conda run -n operational python source_fig08_hindcast_skill.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = "project/manuscript/"
RECIPE = "src/figures/recipe_graph_figure_and_table.py"
LEAD_METRICS = ("project/Experiments/"
                "EXTEND_2026_AUG31_EVENTS_20260918/outputs/figure_inputs_jan_aug/"
                "fig08_lead_metrics_mean3_jan_aug.csv")
REV8 = PAPER + "20260920_Revision_8"
FIGURES = os.path.join(REV8, "figures")
WORK = os.path.join(REV8, "work", "fig08")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

if not os.path.isfile(LEAD_METRICS):
    raise SystemExit("[FATAL] missing the January to August lead metrics: " + LEAD_METRICS
                     + "\n  Run figure_inputs.py first.")

SUBSTITUTIONS = [
    # ---- inputs and outputs
    ('FROZEN = ("project/hpc/Experiments/"\n'
     '          "REFORECAST_LSTM_CLEAN_20260729/frozen_stgnn/reforecast_2026H1_lead_metrics.csv")',
     'FROZEN = "' + LEAD_METRICS + '"', 1, "input lead metrics"),
    ('OUT = ("project/manuscript/"\n'
     '       "20260811_Revision/Work/verified")', 'OUT = "' + FIGURES + '"', 1, "output directory"),
    ('FIGNAME = "fig07_reforecast_frozen_stgnn_51gauges"', 'FIGNAME = "fig08_hindcast_rev8"', 1,
     "figure name"),
    ("table4_frozen_stgnn.csv", "table4_hindcast_rev8.csv", 2, "table name"),
    # ---- a brief title, close to the panels, and larger type, as in revision 6
    ('    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12.2, 7.8))',
     '    plt.rcParams.update({"font.size": 13.0, "axes.titlesize": 15.5, "axes.labelsize": 14.0,\n'
     '                         "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5})\n'
     '    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12.2, 8.1))\n'
     '    fig.suptitle("ST-GNN hindcast skill", fontsize=17.5, fontweight="bold", y=0.996)', 1,
     "brief title and larger type"),
    # ---- one metric set: KGE joins the lead summary, MAE leaves it
    ('    lead_summary = (d.groupby("lead_hours")[["RMSE_m", "MAE_m", "NSE", "Pearson_r"]]\n'
     '                     .median()\n'
     '                     .rename(columns={"RMSE_m": "median_RMSE_m", "MAE_m": "median_MAE_m",\n'
     '                                      "NSE": "median_NSE", "Pearson_r": "median_Pearson_r"})\n'
     '                     .reset_index())',
     '    lead_summary = (d.groupby("lead_hours")[["RMSE_m", "NSE", "Pearson_r", "KGE"]]\n'
     '                     .median()\n'
     '                     .rename(columns={"RMSE_m": "median_RMSE_m", "KGE": "median_KGE",\n'
     '                                      "NSE": "median_NSE", "Pearson_r": "median_Pearson_r"})\n'
     '                     .reset_index())', 1, "lead summary columns"),
    # ---- panel (a): the MAE line goes
    ('    a.plot(lead_summary["lead_hours"], lead_summary["median_MAE_m"],\n'
     '           color=FORECAST_COLOR, linewidth=2.4, label="Median MAE")\n', '', 1, "MAE line removed"),
    # ---- panel (b): the KGE line joins, and Pearson r is named correlation
    ('    b.plot(lead_summary["lead_hours"], lead_summary["median_Pearson_r"],\n'
     '           color="#F0A830", linewidth=2.4, label="Median Pearson r")',
     '    b.plot(lead_summary["lead_hours"], lead_summary["median_Pearson_r"],\n'
     '           color="#F0A830", linewidth=2.4, label="Median correlation")\n'
     '    b.plot(lead_summary["lead_hours"], lead_summary["median_KGE"],\n'
     '           color=FORECAST_COLOR, linewidth=2.4, label="Median KGE")', 1, "KGE line added"),
]

source = open(RECIPE, encoding="utf-8").read()
for old, new, expected, name in SUBSTITUTIONS:
    count = source.count(old)
    if count != expected:
        raise Exception(f"expected {expected} '{name}' target(s), found {count}")
    source = source.replace(old, new)
print("[VERIFY] recipe_graph_figure_and_table.py applied", len(SUBSTITUTIONS), "counted substitutions:",
      ", ".join(entry[3] for entry in SUBSTITUTIONS), flush=True)

plt.show = lambda *args, **kwargs: None
try:
    exec(compile(source, RECIPE, "exec"), {"__file__": RECIPE, "__name__": "__main__"})
except SystemExit as stop:  # the recipe ends by calling sys.exit
    if stop.code not in (0, None):
        raise
plt.close("all")

for extension in (".png", ".pdf"):
    produced = os.path.join(FIGURES, "fig08_hindcast_rev8" + extension)
    if os.path.isfile(produced):
        print("[SAVED]", produced, flush=True)
print("REV8_FIG08_DONE", flush=True)
