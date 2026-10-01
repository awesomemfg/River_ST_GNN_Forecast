"""January to August inputs for the appendix figure recipes, in the layout each one expects.

Why this exists
---------------
The appendix keeps four of its seven figures. A1 is a terrain map and does not change. B2, C1 and C3
are built from the June real-time simulation and go with it. Three need rebuilding on the new period:

  B1  map of h+24 hindcast skill       reads live_fixed_lead_metrics_inparish51.csv
  B3  map of the rainfall penalty      reads the by-seed per-gauge forcing table, already built by
                                       rainfall_and_crest_figure_inputs.py
  C2  all-51 montage of the hindcast   reads the scored hindcast trace directly, no adapter needed

So only B1 needs a file written here. Its recipe reads lowercase column names and only the leads it
plots, which is a different layout from the scored per-gauge tables:

  gauge, lead_hours, rmse_m, correlation, nse, kge

and the January to June file holds 102 rows, which is 51 gauges at the 6 h and 24 h leads. Each metric
is averaged over the three training seeds first, which is the order every caption in the manuscript
states.

Output: ../outputs/figure_inputs_jan_aug/appendixB1_live_fixed_lead_metrics_inparish51_jan_aug.csv

Usage:
  conda run -n operational python -u appendix_figure_inputs.py
"""
import json
import os

import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUTS = os.path.join(EXPERIMENT_ROOT, "outputs")
TARGET = os.path.join(OUTPUTS, "figure_inputs_jan_aug")
PER_GAUGE = os.path.join(OUTPUTS, "event_conditioned_per_gauge_metrics.csv")
INPARISH = ("project/Experiments/"
            "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json")

LEADS = [6, 24]
RENAME = {"gauge": "gauge", "RMSE_m": "rmse_m", "Pearson_r": "correlation",
          "NSE": "nse", "KGE": "kge"}

os.makedirs(TARGET, exist_ok=True)
if not os.path.exists(PER_GAUGE):
    raise SystemExit("[FATAL] missing " + PER_GAUGE)

gauges = sorted(json.load(open(INPARISH))["nodes"])
frame = pd.read_csv(PER_GAUGE)
frame = frame[(frame["origin_set"] == "all") & (frame["model"] == "ST-GNN")
              & frame["lead_hours"].isin(LEADS) & frame["gauge"].isin(gauges)]
if frame.empty:
    raise SystemExit("[FATAL] no ST-GNN rows at leads " + str(LEADS))

# Average each gauge over the three training seeds, then rename to the recipe's own columns.
averaged = frame.groupby(["gauge", "lead_hours"], as_index=False)[
    ["RMSE_m", "Pearson_r", "NSE", "KGE"]].mean()
averaged = averaged.rename(columns=RENAME)
averaged["lead_hours"] = averaged["lead_hours"].astype(int)
averaged = averaged[["gauge", "lead_hours", "rmse_m", "correlation", "nse", "kge"]]
averaged = averaged.sort_values(["gauge", "lead_hours"]).reset_index(drop=True)

path = os.path.join(TARGET, "appendixB1_live_fixed_lead_metrics_inparish51_jan_aug.csv")
averaged.to_csv(path, index=False)

print("[B1] rows", len(averaged), "| gauges", averaged["gauge"].nunique(),
      "| leads", sorted(averaged["lead_hours"].unique()), flush=True)
for lead in LEADS:
    block = averaged[averaged["lead_hours"] == lead]
    print(f"[B1] h+{lead}: median RMSE {block['rmse_m'].median():.4f} m, "
          f"median NSE {block['nse'].median():.4f}, median correlation {block['correlation'].median():.4f}",
          flush=True)
print("[B1] the h+24 row must match Table 4's ST-GNN column, 0.1459 m and 0.7207", flush=True)
print("[B1] wrote", path, flush=True)
print("APPENDIX_INPUTS_DONE", flush=True)
