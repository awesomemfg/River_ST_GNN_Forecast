"""Appendix Tables A1, B1, and B2 rebuilt from the anchor-fix rerun, for checking the MFG3 appendix (2026-09-28).

All three use the hindcast origins of 1 January to 31 August 2026 (5,832 hourly origins), all origins, 51 in-parish
gauges. Values are medians across gauges; A1 and B2 average each gauge over the three seeds first, B1 keeps each seed.
  A1  six graphs, RMSE, correlation, NSE, KGE at h+3, h+6, h+12, h+24   <- event_conditioned_summary_graphs_jan_aug.csv
  B1  each trained model and seed at h+24, plus persistence              <- event_conditioned_per_gauge_metrics.csv
  B2  every rainfall forcing at h+24                                     <- event_table6_skill_by_forcing_jan_aug_members.csv
Output: ../work/appendix_tables_rev14/tableA1.csv, tableB1.csv, tableB2.csv, and a printed copy of each.
"""
import os

import pandas as pd

OUTPUTS = "project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924/outputs/"
TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "appendix_tables_rev14")
METRICS = ["RMSE_m", "Pearson_r", "NSE", "KGE"]
GRAPH_NAMES = {"gObsLagLead": "Observation lead-lag (selected)", "gObsBasin": "Observation, within-basin",
               "gDem": "DEM downslope", "gHecras": "HEC-RAS hydraulic", "gDemBasin": "DEM, within-basin",
               "gIdentity": "Identity (no exchange between gauges)"}
os.makedirs(TARGET, exist_ok=True)
pd.set_option("display.width", 200)

graphs = pd.read_csv(OUTPUTS + "event_conditioned_summary_graphs_jan_aug.csv")
graphs = graphs[(graphs["origin_set"] == "all") & graphs["model"].isin(GRAPH_NAMES)]
table_a1 = graphs.assign(graph=graphs["model"].map(GRAPH_NAMES),
                         order=graphs["model"].map({key: i for i, key in enumerate(GRAPH_NAMES)}))
table_a1 = table_a1.sort_values(["order", "lead_hours"])[["graph", "lead_hours"] + METRICS].round(3)
table_a1.to_csv(os.path.join(TARGET, "tableA1.csv"), index=False)
print("== Table A1\n" + table_a1.to_string(index=False))

per_gauge = pd.read_csv(OUTPUTS + "event_conditioned_per_gauge_metrics.csv")
per_gauge = per_gauge[(per_gauge["origin_set"] == "all") & (per_gauge["lead_hours"] == 24)]
table_b1 = per_gauge.groupby(["model", "seed"])[METRICS].median().reset_index().round(3)
table_b1.to_csv(os.path.join(TARGET, "tableB1.csv"), index=False)
print("== Table B1\n" + table_b1.to_string(index=False))

forcings = pd.read_csv(OUTPUTS + "event_table6_skill_by_forcing_jan_aug_members.csv")
table_b2 = forcings[forcings["origin_set"] == "all"][["forcing"] + METRICS].round(3)
table_b2.to_csv(os.path.join(TARGET, "tableB2.csv"), index=False)
print("== Table B2 (all origins, h+24; persistence is 0.171 / 0.821 / 0.641 / 0.821 from Table 4)\n"
      + table_b2.to_string(index=False))
