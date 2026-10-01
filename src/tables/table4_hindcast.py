"""Table 4 of Revision 14: hindcast RMSE at four leads and the 24 h correlation, NSE and KGE, after the anchor fix.

Source: EXTEND_2026_AUG31_ANCHORFIX_20260924/outputs/event_conditioned_summary.csv, origin set "all" (5,832 hourly
origins, 1 January to 31 August 2026). Its values are already the paper's rule: each metric averaged per gauge over
the three seeds, then the median over the 51 in-parish gauges. Values are rounded to three decimals, as in the
Revision 13 table. The two header rows copy the layout Farid gave on 2026-09-25.

Output: ../work/hindcast_table4_all_leads_rev14.csv
"""
import csv
import os

import pandas as pd

SUMMARY = ("project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924/"
           "outputs/event_conditioned_summary.csv")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "hindcast_table4_all_leads_rev14.csv")
MODELS = [("ST-GNN", "ST-GNN"), ("LSTM", "LSTM"), ("GRU", "GRU"), ("Persistence", "persistence")]

summary = pd.read_csv(SUMMARY)
summary = summary[summary["origin_set"] == "all"].set_index(["model", "lead_hours"])

rows = [
    ["Model", "RMSE", "", "", "", "At the 24 h lead", "", ""],
    ["", "3 h", "6 h", "12 h", "24 h", "Corr.", "NSE", "KGE"],
]
for label, key in MODELS:
    rmse = [f"{summary.loc[(key, lead), 'RMSE_m']:.3f}" for lead in (3, 6, 12, 24)]
    at24 = summary.loc[(key, 24)]
    rows.append([label] + rmse + [f"{at24['Pearson_r']:.3f}", f"{at24['NSE']:.3f}", f"{at24['KGE']:.3f}"])

with open(OUTPUT, "w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerows(rows)
for row in rows:
    print("\t".join(row))
print("[table4] wrote", os.path.abspath(OUTPUT))
