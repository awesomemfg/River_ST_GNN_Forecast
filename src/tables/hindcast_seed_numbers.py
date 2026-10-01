"""Seed-level hindcast numbers for the Sect. 3.2 paragraph after the anchor fix, with the old run as a method check.

For each experiment (old = EXTEND_2026_AUG31_EVENTS_20260918, new = EXTEND_2026_AUG31_ANCHORFIX_20260924) it reads
outputs/event_conditioned_per_gauge_metrics.csv, origin set "all", lead 24 h, the 51 in-parish gauges, and prints:
  1. per model and seed, the median h+24 RMSE over the gauges, and the range over the three seeds;
  2. per seed, the number of gauges where the ST-GNN h+24 RMSE is below the LSTM's (same seed paired).
The old numbers must reproduce the Revision 13 sentence (ST-GNN 0.133 to 0.154 m, GRU 0.157 to 0.169 m,
LSTM 0.160 to 0.166 m, 29 to 46 gauges); only then are the new numbers quoted.

Output: ../work/hindcast_seed_numbers_rev14.csv
"""
import os

import pandas as pd

EXPERIMENTS = "project/Experiments/"
RUNS = {"old": EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918",
        "new": EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924"}
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "hindcast_seed_numbers_rev14.csv")

rows = []
for label, root in RUNS.items():
    table = pd.read_csv(os.path.join(root, "outputs", "event_conditioned_per_gauge_metrics.csv"))
    table = table[(table["origin_set"] == "all") & (table["lead_hours"] == 24)]
    print(f"== {label}: gauges {table['gauge'].nunique()}")
    medians = table.groupby(["model", "seed"])["RMSE_m"].median()
    for model in ["ST-GNN", "GRU", "LSTM"]:
        values = medians.loc[model]
        print(f"   {model:7s} seed medians " + " ".join(f"{v:.4f}" for v in values)
              + f"  range {values.min():.3f} to {values.max():.3f}")
        for seed, value in values.items():
            rows.append({"run": label, "item": "median_h24_rmse_m", "model": model, "seed": seed, "value": value})
    wide = table.pivot_table(index=["seed", "gauge"], columns="model", values="RMSE_m")
    for seed, block in wide.groupby(level="seed"):
        wins = int((block["ST-GNN"] < block["LSTM"]).sum())
        print(f"   seed {seed}: ST-GNN below LSTM at {wins} of {len(block)} gauges")
        rows.append({"run": label, "item": "stgnn_beats_lstm_gauges", "model": "ST-GNN vs LSTM", "seed": seed,
                     "value": wins})

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print("[seed-numbers] wrote", os.path.abspath(OUTPUT))
