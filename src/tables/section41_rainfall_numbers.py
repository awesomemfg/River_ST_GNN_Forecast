"""Numbers behind the Sect. 4.1 rainfall paragraphs, old run against the anchor-fix run.

Per run (old = EXTEND_2026_AUG31_EVENTS_20260918, new = EXTEND_2026_AUG31_ANCHORFIX_20260924), from outputs/:
  1. event_table6_skill_by_forcing_jan_aug_members.csv: median h+24 RMSE, NSE and KGE of every forcing (Table B2),
     all origins and event-only (pump_at_issue), with a flag for forcings whose all-time RMSE is below the
     issue-time HRRR;
  2. event_fig13_per_gauge_h24_jan_aug.csv: the per-gauge NSE after the seed mean, as Figure 14 draws it, and its
     25th and 75th percentiles for observed rain and issue-time HRRR, all origins and event-only.
The old numbers must reproduce the Revision 13 text before the new ones are quoted.

Output: ../work/sect41_rainfall_numbers_rev14.csv
"""
import os

import numpy as np
import pandas as pd

EXPERIMENTS = "project/Experiments/"
RUNS = {"old": EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918",
        "new": EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924"}
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "sect41_rainfall_numbers_rev14.csv")
HRRR = "HRRR, issue time"
pd.set_option("display.width", 200)

rows = []
for label, root in RUNS.items():
    outputs = os.path.join(root, "outputs")
    table = pd.read_csv(os.path.join(outputs, "event_table6_skill_by_forcing_jan_aug_members.csv"))
    table = table[table["origin_set"].isin(["all", "pump_at_issue"])]
    wide = table.pivot_table(index="forcing", columns="origin_set", values=["RMSE_m", "NSE", "KGE"])
    hrrr_all = wide.loc[HRRR, ("RMSE_m", "all")]
    wide[("below_hrrr_all", "")] = wide[("RMSE_m", "all")] < hrrr_all
    print(f"== {label}: Table B2, median h+24")
    print(wide.round(4).to_string())
    for forcing, values in wide.iterrows():
        for (metric, origin_set), value in values.items():
            rows.append({"run": label, "item": "tableB2", "forcing": forcing, "metric": metric,
                         "origin_set": origin_set, "value": float(value)})

    per_gauge = pd.read_csv(os.path.join(outputs, "event_fig13_per_gauge_h24_jan_aug.csv"))
    seed_mean = per_gauge.groupby(["forcing", "origin_set", "gauge"], as_index=False)[["RMSE_m", "NSE"]].mean()
    # 2026-09-25: RMSE added next to NSE, and the six-model mean added to observed rain and issue-time HRRR.
    for metric in ["RMSE_m", "NSE"]:
        print(f"== {label}: Figure 14 {metric} median and interquartile range, h+24")
        for forcing in ["Observed rain", HRRR, "Ensemble mean, 6"]:
            for origin_set in ["all", "pump_at_issue"]:
                values = seed_mean[(seed_mean["forcing"] == forcing)
                                   & (seed_mean["origin_set"] == origin_set)][metric].dropna().to_numpy()
                low, middle, high = np.percentile(values, [25, 50, 75])
                print(f"   {forcing:18s} {origin_set:14s} median {middle:.3f}  {low:.3f} to {high:.3f}  "
                      f"(n {len(values)})")
                rows += [{"run": label, "item": f"fig14_{metric}_q25", "forcing": forcing,
                          "origin_set": origin_set, "value": low},
                         {"run": label, "item": f"fig14_{metric}_q75", "forcing": forcing,
                          "origin_set": origin_set, "value": high}]

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print("[sect41] wrote", os.path.abspath(OUTPUT))
