"""Numbers behind the two Sect. 3.2 paragraphs that describe Figure 8, old run against the anchor-fix run.

Input per run: outputs/figure_inputs_jan_aug/fig08_lead_metrics_mean3_jan_aug.csv (per gauge and lead, the mean over
the three seeds), which is what Figure 8 draws. Printed per run:
  1. median RMSE, correlation, NSE and KGE over the 51 gauges at h+3, h+6, h+12 and h+24, the drop of each skill
     metric from h+3 to h+24, and the RMSE growth per hour between those leads;
  2. at h+24: median RMSE, gauges with RMSE below 0.25 m, gauges with NSE above 0, and the five highest RMSE gauges.

Output: ../work/figure8_text_numbers_rev14.csv
"""
import os

import pandas as pd

EXPERIMENTS = "project/Experiments/"
RUNS = {"old": EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918",
        "new": EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924"}
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "figure8_text_numbers_rev14.csv")
LEADS = [3.0, 6.0, 12.0, 24.0]
METRICS = ["RMSE_m", "Pearson_r", "NSE", "KGE"]

rows = []
for label, root in RUNS.items():
    table = pd.read_csv(os.path.join(root, "outputs", "figure_inputs_jan_aug", "fig08_lead_metrics_mean3_jan_aug.csv"))
    medians = table.groupby("lead_hours")[METRICS].median().loc[LEADS]
    print(f"== {label}: gauges {table['node'].nunique()}")
    print(medians.round(3).to_string())
    for metric in ["Pearson_r", "NSE", "KGE"]:
        drop = medians.loc[3.0, metric] - medians.loc[24.0, metric]
        print(f"   {metric} drop h+3 to h+24: {drop:.3f}")
        rows.append({"run": label, "item": f"{metric}_drop_h3_h24", "value": drop})
    for start, end in zip(LEADS[:-1], LEADS[1:]):
        rate = (medians.loc[end, "RMSE_m"] - medians.loc[start, "RMSE_m"]) / (end - start)
        print(f"   RMSE growth h+{start:.0f} to h+{end:.0f}: {1000 * rate:.1f} mm per hour")
        rows.append({"run": label, "item": f"rmse_growth_mm_per_h_{start:.0f}_{end:.0f}", "value": 1000 * rate})
    day = table[table["lead_hours"] == 24.0]
    below = int((day["RMSE_m"] < 0.25).sum())
    positive = int((day["NSE"] > 0).sum())
    print(f"   h+24 median RMSE {day['RMSE_m'].median():.4f} m, RMSE below 0.25 m at {below} of {len(day)}, "
          f"NSE above 0 at {positive} of {len(day)}")
    print("   highest h+24 RMSE:", ", ".join(f"{r.node} {r.RMSE_m:.3f}" for r in
                                             day.nlargest(5, "RMSE_m").itertuples()))
    rows += [{"run": label, "item": "h24_median_rmse_m", "value": day["RMSE_m"].median()},
             {"run": label, "item": "h24_gauges_rmse_below_0.25", "value": below},
             {"run": label, "item": "h24_gauges_nse_positive", "value": positive}]
    for lead in LEADS:
        for metric in METRICS:
            rows.append({"run": label, "item": f"median_{metric}_h{lead:.0f}", "value": medians.loc[lead, metric]})

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print("[figure8] wrote", os.path.abspath(OUTPUT))
