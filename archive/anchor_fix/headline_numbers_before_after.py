"""Put the headline numbers of the Results section side by side: Revision 13 (before the anchor fix) and Revision 14.

Rule of the paper: each metric is averaged per gauge over the three seeds, then the median is taken over the 51
in-parish gauges. The event-conditioned summaries already follow that rule, so they are read directly. The simulated
forecast tables hold one row per model, seed, gauge and lead, so the rule is applied here.

Tables compared (old source -> new source):
  hindcast, all origins and event-only   EXTEND_*/outputs/event_conditioned_summary.csv (origin_set all, pump_at_issue)
  simulated forecast, all origins        EXTEND_*/outputs/summary_hrrr_jan_aug_postproc/fixed_lead_per_gauge.csv
  simulated forecast, event-only         Revision 9 or 14 work/event_model_comparison/event_replay_fixed_lead_per_gauge.csv
  simulated forecast, rise groups        EXTEND_*/outputs/summary_hrrr_jan_aug_postproc/rise_strata.csv (seed mean)

Output: ../work/headline_numbers_rev13_vs_rev14.csv, one row per (table, model, lead or group, metric), with the
old and new values and the change. Values are not rounded here; rounding is left to the manuscript.
"""
import os

import pandas as pd

EXPERIMENTS = "project/Experiments/"
PAPER = "project/manuscript/"
SOURCES = {
    "old": {
        "experiment": EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918/outputs/",
        "event_work": PAPER + "20260921_Revision_9/work/event_model_comparison/",
    },
    "new": {
        "experiment": EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924/outputs/",
        "event_work": PAPER + "20260924_Revision_14/work/event_model_comparison/",
    },
}
OUTPUT = PAPER + "20260924_Revision_14/work/headline_numbers_rev13_vs_rev14.csv"
METRICS = ["RMSE_m", "Pearson_r", "NSE", "KGE"]


def hindcast(folder):
    summary = pd.read_csv(folder["experiment"] + "event_conditioned_summary.csv")
    summary = summary[summary["origin_set"].isin(["all", "pump_at_issue"])]
    long = summary.melt(id_vars=["model", "origin_set", "lead_hours"], value_vars=METRICS, var_name="metric")
    long["table"] = "hindcast_" + long.pop("origin_set")
    return long.rename(columns={"lead_hours": "lead_or_group"})


def seed_mean_then_median(per_gauge, table):
    by_gauge = per_gauge.groupby(["model", "gauge", "lead_hours"])[METRICS].mean().reset_index()
    median = by_gauge.groupby(["model", "lead_hours"])[METRICS].median().reset_index()
    long = median.melt(id_vars=["model", "lead_hours"], value_vars=METRICS, var_name="metric")
    long["table"] = table
    return long.rename(columns={"lead_hours": "lead_or_group"})


def simulated(folder):
    all_origins = pd.read_csv(folder["experiment"] + "summary_hrrr_jan_aug_postproc/fixed_lead_per_gauge.csv")
    events = pd.read_csv(folder["event_work"] + "event_replay_fixed_lead_per_gauge.csv")
    return pd.concat([seed_mean_then_median(all_origins, "simulated_all"),
                      seed_mean_then_median(events, "simulated_event")])


def rise_groups(folder):
    strata = pd.read_csv(folder["experiment"] + "summary_hrrr_jan_aug_postproc/rise_strata.csv")
    mean = strata.groupby(["model", "rise_group"])[["median_rmse24_m", "beats_persist_frac"]].mean().reset_index()
    long = mean.melt(id_vars=["model", "rise_group"], var_name="metric")
    long["table"] = "simulated_rise_groups"
    return long.rename(columns={"rise_group": "lead_or_group"})


frames = {}
for label, folder in SOURCES.items():
    frame = pd.concat([hindcast(folder), simulated(folder), rise_groups(folder)])
    frame["lead_or_group"] = frame["lead_or_group"].astype(str)
    frames[label] = frame.rename(columns={"value": label})

keys = ["table", "model", "lead_or_group", "metric"]
merged = frames["old"].merge(frames["new"], on=keys, how="outer")
merged["change"] = merged["new"] - merged["old"]
merged = merged.sort_values(keys)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
merged.to_csv(OUTPUT, index=False)
print(f"[headline] {len(merged)} rows, {int((merged['change'].abs() > 0.0005).sum())} move by more than 0.0005")
print("[headline] wrote", OUTPUT)
