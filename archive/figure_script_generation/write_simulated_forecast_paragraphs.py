"""Write the Section 3.3 replay paragraphs from the postprocessing summary.

Reads the per-gauge file written by summarize_simulated_forecast.py. For each model and
lead, it averages the three training seeds at each gauge, then takes the median across
the 51 in-parish gauges. That is the same order used for Tables 3 and 4.

The identity-graph comparison is not in these archives, so these paragraphs do not
repeat the raw-hindcast identity intervals.

Usage:
  conda run -n operational python -u write_simulated_forecast_paragraphs.py
"""
import os

import pandas as pd

SUMMARY = (
    "project/Experiments/"
    "EXTEND_2026_AUG31_EVENTS_20260918/outputs/summary_hrrr_jan_aug_postproc/"
    "fixed_lead_per_gauge.csv"
)
OUT = (
    "project/manuscript/"
    "20260921_Revision_9/work/replay_paragraphs.txt"
)

if not os.path.isfile(SUMMARY):
    raise SystemExit("[FATAL] replay summary is not ready: " + SUMMARY)

frame = pd.read_csv(SUMMARY)
print("[load]", SUMMARY, "rows", len(frame), "models", sorted(frame["model"].unique()), flush=True)

grouped = frame.groupby(["model", "gauge", "lead_hours"], as_index=False)[
    ["RMSE_m", "NSE", "Pearson_r", "KGE"]
].mean()
median = grouped.groupby(["model", "lead_hours"]).median(numeric_only=True)


def cell(model, lead, column):
    value = float(median.loc[(model, lead), column])
    return f"{value:.3f}"


stgnn_24 = cell("ST-GNN", 24, "RMSE_m")
lstm_24 = cell("LSTM", 24, "RMSE_m")
gru_24 = cell("GRU", 24, "RMSE_m")
persist_24 = cell("Persistence", 24, "RMSE_m")

first = (
    "Section 3.2 gives the models observed rainfall over the forecast horizon, with "
    "postprocessing off. A real forecast does not have that rainfall, and the operational "
    "system does not publish the raw output. We therefore replayed the same 5,832 origins "
    "with the rainfall forecast available at each issue time and with the postprocessing in "
    "Sect. 2.5.2. The median h+24 RMSE of the ST-GNN was "
    + stgnn_24
    + " m, against 0.146 m in the observed-rainfall hindcast. Its median NSE was "
    + cell("ST-GNN", 24, "NSE")
    + ", its median correlation was "
    + cell("ST-GNN", 24, "Pearson_r")
    + ", and its median KGE was "
    + cell("ST-GNN", 24, "KGE")
    + ". At h+3, h+6 and h+12 the median RMSE was "
    + cell("ST-GNN", 3, "RMSE_m")
    + ", "
    + cell("ST-GNN", 6, "RMSE_m")
    + " and "
    + cell("ST-GNN", 12, "RMSE_m")
    + " m."
)

second = (
    "With the operational postprocessing the trained models stay close. At h+3, h+6, h+12 "
    "and h+24 the median RMSE was "
    + cell("ST-GNN", 3, "RMSE_m")
    + ", "
    + cell("ST-GNN", 6, "RMSE_m")
    + ", "
    + cell("ST-GNN", 12, "RMSE_m")
    + " and "
    + stgnn_24
    + " m for the ST-GNN, against "
    + cell("LSTM", 3, "RMSE_m")
    + ", "
    + cell("LSTM", 6, "RMSE_m")
    + ", "
    + cell("LSTM", 12, "RMSE_m")
    + " and "
    + lstm_24
    + " m for the LSTM, and "
    + cell("GRU", 3, "RMSE_m")
    + ", "
    + cell("GRU", 6, "RMSE_m")
    + ", "
    + cell("GRU", 12, "RMSE_m")
    + " and "
    + gru_24
    + " m for the GRU. Persistence, which holds the issue-time stage, had a median h+24 "
    "RMSE of "
    + persist_24
    + " m and a median KGE of "
    + cell("Persistence", 24, "KGE")
    + ". At h+24 the ST-GNN NSE was "
    + cell("ST-GNN", 24, "NSE")
    + " and its KGE was "
    + cell("ST-GNN", 24, "KGE")
    + ", against "
    + cell("LSTM", 24, "NSE")
    + " and "
    + cell("LSTM", 24, "KGE")
    + " for the LSTM and "
    + cell("GRU", 24, "NSE")
    + " and "
    + cell("GRU", 24, "KGE")
    + " for the GRU."
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    handle.write(first + "\n\n" + second + "\n")
print("[save]", OUT, flush=True)
print(first, flush=True)
print(second, flush=True)
print("REPLAY_PARAGRAPHS_DONE", flush=True)
