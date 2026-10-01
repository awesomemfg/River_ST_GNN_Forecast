"""Paired bootstrap of the graph effect and the recurrent-cell effect, January to August 2026.

Why paired
----------
A median of one model against a median of another mixes in the spread between training runs, which
is about 0.022 m for this model. Pairing on the gauge cancels that spread, because both models are
compared on the same gauge, and a paired comparison with three seeds carries more information than
an unpaired comparison with many more. This is the same construction the manuscript already uses for
the graph effect, where gauges are resampled to give a 95 percent interval.

Two differences are measured, both in meters, both positive when the better model is on the right:

  graph effect  GRU minus ST-GNN     identical models apart from message passing
  cell effect   LSTM minus GRU       identical models apart from the recurrent cell

The GRU here is the System B ST-GNN fitted on an identity adjacency, so the graph message reduces to
the gauge's own hidden state and no information passes between gauges.

Per-gauge RMSE is averaged over the three training seeds first, then gauges are resampled with
replacement, which is the aggregation order locked in revision 7.

Usage:
  conda run -n operational python -u bootstrap_graph_and_cell_effects.py
"""
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

ARMS = [("", "observed"), ("_hrrr", "forecast")]
COMPARISONS = [
    ("graph effect (GRU minus ST-GNN)", "GRU", "ST-GNN"),
    ("cell effect (LSTM minus GRU)", "LSTM", "GRU"),
    ("model effect (persistence minus ST-GNN)", "persistence", "ST-GNN"),
]
RESAMPLES = 10000
SEED = 20260918

rng = np.random.default_rng(SEED)
rows = []

for tag, arm in ARMS:
    path = os.path.join(OUTPUT_DIRECTORY, f"event_conditioned_per_gauge_metrics{tag}.csv")
    frame = pd.read_csv(path)
    seed_mean = (
        frame.groupby(["model", "origin_set", "lead_hours", "gauge"], as_index=False)["RMSE_m"]
        .mean()
    )
    for origin_set in sorted(seed_mean["origin_set"].unique()):
        for lead_hours in sorted(seed_mean["lead_hours"].unique()):
            block = seed_mean[
                (seed_mean["origin_set"] == origin_set) & (seed_mean["lead_hours"] == lead_hours)
            ]
            wide = block.pivot(index="gauge", columns="model", values="RMSE_m").dropna()
            if wide.empty:
                continue
            for label, worse_model, better_model in COMPARISONS:
                if worse_model not in wide.columns or better_model not in wide.columns:
                    continue
                differences = (wide[worse_model] - wide[better_model]).to_numpy(dtype=float)
                gauge_count = len(differences)
                if gauge_count == 0:
                    continue
                draws = rng.integers(0, gauge_count, size=(RESAMPLES, gauge_count))
                medians = np.median(differences[draws], axis=1)
                rows.append(
                    {
                        "arm": arm,
                        "origin_set": origin_set,
                        "lead_hours": lead_hours,
                        "comparison": label,
                        "gauges": gauge_count,
                        "median_difference_m": float(np.median(differences)),
                        "ci_low_m": float(np.percentile(medians, 2.5)),
                        "ci_high_m": float(np.percentile(medians, 97.5)),
                        "gauges_favoring_better_model": int((differences > 0).sum()),
                    }
                )

table = pd.DataFrame(rows)
table["excludes_zero"] = (table["ci_low_m"] > 0) | (table["ci_high_m"] < 0)
table_path = os.path.join(OUTPUT_DIRECTORY, "graph_and_cell_effect_bootstrap.csv")
table.to_csv(table_path, index=False)
print("[bootstrap] wrote", table_path, table.shape, flush=True)

report_path = os.path.join(OUTPUT_DIRECTORY, "GRAPH_AND_CELL_EFFECT_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Paired bootstrap of the graph effect and the cell effect\n")
    handle.write("1 January to 31 August 2026, 5,832 hourly origins, 51 scored gauges\n")
    handle.write(f"resamples: {RESAMPLES}, random seed: {SEED}\n")
    handle.write("positive values mean the model named second is better\n\n")
    for _, arm in ARMS:
        for label, _, _ in COMPARISONS:
            block = table[(table["arm"] == arm) & (table["comparison"] == label)]
            if block.empty:
                continue
            handle.write(f"--- {arm} rain, {label} ---\n")
            handle.write(
                block[
                    [
                        "origin_set",
                        "lead_hours",
                        "median_difference_m",
                        "ci_low_m",
                        "ci_high_m",
                        "gauges_favoring_better_model",
                        "excludes_zero",
                    ]
                ].to_string(index=False)
            )
            handle.write("\n\n")
print("[bootstrap] wrote", report_path, flush=True)
print("BOOTSTRAP_DONE", flush=True)
