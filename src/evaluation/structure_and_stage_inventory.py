"""Inventory the structure telemetry and the observed stage record for 1 January to 31 August 2026.

This answers three questions before any model is run again:

  1. Which gate, pump, and weir columns exist, in which basin, and how completely are they reported
     between 1 January and 31 August 2026.
  2. How often a gate or a pump reads on, and how that would translate into events if an event is
     "any structure in the parish reads on".
  3. What the 80th, 90th, and 95th percentile of the observed stage record is at every scored gauge,
     and how many hours sit above each of them.

Column conventions, taken from the deployed trainer and verified in
project/Experiments/STRUCT_REPRESENTATION_AB_20260727/scripts/verify_struct_claims.py:
  ST<n>_status   gate status, binary, 1 = open
  PD<n>_status   pump status, binary, 1 = running
  WS<n>_elev_ft  weir crest elevation in feet
The first two characters of a column name are the basin code (BC, BM, HB, MB).

Read only. Writes CSV summaries and a report next to this script's experiment.

Usage:
  conda run -n operational python -u structure_and_stage_inventory.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

MATRIX = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/in_parish_gauges.json"
)
WINDOW_START = "2026-01-01 00:00"
WINDOW_END = "2026-08-31 23:45"
PERCENTILES = [80.0, 90.0, 95.0]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[inventory]", *parts, flush=True)


frame = pd.read_pickle(MATRIX)
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
frame = frame.sort_index()
log("matrix", frame.shape, frame.index.min(), "->", frame.index.max())

window = frame.loc[WINDOW_START:WINDOW_END]
log("window", window.shape, window.index.min(), "->", window.index.max())

columns = [str(name) for name in window.columns]
gate_columns = sorted(name for name in columns if re.search(r"ST\d+_status$", name))
pump_columns = sorted(name for name in columns if re.search(r"PD\d+_status$", name))
weir_columns = sorted(name for name in columns if re.search(r"WS\d+_elev_ft$", name))
log("gate columns", len(gate_columns), "pump columns", len(pump_columns), "weir columns", len(weir_columns))

structure_rows = []
for kind, names in (("gate", gate_columns), ("pump", pump_columns), ("weir", weir_columns)):
    for name in names:
        series = pd.to_numeric(window[name], errors="coerce")
        reported = series.notna()
        row = {
            "kind": kind,
            "column": name,
            "basin": name[:2],
            "steps": int(len(series)),
            "steps_reported": int(reported.sum()),
            "reported_fraction": float(reported.mean()),
            "first_reported": str(series[reported].index.min()) if reported.any() else "",
            "last_reported": str(series[reported].index.max()) if reported.any() else "",
            "min": float(np.nanmin(series.values)) if reported.any() else np.nan,
            "max": float(np.nanmax(series.values)) if reported.any() else np.nan,
        }
        if kind in ("gate", "pump"):
            on = series.fillna(0.0) > 0.5
            row["steps_on"] = int(on.sum())
            row["on_fraction"] = float(on.mean())
            row["distinct_values"] = json.dumps(
                sorted(float(v) for v in pd.unique(series.dropna().values))[:10]
            )
        structure_rows.append(row)

structure_table = pd.DataFrame(structure_rows)
structure_path = os.path.join(OUTPUT_DIRECTORY, "structure_column_inventory_2026_jan_aug.csv")
structure_table.to_csv(structure_path, index=False)
log("wrote", structure_path)

# Parish-wide "any structure on" series, at 15 min resolution and rolled up to hourly origins.
gate_any = pd.Series(False, index=window.index)
pump_any = pd.Series(False, index=window.index)
for name in gate_columns:
    gate_any = gate_any | (pd.to_numeric(window[name], errors="coerce").fillna(0.0) > 0.5)
for name in pump_columns:
    pump_any = pump_any | (pd.to_numeric(window[name], errors="coerce").fillna(0.0) > 0.5)
any_on = gate_any | pump_any

hourly = pd.DataFrame(
    {
        "gate_any_on": gate_any.astype(int),
        "pump_any_on": pump_any.astype(int),
        "any_on": any_on.astype(int),
    }
)
hourly_origin = hourly.resample("h").max()
hourly_path = os.path.join(OUTPUT_DIRECTORY, "structure_any_on_hourly_2026_jan_aug.csv")
hourly_origin.to_csv(hourly_path)
log("wrote", hourly_path)

# Run lengths of the "any structure on" condition, so the event duration question has numbers.
def run_lengths(flag_series):
    values = flag_series.to_numpy(dtype=bool)
    lengths = []
    count = 0
    for value in values:
        if value:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return np.asarray(lengths, dtype=float)

report_lines = []
for label, series in (
    ("gate_any_on", hourly_origin["gate_any_on"]),
    ("pump_any_on", hourly_origin["pump_any_on"]),
    ("any_on", hourly_origin["any_on"]),
):
    lengths = run_lengths(series > 0)
    report_lines.append(
        f"{label}: hours on {int((series > 0).sum())} of {len(series)} "
        f"({float((series > 0).mean()) * 100:.1f} %), episodes {len(lengths)}, "
        f"median episode {np.median(lengths) if len(lengths) else float('nan'):.1f} h, "
        f"p90 episode {np.percentile(lengths, 90) if len(lengths) else float('nan'):.1f} h, "
        f"longest {lengths.max() if len(lengths) else float('nan'):.1f} h"
    )
    log(report_lines[-1])

# Observed stage percentiles at the 51 scored in-parish gauges.
with open(INPARISH_MANIFEST, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if isinstance(manifest, dict):
    for key in ("inparish51", "gauges", "nodes", "stations"):
        if key in manifest:
            scored_gauges = [str(value) for value in manifest[key]]
            break
    else:
        scored_gauges = [str(value) for value in list(manifest.values())[0]]
else:
    scored_gauges = [str(value) for value in manifest]
log("scored gauges", len(scored_gauges))

stage_rows = []
for gauge in scored_gauges:
    column = gauge + "_stage_ft"
    if column not in window.columns:
        stage_rows.append({"gauge": gauge, "column": column, "present": False})
        continue
    series = pd.to_numeric(window[column], errors="coerce")
    valid = series.dropna()
    row = {
        "gauge": gauge,
        "column": column,
        "present": True,
        "steps_reported": int(len(valid)),
        "reported_fraction": float(series.notna().mean()),
    }
    for percentile in PERCENTILES:
        threshold = float(np.percentile(valid.values, percentile)) if len(valid) else np.nan
        row[f"p{int(percentile)}_stage_ft"] = threshold
        row[f"hours_above_p{int(percentile)}"] = (
            float((valid > threshold).sum()) / 4.0 if len(valid) else np.nan
        )
    stage_rows.append(row)

stage_table = pd.DataFrame(stage_rows)
stage_path = os.path.join(OUTPUT_DIRECTORY, "stage_percentiles_2026_jan_aug.csv")
stage_table.to_csv(stage_path, index=False)
log("wrote", stage_path)

report_path = os.path.join(OUTPUT_DIRECTORY, "INVENTORY_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Structure telemetry and stage percentile inventory\n")
    handle.write(f"matrix: {MATRIX}\n")
    handle.write(f"window: {WINDOW_START} to {WINDOW_END} UTC\n")
    handle.write(f"gate columns: {len(gate_columns)}\n")
    handle.write(f"pump columns: {len(pump_columns)}\n")
    handle.write(f"weir columns: {len(weir_columns)}\n")
    for line in report_lines:
        handle.write(line + "\n")
    handle.write(f"scored gauges found in matrix: {int(stage_table['present'].sum())}\n")
log("wrote", report_path)
print("INVENTORY_DONE", flush=True)
