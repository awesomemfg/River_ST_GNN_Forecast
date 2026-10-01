"""How much do the pump event and the high-stage event actually overlap?

The paper has to choose a primary event definition. Pumps running is the operational state; a stage
percentile is the hydrologic one. If the two select nearly the same origins, the choice is cosmetic.
If they are largely disjoint, they measure different things and the paper should carry both.

This reports, over 1 January to 31 August 2026 and the 51 scored gauges:

  the size of each event set, as origins and as gauge-origin pairs
  the overlap between each pair of definitions, both as a Jaccard index and as conditional shares
  how much of the pump set is also high stage, and how much of the high-stage set is also pumping

Both stage thresholds come from the record through 31 December 2025, so the definition never sees
the scored period.

Read only. Writes a small table and a report.

Usage:
  conda run -n operational python -u event_definition_overlap.py
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

ORIGIN_START = "2026-01-01 00:00"
ORIGIN_END = "2026-08-31 23:00"
LB_W = 96
EXCLUDED_GAUGES = {"MBSA4570"}
THRESHOLD_RECORD_END = "2025-12-31 23:45"
EVENT_PERCENTILES = [80.0, 90.0]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[compare]", *parts, flush=True)


frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
index_times = pd.DatetimeIndex(frame.index)
row_of = {time_point: position for position, time_point in enumerate(index_times)}

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
scored_gauges = [gauge for gauge in scored_gauges if gauge not in EXCLUDED_GAUGES]
log("scored gauges", len(scored_gauges))

stage_series = {}
for gauge in scored_gauges:
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    stage_series[gauge] = series.mask(trailing < 1e-6)
S = np.column_stack([stage_series[gauge].values for gauge in scored_gauges]).astype(np.float32)

threshold_slice = frame.index <= pd.Timestamp(THRESHOLD_RECORD_END)
thresholds = {}
for percentile in EVENT_PERCENTILES:
    values = np.full(len(scored_gauges), np.nan)
    for position, gauge in enumerate(scored_gauges):
        finite = stage_series[gauge].values[threshold_slice]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values[position] = float(np.percentile(finite, percentile))
    thresholds[percentile] = values

pump_columns = sorted(str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name)))
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)

origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="h")
origins = pd.DatetimeIndex([t for t in origins if t in row_of])
positions = np.asarray([row_of[t] for t in origins])
steps = positions[:, None] + 1 + np.arange(LB_W)[None, :]
window_observed = S[steps]
peak = np.nanmax(np.where(np.isfinite(window_observed), window_observed, -np.inf), axis=1)
log("origins", len(origins), origins.min(), "->", origins.max())

gauge_count = len(scored_gauges)
masks = {
    "pump_window": np.repeat(pump_on[steps].any(axis=1)[:, None], gauge_count, axis=1),
    "pump_at_issue": np.repeat(pump_on[positions][:, None], gauge_count, axis=1),
}
for percentile in EVENT_PERCENTILES:
    masks[f"stage_p{int(percentile)}"] = peak >= thresholds[percentile][None, :]

valid = np.isfinite(peak)
total_pairs = int(valid.sum())
log("scorable gauge-origin pairs", total_pairs)

size_rows = []
for name, mask in masks.items():
    selected = mask & valid
    size_rows.append(
        {
            "definition": name,
            "gauge_origin_pairs": int(selected.sum()),
            "share_of_pairs": float(selected.sum() / total_pairs),
            "origins_touched": int((selected.any(axis=1)).sum()),
            "share_of_origins": float((selected.any(axis=1)).sum() / len(origins)),
        }
    )
sizes = pd.DataFrame(size_rows)
log("set sizes")
log(sizes.round(4).to_string(index=False))

overlap_rows = []
names = list(masks.keys())
for first_index, first in enumerate(names):
    for second in names[first_index + 1:]:
        a = masks[first] & valid
        b = masks[second] & valid
        intersection = int((a & b).sum())
        union = int((a | b).sum())
        overlap_rows.append(
            {
                "definition_a": first,
                "definition_b": second,
                "both": intersection,
                "either": union,
                "jaccard": float(intersection / union) if union else np.nan,
                "share_of_a_also_b": float(intersection / a.sum()) if a.sum() else np.nan,
                "share_of_b_also_a": float(intersection / b.sum()) if b.sum() else np.nan,
            }
        )
overlaps = pd.DataFrame(overlap_rows)
log("overlaps")
log(overlaps.round(4).to_string(index=False))

sizes_path = os.path.join(OUTPUT_DIRECTORY, "event_definition_sizes.csv")
overlaps_path = os.path.join(OUTPUT_DIRECTORY, "event_definition_overlaps.csv")
sizes.to_csv(sizes_path, index=False)
overlaps.to_csv(overlaps_path, index=False)

report_path = os.path.join(OUTPUT_DIRECTORY, "EVENT_DEFINITION_COMPARISON.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Do the pump event and the high-stage event select the same origins?\n")
    handle.write(f"window: {ORIGIN_START} to {ORIGIN_END}, {len(origins)} origins, "
                 f"{len(scored_gauges)} gauges\n")
    handle.write(f"stage thresholds from the record through {THRESHOLD_RECORD_END}\n\n")
    handle.write("set sizes\n")
    handle.write(sizes.to_string(index=False))
    handle.write("\n\noverlaps\n")
    handle.write(overlaps.to_string(index=False))
    handle.write("\n")
log("wrote", report_path)
print("EVENT_DEFINITION_COMPARISON_DONE", flush=True)
