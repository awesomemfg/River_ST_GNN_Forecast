"""Per-gauge h+24 skill of every reduced gauge network, on event origins as well as all origins.

This rebuilds the table behind Figure 16 with the event definitions added. The published version
answers "how many in-parish gauges does the forecast need" over every origin. The event version asks
the same question during pump operation and at high stage, which is when the answer matters.

Construction follows the published one exactly. Within a draw, the ten-gauge arm defines a fixed core
and every larger size of that draw is scored on those same ten gauges, so the sizes are comparable.
The manifest confirms the nesting: each draw's ten core gauges are contained in its 20-, 30- and
40-gauge arms. The 51-gauge in-parish parent and the 68-gauge control are scored on the same core.

  size 68  the full control, from ../runs
  size 51  inparish51, from ../runs_ladder
  size 40, 30, 20, 10  ipsub<size>_d<draw>, from ../runs_ladder

Origin sets are the ones agreed for this experiment: all, pump_at_issue, stage_p80 and stage_p90,
with both stage thresholds taken from the record through 31 December 2025.

No model is run again. This reads the January to August archives.

Output: ../outputs/event_network_levels_h24.csv, with one row per origin set, size, draw, seed and
gauge, carrying RMSE, correlation, NSE and KGE.

Usage:
  conda run -n operational python -u gauge_network_skill.py
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
CONTROL_RUNS = os.path.join(EXPERIMENT_ROOT, "runs")
LADDER_RUNS = os.path.join(EXPERIMENT_ROOT, "runs_ladder")
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

BOUNDARY_MANIFEST = (
    "project/hpc/Experiments/"
    "SYSTEM_B_BOUNDARY_INPARISH_20260914/frozen_assets/ASSET_MANIFEST.json"
)
MATRIX = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)

SEEDS = [101, 202, 303]
DRAWS = [1, 2, 3]
REDUCED_SIZES = [40, 30, 20, 10]
LB_W = 96
QUANTILE = 0.8
FT2M = 0.3048
THRESHOLD_RECORD_END = "2025-12-31 23:45"
EVENT_PERCENTILES = [80.0, 90.0]
MINIMUM_SAMPLES = 30

# The rainfall forcing is an argument because Section 4.2 reports the network experiments under both
# observed rain and the rainfall forecast. The defaults reproduce the observed-rain table exactly.
parser = argparse.ArgumentParser()
parser.add_argument("--ladder-dir", default=LADDER_RUNS)
parser.add_argument("--forcing", default="observed", help="the token in the archive file names")
parser.add_argument("--tag", default="", help="suffix for the output file")
arguments = parser.parse_args()
LADDER_RUNS = arguments.ladder_dir
FORCING = arguments.forcing
TAG = arguments.tag

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[levels]", *parts, flush=True)


with open(BOUNDARY_MANIFEST, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
arms = manifest["arms"]
inparish_nodes = [str(node) for node in arms["inparish51"]["nodes"]]
if len(inparish_nodes) != 51:
    raise SystemExit("[FATAL] the in-parish parent does not hold 51 gauges")
cores = {draw: [str(node) for node in arms["ipsub10_d" + str(draw)]["nodes"]] for draw in DRAWS}
for draw in DRAWS:
    for size in (20, 30, 40):
        bigger = set(str(node) for node in arms["ipsub" + str(size) + "_d" + str(draw)]["nodes"])
        if not set(cores[draw]) <= bigger:
            raise SystemExit(f"[FATAL] draw {draw} core is not nested inside ipsub{size}")
log("cores confirmed nested; core size", {draw: len(cores[draw]) for draw in DRAWS})

frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
row_of = {time_point: position for position, time_point in enumerate(pd.DatetimeIndex(frame.index))}
log("matrix", frame.shape)

all_core_gauges = sorted({gauge for draw in DRAWS for gauge in cores[draw]})
stage_series = {}
for gauge in all_core_gauges:
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    stage_series[gauge] = series.mask(trailing < 1e-6)
S = np.column_stack([stage_series[gauge].values for gauge in all_core_gauges]).astype(np.float32)
gauge_index = {gauge: position for position, gauge in enumerate(all_core_gauges)}

threshold_slice = frame.index <= pd.Timestamp(THRESHOLD_RECORD_END)
thresholds = {}
for percentile in EVENT_PERCENTILES:
    values = np.full(len(all_core_gauges), np.nan)
    for position, gauge in enumerate(all_core_gauges):
        finite = stage_series[gauge].values[threshold_slice]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values[position] = float(np.percentile(finite, percentile))
    thresholds[percentile] = values

pump_columns = sorted(str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name)))
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)


def metrics(true_values, predicted_values):
    mask = np.isfinite(true_values) & np.isfinite(predicted_values)
    count = int(mask.sum())
    result = {"n": count, "RMSE_m": np.nan, "Pearson_r": np.nan, "NSE": np.nan, "KGE": np.nan}
    if count < MINIMUM_SAMPLES:
        return result
    true_selected = true_values[mask].astype(np.float64)
    predicted_selected = predicted_values[mask].astype(np.float64)
    difference = predicted_selected - true_selected
    result["RMSE_m"] = float(np.sqrt(np.mean(difference ** 2))) * FT2M
    variance_true = float(np.var(true_selected))
    variance_predicted = float(np.var(predicted_selected))
    if variance_true <= 0.0:
        return result
    result["NSE"] = float(
        1.0 - np.sum(difference ** 2) / np.sum((true_selected - true_selected.mean()) ** 2)
    )
    if variance_predicted <= 0.0:
        return result
    correlation = float(np.corrcoef(true_selected, predicted_selected)[0, 1])
    result["Pearson_r"] = correlation
    mean_true = float(true_selected.mean())
    if abs(mean_true) <= 1e-9:
        return result
    alpha = float(np.sqrt(variance_predicted) / np.sqrt(variance_true))
    beta = float(predicted_selected.mean() / mean_true)
    result["KGE"] = float(
        1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    )
    return result


def score_archive(path, core_gauges):
    """Per-gauge h+24 metrics under every origin set, for the given core gauges."""
    archive = np.load(path, allow_pickle=True)
    origins = pd.to_datetime(archive["origins_utc"])
    if origins.tz is not None:
        origins = origins.tz_convert("UTC").tz_localize(None)
    nodes = [str(value) for value in archive["nodes"]]
    node_position = {node: position for position, node in enumerate(nodes)}
    missing = [gauge for gauge in core_gauges if gauge not in node_position]
    if missing:
        archive.close()
        raise SystemExit("[FATAL] " + os.path.basename(path) + " lacks core gauges: " + ",".join(missing))
    quantile_index = [float(q) for q in archive["quantiles_saved"]].index(QUANTILE)
    columns = [node_position[gauge] for gauge in core_gauges]
    predicted = archive["pred_ft"][:, LB_W - 1, :, quantile_index][:, columns].astype(np.float32)
    archive.close()

    positions = np.asarray([row_of[time_point] for time_point in origins])
    observed_columns = [gauge_index[gauge] for gauge in core_gauges]
    steps = positions[:, None] + 1 + np.arange(LB_W)[None, :]
    window_observed = S[steps][:, :, observed_columns]
    observed = S[positions + LB_W][:, observed_columns]
    peak = np.nanmax(np.where(np.isfinite(window_observed), window_observed, -np.inf), axis=1)

    selections = {
        "all": np.ones((len(origins), len(core_gauges)), dtype=bool),
        "pump_at_issue": np.repeat(pump_on[positions][:, None], len(core_gauges), axis=1),
    }
    for percentile in EVENT_PERCENTILES:
        selections[f"stage_p{int(percentile)}"] = peak >= thresholds[percentile][observed_columns][None, :]

    rows = []
    for origin_set, selection in selections.items():
        for position, gauge in enumerate(core_gauges):
            chosen = selection[:, position]
            values = metrics(observed[chosen, position], predicted[chosen, position])
            values.update({"origin_set": origin_set, "node": gauge})
            rows.append(values)
    return rows


records = []
for draw in DRAWS:
    core = cores[draw]
    for seed in SEEDS:
        control_path = os.path.join(CONTROL_RUNS, f"stgnn_seed{seed}_{FORCING}_jan_aug.npz")
        parent_path = os.path.join(LADDER_RUNS, f"stgnn_inparish51_seed{seed}_{FORCING}_jan_aug.npz")
        jobs = [(68, control_path), (51, parent_path)]
        for size in REDUCED_SIZES:
            jobs.append(
                (
                    size,
                    os.path.join(
                        LADDER_RUNS, f"stgnn_ipsub{size}_d{draw}_seed{seed}_{FORCING}_jan_aug.npz"
                    ),
                )
            )
        for size, path in jobs:
            if not os.path.exists(path):
                log("missing archive, skipped:", os.path.basename(path))
                continue
            for row in score_archive(path, core):
                row.update({"size": size, "draw": draw, "seed": seed})
                records.append(row)
            log("draw", draw, "seed", seed, "size", size, "scored")

levels = pd.DataFrame(records)
if levels.empty:
    raise SystemExit("[FATAL] no archives were scored; the ladder runs may not have finished")
levels_path = os.path.join(OUTPUT_DIRECTORY, f"event_network_levels_h24{TAG}.csv")
levels.to_csv(levels_path, index=False)
log("wrote", levels_path, levels.shape)

summary = (
    levels.groupby(["origin_set", "size"], as_index=False)["RMSE_m"].median().sort_values(
        ["origin_set", "size"], ascending=[True, False]
    )
)
log("median h+24 RMSE by origin set and size")
log(summary.round(4).to_string(index=False))
print("EVENT_NETWORK_LEVELS_DONE", flush=True)
