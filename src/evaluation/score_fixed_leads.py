"""Score the January to August 2026 hindcast under five origin sets.

The origin sets share one archive per model and seed. No model is run again here.

  all             every hourly origin from 1 January to 31 August 2026.

  pump_window     origins whose own 24 h verification window contains at least one 15 min step with
                  any pump in the parish running. This is the loose reading of a pump event: a three
                  hour pump episode marks every origin in the preceding 24 hours, so the set covers
                  well over half the year.

  pump_at_issue   origins with any pump already running at issue time. This is the tighter reading
                  and the operationally honest one, because it is what the forecaster knows when the
                  forecast goes out.

  Gates are never used as a trigger. All 13 gate columns read open for most of the record, so "any
  gate open" is true for every hour of 2026 and separates nothing. Weirs report a crest elevation,
  not an on or off state, so they cannot trigger an event either. See
  outputs/structure_column_inventory_2026_jan_aug.csv.

  stage_p80       per gauge and origin: the observed stage anywhere in that origin's 24 h window
  stage_p90       reaches or passes that gauge's 80th or 90th percentile.

                  Both thresholds come from the observed record through 31 December 2025, which is
                  the period the models were trained on, so the event definition never sees the
                  scored period. This follows Nearing et al. (2022), Hydrol. Earth Syst. Sci., 26,
                  5493-5513, Appendix E, which defines peaks as observations above the 80th flow
                  percentile in a given basin.

Neither stage set fixes an event duration. The event window is the forecast's own 24 h window.

Observations use the same causal trailing flatline rule as System B training, inference, and the
January to June scorer, so the two periods stay comparable.

Metrics are RMSE, Pearson correlation, Nash-Sutcliffe Efficiency (NSE) and Kling-Gupta Efficiency
(KGE) at the fixed leads h+3, h+6, h+12 and h+24, which is the manuscript's uniform metric set.
Per-gauge values are averaged over the training seeds first and then taken as a median across gauges,
which is the aggregation order locked in revision 7.

Usage:
  conda run -n operational python -u score_fixed_leads.py
  conda run -n operational python -u score_fixed_leads.py --pattern "*_hrrr_jan_aug.npz" --tag _hrrr
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
RUN_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "runs")
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

FT2M = 0.3048
LB_W = 96
QUANTILE = 0.8
EXCLUDED_GAUGES = {"MBSA4570"}
THRESHOLD_RECORD_END = "2025-12-31 23:45"
EVENT_PERCENTILES = [80.0, 90.0]
MINIMUM_SAMPLES = 30
LEADS = [(3, 12), (6, 24), (12, 48), (24, 96)]
MODEL_NAMES = {"stgnn": "ST-GNN", "lstm": "LSTM", "gru": "GRU"}

# The archive pattern and an output tag are arguments so that the observed-rain arm and the
# forecast-rain arm can be scored with one script without overwriting each other's results.
parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="*_observed_jan_aug.npz")
parser.add_argument("--tag", default="")
parser.add_argument("--runs-dir", default="", help="defaults to this experiment's runs directory")
args = parser.parse_args()
RUN_DIRECTORY = args.runs_dir or RUN_DIRECTORY

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[event-score]", *parts, flush=True)


frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
index_times = pd.DatetimeIndex(frame.index)
row_of = {time_point: position for position, time_point in enumerate(index_times)}
log("matrix", frame.shape, index_times.min(), "->", index_times.max())

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

# Observed stage, with the causal trailing flatline rule.
stage_masked = {}
for gauge in scored_gauges:
    column = gauge + "_stage_ft"
    series = pd.to_numeric(frame[column], errors="coerce")
    trailing_standard_deviation = series.rolling(window=96, center=False, min_periods=96).std()
    stage_masked[gauge] = series.mask(trailing_standard_deviation < 1e-6)

S = np.column_stack([stage_masked[gauge].values for gauge in scored_gauges]).astype(np.float32)
S_RAW = np.column_stack(
    [pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce").values for gauge in scored_gauges]
).astype(np.float32)

# Event thresholds from the record through 31 December 2025 only.
threshold_slice = frame.index <= pd.Timestamp(THRESHOLD_RECORD_END)
thresholds = {percentile: np.full(len(scored_gauges), np.nan) for percentile in EVENT_PERCENTILES}
threshold_rows = []
for position, gauge in enumerate(scored_gauges):
    values = stage_masked[gauge].values[threshold_slice]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        continue
    row = {
        "gauge": gauge,
        "threshold_source_end": THRESHOLD_RECORD_END,
        "n_steps_used": int(len(values)),
    }
    for percentile in EVENT_PERCENTILES:
        threshold = float(np.percentile(values, percentile))
        thresholds[percentile][position] = threshold
        row[f"p{int(percentile)}_stage_ft"] = threshold
        row[f"p{int(percentile)}_stage_m"] = threshold * FT2M
    threshold_rows.append(row)
threshold_path = os.path.join(OUTPUT_DIRECTORY, "event_stage_thresholds_through_2025.csv")
pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
log("wrote", threshold_path, "gauges with thresholds", len(threshold_rows))

# Parish-wide pump activity at the matrix's own 15 min resolution.
pump_columns = sorted(str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name)))
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)
log("pump columns", len(pump_columns), "steps with any pump running", int(pump_on.sum()))


def metrics_from_pairs(true_values, predicted_values):
    """RMSE in meters, Pearson correlation, NSE and KGE for one gauge, lead and origin set."""
    mask = np.isfinite(true_values) & np.isfinite(predicted_values)
    count = int(mask.sum())
    empty = {"n": count, "RMSE_m": np.nan, "Pearson_r": np.nan, "NSE": np.nan, "KGE": np.nan}
    if count < MINIMUM_SAMPLES:
        return empty
    true_selected = true_values[mask].astype(np.float64)
    predicted_selected = predicted_values[mask].astype(np.float64)
    difference = predicted_selected - true_selected
    result = dict(empty)
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


def score_one_archive(label, model, seed, origins, predictions):
    """Return per-gauge rows for one archive under every origin set."""
    positions = np.asarray([row_of[time_point] for time_point in origins])
    if np.any(positions + LB_W >= len(frame)):
        raise ValueError("The evaluation matrix ends before some verifying windows close: " + label)
    steps = positions[:, None] + 1 + np.arange(LB_W)[None, :]
    observed = S[steps]

    pump_window = pump_on[steps].any(axis=1)
    pump_at_issue = pump_on[positions]
    observed_peak = np.nanmax(np.where(np.isfinite(observed), observed, -np.inf), axis=1)
    stage_events = {
        percentile: observed_peak >= thresholds[percentile][None, :]
        for percentile in EVENT_PERCENTILES
    }

    log(
        label,
        "origins",
        len(origins),
        "| pump_window",
        f"{int(pump_window.sum())} ({float(pump_window.mean()) * 100:.1f} %)",
        "| pump_at_issue",
        f"{int(pump_at_issue.sum())} ({float(pump_at_issue.mean()) * 100:.1f} %)",
        "| stage pairs",
        {
            int(percentile): int(stage_events[percentile].sum())
            for percentile in EVENT_PERCENTILES
        },
    )

    rows = []
    for lead_hours, lead_step in LEADS:
        true_lead = observed[:, lead_step - 1, :]
        predicted_lead = predictions[:, lead_step - 1, :]
        for gauge_position, gauge in enumerate(scored_gauges):
            true_column = true_lead[:, gauge_position]
            predicted_column = predicted_lead[:, gauge_position]
            selections = {
                "all": np.ones(len(origins), dtype=bool),
                "pump_window": pump_window,
                "pump_at_issue": pump_at_issue,
            }
            for percentile in EVENT_PERCENTILES:
                selections[f"stage_p{int(percentile)}"] = stage_events[percentile][:, gauge_position]
            for mask_name, selection in selections.items():
                values = metrics_from_pairs(true_column[selection], predicted_column[selection])
                values.update(
                    {
                        "model": model,
                        "seed": seed,
                        "archive": label,
                        "origin_set": mask_name,
                        "lead_hours": lead_hours,
                        "gauge": gauge,
                        "origins_in_set": int(selection.sum()),
                    }
                )
                rows.append(values)
    return rows


archives = []
for path in sorted(glob.glob(os.path.join(RUN_DIRECTORY, args.pattern))):
    name = os.path.basename(path)
    # Everything before "_seed" names the model or the rainfall forcing. Splitting on the first
    # underscore instead would merge gfs_global with gfs_hrrr, which are different forcings.
    label = name.split("_seed")[0]
    seed_match = re.search(r"seed(\d+)", name)
    archives.append(
        (path, name, MODEL_NAMES.get(label, label), int(seed_match.group(1)) if seed_match else -1)
    )
if not archives:
    raise SystemExit("[FATAL] no run archives matching " + args.pattern + " in " + RUN_DIRECTORY)
log("archives", [name for _, name, _, _ in archives])

all_rows = []
reference_origins = None
for path, name, model, seed in archives:
    archive = np.load(path, allow_pickle=True)
    origins = pd.to_datetime(archive["origins_utc"])
    if origins.tz is not None:
        origins = origins.tz_convert("UTC").tz_localize(None)
    archive_nodes = [str(value) for value in archive["nodes"]]
    node_position = {node: position for position, node in enumerate(archive_nodes)}
    columns = [node_position[gauge] for gauge in scored_gauges]
    if "pred_ft" in archive:
        quantile_index = [float(q) for q in archive["quantiles_saved"]].index(QUANTILE)
        predictions = archive["pred_ft"][:, :, :, quantile_index][:, :, columns].astype(np.float32)
    else:
        predictions = archive["predictions_ft"][:, :, columns].astype(np.float32)
    all_rows.extend(score_one_archive(name, model, seed, origins, predictions))
    if reference_origins is None:
        reference_origins = origins
    del archive, predictions

# Persistence: the stage reported at issue time, held across the whole horizon.
positions = np.asarray([row_of[time_point] for time_point in reference_origins])
persistence_predictions = np.repeat(S_RAW[positions][:, None, :], LB_W, axis=1)
all_rows.extend(
    score_one_archive("persistence", "persistence", -1, reference_origins, persistence_predictions)
)

per_gauge = pd.DataFrame(all_rows)
per_gauge_path = os.path.join(OUTPUT_DIRECTORY, f"event_conditioned_per_gauge_metrics{args.tag}.csv")
per_gauge.to_csv(per_gauge_path, index=False)
log("wrote", per_gauge_path, per_gauge.shape)

# Seed mean first, then the median across gauges, which is the manuscript's locked order.
metric_names = ["RMSE_m", "Pearson_r", "NSE", "KGE"]
seed_mean = (
    per_gauge.groupby(["model", "origin_set", "lead_hours", "gauge"], as_index=False)[metric_names]
    .mean()
)
summary = (
    seed_mean.groupby(["model", "origin_set", "lead_hours"], as_index=False)[metric_names]
    .median()
    .sort_values(["origin_set", "lead_hours", "model"])
)
sample_counts = (
    per_gauge.groupby(["model", "origin_set", "lead_hours"], as_index=False)["n"]
    .median()
    .rename(columns={"n": "median_n_per_gauge"})
)
summary = summary.merge(sample_counts, on=["model", "origin_set", "lead_hours"], how="left")
summary_path = os.path.join(OUTPUT_DIRECTORY, f"event_conditioned_summary{args.tag}.csv")
summary.to_csv(summary_path, index=False)
log("wrote", summary_path)

report_path = os.path.join(OUTPUT_DIRECTORY, f"EVENT_SCORING_REPORT{args.tag}.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Event-conditioned scoring, 1 January to 31 August 2026\n")
    handle.write(f"archive pattern: {args.pattern}\n")
    handle.write(f"matrix: {MATRIX}\n")
    handle.write(f"scored gauges: {len(scored_gauges)}\n")
    handle.write(f"origins per archive: {len(reference_origins)}\n")
    handle.write(f"pump columns used: {len(pump_columns)}\n")
    handle.write(
        "stage thresholds: 80th and 90th percentile per gauge from the record through "
        f"{THRESHOLD_RECORD_END}\n"
    )
    handle.write("gates are not used as a trigger: every gate column reads open for most of the record\n\n")
    handle.write(summary.to_string(index=False))
    handle.write("\n")
log("wrote", report_path)
print("EVENT_SCORING_DONE", flush=True)
