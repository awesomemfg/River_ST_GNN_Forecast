"""Does the ST-GNN's lead over the LSTM depend on how good the rainfall forecast is?

The two models were compared under only two rainfall inputs: observed rain, where the ST-GNN leads
clearly, and issue-time HRRR, the worst product available, where they tie at h+24. That left the
question of whether the tie is a property of the architecture or of that particular rainfall product.

This scores both models on the same eight rainfall inputs, over the same 4,344 January to June
origins and the same 51 gauges, and pairs them gauge by gauge. Gauges are then resampled to give a
95 percent interval on the median paired difference, which is how the manuscript reports the graph
effect. Pairing matters here: the spread between training runs of one model is about 0.022 m, larger
than the differences being measured, and pairing cancels it.

  difference = LSTM minus ST-GNN, so a positive value means the ST-GNN is better.

The quality axis is the ST-GNN's own error under that same forcing, which is a measured ranking
rather than an assumed one.

No model is run again.

Output: ../outputs/lstm_vs_stgnn_across_forcings.csv and LSTM_VS_STGNN_REPORT.txt

Usage:
  conda run -n operational python -u lstm_vs_stgnn_by_forcing.py
"""
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
LSTM_MEMBER_RUNS = os.path.join(EXPERIMENT_ROOT, "runs_members_lstm")

SYSTEM_B_RUNS = (
    "project/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913/runs"
)
ENSEMBLE_RUNS = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/runs"
)
XUE_RUNS = (
    "project/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/runs"
)
MATRIX = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/in_parish_gauges.json"
)

SEEDS = [101, 202, 303]
LB_W = 96
QUANTILE = 0.8
FT2M = 0.3048
EXCLUDED_GAUGES = {"MBSA4570"}
LEADS = [(12, 48), (24, 96)]
RESAMPLES = 10000
BOOTSTRAP_SEED = 20260918

# label -> (ST-GNN archive pattern, LSTM archive pattern)
RUNGS = [
    ("observed rain",
     os.path.join(SYSTEM_B_RUNS, "stgnn_obs_pre2026_seed{seed}_observed.npz"),
     os.path.join(XUE_RUNS, "lstm_s{seed}_observed_off.npz")),
    ("JMA",
     os.path.join(ENSEMBLE_RUNS, "jma_seamless_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_jma_seamless_seed{seed}.npz")),
    ("NBM",
     os.path.join(ENSEMBLE_RUNS, "ncep_nbm_conus_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_ncep_nbm_conus_seed{seed}.npz")),
    ("GEM",
     os.path.join(ENSEMBLE_RUNS, "gem_global_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_gem_global_seed{seed}.npz")),
    ("ECMWF IFS",
     os.path.join(ENSEMBLE_RUNS, "ecmwf_ifs025_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_ecmwf_ifs025_seed{seed}.npz")),
    ("GFS",
     os.path.join(ENSEMBLE_RUNS, "gfs_global_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_gfs_global_seed{seed}.npz")),
    ("HRRR (Open-Meteo)",
     os.path.join(ENSEMBLE_RUNS, "gfs_hrrr_seed{seed}.npz"),
     os.path.join(LSTM_MEMBER_RUNS, "lstm_gfs_hrrr_seed{seed}.npz")),
    ("HRRR (issue time)",
     os.path.join(SYSTEM_B_RUNS, "stgnn_obs_pre2026_seed{seed}_hrrr.npz"),
     os.path.join(XUE_RUNS, "lstm_s{seed}_hrrr_off.npz")),
]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
rng = np.random.default_rng(BOOTSTRAP_SEED)


def log(*parts):
    print("[head-to-head]", *parts, flush=True)


frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
row_of = {time_point: position for position, time_point in enumerate(pd.DatetimeIndex(frame.index))}

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

stage_columns = []
for gauge in scored_gauges:
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    stage_columns.append(series.mask(trailing < 1e-6).values)
S = np.column_stack(stage_columns).astype(np.float32)


def per_gauge_rmse(path, lead_step):
    archive = np.load(path, allow_pickle=True)
    origins = pd.to_datetime(archive["origins_utc"])
    if origins.tz is not None:
        origins = origins.tz_convert("UTC").tz_localize(None)
    if "pred_ft" in archive:
        nodes = [str(value) for value in archive["nodes"]]
        node_position = {node: position for position, node in enumerate(nodes)}
        columns = [node_position[gauge] for gauge in scored_gauges]
        quantile_index = [float(q) for q in archive["quantiles_saved"]].index(QUANTILE)
        predicted = archive["pred_ft"][:, lead_step - 1, :, quantile_index][:, columns]
    else:
        nodes = [str(value) for value in archive["target_gauges"]]
        node_position = {node: position for position, node in enumerate(nodes)}
        columns = [node_position[gauge] for gauge in scored_gauges]
        predicted = archive["predictions_ft"][:, lead_step - 1, :][:, columns]
    predicted = predicted.astype(np.float32)
    archive.close()
    positions = np.asarray([row_of[time_point] for time_point in origins])
    observed = S[positions + lead_step]
    mask = np.isfinite(observed) & np.isfinite(predicted)
    difference = np.where(mask, predicted - observed, 0.0).astype(np.float64)
    count = mask.sum(axis=0)
    rmse = np.full(len(scored_gauges), np.nan)
    valid = count > 0
    rmse[valid] = np.sqrt((difference ** 2).sum(axis=0)[valid] / count[valid]) * FT2M
    return rmse, len(origins)


rows = []
for label, stgnn_pattern, lstm_pattern in RUNGS:
    for lead_hours, lead_step in LEADS:
        stgnn_values = []
        lstm_values = []
        origin_count = None
        missing = False
        for seed in SEEDS:
            stgnn_path = stgnn_pattern.format(seed=seed)
            lstm_path = lstm_pattern.format(seed=seed)
            if not (os.path.exists(stgnn_path) and os.path.exists(lstm_path)):
                log("missing archive for", label, "seed", seed)
                missing = True
                break
            a, origin_count = per_gauge_rmse(stgnn_path, lead_step)
            b, _ = per_gauge_rmse(lstm_path, lead_step)
            stgnn_values.append(a)
            lstm_values.append(b)
        if missing:
            continue
        stgnn_mean = np.nanmean(np.vstack(stgnn_values), axis=0)
        lstm_mean = np.nanmean(np.vstack(lstm_values), axis=0)
        paired = lstm_mean - stgnn_mean
        paired = paired[np.isfinite(paired)]
        draws = rng.integers(0, len(paired), size=(RESAMPLES, len(paired)))
        medians = np.median(paired[draws], axis=1)
        rows.append({
            "forcing": label,
            "lead_hours": lead_hours,
            "origins": origin_count,
            "gauges": len(paired),
            "stgnn_median_rmse_m": float(np.nanmedian(stgnn_mean)),
            "lstm_median_rmse_m": float(np.nanmedian(lstm_mean)),
            "stgnn_advantage_m": float(np.median(paired)),
            "ci_low_m": float(np.percentile(medians, 2.5)),
            "ci_high_m": float(np.percentile(medians, 97.5)),
            "gauges_favoring_stgnn": int((paired > 0).sum()),
        })
        log(label, f"h+{lead_hours}", "advantage", round(rows[-1]["stgnn_advantage_m"], 4))

table = pd.DataFrame(rows)
table["excludes_zero"] = (table["ci_low_m"] > 0) | (table["ci_high_m"] < 0)
table_path = os.path.join(OUTPUT_DIRECTORY, "lstm_vs_stgnn_across_forcings.csv")
table.to_csv(table_path, index=False)
log("wrote", table_path)

report_path = os.path.join(OUTPUT_DIRECTORY, "LSTM_VS_STGNN_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("ST-GNN against the LSTM across eight rainfall inputs\n")
    handle.write("1 January to 30 June 2026, 4,344 origins, 51 gauges, three seeds\n")
    handle.write("advantage = LSTM minus ST-GNN, paired on the gauge; positive favors the ST-GNN\n\n")
    for lead_hours, _ in LEADS:
        block = table[table["lead_hours"] == lead_hours].sort_values("stgnn_median_rmse_m")
        if block.empty:
            continue
        handle.write(f"--- h+{lead_hours} ---\n")
        handle.write(block[["forcing", "stgnn_median_rmse_m", "lstm_median_rmse_m",
                            "stgnn_advantage_m", "ci_low_m", "ci_high_m",
                            "gauges_favoring_stgnn", "excludes_zero"]].to_string(index=False))
        handle.write("\n")
        if len(block) >= 3:
            correlation = float(np.corrcoef(block["stgnn_median_rmse_m"], block["stgnn_advantage_m"])[0, 1])
            handle.write(f"correlation between forcing error and the ST-GNN advantage: {correlation:+.3f}\n")
        handle.write("\n")
log("wrote", report_path)
print("LSTM_VS_STGNN_DONE", flush=True)
