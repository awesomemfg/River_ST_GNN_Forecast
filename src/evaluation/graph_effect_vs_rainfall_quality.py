"""Does the graph effect track rainfall-forecast quality?

The graph effect is the paired difference GRU minus ST-GNN, positive when message passing helps. It
is clearly positive with observed rain and reverses at h+24 with issue-time HRRR rain. If that is a
property of forcing quality, the effect should fall smoothly as the rain forecast gets worse, rather
than switching between two unrelated rain products.

Eight rungs are available on the same 4,344 January to June origins and the same 51 gauges:

  observed rain, the six weather-model forecasts, and the issue-time HRRR stitch.

For each rung this script scores both models per gauge, averages over the three seeds, pairs on the
gauge, and bootstraps the median paired difference by resampling gauges. It then reports the graph
effect against the ST-GNN's own error under that same forcing, which is the quality axis.

Observations use the causal trailing flatline rule, as everywhere else in this experiment.

Usage:
  conda run -n operational python -u graph_effect_vs_rainfall_quality.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
MEMBER_RUN_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "runs_members")

SYSTEM_B = "project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
ENSEMBLE_RUNS = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/runs"
)
MIKE_COMPARISON_RUNS = (
    "project/hpc/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913/runs/comparisons"
)
LOCAL_RUNS = os.path.join(SYSTEM_B, "runs")

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
SEEDS = [101, 202, 303]
LEADS = [(12, 48), (24, 96)]
RESAMPLES = 10000
BOOTSTRAP_SEED = 20260918

# label -> (ST-GNN archive pattern, GRU archive pattern). {seed} is filled per seed.
RUNGS = [
    (
        "observed rain",
        os.path.join(LOCAL_RUNS, "stgnn_obs_pre2026_seed{seed}_observed.npz"),
        os.path.join(MIKE_COMPARISON_RUNS, "stgnn_identity_seed{seed}_observed.npz"),
    ),
    (
        "JMA",
        os.path.join(ENSEMBLE_RUNS, "jma_seamless_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_jma_seamless_seed{seed}.npz"),
    ),
    (
        "NBM",
        os.path.join(ENSEMBLE_RUNS, "ncep_nbm_conus_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_ncep_nbm_conus_seed{seed}.npz"),
    ),
    (
        "GEM",
        os.path.join(ENSEMBLE_RUNS, "gem_global_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_gem_global_seed{seed}.npz"),
    ),
    (
        "ECMWF IFS",
        os.path.join(ENSEMBLE_RUNS, "ecmwf_ifs025_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_ecmwf_ifs025_seed{seed}.npz"),
    ),
    (
        "GFS",
        os.path.join(ENSEMBLE_RUNS, "gfs_global_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_gfs_global_seed{seed}.npz"),
    ),
    (
        "HRRR (Open-Meteo)",
        os.path.join(ENSEMBLE_RUNS, "gfs_hrrr_seed{seed}.npz"),
        os.path.join(MEMBER_RUN_DIRECTORY, "gru_gfs_hrrr_seed{seed}.npz"),
    ),
    (
        "HRRR (issue time)",
        os.path.join(LOCAL_RUNS, "stgnn_obs_pre2026_seed{seed}_hrrr.npz"),
        os.path.join(MIKE_COMPARISON_RUNS, "stgnn_identity_seed{seed}_hrrr.npz"),
    ),
]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
rng = np.random.default_rng(BOOTSTRAP_SEED)


def log(*parts):
    print("[quality]", *parts, flush=True)


frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
index_times = pd.DatetimeIndex(frame.index)
row_of = {time_point: position for position, time_point in enumerate(index_times)}
log("matrix", frame.shape)

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
    """Median-free per-gauge RMSE in meters at one fixed lead, for one archive."""
    archive = np.load(path, allow_pickle=True)
    origins = pd.to_datetime(archive["origins_utc"])
    if origins.tz is not None:
        origins = origins.tz_convert("UTC").tz_localize(None)
    nodes = [str(value) for value in archive["nodes"]]
    node_position = {node: position for position, node in enumerate(nodes)}
    columns = [node_position[gauge] for gauge in scored_gauges]
    quantile_index = [float(q) for q in archive["quantiles_saved"]].index(QUANTILE)
    predicted = archive["pred_ft"][:, lead_step - 1, :, quantile_index][:, columns].astype(np.float32)
    positions = np.asarray([row_of[time_point] for time_point in origins])
    observed = S[positions + lead_step]
    archive.close()
    mask = np.isfinite(observed) & np.isfinite(predicted)
    difference = np.where(mask, predicted - observed, 0.0).astype(np.float64)
    count = mask.sum(axis=0)
    rmse = np.full(len(scored_gauges), np.nan)
    valid = count > 0
    rmse[valid] = np.sqrt((difference ** 2).sum(axis=0)[valid] / count[valid]) * FT2M
    return rmse, len(origins)


rows = []
for label, stgnn_pattern, gru_pattern in RUNGS:
    for lead_hours, lead_step in LEADS:
        stgnn_values = []
        gru_values = []
        missing = False
        origin_count = None
        for seed in SEEDS:
            stgnn_path = stgnn_pattern.format(seed=seed)
            gru_path = gru_pattern.format(seed=seed)
            if not (os.path.exists(stgnn_path) and os.path.exists(gru_path)):
                log("missing archive for", label, "seed", seed)
                missing = True
                break
            stgnn_rmse, origin_count = per_gauge_rmse(stgnn_path, lead_step)
            gru_rmse, _ = per_gauge_rmse(gru_path, lead_step)
            stgnn_values.append(stgnn_rmse)
            gru_values.append(gru_rmse)
        if missing:
            continue
        stgnn_mean = np.nanmean(np.vstack(stgnn_values), axis=0)
        gru_mean = np.nanmean(np.vstack(gru_values), axis=0)
        paired = gru_mean - stgnn_mean
        paired = paired[np.isfinite(paired)]
        draws = rng.integers(0, len(paired), size=(RESAMPLES, len(paired)))
        medians = np.median(paired[draws], axis=1)
        rows.append(
            {
                "forcing": label,
                "lead_hours": lead_hours,
                "origins": origin_count,
                "gauges": len(paired),
                "stgnn_median_rmse_m": float(np.nanmedian(stgnn_mean)),
                "gru_median_rmse_m": float(np.nanmedian(gru_mean)),
                "graph_effect_m": float(np.median(paired)),
                "ci_low_m": float(np.percentile(medians, 2.5)),
                "ci_high_m": float(np.percentile(medians, 97.5)),
                "gauges_helped_by_graph": int((paired > 0).sum()),
            }
        )
        log(label, f"h+{lead_hours}", "graph effect", round(rows[-1]["graph_effect_m"], 4))

table = pd.DataFrame(rows)
table["excludes_zero"] = (table["ci_low_m"] > 0) | (table["ci_high_m"] < 0)
table_path = os.path.join(OUTPUT_DIRECTORY, "graph_effect_versus_forcing_quality.csv")
table.to_csv(table_path, index=False)
log("wrote", table_path)

report_path = os.path.join(OUTPUT_DIRECTORY, "GRAPH_EFFECT_VERSUS_QUALITY_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Graph effect against rainfall-forecast quality\n")
    handle.write("1 January to 30 June 2026, 4,344 hourly origins, 51 gauges, three seeds\n")
    handle.write("graph effect = GRU minus ST-GNN, paired on the gauge, positive when the graph helps\n")
    handle.write("quality axis = the ST-GNN's own median RMSE under the same forcing\n\n")
    for lead_hours, _ in LEADS:
        block = table[table["lead_hours"] == lead_hours].sort_values("stgnn_median_rmse_m")
        if block.empty:
            continue
        handle.write(f"--- h+{lead_hours} ---\n")
        handle.write(
            block[
                [
                    "forcing",
                    "stgnn_median_rmse_m",
                    "gru_median_rmse_m",
                    "graph_effect_m",
                    "ci_low_m",
                    "ci_high_m",
                    "gauges_helped_by_graph",
                    "excludes_zero",
                ]
            ].to_string(index=False)
        )
        handle.write("\n")
        if len(block) >= 3:
            correlation = float(
                np.corrcoef(block["stgnn_median_rmse_m"], block["graph_effect_m"])[0, 1]
            )
            handle.write(
                f"Pearson correlation between forcing error and graph effect: {correlation:.3f}\n"
            )
            handle.write("A negative value means the graph helps less as the rain forecast worsens.\n")
        handle.write("\n")
log("wrote", report_path)
print("GRAPH_EFFECT_QUALITY_DONE", flush=True)
