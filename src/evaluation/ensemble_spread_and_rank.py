"""Spread against error and the rank of the observation within the six-member ensemble, January to August 2026.

The Revision 13 text quoted these statistics from
project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914/scripts/
10_fig_spread_diagnostics.py, which covers 1 January to 30 June and predates the anchor fix. This script applies the
same computation to the January to August member runs: the six weather-model members (GFS-HRRR blend, GFS, NBM,
ECMWF IFS, GEM, JMA) at the fixed h+24 lead, 51 in-parish gauges, each seed's six members forming one ensemble and
the pairs of the three seeds pooled. Spread is the member standard deviation (ddof 1), error is the ensemble mean
minus the observed stage, and the rank of the observation among the six members uses random tie-breaking (seed 0).
Event-only origins are those with any parish pump running at issue time, built as rainfall_forcing_skill.py
builds them.

Usage: python ensemble_spread_and_rank.py [EXPERIMENT_ROOT]   (default: this experiment)
Output: <EXPERIMENT_ROOT>/outputs/spread_rank_jan_aug.csv, printed summary
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = ("project/hpc/Experiments/"
          "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/global_features_all_stations_feature_engineered_20260911.pkl")
INPARISH_MANIFEST = ("project/Experiments/INPARISH51_20260813/"
                     "outputs/in_parish_gauges.json")
EXCLUDED_GAUGES = {"MBSA4570"}
SEEDS = [101, 202, 303]
MEMBERS = ["gfs_hrrr", "gfs_global", "ncep_nbm_conus", "ecmwf_ifs025", "gem_global", "jma_seamless"]

with open(INPARISH_MANIFEST, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
nodes = manifest["nodes"] if isinstance(manifest, dict) and "nodes" in manifest else list(manifest.values())[0]
keep = {str(node) for node in nodes} - EXCLUDED_GAUGES

frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
pump_columns = [str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name))]
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)
pump_at = pd.Series(pump_on, index=pd.DatetimeIndex(frame.index))

blocks = []
for seed in SEEDS:
    members = []
    for tag in MEMBERS:
        path = os.path.join(EXPERIMENT_ROOT, "scored_members_jan_aug", f"{tag}_seed{seed}",
                            "reforecast_2026H1_h96_timeseries.csv")
        table = pd.read_csv(path, usecols=["timestamp_utc", "node", "observed_stage_m", "predicted_stage_m"],
                            parse_dates=["timestamp_utc"])
        table = table[table["node"].isin(keep)].set_index(["timestamp_utc", "node"])
        members.append(table.rename(columns={"predicted_stage_m": tag}))
    block = members[0][["observed_stage_m"]].copy()
    for tag, table in zip(MEMBERS, members):
        block[tag] = table[tag]
    blocks.append(block.reset_index())
pairs = pd.concat(blocks, ignore_index=True)
origins = pairs["timestamp_utc"] - pd.Timedelta(hours=24)
if origins.dt.tz is not None:
    origins = origins.dt.tz_convert("UTC").dt.tz_localize(None)
pairs["pump_at_issue"] = pump_at.reindex(origins).to_numpy()

rows = []
for origin_set in ["all", "pump_at_issue"]:
    subset = pairs if origin_set == "all" else pairs[pairs["pump_at_issue"] == True]  # noqa: E712
    ensemble = subset[MEMBERS].to_numpy(dtype="float64")
    observed = subset["observed_stage_m"].to_numpy(dtype="float64")
    ok = np.isfinite(observed) & np.isfinite(ensemble).all(axis=1)
    ensemble = ensemble[ok]
    observed = observed[ok]
    mean = ensemble.mean(axis=1)
    spread = ensemble.std(axis=1, ddof=1)
    rmse = float(np.sqrt(np.mean((mean - observed) ** 2)))
    generator = np.random.default_rng(0)
    below = (ensemble < observed[:, None]).sum(axis=1)
    equal = (ensemble == observed[:, None]).sum(axis=1)
    rank = below + np.array([generator.integers(0, e + 1) if e > 0 else 0 for e in equal])
    frequency = np.bincount(rank, minlength=len(MEMBERS) + 1)[: len(MEMBERS) + 1] / len(rank) * 100.0
    row = {"origin_set": origin_set, "pairs": int(len(observed)),
           # rank 0: no member below the observation, so the observation is below all six members
           "below_all_pct": frequency[0], "above_all_pct": frequency[-1],
           "outside_pct": frequency[0] + frequency[-1], "reliable_outside_pct": 200.0 / (len(MEMBERS) + 1),
           "rmse_ensemble_mean_m": rmse, "mean_spread_m": float(spread.mean()),
           "error_to_spread": rmse / float(spread.mean())}
    rows.append(row)
    print(f"[spread] {origin_set:14s} pairs {row['pairs']:,}  observed below all six {row['below_all_pct']:.1f} %  "
          f"above all six {row['above_all_pct']:.1f} %  outside {row['outside_pct']:.1f} % "
          f"(reliable {row['reliable_outside_pct']:.1f} %)  RMSE of mean {rmse:.4f} m  "
          f"mean spread {row['mean_spread_m']:.4f} m  ratio {row['error_to_spread']:.1f}")

output = os.path.join(EXPERIMENT_ROOT, "outputs", "spread_rank_jan_aug.csv")
pd.DataFrame(rows).to_csv(output, index=False)
print("[spread] wrote", output)
