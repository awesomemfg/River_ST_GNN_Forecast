"""January to August inputs for the manuscript's figure recipes, in the shape each recipe expects.

Why this exists
---------------
The figure recipes are not rewritten. Each one is re-run unchanged and handed a file that looks
exactly like the January to June file it already reads. This builds those files.

  Figure 8   reads reforecast_2026H1_lead_metrics.csv from a mean-over-seeds scored directory.
             The scored directories here are per seed, so the three are averaged on (lead, gauge)
             first, which is what the caption states: per-gauge metrics are averaged over the three
             trained models before the median is taken.

  Figure 16  reads a fixed-core levels table with the columns
             draw, seed, size, node, RMSE_m, NSE, Pearson_r, KGE.
             event_network_levels_h24.csv holds every retained node, not the fixed core, so each
             draw is restricted to the ten gauges its own 10-gauge network keeps. Taking the core
             from this file rather than from the earlier experiment keeps the table self-consistent:
             the comparison is only fair if every size is scored on gauges they all share.

Figures 13 and 14 need a per-seed rise-band summary in a different layout and are not built here.

Outputs, in ../outputs/figure_inputs_jan_aug/:
  fig08_lead_metrics_mean3_jan_aug.csv
  fig16_fixed_core_levels_jan_aug.csv
  fig16_parent_seed_floors_jan_aug.csv

Usage:
  conda run -n operational python -u figure_inputs.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
SCORED = os.path.join(EXPERIMENT_ROOT, "scored_jan_aug")
OUTPUTS = os.path.join(EXPERIMENT_ROOT, "outputs")
TARGET = os.path.join(OUTPUTS, "figure_inputs_jan_aug")
INPARISH = ("project/Experiments/"
            "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json")

SEEDS = [101, 202, 303]
AVERAGED = ["RMSE_m", "MAE_m", "bias_m", "ubRMSE_m", "NSE", "Pearson_r", "KGE"]
KEYS = ["lead_step", "lead_hours", "node"]

os.makedirs(TARGET, exist_ok=True)
gauges = sorted(json.load(open(INPARISH))["nodes"])
print(f"[inputs] in-parish gauges: {len(gauges)}", flush=True)


# ---------------------------------------------------------------- Figure 8
frames = []
for seed in SEEDS:
    path = os.path.join(SCORED, f"observed_seed{seed}", "reforecast_2026H1_lead_metrics.csv")
    if not os.path.exists(path):
        raise SystemExit("[FATAL] missing " + path)
    frame = pd.read_csv(path)
    frame = frame[frame["node"].isin(gauges)]
    frames.append(frame)
    print(f"[inputs] seed {seed}: {len(frame)} rows, {frame['node'].nunique()} gauges,"
          f" leads {sorted(frame['lead_hours'].unique())[:3]}...", flush=True)

stacked = pd.concat(frames, ignore_index=True)
columns = [column for column in AVERAGED if column in stacked.columns]
lead_metrics = stacked.groupby(KEYS, as_index=False)[columns].mean()
# The recipe also reads these, so they are carried through unchanged from the first seed.
for column in ("target", "source_stage_column", "quantile"):
    if column in frames[0].columns:
        lookup = frames[0].drop_duplicates("node").set_index("node")[column]
        lead_metrics[column] = lead_metrics["node"].map(lookup)
if "n" in stacked.columns:
    lead_metrics["n"] = stacked.groupby(KEYS, as_index=False)["n"].mean()["n"]

figure8 = os.path.join(TARGET, "fig08_lead_metrics_mean3_jan_aug.csv")
lead_metrics.to_csv(figure8, index=False)
print(f"[inputs] wrote {figure8} {lead_metrics.shape}", flush=True)
at24 = lead_metrics[np.isclose(lead_metrics["lead_hours"], 24.0)]
print(f"[inputs]   at h+24: {len(at24)} gauges, median RMSE "
      f"{at24['RMSE_m'].median():.4f} m, median NSE {at24['NSE'].median():.4f}", flush=True)


# ---------------------------------------------------------------- Figure 16
levels_path = os.path.join(OUTPUTS, "event_network_levels_h24.csv")
if not os.path.exists(levels_path):
    raise SystemExit("[FATAL] missing " + levels_path)
levels = pd.read_csv(levels_path)
levels = levels[levels["origin_set"] == "all"]

pieces = []
for draw in sorted(levels["draw"].unique()):
    scope = levels[levels["draw"] == draw]
    core = sorted(scope[scope["size"] == 10]["node"].unique())
    if len(core) == 0:
        print(f"[inputs] draw {draw}: no 10-gauge network, skipped", flush=True)
        continue
    kept = scope[scope["node"].isin(core)]
    print(f"[inputs] draw {draw}: fixed core of {len(core)} gauges, {len(kept)} rows", flush=True)
    pieces.append(kept)

core_levels = pd.concat(pieces, ignore_index=True)
core_levels = core_levels[["draw", "seed", "size", "node", "RMSE_m", "NSE", "Pearson_r", "KGE"]]
core_levels = core_levels.sort_values(["draw", "seed", "size", "node"], ascending=[True, True, False, True])
figure16 = os.path.join(TARGET, "fig16_fixed_core_levels_jan_aug.csv")
core_levels.to_csv(figure16, index=False)
print(f"[inputs] wrote {figure16} {core_levels.shape}", flush=True)
print(core_levels.groupby("size")["RMSE_m"].median().round(4).to_string(), flush=True)


def seed_floor(block):
    """Median over gauges of the largest absolute difference between two training runs."""
    values = []
    for first, second in itertools.combinations(SEEDS, 2):
        a = block[block["seed"] == first].set_index("node")["RMSE_m"]
        b = block[block["seed"] == second].set_index("node")["RMSE_m"]
        common = a.index.intersection(b.index)
        if len(common):
            values.append(float((a.loc[common] - b.loc[common]).abs().median()))
    return float(max(values)) if values else np.nan


floor_rows = []
for draw in sorted(core_levels["draw"].unique()):
    scope = core_levels[core_levels["draw"] == draw]
    for size in sorted(scope["size"].unique(), reverse=True):
        block = scope[scope["size"] == size]
        floor_rows.append({
            "forcing": "observed",
            "draw": int(draw),
            "size": int(size),
            "lead_hours": 24.0,
            "parent_seed_floor_m": seed_floor(block),
            "core_gauge_count": int(block["node"].nunique()),
            "core_nodes": "|".join(sorted(block["node"].unique())),
        })
floors = pd.DataFrame(floor_rows)
figure16_floors = os.path.join(TARGET, "fig16_parent_seed_floors_jan_aug.csv")
floors.to_csv(figure16_floors, index=False)
print(f"[inputs] wrote {figure16_floors} {floors.shape}", flush=True)
print(floors[floors["size"] == 68][["draw", "parent_seed_floor_m"]].round(4).to_string(index=False),
      flush=True)
print("FIGURE_INPUTS_DONE", flush=True)
