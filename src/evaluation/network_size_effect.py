"""The gauge-count question asked correctly: each reduced size against the 51-gauge in-parish parent.

Why this exists
---------------
The event levels table reports each size against the full 68-gauge control. That difference mixes two
separate effects:

  68 -> 51  the boundary effect, dropping the 17 gauges outside the parish
  51 -> 10  the size effect, thinning the in-parish network

The published claim, that reducing the in-parish network costs nothing at the gauges that remain, is
about the second effect only, measured against the 51-gauge parent. This computes that, pairing on
the gauge, the draw and the training seed, which is how the published analysis does it
(finalize_system_b_inparish_ladder.py).

The comparison is judged against the parent's own paired seed floor: the largest median absolute
difference between two training runs of the same parent model, on the same gauges. A size effect
smaller than that floor cannot be separated from the noise of retraining.

Input:  ../outputs/event_network_levels_h24.csv
Output: ../outputs/size_effect_against_parent.csv and SIZE_EFFECT_REPORT.txt

Usage:
  conda run -n operational python -u network_size_effect.py
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
# The levels file and an output tag are arguments, so the same comparison runs for both rainfall
# forcings. Without them this script silently reads the observed-rain levels whatever it is asked
# for. The defaults reproduce the observed-rain result exactly.
parser = argparse.ArgumentParser()
parser.add_argument("--levels", default=os.path.join(OUTPUT_DIRECTORY, "event_network_levels_h24.csv"))
parser.add_argument("--tag", default="")
arguments = parser.parse_args()
LEVELS = arguments.levels
TAG = arguments.tag

SEEDS = [101, 202, 303]
DRAWS = [1, 2, 3]
REDUCED_SIZES = [40, 30, 20, 10]
PARENT_SIZE = 51
CONTROL_SIZE = 68

if not os.path.exists(LEVELS):
    raise SystemExit("[FATAL] missing " + LEVELS)
levels = pd.read_csv(LEVELS)


def paired_seed_floor(block):
    """Largest median absolute difference between two training runs of the same model."""
    values = []
    for first, second in itertools.combinations(SEEDS, 2):
        a = block[block["seed"] == first].set_index("node")["RMSE_m"]
        b = block[block["seed"] == second].set_index("node")["RMSE_m"]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        values.append(float((a.loc[common] - b.loc[common]).abs().median()))
    return float(max(values)) if values else np.nan


rows = []
for origin_set in sorted(levels["origin_set"].unique()):
    scope = levels[levels["origin_set"] == origin_set]
    for draw in DRAWS:
        draw_scope = scope[scope["draw"] == draw]
        parent_block = draw_scope[draw_scope["size"] == PARENT_SIZE]
        control_block = draw_scope[draw_scope["size"] == CONTROL_SIZE]
        if parent_block.empty:
            continue
        parent_floor = paired_seed_floor(parent_block)

        # The boundary effect, for context: the parent against the full control.
        boundary_effects = []
        for seed in SEEDS:
            parent_seed = parent_block[parent_block["seed"] == seed].set_index("node")["RMSE_m"]
            control_seed = control_block[control_block["seed"] == seed].set_index("node")["RMSE_m"]
            common = parent_seed.index.intersection(control_seed.index)
            if len(common):
                boundary_effects.append(float((parent_seed.loc[common] - control_seed.loc[common]).median()))

        for size in REDUCED_SIZES:
            arm_block = draw_scope[draw_scope["size"] == size]
            if arm_block.empty:
                continue
            effects = []
            for seed in SEEDS:
                arm_seed = arm_block[arm_block["seed"] == seed].set_index("node")["RMSE_m"]
                parent_seed = parent_block[parent_block["seed"] == seed].set_index("node")["RMSE_m"]
                common = arm_seed.index.intersection(parent_seed.index)
                if len(common) == 0:
                    continue
                effects.append(float((arm_seed.loc[common] - parent_seed.loc[common]).median()))
            if not effects:
                continue
            rows.append({
                "origin_set": origin_set,
                "draw": draw,
                "size": size,
                "median_effect_vs_parent_m": float(np.median(effects)),
                "min_effect_m": float(np.min(effects)),
                "max_effect_m": float(np.max(effects)),
                "parent_seed_floor_m": parent_floor,
                "within_parent_seed_floor": bool(abs(float(np.median(effects))) <= parent_floor),
                "boundary_effect_parent_vs_control_m": float(np.median(boundary_effects))
                if boundary_effects else np.nan,
            })

effects_table = pd.DataFrame(rows)
effects_path = os.path.join(OUTPUT_DIRECTORY, f"size_effect_against_parent{TAG}.csv")
effects_table.to_csv(effects_path, index=False)
print("[size] wrote", effects_path, effects_table.shape, flush=True)

pooled = (
    effects_table.groupby(["origin_set", "size"], as_index=False)
    .agg(median_effect_m=("median_effect_vs_parent_m", "median"),
         worst_draw_effect_m=("median_effect_vs_parent_m", "max"),
         parent_seed_floor_m=("parent_seed_floor_m", "median"),
         all_draws_within_floor=("within_parent_seed_floor", "all"))
    .sort_values(["origin_set", "size"], ascending=[True, False])
)
report_path = os.path.join(OUTPUT_DIRECTORY, f"SIZE_EFFECT_REPORT{TAG}.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("In-parish size effect, paired against the 51-gauge parent, h+24\n")
    handle.write("positive means the reduced network is worse than the parent\n")
    handle.write("a size effect below the parent seed floor cannot be separated from retraining noise\n\n")
    handle.write(pooled.to_string(index=False))
    handle.write("\n\nboundary effect, parent against the full 68-gauge control, by origin set\n")
    boundary = (
        effects_table.groupby("origin_set", as_index=False)["boundary_effect_parent_vs_control_m"]
        .median()
    )
    handle.write(boundary.to_string(index=False))
    handle.write("\n")
print(pooled.round(4).to_string(index=False), flush=True)
print("[size] wrote", report_path, flush=True)
print("SIZE_EFFECT_DONE", flush=True)
