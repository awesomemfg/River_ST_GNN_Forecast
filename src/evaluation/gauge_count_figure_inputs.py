"""The decision file Figure 16 reads, written for January to August 2026.

Why this exists
---------------
build_rev6_fig16_gauge_count.py reads three inputs. Two are per-gauge tables that
figure_inputs.py already writes. The third is a small JSON of the boundary result:

  boundary[0].mean_seed_median_delta_RMSE_m   the mean over seeds of the median paired change when
                                              the 17 supporting gauges are removed
  boundary[0].control_seed_floor_m            the seed floor of the FULL 68-gauge model
  boundary[0].all_three_seeds_worse           whether every seed got worse

Revision 7's file holds the January to June values, 0.0238 m against a floor of 0.0222 m. The values
here come from section42_network_numbers.py and SECTION42_NUMBERS_REPORT.txt.

One consistency point worth stating. The recipe computes the shaded band of panel (a) itself, from
the 68-gauge rows of the levels file, and separately reads control_seed_floor_m from this file. The
caption of Figure 16 quotes that floor as 0.020 m, and Section 4.2 quotes 0.0201 m. All three have to
be the same number or the figure contradicts its own caption.

Output: ../outputs/figure_inputs_jan_aug/fig16_locked_decision_jan_aug.json

Usage:
  conda run -n operational python -u gauge_count_figure_inputs.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUTS = os.path.join(EXPERIMENT_ROOT, "outputs")
TARGET = os.path.join(OUTPUTS, "figure_inputs_jan_aug")
LEVELS = os.path.join(OUTPUTS, "event_network_levels_h24.csv")

SEEDS = [101, 202, 303]
PARENT_SIZE = 51
CONTROL_SIZE = 68

os.makedirs(TARGET, exist_ok=True)
if not os.path.exists(LEVELS):
    raise SystemExit("[FATAL] missing " + LEVELS)

levels = pd.read_csv(LEVELS)
levels = levels[levels["origin_set"] == "all"]
parent = levels[levels["size"] == PARENT_SIZE]
control = levels[levels["size"] == CONTROL_SIZE]


def paired_effect(seed):
    """Median over shared gauges of the 51-gauge model minus the 68-gauge model, at one seed."""
    a = parent[parent["seed"] == seed].set_index("node")["RMSE_m"]
    b = control[control["seed"] == seed].set_index("node")["RMSE_m"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.nan
    return float((a.loc[common] - b.loc[common]).median())


def seed_floor(block):
    """Median over gauges of the range of RMSE across the three training runs of one model."""
    wide = block.pivot_table(index="node", columns="seed", values="RMSE_m")
    present = [seed for seed in SEEDS if seed in wide.columns]
    if len(present) < 2:
        return float("nan")
    return float((wide[present].max(axis=1) - wide[present].min(axis=1)).median())


effects = [paired_effect(seed) for seed in SEEDS]
effects = [value for value in effects if np.isfinite(value)]
control_floor = seed_floor(control)
mean_effect = float(np.mean(effects))

decision = {
    "primary_forcing": "observed",
    "primary_lead_step": 96,
    "primary_lead_hours": 24.0,
    "period": "1 January to 31 August 2026, 5,832 hourly origins",
    "boundary": [{
        "forcing": "observed",
        "lead_step": 96,
        "lead_hours": 24.0,
        "mean_seed_median_delta_RMSE_m": mean_effect,
        "minimum_seed_median_delta_RMSE_m": float(np.min(effects)),
        "maximum_seed_median_delta_RMSE_m": float(np.max(effects)),
        "all_three_seeds_worse": bool(all(value > 0 for value in effects)),
        "control_seed_floor_m": control_floor,
        "important_by_locked_rule": bool(mean_effect > control_floor),
    }],
    "smallest_tested_adequate_inparish_size": 10,
    "terminology": {
        "boundary_effect": "the 51 in-parish gauges against the full 68-gauge network",
        "seed_floor": "the median over gauges of the range of RMSE across the three training runs",
    },
}

path = os.path.join(TARGET, "fig16_locked_decision_jan_aug.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(decision, handle, indent=1)

# The recipe filters the floors table on lead_step == 96, not on lead_hours. Writing only lead_hours
# leaves that filter matching nothing, and the seed floor silently becomes NaN, so the figure draws
# with no noise band instead of failing. The column is added here rather than relaxing the filter.
floors_path = os.path.join(TARGET, "fig16_parent_seed_floors_jan_aug.csv")
if os.path.exists(floors_path):
    floors = pd.read_csv(floors_path)
    if "lead_step" not in floors.columns:
        floors.insert(floors.columns.get_loc("lead_hours"), "lead_step", 96)
        floors.to_csv(floors_path, index=False)
        print("[fig16] added lead_step=96 to", floors_path, flush=True)
    observed_at_h24 = floors[(floors["forcing"] == "observed") & (floors["lead_step"] == 96)]
    print(f"[fig16] floors the recipe will read: {len(observed_at_h24)} rows, "
          f"median {observed_at_h24['parent_seed_floor_m'].median():.4f} m", flush=True)

print("[fig16] per-seed boundary effect:", ", ".join(f"{value:.4f}" for value in effects), flush=True)
print(f"[fig16] mean {mean_effect:.4f} m, control seed floor {control_floor:.4f} m", flush=True)
print(f"[fig16] every seed worse: {all(value > 0 for value in effects)}", flush=True)
print(f"[fig16] the caption and Section 4.2 must both say {control_floor:.3f} m", flush=True)
print("[fig16] wrote", path, flush=True)
print("FIG16_DECISION_DONE", flush=True)
