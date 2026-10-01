"""The numbers Section 4.2 quotes about the gauge network, on January to August 2026.

Why this exists
---------------
Section 4.2 states three things that no file holds for this period:

  the boundary effect   the mean increase in h+24 RMSE when the 17 supporting gauges are removed,
                        per seed, so the section can give a range and not only a middle value
  the seed floor        the spread between training seeds of the FULL 68-gauge model. Section 4.2
                        compares the boundary effect against this, and the comparison decides
                        whether the supporting gauges matter at all. network_size_effect.py
                        computes the floor of the 51-gauge parent instead, which answers a different
                        question.
  the size effect       the median change of each draw against the 51-gauge parent, across every
                        draw and size, which the section quotes as a range

The published sentence says both boundary increases exceeded the full-network seed floor. On this
period the boundary effect with observed rainfall is much smaller than revision 7 reported, so that
comparison has to be made again rather than assumed.

Definitions follow network_size_effect.py:
  effect     median over gauges of the paired difference in per-gauge h+24 RMSE
  seed floor median over gauges of the range of RMSE across the three seeds of one model

Input:  ../outputs/event_network_levels_h24.csv and event_network_levels_h24_hrrr.csv
Output: ../outputs/section42_numbers_jan_aug.csv and SECTION42_NUMBERS_REPORT.txt

Usage:
  conda run -n operational python -u section42_network_numbers.py
"""
import itertools
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

SEEDS = [101, 202, 303]
DRAWS = [1, 2, 3]
SIZES = [40, 30, 20, 10]
PARENT_SIZE = 51
CONTROL_SIZE = 68
ORIGIN_SET = "all"

pd.set_option("display.width", 200)

lines = []


def out(*parts):
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    lines.append(text)


def seed_floor(block):
    """Median over gauges of the range of RMSE across the three seeds of one model."""
    wide = block.pivot_table(index="node", columns="seed", values="RMSE_m")
    present = [seed for seed in SEEDS if seed in wide.columns]
    if len(present) < 2:
        return np.nan
    return float((wide[present].max(axis=1) - wide[present].min(axis=1)).median())


def paired_effect(arm, reference, seed):
    """Median over shared gauges of (arm minus reference) at one seed."""
    a = arm[arm["seed"] == seed].set_index("node")["RMSE_m"]
    b = reference[reference["seed"] == seed].set_index("node")["RMSE_m"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.nan
    return float((a.loc[common] - b.loc[common]).median())


rows = []
for tag, label in [("", "observed rainfall"), ("_hrrr", "issue-time HRRR")]:
    path = os.path.join(OUTPUT_DIRECTORY, f"event_network_levels_h24{tag}.csv")
    if not os.path.exists(path):
        out(f"[missing] {path}")
        continue
    levels = pd.read_csv(path)
    levels = levels[levels["origin_set"] == ORIGIN_SET]

    out("")
    out("=" * 100)
    out(f"{label}")
    out("=" * 100)

    # The full network's own seed floor, which is what the section compares the boundary against.
    control = levels[levels["size"] == CONTROL_SIZE]
    parent = levels[levels["size"] == PARENT_SIZE]
    control_floor = seed_floor(control)
    parent_floor = seed_floor(parent)
    out(f"seed floor of the 68-gauge network : {control_floor:.4f} m")
    out(f"seed floor of the 51-gauge network : {parent_floor:.4f} m")

    # The boundary effect per seed: the 51-gauge parent against the 68-gauge control.
    effects = [paired_effect(parent, control, seed) for seed in SEEDS]
    effects = [value for value in effects if np.isfinite(value)]
    mean_effect = float(np.mean(effects))
    out("")
    out("boundary effect, 51 gauges against 68, per seed: "
        + ", ".join(f"{value:.4f}" for value in effects))
    out(f"  mean {mean_effect:.4f} m, range {min(effects):.4f} to {max(effects):.4f} m")
    out(f"  every seed positive: {all(value > 0 for value in effects)}")
    out(f"  mean effect exceeds the 68-gauge seed floor: {mean_effect > control_floor}")
    if mean_effect <= control_floor:
        out("  The increase does NOT clear the spread between training runs of the same model, so")
        out("  it cannot be separated from retraining noise on this period.")

    # Median per-gauge RMSE at each network size, which the section lists.
    medians = levels.groupby("size", as_index=False)["RMSE_m"].median().sort_values("size",
                                                                                   ascending=False)
    out("")
    out("median per-gauge h+24 RMSE by network size")
    out(medians.round(4).to_string(index=False))

    # Each draw against the 51-gauge parent, which the section quotes as a range.
    draw_effects = []
    for draw in DRAWS:
        scope = levels[levels["draw"] == draw]
        draw_parent = scope[scope["size"] == PARENT_SIZE]
        for size in SIZES:
            arm = scope[scope["size"] == size]
            if arm.empty or draw_parent.empty:
                continue
            values = [paired_effect(arm, draw_parent, seed) for seed in SEEDS]
            values = [value for value in values if np.isfinite(value)]
            if not values:
                continue
            draw_effects.append({"forcing": label, "draw": draw, "size": size,
                                 "median_effect_vs_parent_m": float(np.median(values))})
    if draw_effects:
        frame = pd.DataFrame(draw_effects)
        low = float(frame["median_effect_vs_parent_m"].min())
        high = float(frame["median_effect_vs_parent_m"].max())
        out("")
        out(f"each draw against the 51-gauge parent: {low:.4f} to {high:.4f} m "
            f"over {len(frame)} draw and size combinations")
        out(f"  every combination within the 51-gauge seed floor: "
            f"{bool((frame['median_effect_vs_parent_m'].abs() <= parent_floor).all())}")
        rows.extend(draw_effects)

    rows.append({"forcing": label, "draw": np.nan, "size": CONTROL_SIZE,
                 "median_effect_vs_parent_m": np.nan})

table = pd.DataFrame(rows)
table_path = os.path.join(OUTPUT_DIRECTORY, "section42_numbers_jan_aug.csv")
table.to_csv(table_path, index=False)
out("")
out("[section42] wrote " + table_path)

report_path = os.path.join(OUTPUT_DIRECTORY, "SECTION42_NUMBERS_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("[section42] wrote", report_path, flush=True)
print("SECTION42_NUMBERS_DONE", flush=True)
