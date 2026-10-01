"""January to August inputs for the Figure 13 and Figure 14 recipe, in the layout it already reads.

Why this exists
---------------
recipe_rainfall_skill_and_crest.py draws Figures 13 and 14 from two files:

  gauge_metrics_24h_mean_over_seeds.csv   forcing, node, rmse, nse, r, kge
                                          one row per rainfall forcing and gauge, at the 24 h lead,
                                          with the three seeds averaged before the boxes are drawn
  window_summary_by_seed.csv              one row per rainfall forcing and seed, holding the rise
                                          counts, mean peak errors and missed-crest shares at three
                                          thresholds and in three bands

Both cover January to June. This writes the same two layouts for January to August. The recipe is
not modified; it is re-run and handed these.

Conventions follow rainfall_forcing_skill.py, which follows the published 06_aggregate.py:
  a rise      the observed rise over the 24 h window, at least 0.1524 m
  a missed    the forecast crest falls more than 0.1524 m below the observed crest, which is
  crest      peak_bias_m below -0.1524
  the bands   0.15 to 0.30 m, 0.30 to 0.61 m, and above 0.61 m

Outputs, in ../outputs/figure_inputs_jan_aug/:
  fig13_gauge_metrics_24h_mean_over_seeds_jan_aug.csv
  fig13_window_summary_by_seed_jan_aug.csv

Usage:
  conda run -n operational python -u rainfall_and_crest_figure_inputs.py
"""
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUTS = os.path.join(EXPERIMENT_ROOT, "outputs")
TARGET = os.path.join(OUTPUTS, "figure_inputs_jan_aug")

SEEDS = [101, 202, 303]
RISE_M = 0.1524
UNDERCALL_M = -0.1524
THRESHOLDS = [0.1524, 0.3048, 0.6096]
BANDS = [("0.15-0.30 m", 0.1524, 0.3048),
         ("0.30-0.61 m", 0.3048, 0.6096),
         ("above 0.61 m", 0.6096, np.inf)]

# The window metrics hold every gauge the run scored, not only the 51 the manuscript reports. Without
# this filter the crest statistics count out-of-parish gauges too, which gives 15,266 strong rises and
# 63.2 percent missed crests for observed rainfall instead of the 12,520 and 57.6 percent that
# Table 6 and Section 4.1 state. The figure would then contradict its own caption.
INPARISH = ("project/Experiments/"
            "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json")
GAUGES = sorted(json.load(open(INPARISH))["nodes"])

# label -> (per-gauge metrics file, model name inside it, scored root, directory template)
# gfs_hrrr is Open-Meteo's blend and is never called plain HRRR. The native issue-time stitch is the
# ST-GNN row of the hrrr per-gauge file and the hrrr_seed directories.
FORCINGS = [
    ("Observed rain", "event_conditioned_per_gauge_metrics.csv", "ST-GNN", "scored_jan_aug", "observed_seed{seed}"),
    ("HRRR, issue time", "event_conditioned_per_gauge_metrics_hrrr.csv", "ST-GNN", "scored_jan_aug", "hrrr_seed{seed}"),
    ("Ensemble mean, 6", "event_conditioned_per_gauge_metrics_ens_jan_aug.csv", "ens6_mean", "scored_jan_aug", "ens6_mean_seed{seed}"),
    ("Ensemble median, 6", "event_conditioned_per_gauge_metrics_ens_jan_aug.csv", "ens6_median", "scored_members_jan_aug", "ens6_median_seed{seed}"),
    ("Ensemble mean, 8", "event_conditioned_per_gauge_metrics_ens_jan_aug.csv", "ens8_mean", "scored_members_jan_aug", "ens8_mean_seed{seed}"),
    ("JMA", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "jma_seamless", "scored_members_jan_aug", "jma_seamless_seed{seed}"),
    ("NBM", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "ncep_nbm_conus", "scored_members_jan_aug", "ncep_nbm_conus_seed{seed}"),
    ("GEM", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "gem_global", "scored_members_jan_aug", "gem_global_seed{seed}"),
    ("GEFS mean", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "gefs_geavg", "scored_members_jan_aug", "gefs_geavg_seed{seed}"),
    ("ECMWF IFS", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "ecmwf_ifs025", "scored_members_jan_aug", "ecmwf_ifs025_seed{seed}"),
    ("GFS", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "gfs_global", "scored_members_jan_aug", "gfs_global_seed{seed}"),
    # The recipe's own MEMBERS list calls this member "HRRR" and reserves "HRRR, issue time" for the
    # native issue-time stitch, so the two are already distinct there. The label has to match the
    # recipe or its lookup fails, which is what KeyError: 'HRRR' meant. It is Open-Meteo's blended
    # GFS and HRRR product, not the native stitch, and it scores 0.1851 m against the native 0.1907 m.
    ("HRRR", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "gfs_hrrr", "scored_members_jan_aug", "gfs_hrrr_seed{seed}"),
    ("GEFS pmm", "event_conditioned_per_gauge_metrics_members_jan_aug.csv", "gefs_pmm", "scored_members_jan_aug", "gefs_pmm_seed{seed}"),
]

os.makedirs(TARGET, exist_ok=True)
cache = {}


def per_gauge(name):
    if name not in cache:
        path = os.path.join(OUTPUTS, name)
        if not os.path.exists(path):
            raise SystemExit("[FATAL] missing " + path)
        cache[name] = pd.read_csv(path)
    return cache[name]


# ---------------------------------------------------------------- the per-gauge box data
rows = []
for label, metrics_file, model, _root, _template in FORCINGS:
    frame = per_gauge(metrics_file)
    block = frame[(frame["origin_set"] == "all") & (frame["lead_hours"] == 24)
                  & (frame["model"] == model)]
    if block.empty:
        print(f"[fig13] no rows for {label} ({model}) in {metrics_file}", flush=True)
        continue
    averaged = block.groupby("gauge", as_index=False)[["RMSE_m", "NSE", "Pearson_r", "KGE"]].mean()
    averaged.insert(0, "forcing", label)
    averaged = averaged.rename(columns={"gauge": "node", "RMSE_m": "rmse", "NSE": "nse",
                                        "Pearson_r": "r", "KGE": "kge"})
    rows.append(averaged[["forcing", "node", "rmse", "nse", "r", "kge"]])
    print(f"[fig13] {label:20s} {len(averaged)} gauges, median RMSE {averaged['rmse'].median():.4f}",
          flush=True)

gauge_metrics = pd.concat(rows, ignore_index=True)
gauge_path = os.path.join(TARGET, "fig13_gauge_metrics_24h_mean_over_seeds_jan_aug.csv")
gauge_metrics.to_csv(gauge_path, index=False)
print(f"[fig13] wrote {gauge_path} {gauge_metrics.shape}", flush=True)


# ---------------------------------------------------------------- the per-gauge, per-seed data
# Appendix Figure B3 reads the by-seed table, not the mean-over-seeds one above. Its layout is
# forcing, seed, node, rmse, nse, r, kge.
seed_rows = []
for label, metrics_file, model, _root, _template in FORCINGS:
    frame = per_gauge(metrics_file)
    block = frame[(frame["origin_set"] == "all") & (frame["lead_hours"] == 24)
                  & (frame["model"] == model)]
    if block.empty:
        continue
    piece = block[["seed", "gauge", "RMSE_m", "NSE", "Pearson_r", "KGE"]].copy()
    piece.insert(0, "forcing", label)
    seed_rows.append(piece.rename(columns={"gauge": "node", "RMSE_m": "rmse", "NSE": "nse",
                                           "Pearson_r": "r", "KGE": "kge"}))

# Persistence uses no rainfall and has no training seed, so it is read once from the observed file
# rather than through the forcing loop.
observed_metrics = per_gauge("event_conditioned_per_gauge_metrics.csv")
persistence = observed_metrics[(observed_metrics["origin_set"] == "all")
                               & (observed_metrics["lead_hours"] == 24)
                               & (observed_metrics["model"] == "persistence")]
if not persistence.empty:
    piece = persistence[["seed", "gauge", "RMSE_m", "NSE", "Pearson_r", "KGE"]].copy()
    piece.insert(0, "forcing", "Persistence")
    seed_rows.append(piece.rename(columns={"gauge": "node", "RMSE_m": "rmse", "NSE": "nse",
                                           "Pearson_r": "r", "KGE": "kge"}))

by_seed = pd.concat(seed_rows, ignore_index=True)[["forcing", "seed", "node", "rmse", "nse", "r", "kge"]]
by_seed_path = os.path.join(TARGET, "fig13_gauge_metrics_24h_by_seed_jan_aug.csv")
by_seed.to_csv(by_seed_path, index=False)
print(f"[B3] wrote {by_seed_path} {by_seed.shape}, forcings {by_seed['forcing'].nunique()}",
      flush=True)


# ---------------------------------------------------------------- the per-seed window summary
def window_statistics(path):
    frame = pd.read_csv(path, usecols=["t0_utc", "gauge", "window_hours", "RMSE_m",
                                       "observed_rise_m", "peak_absolute_error_m", "peak_bias_m"])
    frame = frame[frame["window_hours"] == 24]
    frame = frame[frame["gauge"].isin(GAUGES)]
    entry = {
        "origin_gauge_count": int(len(frame)),
        "mean_0_24h_RMSE_m": float(frame["RMSE_m"].mean()),
        "median_0_24h_RMSE_m": float(frame["RMSE_m"].median()),
    }
    rises = frame.dropna(subset=["observed_rise_m"])
    for threshold in THRESHOLDS:
        selected = rises[rises["observed_rise_m"] >= threshold]
        key = f"{threshold:.4f}"
        entry[f"n_rises_ge_{key}"] = int(len(selected))
        entry[f"mean_peak_abs_error_m_ge_{key}"] = float(selected["peak_absolute_error_m"].mean())
        entry[f"missed_crest_pct_ge_{key}"] = (
            100.0 * float((selected["peak_bias_m"] < UNDERCALL_M).mean()) if len(selected) else np.nan)
    for name, low, high in BANDS:
        selected = rises[(rises["observed_rise_m"] >= low) & (rises["observed_rise_m"] < high)]
        entry[f"n_band_{name}"] = int(len(selected))
        entry[f"mean_peak_abs_error_m_band_{name}"] = float(selected["peak_absolute_error_m"].mean())
        entry[f"missed_crest_pct_band_{name}"] = (
            100.0 * float((selected["peak_bias_m"] < UNDERCALL_M).mean()) if len(selected) else np.nan)
    return entry


summary_rows = []
for label, _metrics_file, _model, root, template in FORCINGS:
    for seed in SEEDS:
        path = os.path.join(EXPERIMENT_ROOT, root, template.format(seed=seed),
                            "reforecast_2026H1_origin_window_metrics.csv")
        if not os.path.exists(path):
            print(f"[fig14] missing {path}", flush=True)
            continue
        entry = {"forcing": label, "kind": "stgnn", "seed": seed}
        entry.update(window_statistics(path))
        summary_rows.append(entry)
    print(f"[fig14] {label} done", flush=True)

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(TARGET, "fig13_window_summary_by_seed_jan_aug.csv")
summary.to_csv(summary_path, index=False)
print(f"[fig14] wrote {summary_path} {summary.shape}", flush=True)

check = summary.groupby("forcing")[["n_rises_ge_0.6096", "missed_crest_pct_ge_0.6096",
                                    "mean_peak_abs_error_m_ge_0.6096"]].mean()
print(check.round(4).to_string(), flush=True)
print("FIG13_FIG14_INPUTS_DONE", flush=True)
