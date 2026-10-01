"""False-peak rate of the ST-GNN ensemble run (six-model mean), January to August 2026.

false_peaks.py covers observed rainfall and issue-time HRRR only. Farid asked on 2026-09-25 for the
ensemble run's rate next to them. The definition, the observed rise, and the gauges are the same as there:
a false peak is a 24 h window in which the observed stage rose less than 0.05 m while the forecast rose at least
0.15 m; both rises are the window maximum minus the observed stage at the issue time; the rate is false peaks
divided by the windows whose observed rise is below 0.05 m. The ensemble archive is
../runs_ens_jan_aug/ens6_mean_seed<S>_jan_aug.npz, which has the same layout as the single-forcing archives
(pred_ft with quantiles [0.5, 0.8, 0.9], issue_stage_ft).

Output: ../outputs/false_peaks_ensemble_jan_aug.csv, printed seed mean per quantile
"""
import json
import os

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORED = os.path.join(EXPERIMENT_ROOT, "scored_jan_aug")
ENSEMBLE_RUNS = os.path.join(EXPERIMENT_ROOT, "runs_ens_jan_aug")
OUTPUT = os.path.join(EXPERIMENT_ROOT, "outputs", "false_peaks_ensemble_jan_aug.csv")
INPARISH_MANIFEST = ("project/Experiments/INPARISH51_20260813/"
                     "outputs/in_parish_gauges.json")
SEEDS = [101, 202, 303]
FT2M = 0.3048
FLAT_M = 0.05
SPIKE_M = 0.1524
EXCLUDED_GAUGES = {"MBSA4570"}
QUANTILE_INDEX = {"P50": 0, "P80": 1, "P90": 2}

gauges = set(json.load(open(INPARISH_MANIFEST))["nodes"]) - EXCLUDED_GAUGES

rows = []
for seed in SEEDS:
    window = pd.read_csv(os.path.join(SCORED, f"observed_seed{seed}", "reforecast_2026H1_origin_window_metrics.csv"),
                         usecols=["t0_utc", "gauge", "window_hours", "observed_rise_m"])
    window = window[(window["window_hours"] == 24) & window["gauge"].isin(gauges)].dropna(subset=["observed_rise_m"])
    window["t0_utc"] = pd.to_datetime(window["t0_utc"])
    observed = window.set_index(["t0_utc", "gauge"])["observed_rise_m"]
    with np.load(os.path.join(ENSEMBLE_RUNS, f"ens6_mean_seed{seed}_jan_aug.npz"), allow_pickle=True) as archive:
        saved = [float(value) for value in archive["quantiles_saved"]]
        nodes = [str(node) for node in archive["nodes"]]
        origins = pd.to_datetime([str(value) for value in archive["origins_utc"]])
        predicted_all = archive["pred_ft"]
        issue = archive["issue_stage_ft"]
    for quantile, index in QUANTILE_INDEX.items():
        if abs(saved[index] - float(quantile[1:]) / 100.0) > 1e-6:
            raise SystemExit(f"[FATAL] seed {seed} stores {saved}, so {quantile} is not at index {index}")
        rise = (np.nanmax(predicted_all[:, :, :, index], axis=1) - issue) * FT2M
        frame = pd.DataFrame(rise, index=origins, columns=nodes)
        frame = frame[[node for node in nodes if node in gauges]]
        forecast = frame.stack(future_stack=True).dropna()
        forecast.index = forecast.index.set_names(["t0_utc", "gauge"])
        joined = pd.concat({"observed": observed, "forecast": forecast}, axis=1).dropna()
        if joined.empty:
            raise SystemExit("[FATAL] the observed and forecast tables share no (origin, gauge) key")
        flat = joined[joined["observed"] < FLAT_M]
        false_peaks = int((flat["forecast"] >= SPIKE_M).sum())
        rows.append({"model": "ST-GNN", "forcing": "six-model mean", "quantile": quantile, "seed": seed,
                     "flat_windows": int(len(flat)), "false_peaks": false_peaks,
                     "false_peak_pct": 100.0 * false_peaks / len(flat)})

table = pd.DataFrame(rows)
table.to_csv(OUTPUT, index=False)
print(table.groupby("quantile")["false_peak_pct"].agg(["mean", "min", "max"]).round(3).to_string())
print(table.pivot_table(index="quantile", columns="seed", values="flat_windows").to_string())
print("[false peaks ensemble] wrote", OUTPUT)
