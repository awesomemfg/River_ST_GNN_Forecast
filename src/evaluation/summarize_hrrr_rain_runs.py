"""The four summary files that Section 3.3's figures read, for January to August with HRRR rainfall.

Why this exists
---------------
Figures 10 and 11 of the manuscript both read the same four files:

  headline_numbers.json        median window RMSE per model and window, with the seed range
  persistence_headline.json    the persistence window RMSE and the matched gauge-cycle count
  fixed_lead_summary.csv       per-seed medians at the four fixed leads, with the NSE quartiles
  persistence_rise_strata.csv  median 24 h RMSE and the beat-persistence share, by rise group

Revision 7 produced them for the June real-time simulation. That experiment is gone, and Section 3.3
is now the same 5,832 origins driven by the issue-time HRRR forecast, so the same four files are
written here from the HRRR archives.

The recipe is copied from paper_rev5/scripts/04_summarize_model_comparison.py, which is the script
that produced the June versions. Its aggregation is unchanged:

  truth        stage in feet, masked where the causal trailing 96-step standard deviation is below
               1e-6, over the 96 steps after the origin
  persistence  the RAW stage at the origin, held flat across the window, built and never scored
  forecast     the P80 line, which is index 1 of the last axis of pred_ft
  window RMSE  per gauge-cycle over the first 12, 24, 48 or 96 steps
  matched      gauge-cycles where every loaded model is finite
  rise         the window maximum of truth minus its first finite value, needing 8 finite steps
  groups       in feet: below 0.30, 0.30 to 1.00, 1.00 to 2.00, and 2.00 and above

Two deliberate differences from the June files
----------------------------------------------
The third model is the GRU, not the model revision 7 carried. The figure recipes read the third
series by the key name "bilstm", so the GRU is written into those keys and the recipes are told to
label it GRU. Nothing else in the figures changes.

The GRU is the identity-graph model: the same architecture with no message passing between gauges.
Including it here means Section 3.3 shows what the graph is worth when the rainfall is a forecast.

Outputs, in ../outputs/summary_hrrr_jan_aug/:
  headline_numbers.json, persistence_headline.json, fixed_lead_summary.csv,
  fixed_lead_per_gauge.csv, persistence_rise_strata.csv, rise_strata.csv, window_rmse.csv

Usage:
  conda run -n operational python -u summarize_hrrr_rain_runs.py
"""
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
RUNS = os.path.join(EXPERIMENT_ROOT, "runs")
OUT = os.path.join(EXPERIMENT_ROOT, "outputs", "summary_hrrr_jan_aug")
PICKLE = ("project/hpc/Experiments/"
          "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
          "global_features_all_stations_feature_engineered_20260911.pkl")
INPARISH = ("project/Experiments/"
            "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json")

SEEDS = [101, 202, 303]
FT2M = 0.3048
LB_W = 96
Q80 = 0.8
WINDOWS = [("0-3h", 12), ("0-6h", 24), ("0-12h", 48), ("0-24h", 96)]
LEADS = {3: 11, 6: 23, 12: 47, 24: 95}
RISE_GROUPS = [("rise < 0.30 ft", -np.inf, 0.30), ("0.30-1.00 ft", 0.30, 1.00),
               ("1.00-2.00 ft", 1.00, 2.00), ("rise >= 2.00 ft", 2.00, np.inf)]
# The figure recipes read the third series by the "bilstm" key. The GRU occupies that slot.
MODEL_KEYS = {"ST-GNN": "stgnn", "LSTM": "lstm", "GRU": "bilstm", "Persistence": "persist"}
ARCHIVES = {"ST-GNN": "stgnn", "LSTM": "lstm", "GRU": "gru"}

os.makedirs(OUT, exist_ok=True)
gauges = sorted(json.load(open(INPARISH))["nodes"])
G = len(gauges)
print("[load] in-parish gauges:", G, flush=True)

loaded = {}
common = None
for model, prefix in ARCHIVES.items():
    for seed in SEEDS:
        path = os.path.join(RUNS, f"{prefix}_seed{seed}_hrrr_jan_aug.npz")
        if not os.path.isfile(path):
            raise SystemExit("[FATAL] missing " + path)
        with np.load(path, allow_pickle=True) as archive:
            times = pd.DatetimeIndex(pd.to_datetime([str(v) for v in archive["origins_utc"]]))
            nodes = [str(n) for n in archive["nodes"]]
            quantiles = [float(v) for v in archive["quantiles_saved"]]
            columns = [nodes.index(g) for g in gauges]
            pred = archive["pred_ft"][:, :, columns, quantiles.index(Q80)].astype(np.float32)
        loaded[(model, seed)] = (times, pred)
        common = times if common is None else common.intersection(times)
        print("[load]", model, seed, pred.shape, flush=True)

origins = pd.DatetimeIndex(sorted(common))
print("[load] common origins:", len(origins), origins[0], "->", origins[-1], flush=True)

frame = pd.read_pickle(PICKLE).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
row_of = {t: i for i, t in enumerate(frame.index)}
S = np.full((len(frame), G), np.nan, np.float32)
S_RAW = np.full_like(S, np.nan)
for i, gauge in enumerate(gauges):
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    S_RAW[:, i] = series.values
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    S[:, i] = series.mask(trailing < 1e-6).values

positions = np.asarray([row_of[t] for t in origins])
truth = S[positions[:, None] + 1 + np.arange(LB_W)[None, :]]

preds = {}
for key, (times, pred) in loaded.items():
    where = {v: i for i, v in enumerate(times)}
    preds[key] = pred[[where[v] for v in origins]]
preds[("Persistence", "")] = np.repeat(S_RAW[positions][:, None, :], LB_W, axis=1)


def window_rmse(pred, steps):
    err = pred[:, :steps, :] - truth[:, :steps, :]
    valid = np.isfinite(err)
    n = valid.sum(axis=1)
    return np.where(n > 0, np.sqrt(np.where(valid, err ** 2, 0.0).sum(axis=1) / np.maximum(n, 1)), np.nan)


wrmse = {key: {w: window_rmse(p, steps) for w, steps in WINDOWS} for key, p in preds.items()}
matched = {w: np.logical_and.reduce([np.isfinite(wrmse[key][w]) for key in preds]) for w, _ in WINDOWS}
print("[calc] matched gauge-cycles at 0-24h:", int(matched["0-24h"].sum()), flush=True)

finite = np.isfinite(truth)
enough = finite.sum(axis=1) >= 8
first_index = np.argmax(finite, axis=1)
first_value = np.take_along_axis(truth, first_index[:, None, :], axis=1)[:, 0, :]
rise_ft = np.where(enough, np.nanmax(np.where(finite, truth, -np.inf), axis=1) - first_value, np.nan)

window_rows = []
strata_rows = []
for (model, seed) in preds:
    for w, _steps in WINDOWS:
        v = wrmse[(model, seed)][w][matched[w]]
        window_rows.append({"model": model, "seed": seed, "window": w, "n_gauge_cycles": int(v.size),
                            "median_rmse_m": float(np.median(v)) * FT2M})
    base = matched["0-24h"] & np.isfinite(rise_ft)
    for label, lo, hi in RISE_GROUPS:
        m = base & (rise_ft >= lo) & (rise_ft < hi)
        mine = wrmse[(model, seed)]["0-24h"][m]
        persist = wrmse[("Persistence", "")]["0-24h"][m]
        strata_rows.append({"model": model, "seed": seed, "rise_group": label, "n": int(m.sum()),
                            "median_rmse24_m": float(np.median(mine)) * FT2M,
                            "beats_persist_frac": float(np.mean(mine < persist))})

fixed_rows = []
for (model, seed), pred in preds.items():
    for lead_hours, step in LEADS.items():
        for i, gauge in enumerate(gauges):
            observed = truth[:, step, i]
            forecast = pred[:, step, i]
            valid = np.isfinite(observed) & np.isfinite(forecast)
            observed = observed[valid].astype(float)
            forecast = forecast[valid].astype(float)
            error = forecast - observed
            nse = correlation = kge = np.nan
            observed_mean = float(np.mean(observed))
            observed_std = float(np.std(observed))
            forecast_std = float(np.std(forecast))
            denominator = float(np.sum((observed - observed_mean) ** 2))
            if denominator > 1.0e-9:
                nse = 1.0 - float(np.sum(error ** 2)) / denominator
            if observed_std > 1.0e-9 and forecast_std > 1.0e-9:
                correlation = float(np.corrcoef(observed, forecast)[0, 1])
            if np.isfinite(correlation) and abs(observed_mean) > 1.0e-9:
                kge = 1.0 - float(np.sqrt((correlation - 1.0) ** 2 + (forecast_std / observed_std - 1.0) ** 2
                                          + (float(np.mean(forecast)) / observed_mean - 1.0) ** 2))
            fixed_rows.append({"model": model, "seed": seed, "gauge": gauge, "lead_hours": lead_hours,
                               "n_pairs": int(valid.sum()),
                               "RMSE_m": float(np.sqrt(np.mean(error ** 2))) * FT2M,
                               "NSE": nse, "Pearson_r": correlation, "KGE": kge})

window = pd.DataFrame(window_rows)
strata = pd.DataFrame(strata_rows)
fixed = pd.DataFrame(fixed_rows)
window.to_csv(os.path.join(OUT, "window_rmse.csv"), index=False)
strata.to_csv(os.path.join(OUT, "rise_strata.csv"), index=False)
fixed.to_csv(os.path.join(OUT, "fixed_lead_per_gauge.csv"), index=False)

per_seed = fixed.groupby(["model", "seed", "lead_hours"]).agg(
    RMSE_median_m=("RMSE_m", "median"), NSE_median=("NSE", "median"),
    NSE_q25=("NSE", lambda x: np.nanpercentile(x, 25)), NSE_q75=("NSE", lambda x: np.nanpercentile(x, 75)),
    KGE_median=("KGE", "median"), r_median=("Pearson_r", "median"),
    NSE_positive_gauges=("NSE", lambda x: int((x > 0).sum()))).reset_index()
per_seed.to_csv(os.path.join(OUT, "fixed_lead_summary.csv"), index=False)


def seed_mean(frame_in, filters, column):
    sub = frame_in
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return float(sub[column].mean()), float(sub[column].min()), float(sub[column].max())


headline = {"source": "summarize_hrrr_rain_runs.py", "forcing": "hrrr", "period": "jan_aug",
            "origins": len(origins), "first_origin": str(origins[0]), "last_origin": str(origins[-1]),
            "gauges": G, "statistic": "median gauge-cycle RMSE; seeded models = mean of per-seed medians",
            "paired": {}}
persistence_headline = {"matched": {}}
for w, _ in WINDOWS:
    entry = {}
    for model, key in MODEL_KEYS.items():
        mean_value, low, high = seed_mean(window, {"model": model, "window": w}, "median_rmse_m")
        entry[key + "_median_rmse_ft"] = mean_value / FT2M
        entry[key + "_seed_range_m"] = [low, high]
        entry[key + "_median_rmse_m"] = mean_value
    headline["paired"][w] = entry
    persistence_headline["matched"][w] = {"median_rmse_ft": entry["persist_median_rmse_ft"],
                                          "n_gauge_cycles": int(matched[w].sum())}
json.dump(headline, open(os.path.join(OUT, "headline_numbers.json"), "w"), indent=1)
json.dump(persistence_headline, open(os.path.join(OUT, "persistence_headline.json"), "w"), indent=1)

recipe_rows = []
for label, _lo, _hi in RISE_GROUPS:
    row = {"rise_group": label}
    for model, key in MODEL_KEYS.items():
        row["median_rmse24_" + key] = seed_mean(strata, {"model": model, "rise_group": label},
                                                "median_rmse24_m")[0] / FT2M
        if model != "Persistence":
            row[key + "_beats_persist_frac"] = seed_mean(strata, {"model": model, "rise_group": label},
                                                         "beats_persist_frac")[0]
    row["n"] = int(strata[(strata["model"] == "Persistence") & (strata["rise_group"] == label)]["n"].iloc[0])
    recipe_rows.append(row)
pd.DataFrame(recipe_rows).to_csv(os.path.join(OUT, "persistence_rise_strata.csv"), index=False)

print(flush=True)
print("[check] median h+24 RMSE, mean over seeds, must match Section 3.3:", flush=True)
at24 = fixed[fixed["lead_hours"] == 24]
for model in ["ST-GNN", "LSTM", "GRU", "Persistence"]:
    per = at24[at24["model"] == model].groupby(["gauge"], as_index=False)["RMSE_m"].mean()
    print(f"   {model:12} {per['RMSE_m'].median():.4f} m", flush=True)
print("   Section 3.3 states 0.191 ST-GNN, 0.189 LSTM, 0.196 GRU, 0.171 persistence", flush=True)
print(flush=True)
print("[out]", OUT, flush=True)
print("SUMMARIZE_HRRR_JAN_AUG_DONE", flush=True)
