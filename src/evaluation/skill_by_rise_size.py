"""How often each model beats persistence in each rise group, January to August 2026.

Why this exists
---------------
Sections 3.3 and 5 state how often the ST-GNN and the LSTM have the lower error than persistence in
each rise group: 62.3 % and 62.1 % in near-flat conditions, 54.3 % and 56.2 % for moderate rises,
65.9 % and 68.9 % between 0.30 and 0.61 m, and 71.9 % and 72.7 % for rises of at least 0.61 m. Those
ten numbers cover January to June and no file holds them for the extended period.

They cannot be taken from the scored tables. Persistence is not an archive and not a column: the
recipe builds it, as the raw stage at the issue time held flat across the whole 24 h window. The
window scorer never writes it.

The recipe
----------
Copied from paper_rev5/scripts/04_summarize_model_comparison.py, lines 122 to 187, so the numbers
stay comparable with the published ones:

  truth        stage in feet, masked where the causal trailing 96-step standard deviation is below
               1e-6, over the 96 steps after the origin
  persistence  the RAW stage at the origin, unmasked, repeated across all 96 steps
  forecast     the P80 line of each archive, which is index 1 of the last axis of pred_ft
  window RMSE  root mean square error of one forecast over its first 96 steps, per gauge-cycle
  matched      gauge-cycles where every model has a finite window RMSE
  rise         the maximum of truth minus its first finite value, needing at least 8 finite steps
  groups       in feet: below 0.30, 0.30 to 1.00, 1.00 to 2.00, and 2.00 and above, which is the
               0.61 m group the manuscript quotes
  statistic    the share of matched gauge-cycles in a group whose window RMSE is below persistence's,
               computed per seed and then averaged over the three seeds

One deliberate difference from the published run
------------------------------------------------
The published run also loaded a third trained model, and "matched" required every loaded model to be
finite at a gauge-cycle. That model is no longer in the manuscript, so it is not loaded here. The
matched set therefore differs slightly from the published one by construction, not by period. This is
stated in the report so the two are never compared as if they were identical.

Outputs:
  ../outputs/rise_strata_jan_aug.csv
  ../outputs/RISE_STRATA_JAN_AUG_REPORT.txt

Usage:
  conda run -n operational python -u skill_by_rise_size.py
"""
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
RUNS = os.path.join(EXPERIMENT_ROOT, "runs")
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

PICKLE = ("project/hpc/Experiments/"
          "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
          "global_features_all_stations_feature_engineered_20260911.pkl")
INPARISH = ("project/Experiments/"
            "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/in_parish_gauges.json")

SEEDS = [101, 202, 303]
FT2M = 0.3048
LB_W = 96
TARGET_QUANTILE = 0.8
MINIMUM_FINITE_STEPS = 8
# Feet, as in the published recipe. 2.00 ft is the 0.61 m group the manuscript quotes.
RISE_GROUPS = [
    ("rise < 0.09 m", -np.inf, 0.30),
    ("0.09-0.30 m", 0.30, 1.00),
    ("0.30-0.61 m", 1.00, 2.00),
    ("rise >= 0.61 m", 2.00, np.inf),
]
MODELS = {"ST-GNN": "stgnn", "LSTM": "lstm"}
FORCING = "observed"

lines = []


def out(*parts):
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    lines.append(text)


gauges = sorted(json.load(open(INPARISH))["nodes"])
G = len(gauges)
out(f"[strata] in-parish gauges: {G}")

loaded = {}
common = None
for model, prefix in MODELS.items():
    for seed in SEEDS:
        path = os.path.join(RUNS, f"{prefix}_seed{seed}_{FORCING}_jan_aug.npz")
        if not os.path.exists(path):
            raise SystemExit("[FATAL] missing archive: " + path)
        with np.load(path, allow_pickle=True) as archive:
            origins = pd.DatetimeIndex(pd.to_datetime([str(v) for v in archive["origins_utc"]]))
            node_list = [str(node) for node in archive["nodes"]]
            saved = [float(value) for value in archive["quantiles_saved"]]
            if TARGET_QUANTILE not in saved:
                raise SystemExit(f"[FATAL] {path} stores {saved}, so P80 is not available")
            quantile_index = saved.index(TARGET_QUANTILE)
            columns = [node_list.index(gauge) for gauge in gauges]
            predicted = archive["pred_ft"][:, :, columns, quantile_index].astype(np.float32)
        loaded[(model, seed)] = (origins, predicted)
        common = origins if common is None else common.intersection(origins)
        out(f"[strata] loaded {model} seed {seed} {predicted.shape}")

origins = pd.DatetimeIndex(sorted(common))
out(f"[strata] common origins: {len(origins)}  {origins[0]} -> {origins[-1]}")

frame = pd.read_pickle(PICKLE).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
row_of = {time_point: position for position, time_point in enumerate(frame.index)}
missing = [time_point for time_point in origins if time_point not in row_of]
if missing:
    raise SystemExit(f"[FATAL] {len(missing)} origins are not in the feature matrix, first {missing[0]}")

masked = np.full((len(frame), G), np.nan, np.float32)
raw = np.full_like(masked, np.nan)
for position, gauge in enumerate(gauges):
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    raw[:, position] = series.values
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    masked[:, position] = series.mask(trailing < 1e-6).values

positions = np.asarray([row_of[time_point] for time_point in origins])
truth = masked[positions[:, None] + 1 + np.arange(LB_W)[None, :]]

predictions = {}
for key, (archive_origins, predicted) in loaded.items():
    where = {value: index for index, value in enumerate(archive_origins)}
    predictions[key] = predicted[[where[value] for value in origins]]
# Persistence holds the raw stage at the issue time across the window. It is built, never scored.
predictions[("Persistence", "")] = np.repeat(raw[positions][:, None, :], LB_W, axis=1)


def window_rmse(predicted):
    error = predicted[:, :LB_W, :] - truth[:, :LB_W, :]
    valid = np.isfinite(error)
    count = valid.sum(axis=1)
    return np.where(count > 0,
                    np.sqrt(np.where(valid, error ** 2, 0.0).sum(axis=1) / np.maximum(count, 1)),
                    np.nan)


errors = {key: window_rmse(predicted) for key, predicted in predictions.items()}
matched = np.logical_and.reduce([np.isfinite(errors[key]) for key in predictions])
out(f"[strata] matched gauge-cycles: {int(matched.sum()):,} of {matched.size:,}")

finite = np.isfinite(truth)
enough = finite.sum(axis=1) >= MINIMUM_FINITE_STEPS
first_index = np.argmax(finite, axis=1)
first_value = np.take_along_axis(truth, first_index[:, None, :], axis=1)[:, 0, :]
rise_ft = np.where(enough, np.nanmax(np.where(finite, truth, -np.inf), axis=1) - first_value, np.nan)

rows = []
base = matched & np.isfinite(rise_ft)
for (model, seed) in predictions:
    if model == "Persistence":
        continue
    for label, low, high in RISE_GROUPS:
        selected = base & (rise_ft >= low) & (rise_ft < high)
        mine = errors[(model, seed)][selected]
        persist = errors[("Persistence", "")][selected]
        rows.append({
            "model": model,
            "seed": seed,
            "rise_group": label,
            "n_gauge_cycles": int(selected.sum()),
            "median_rmse24_m": float(np.median(mine)) * FT2M,
            "median_persistence_rmse24_m": float(np.median(persist)) * FT2M,
            "beats_persistence_pct": 100.0 * float(np.mean(mine < persist)),
        })

strata = pd.DataFrame(rows)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
strata_path = os.path.join(OUTPUT_DIRECTORY, "rise_strata_jan_aug.csv")
strata.to_csv(strata_path, index=False)
out("[strata] wrote " + strata_path + f" {strata.shape}")

pooled = (
    strata.groupby(["rise_group", "model"], as_index=False)
    .agg(beats_persistence_pct=("beats_persistence_pct", "mean"),
         lowest_seed_pct=("beats_persistence_pct", "min"),
         highest_seed_pct=("beats_persistence_pct", "max"),
         median_rmse24_m=("median_rmse24_m", "mean"),
         n_gauge_cycles=("n_gauge_cycles", "mean"))
)
order = {label: index for index, (label, _, _) in enumerate(RISE_GROUPS)}
pooled = pooled.sort_values(["rise_group", "model"], key=lambda s: s.map(order) if s.name == "rise_group" else s)

out("")
out("=" * 100)
out("Share of gauge-cycles with a lower 24 h window RMSE than persistence, by rise group")
out("1 January to 31 August 2026, observed rainfall, P80 line, seed mean of three seeds")
out("=" * 100)
out(pooled.round(3).to_string(index=False))
out("")
out("The published January to June values were, for the ST-GNN and the LSTM:")
out("  near flat 62.3 and 62.1, moderate 54.3 and 56.2, 0.30-0.61 m 65.9 and 68.9,")
out("  0.61 m and above 71.9 and 72.7.")
out("")
out("The published run also loaded a third trained model that the manuscript no longer contains, and")
out("it required every loaded model to be finite at a gauge-cycle. That model is not loaded here, so")
out("the matched set differs slightly by construction as well as by period.")

report_path = os.path.join(OUTPUT_DIRECTORY, "RISE_STRATA_JAN_AUG_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("[strata] wrote", report_path, flush=True)
print("RISE_STRATA_JAN_AUG_DONE", flush=True)
