"""False peaks over January to August 2026, for the last paragraph of Section 4.1.

The definition, unchanged from
project/Experiments/SYSTEM_B_EVENTS_CROSSCHECK_20260914/VERDICT.md:

  a false peak is a 24 h forecast window in which the observed stage rose less than 0.05 m while the
  P80 forecast rose at least 0.15 m. Both rises are the window maximum minus the observed stage at
  the issue time. The rate is false peaks divided by the windows whose observed rise is below 0.05 m.

Why this runs at all
--------------------
The manuscript quotes false-peak rates for January to June. No January to August version exists. The
paragraph cannot simply be renumbered.

No model runs again. Two files supply everything:

  the archive   runs/<kind>_seed<S>_<forcing>_jan_aug.npz holds pred_ft with shape
                (origins, 96, gauges, 3) and quantiles_saved [0.5, 0.8, 0.9], so the P80 line is
                index 1 of the last axis, and issue_stage_ft holds the stage at the issue time. The
                forecast rise is therefore the window maximum of that line minus the issue stage.
  the scorer    scored_jan_aug/observed_seed<S>/reforecast_2026H1_origin_window_metrics.csv holds
                observed_rise_m per origin and gauge.

observed_rise_m describes the observed record, not the model, so one scored directory serves every
model. That is why the LSTM and the GRU need no window scoring of their own.

Two details that will silently give wrong answers if ignored:
  observed_rise_m is NaN unless window_hours is 24, so the rows must be filtered to that window.
  t0_utc in the window metrics is the issue time. The time series file is stamped with the verifying
  time instead, which is 24 h later, and is not used here.

The P50 and P90 rates are reported as well, because the paragraph ends by saying that the published
quantile matters.

Outputs:
  ../outputs/false_peaks_jan_aug.csv
  ../outputs/FALSE_PEAKS_JAN_AUG_REPORT.txt

Usage:
  conda run -n operational python -u false_peaks.py
"""
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
RUNS = os.path.join(EXPERIMENT_ROOT, "runs")
SCORED = os.path.join(EXPERIMENT_ROOT, "scored_jan_aug")
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")
INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/in_parish_gauges.json"
)

SEEDS = [101, 202, 303]
FT2M = 0.3048
FLAT_M = 0.05
SPIKE_M = 0.1524
EXCLUDED_GAUGES = {"MBSA4570"}
QUANTILE_INDEX = {"P50": 0, "P80": 1, "P90": 2}

MODELS = [("stgnn", "ST-GNN"), ("lstm", "LSTM"), ("gru", "GRU (identity graph)")]
FORCINGS = [("observed", "observed rainfall"), ("hrrr", "issue-time HRRR")]

pd.set_option("display.width", 200)

lines = []


def out(*parts):
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    lines.append(text)


gauges = sorted(set(json.load(open(INPARISH_MANIFEST))["nodes"]) - EXCLUDED_GAUGES)
out(f"[false peaks] in-parish gauges: {len(gauges)}")


def observed_rise_table(seed):
    """Observed rise per origin and gauge, from the 24 h window rows of the scorer.

    The origin is parsed to a timestamp rather than kept as text. The archive and the scorer both
    write the origin as a 19-character string, but not necessarily in the same shape:
    "2026-01-01 00:00:00" and "2026-01-01T00:00:00" are both 19 characters and never compare equal
    as text. Comparing the parsed timestamps removes that trap.
    """
    path = os.path.join(SCORED, f"observed_seed{seed}", "reforecast_2026H1_origin_window_metrics.csv")
    if not os.path.exists(path):
        raise SystemExit("[FATAL] missing window metrics: " + path)
    frame = pd.read_csv(path, usecols=["t0_utc", "gauge", "window_hours", "observed_rise_m"])
    frame = frame[frame["window_hours"] == 24]
    frame = frame[frame["gauge"].isin(gauges)]
    frame = frame.dropna(subset=["observed_rise_m"])
    if frame.empty:
        raise SystemExit(
            "[FATAL] no 24 h window rows with an observed rise in " + path
            + "\n  observed_rise_m is NaN unless window_hours is 24. If it is NaN there too, the"
              " scorer did not compute window statistics for this run.")
    frame["t0_utc"] = pd.to_datetime(frame["t0_utc"])
    return frame.set_index(["t0_utc", "gauge"])["observed_rise_m"]


def forecast_rise_table(kind, forcing, seed, quantile):
    """Forecast rise per origin and gauge: the window maximum of one quantile minus the issue stage."""
    path = os.path.join(RUNS, f"{kind}_seed{seed}_{forcing}_jan_aug.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as archive:
        saved = [float(value) for value in archive["quantiles_saved"]]
        index = QUANTILE_INDEX[quantile]
        if abs(saved[index] - float(quantile[1:]) / 100.0) > 1e-6:
            raise SystemExit(f"[FATAL] {path} stores {saved}, so {quantile} is not at index {index}")
        nodes = [str(node) for node in archive["nodes"]]
        origins = pd.to_datetime([str(value) for value in archive["origins_utc"]])
        predicted = archive["pred_ft"][:, :, :, index]
        issue = archive["issue_stage_ft"]
    rise_ft = np.nanmax(predicted, axis=1) - issue
    frame = pd.DataFrame(rise_ft * FT2M, index=origins, columns=nodes)
    frame = frame[[node for node in nodes if node in set(gauges)]]
    stacked = frame.stack(future_stack=True).dropna()
    stacked.index = stacked.index.set_names(["t0_utc", "gauge"])
    return stacked


rows = []
for kind, model_label in MODELS:
    for forcing, forcing_label in FORCINGS:
        for seed in SEEDS:
            observed = observed_rise_table(seed)
            quantiles = ["P50", "P80", "P90"] if kind == "stgnn" else ["P80"]
            for quantile in quantiles:
                forecast = forecast_rise_table(kind, forcing, seed, quantile)
                if forecast is None:
                    out(f"[skip] no archive for {kind} {forcing} seed {seed}")
                    continue
                joined = pd.concat({"observed": observed, "forecast": forecast}, axis=1).dropna()
                # An empty join must stop the run. Without this the script reports "no rates" for a
                # key mismatch, which reads like a data gap instead of a bug.
                if joined.empty:
                    raise SystemExit(
                        "[FATAL] the observed and forecast tables share no (origin, gauge) key.\n"
                        f"  observed rows {len(observed)}, sample keys {observed.index[:2].tolist()}\n"
                        f"  forecast rows {len(forecast)}, sample keys {forecast.index[:2].tolist()}\n"
                        "  Both sides must use the same timestamp type and the same gauge codes.")
                flat = joined[joined["observed"] < FLAT_M]
                if flat.empty:
                    continue
                false_peaks = int((flat["forecast"] >= SPIKE_M).sum())
                rows.append({
                    "model": model_label,
                    "forcing": forcing_label,
                    "quantile": quantile,
                    "seed": seed,
                    "flat_windows": int(len(flat)),
                    "false_peaks": false_peaks,
                    "false_peak_pct": 100.0 * false_peaks / len(flat),
                })

table = pd.DataFrame(rows)
if table.empty:
    raise SystemExit("[FATAL] no false-peak rates were computed")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
table_path = os.path.join(OUTPUT_DIRECTORY, "false_peaks_jan_aug.csv")
table.to_csv(table_path, index=False)
out("[false peaks] wrote " + table_path + f" {table.shape}")

pooled = (
    table.groupby(["model", "forcing", "quantile"], as_index=False)
    .agg(false_peak_pct=("false_peak_pct", "mean"),
         lowest_seed_pct=("false_peak_pct", "min"),
         highest_seed_pct=("false_peak_pct", "max"),
         flat_windows=("flat_windows", "mean"))
)

out("")
out("=" * 100)
out("False-peak rate, 1 January to 31 August 2026, 24 h windows, 51 in-parish gauges")
out("a false peak is a flat window (observed rise below 0.05 m) with a forecast rise of at least 0.15 m")
out("=" * 100)
out(pooled.round(3).to_string(index=False))

out("")
out("per seed, P80 only")
out(table[table["quantile"] == "P80"]
    .pivot_table(index=["model", "forcing"], columns="seed", values="false_peak_pct")
    .round(3).to_string())

report_path = os.path.join(OUTPUT_DIRECTORY, "FALSE_PEAKS_JAN_AUG_REPORT.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("[false peaks] wrote", report_path, flush=True)
print("FALSE_PEAKS_JAN_AUG_DONE", flush=True)
