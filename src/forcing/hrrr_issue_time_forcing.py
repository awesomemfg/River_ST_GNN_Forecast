"""Turn the archived HRRR cycles into one 24 h rainfall and wind forecast per hourly forecast origin.

For an origin t0 the forecast may use only HRRR cycles that were already published at t0. A cycle is
treated as available LAG hours after its initialization time. Two assembly rules are built:

  stitch    (what Open-Meteo serves): each future hour comes from the newest available cycle that
            reaches it. Hourly cycles reach 18 h; beyond that the newest 00/06/12/18 UTC cycle (48 h) is used.
  synoptic  every hour of the window comes from the newest available 00/06/12/18 UTC cycle.

A missing cycle falls back to the next older cycle of the same kind (the fallback count is recorded).
Rain: hourly accumulation spread evenly over its four 15 min steps (the same convention as the earlier
Open-Meteo archive in Experiments/QPF_ENSEMBLE_20260831). Wind: hourly 10 m speed, linearly interpolated
to 15 min.

Output: data/forcing/hrrr_issue_time_<variant>.npz
    origins_utc [4344]            hourly origins 2026-01-01 00 UTC to 2026-06-30 23 UTC
    rain_in     [4344, 96, 44]    forecast 15 min rain (inches) for steps t0+15 min .. t0+24 h
    wind_mph    [4344, 96, 6]     forecast 10 m wind speed (mph) at the same steps
    rain_codes, wind_codes, variant, lag_hours, rule
    fallback_hours [4344]         total cycle-fallback hours used by the origin (0 = all cycles present)
and data/forcing/forcing_summary.csv.

Run: conda run -n operational python -u hrrr_issue_time_forcing.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

EXP = "project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
CYCLE_DIR = os.path.join(EXP, "data", "cycles")
OUT_DIR = os.path.join(EXP, "data", "forcing")
os.makedirs(OUT_DIR, exist_ok=True)

# The origin window, the variant list and an output suffix are arguments so that a longer period can
# be built without touching the January to June files. Those files are hashed in the System B asset
# manifest and their hash is recorded in every run's metadata, so they must never be overwritten.
# The defaults reproduce the original January to June build exactly.
parser = argparse.ArgumentParser()
parser.add_argument("--origin-start", default="2026-01-01 00:00")
parser.add_argument("--origin-end", default="2026-06-30 23:00")
parser.add_argument("--variants", default="stitch_L1,stitch_L2,stitch_L3,synoptic_L2")
parser.add_argument("--suffix", default="")
args = parser.parse_args()

ORIGINS = pd.date_range(args.origin_start, args.origin_end, freq="h", tz="UTC")
ALL_VARIANTS = {
    "stitch_L1": ("stitch_L1", 1, "stitch"),
    "stitch_L2": ("stitch_L2", 2, "stitch"),
    "stitch_L3": ("stitch_L3", 3, "stitch"),
    "synoptic_L2": ("synoptic_L2", 2, "synoptic"),
}
VARIANTS = [ALL_VARIANTS[name.strip()] for name in args.variants.split(",") if name.strip()]
HOURLY_REACH = 18
MAX_FALLBACK = 6


def log(*parts):
    print("[forcing]", *parts, flush=True)


files = sorted(glob.glob(os.path.join(CYCLE_DIR, "hrrr_points_*.npz")))
if not files:
    raise SystemExit("[FATAL] no cycle files; run hrrr_fetch_points.py first")
cycle_times = []
apcp_list = []
ugrd_list = []
vgrd_list = []
rain_codes = None
wind_codes = None
for path in files:
    z = np.load(path)
    cycle_times.extend(pd.to_datetime(z["cycle_utc"]))
    apcp_list.append(z["apcp_mm"])
    ugrd_list.append(z["ugrd_ms"])
    vgrd_list.append(z["vgrd_ms"])
    rain_codes = [str(c) for c in z["rain_codes"]]
    wind_codes = [str(c) for c in z["wind_codes"]]
cycle_index = pd.DatetimeIndex(cycle_times)
if cycle_index.tz is None:
    cycle_index = cycle_index.tz_localize("UTC")
APCP = np.concatenate(apcp_list)
WIND = np.sqrt(np.concatenate(ugrd_list) ** 2 + np.concatenate(vgrd_list) ** 2) * 2.2369363
position = {t: i for i, t in enumerate(cycle_index)}
present = ~np.all(np.isnan(APCP[:, 0, :]), axis=1)
log(f"cycles loaded {len(cycle_index)}  present {int(present.sum())}  "
    f"range {cycle_index.min()} -> {cycle_index.max()}")


def find_cycle(target, synoptic_only):
    """Newest present cycle at or before target (of the requested kind); returns (index, fallback_hours)."""
    candidate = target
    for step in range(MAX_FALLBACK + 1):
        if (not synoptic_only) or candidate.hour % 6 == 0:
            i = position.get(candidate)
            if i is not None and present[i]:
                return i, int((target - candidate) / pd.Timedelta(hours=1))
        candidate = candidate - pd.Timedelta(hours=6 if synoptic_only else 1)
    raise SystemExit(f"[FATAL] no usable cycle near {target}")


def latest_synoptic(time_point):
    return time_point.floor("6h")


summary_rows = []
for variant, lag, rule in VARIANTS:
    rain = np.zeros((len(ORIGINS), 96, len(rain_codes)), dtype=np.float32)
    wind = np.zeros((len(ORIGINS), 96, len(wind_codes)), dtype=np.float32)
    fallback = np.zeros(len(ORIGINS), dtype=np.int32)
    for o, t0 in enumerate(ORIGINS):
        available = t0 - pd.Timedelta(hours=lag)
        i_syn, fb_syn = find_cycle(latest_synoptic(available), True)
        c_syn = cycle_index[i_syn]
        if rule == "stitch":
            i_hr, fb_hr = find_cycle(available, False)
            c_hr = cycle_index[i_hr]
        fallback[o] = fb_syn + (fb_hr if rule == "stitch" else 0)
        hourly_rain = np.zeros((24, len(rain_codes)), dtype=np.float32)
        hourly_wind = np.zeros((25, len(wind_codes)), dtype=np.float32)
        for j in range(0, 25):
            valid = t0 + pd.Timedelta(hours=j)
            use_hourly = False
            if rule == "stitch":
                fxx_hr = int((valid - c_hr) / pd.Timedelta(hours=1))
                use_hourly = 1 <= fxx_hr <= HOURLY_REACH
            if use_hourly:
                i_use = i_hr
                fxx = fxx_hr
            else:
                i_use = i_syn
                fxx = int((valid - c_syn) / pd.Timedelta(hours=1))
            if not 1 <= fxx <= 48:
                raise SystemExit(f"[FATAL] {variant} origin {t0} hour {j}: forecast hour {fxx} out of range")
            hourly_wind[j] = WIND[i_use, fxx - 1]
            if j >= 1:
                hourly_rain[j - 1] = APCP[i_use, fxx - 1] / 25.4
        if not np.all(np.isfinite(hourly_rain)):
            raise SystemExit(f"[FATAL] {variant} origin {t0}: non-finite rain")
        rain[o] = np.repeat(hourly_rain / 4.0, 4, axis=0)
        step_hours = np.arange(1, 97) / 4.0
        for w in range(len(wind_codes)):
            wind[o, :, w] = np.interp(step_hours, np.arange(0, 25), hourly_wind[:, w])
    out_path = os.path.join(OUT_DIR, f"hrrr_issue_time_{variant}{args.suffix}.npz")
    np.savez_compressed(out_path, origins_utc=np.asarray([t.isoformat() for t in ORIGINS]),
                        rain_in=rain, wind_mph=wind, rain_codes=np.asarray(rain_codes),
                        wind_codes=np.asarray(wind_codes), variant=variant, lag_hours=lag, rule=rule,
                        fallback_hours=fallback)
    summary_rows.append({"variant": variant, "lag_hours": lag, "rule": rule,
                         "origins": len(ORIGINS),
                         "origins_with_fallback": int((fallback > 0).sum()),
                         "mean_24h_rain_in_per_gauge": float(rain.sum(axis=1).mean()),
                         "max_15min_rain_in": float(rain.max()),
                         "path": out_path})
    log(variant, "written", out_path, "fallback origins", int((fallback > 0).sum()))

summary_path = os.path.join(OUT_DIR, f"forcing_summary{args.suffix}.csv")
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
log("summary", summary_path)
print("FORCING_BUILD_DONE", flush=True)
