#!/usr/bin/env python3
"""
Download one archived rainfall-forecast record per numerical weather prediction model, so the
reforecast can be re-run once per model and the results combined into a forcing ensemble.

WHAT THIS IS FOR
----------------
The manuscript's Sect. 5.1 currently contrasts a single archived rainfall forecast against observed
rainfall. This builds the material for contrasting observed rainfall against an ENSEMBLE of
independent operational forecasts instead, which is a fairer picture of what a forecaster actually
has available and lets the spread between weather centres be reported rather than hidden.

The ensembling dimension is the WEATHER MODEL, not the model's own output quantiles.

EXACTLY MATCHES THE EXISTING SINGLE-FORCING RECORD
--------------------------------------------------
Format, units, grid and station list are identical to
Experiments/REFORECAST_2026H1/<SERVER>/fetch_archived_qpf_2026H1.py, so each pickle written here is
a drop-in replacement for that file via the REFORECAST_QPF_PICKLE environment variable:

  - variable precipitation_previous_day1, the value predicted about 24 h before each valid time
  - hourly values spread uniformly onto the 15-min grid (hourly / 4)
  - columns <CODE>_rain_in_qpf in inches, 15-min UTC index
  - the same rain stations, taken from the nearest_rain column of stage_to_rain_gauge.csv
  - the same coordinates, parsed out of the production downloader

The single existing record used models="best_match". That was measured to be bit-identical to
gfs_hrrr, so the manuscript's existing forecast-rainfall run is in fact the HRRR member of this
ensemble, and it is re-fetched here under its own name rather than assumed.

WHY THE REQUESTS ARE BATCHED BY COORDINATE
------------------------------------------
Open-Meteo accepts many latitudes and longitudes in one request. Fetching each station separately
would be about 3,400 requests; batching them is about 230, which is both far faster and much
gentler on a free public API.

Run:
    conda run -n operational python weather_model_fetch.py
"""

import ast
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

DOWNLOAD_OPERATIONAL_PY = "project/Inference/Download_Operational.py"
STAGE_TO_RAIN_CSV = "src/data/stage_to_rain_gauge.csv"
EXP = "project/Experiments/QPF_ENSEMBLE_20260831"
OUT = os.path.join(EXP, "outputs")
QPF_DIR = os.environ.get("QPFENS_QPF_DIR", os.path.join(EXP, "qpf"))
os.makedirs(QPF_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLE = "precipitation_previous_day1"
# Defaults are the manuscript's six-month window and are unchanged. They can be overridden by
# environment variable so the same fetcher can build a matched comparison window for the HREF
# member, which is collected live and therefore covers recent dates instead.
START_DATE = os.environ.get("QPFENS_START", "2026-01-01")
END_DATE = os.environ.get("QPFENS_END", "2026-07-02")
BATCH = 15                 # coordinates per request
SLEEP = 1.5
MAX_RETRY = 4
TIMEOUT = 180

# One member per forecasting centre or system. Duplicate Open-Meteo aliases were collapsed by
# select_ensemble_members.py; KNMI and MET Norway were dropped because their limited-area domains
# do not cover Louisiana, and gem_seamless was dropped as the same centre as gem_global.
MEMBERS = [
    ("gfs_hrrr",                       "HRRR (NOAA, 3 km CONUS)"),
    ("gfs_global",                     "GFS (NOAA, 0.25 deg global)"),
    ("ncep_nam_conus",                 "NAM (NOAA, CONUS mesoscale)"),
    ("ncep_nbm_conus",                 "NBM (NOAA National Blend of Models)"),
    ("gfs_graphcast025",               "GraphCast (machine learning, run by NOAA)"),
    ("ecmwf_ifs025",                   "IFS (ECMWF, 0.25 deg global)"),
    ("icon_global",                    "ICON (DWD, global)"),
    ("gem_global",                     "GEM (Environment Canada, global)"),
    ("ukmo_global_deterministic_10km", "UM (UK Met Office, 10 km global)"),
    ("meteofrance_seamless",           "ARPEGE/AROME (Meteo-France)"),
    ("jma_seamless",                   "GSM/MSM (Japan Meteorological Agency)"),
]


def log(*a):
    print("[qpf-ens]", *a, flush=True)


def resolve_coords(code, node_coords):
    if code in node_coords:
        return node_coords[code], "direct"
    for alias in (code.replace("RA", "SA", 1), code.replace("RU", "SU", 1)):
        if alias in node_coords:
            return node_coords[alias], "alias:" + alias
    pref = [k for k in node_coords if k[:4] == code[:4]]
    if pref:
        return node_coords[pref[0]], "prefix:" + pref[0]
    return None, "missing"


def request_batch(model, lats, lons, start, end):
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{v:.6f}" for v in lats),
        "longitude": ",".join(f"{v:.6f}" for v in lons),
        "hourly": VARIABLE,
        "start_date": start,
        "end_date": end,
        "models": model,
        "timezone": "UTC",
        "precipitation_unit": "inch",
    })
    url = f"{API_URL}?{q}"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                payload = json.loads(r.read().decode())
            # a multi-coordinate request returns a list; a single one returns a dict
            return payload if isinstance(payload, list) else [payload]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:160]
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRY:
                wait = SLEEP * (2 ** attempt)
                log(f"    HTTP {e.code}, retry {attempt}/{MAX_RETRY} in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise SystemExit(f"[FATAL] {model}: HTTP {e.code} {body}")
        except Exception as e:
            if attempt < MAX_RETRY:
                wait = SLEEP * (2 ** attempt)
                log(f"    {type(e).__name__}, retry {attempt}/{MAX_RETRY} in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise SystemExit(f"[FATAL] {model}: {type(e).__name__}: {e}")


def main():
    log("parsing coordinates from", DOWNLOAD_OPERATIONAL_PY)
    src = open(DOWNLOAD_OPERATIONAL_PY).read()
    m = re.search(r"HARDCODED_NODE_COORDS\s*=\s*(\{.*?\n\})", src, re.S)
    if m is None:
        raise SystemExit("[FATAL] HARDCODED_NODE_COORDS not found")
    node_coords = ast.literal_eval(m.group(1))

    rain_codes = sorted(set(pd.read_csv(STAGE_TO_RAIN_CSV)["nearest_rain"].astype(str)))
    coords, used, missing = [], [], []
    for c in rain_codes:
        xy, how = resolve_coords(c, node_coords)
        if xy is None:
            missing.append(c)
            continue
        lat, lon = (xy["lat"], xy["lon"]) if isinstance(xy, dict) else (xy[0], xy[1])
        coords.append((float(lat), float(lon)))
        used.append(c)
    log(f"{len(used)} rain stations resolved, {len(missing)} missing {missing}")

    grid = pd.date_range(START_DATE, END_DATE, freq="15min", tz="UTC", inclusive="left")
    edges = pd.date_range(START_DATE, END_DATE, freq="MS").strftime("%Y-%m-%d").tolist()
    if edges[0] != START_DATE:
        edges = [START_DATE] + edges
    if edges[-1] != END_DATE:
        edges = edges + [END_DATE]
    chunks = list(zip(edges[:-1], edges[1:]))
    batches = [(used[i:i + BATCH], coords[i:i + BATCH]) for i in range(0, len(used), BATCH)]
    log(f"{len(MEMBERS)} members x {len(chunks)} chunks x {len(batches)} batches = "
        f"{len(MEMBERS) * len(chunks) * len(batches)} requests")

    report = []
    for model, label in MEMBERS:
        path = os.path.join(QPF_DIR, f"qpf_{model}.pkl")
        if os.path.exists(path):
            log(f"{model}: already fetched -> {path}")
            continue
        log(f"=== {model}  ({label})")
        out = pd.DataFrame(index=grid)
        for codes, cc in batches:
            got = {c: [] for c in codes}
            for s, e in chunks:
                payloads = request_batch(model, [a for a, _ in cc], [b for _, b in cc], s, e)
                if len(payloads) != len(codes):
                    raise SystemExit(f"[FATAL] {model}: asked {len(codes)} coords, "
                                     f"got {len(payloads)} payloads")
                for c, p in zip(codes, payloads):
                    h = p.get("hourly") or {}
                    key = VARIABLE if VARIABLE in h else next(
                        (k for k in h if k.startswith("precipitation")), None)
                    if key is None:
                        raise SystemExit(f"[FATAL] {model}/{c}: no precipitation key")
                    got[c].append(pd.Series(
                        h[key], index=pd.to_datetime(h["time"]).tz_localize("UTC"),
                        dtype="float64"))
                time.sleep(SLEEP)
            for c in codes:
                ser = pd.concat(got[c]).sort_index()
                ser = ser[~ser.index.duplicated(keep="last")]
                # hourly total spread uniformly across the four 15-min steps of that hour
                q = ser.reindex(grid, method="ffill", limit=3) / 4.0
                out[f"{c}_rain_in_qpf"] = q.astype("float32")
            log(f"  batch of {len(codes)} done ({codes[0]} .. {codes[-1]})")

        out.to_pickle(path)
        tot = float(np.nansum(out.to_numpy()))
        cov = float(np.isfinite(out.to_numpy()).mean())
        log(f"  wrote {path}  shape {out.shape}  finite {cov:.1%}  basin-total {tot:.1f} in")
        report.append({"model": model, "label": label, "path": path,
                       "rows": int(out.shape[0]), "cols": int(out.shape[1]),
                       "finite_fraction": cov, "total_inches_all_stations": tot})

    if report:
        pd.DataFrame(report).to_csv(os.path.join(OUT, "qpf_fetch_report.csv"), index=False)
        log("wrote " + os.path.join(OUT, "qpf_fetch_report.csv"))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
