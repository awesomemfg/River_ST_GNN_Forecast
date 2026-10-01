#!/usr/bin/env python3
"""
Fetch Global Ensemble Forecast System (GEFS) rainfall at the parish rain gauges, for the same
January-June 2026 window the rest of the ensemble is scored on.

WHY GEFS, AND WHY IT IS DIFFERENT FROM EVERY MEMBER ALREADY HERE
----------------------------------------------------------------
Every current member (HRRR, GFS, NBM, IFS, GEM, JMA) is a single deterministic run of one weather
model. The ensemble built from them measures disagreement BETWEEN weather centres. GEFS is a true
ensemble in its own right: a control run plus 30 perturbed members from one centre, which measures
the uncertainty WITHIN a single forecasting system. Those are different quantities and neither
stands in for the other.

GEFS was chosen over HREF, REFS and the AI variants for one reason: it is the only candidate whose
archive actually covers the scored window. HREF has no archive at all (see VERDICT_HREF.md), REFS
is not operational until 6 October 2026, and Open-Meteo's ensemble archive begins 2026-06-08.
NBM is already a member.

WHAT IS FETCHED, AND THE LEAD IT IS HELD TO
-------------------------------------------
The other members come from Open-Meteo's precipitation_previous_day1, which is rainfall predicted
about 24 hours before it applies. GEFS is held to the same lead rather than given a shorter one:

    for cycle C in 00/06/12/18 UTC, take forecast hours 27 and 30, whose precipitation buckets are
    24-27 h and 27-30 h. Each cycle therefore supplies the six valid hours from C+24 to C+30, at a
    lead of 24 to 30 hours, and four cycles cover the whole day.

GEFS publishes precipitation in three-hour buckets, so each bucket is spread evenly across its
twelve 15-minute steps, the same treatment the hourly members get.

Only the precipitation record is pulled from each file, addressed by byte range out of the GRIB
index, so a member costs a few hundred megabytes instead of many gigabytes.

MEMBERS
-------
    geavg           the ensemble mean, the apples-to-apples seventh forcing member
    gec00           the control run
    gep01..gep30    the perturbed members, which are what make a spaghetti plot mean anything

Run:
    conda run -n operational python gefs_fetch.py --members geavg
    conda run -n operational python gefs_fetch.py --members gec00,gep01,gep02 --workers 8
"""

import argparse
import ast
import datetime as dt
import json
import os
import re
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

EXP = "project/Experiments/QPF_ENSEMBLE_20260831"
QPF = os.path.join(EXP, "qpf")
CACHE = os.path.join(EXP, "gefs", "cache")
OUT = os.path.join(EXP, "outputs")

DOWNLOAD_OPERATIONAL_PY = "project/Inference/Download_Operational.py"
STAGE_TO_RAIN_CSV = "src/data/stage_to_rain_gauge.csv"

BASE = ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.{day}/{cycle:02d}/atmos/pgrb2sp25/"
        "{member}.t{cycle:02d}z.pgrb2s.0p25.f{fxx:03d}")

START_DATE = "2026-01-01"
END_DATE = "2026-07-02"
CYCLES = (0, 6, 12, 18)
LEADS = (27, 30)                 # buckets 24-27 h and 27-30 h
BUCKET_HOURS = 3
MM_PER_INCH = 25.4
TIMEOUT = 90
MAX_RETRY = 4

for d in (QPF, CACHE, OUT):
    os.makedirs(d, exist_ok=True)

_print_lock = threading.Lock()
_grid_lock = threading.Lock()
_grid = {"idx": None}


def log(*a):
    with _print_lock:
        print("[gefs]", *a, flush=True)


def resolve_coords(code, node_coords):
    """The same resolution ladder the Open-Meteo members use, so the station set matches exactly."""
    if code in node_coords:
        return node_coords[code]
    for alias in (code.replace("RA", "SA", 1), code.replace("RU", "SU", 1)):
        if alias in node_coords:
            return node_coords[alias]
    pref = [k for k in node_coords if k[:4] == code[:4]]
    if pref:
        return node_coords[pref[0]]
    return None


def station_table():
    src = open(DOWNLOAD_OPERATIONAL_PY).read()
    m = re.search(r"HARDCODED_NODE_COORDS\s*=\s*(\{.*?\n\})", src, re.S)
    if m is None:
        raise SystemExit("[FATAL] HARDCODED_NODE_COORDS not found")
    node_coords = ast.literal_eval(m.group(1))

    rain_codes = sorted(set(pd.read_csv(STAGE_TO_RAIN_CSV)["nearest_rain"].astype(str)))
    codes = []
    lats = []
    lons = []
    for c in rain_codes:
        xy = resolve_coords(c, node_coords)
        if xy is None:
            continue
        lat, lon = (xy["lat"], xy["lon"]) if isinstance(xy, dict) else (xy[0], xy[1])
        codes.append(c)
        lats.append(float(lat))
        lons.append(float(lon))
    return codes, np.asarray(lats), np.asarray(lons)


def http(url, byte_range=None):
    """Fetch, backing off when S3 throttles.

    Without a pause between attempts a throttled request simply burns its retries at full speed and
    is lost. S3 answers 503 SlowDown under concurrency, so the wait grows with each attempt and is
    jittered to stop all the workers retrying in step with each other.
    """
    req = urllib.request.Request(url)
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    last = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            last = e
            if e.code in (500, 502, 503, 504) and attempt < MAX_RETRY:
                time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.6 + 0.8 * random.random()))
                continue
        except Exception as e:
            last = e
            if attempt < MAX_RETRY:
                time.sleep(min(20.0, 1.0 * (2 ** attempt)) * (0.6 + 0.8 * random.random()))
                continue
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def apcp_range(url):
    """Byte range of the precipitation record, from the GRIB index."""
    raw = http(url + ".idx")
    if raw is None:
        return None
    lines = raw.decode("utf-8", "replace").splitlines()
    for i, ln in enumerate(lines):
        if ":APCP:" in ln:
            start = int(ln.split(":")[1])
            if i + 1 < len(lines):
                end = int(lines[i + 1].split(":")[1]) - 1
            else:
                end = start + 40_000_000
            return start, end
    return None


def ensure_grid(blob, lats, lons):
    """Nearest GEFS grid point per station, computed once. Matching is done on the unit sphere."""
    with _grid_lock:
        if _grid["idx"] is not None:
            return _grid["idx"]
        import pygrib
        tmp = os.path.join(CACHE, "_grid_sample.grib2")
        with open(tmp, "wb") as f:
            f.write(blob)
        with pygrib.open(tmp) as gr:
            glats, glons = gr[1].latlons()
        os.remove(tmp)
        glons = np.where(glons > 180.0, glons - 360.0, glons)

        def unit(la, lo):
            la = np.radians(la)
            lo = np.radians(lo)
            return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

        from scipy.spatial import cKDTree
        tree = cKDTree(unit(glats.ravel(), glons.ravel()))
        _dist, idx = tree.query(unit(lats, lons), k=1)
        gl = glats.ravel()[idx]
        go = glons.ravel()[idx]
        dlat = np.radians(gl - lats)
        dlon = np.radians(go - lons)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lats)) * np.cos(np.radians(gl)) * np.sin(dlon / 2) ** 2)
        km = 6371.0088 * 2 * np.arcsin(np.sqrt(a))
        log(f"grid {glats.shape}, station match mean {km.mean():.1f} km, worst {km.max():.1f} km")
        _grid["idx"] = idx
        _grid["km_mean"] = float(km.mean())
        _grid["km_max"] = float(km.max())
        return idx


def read_at_stations(blob, idx):
    import pygrib
    tmp = os.path.join(CACHE, f"_rec_{threading.get_ident()}.grib2")
    with open(tmp, "wb") as f:
        f.write(blob)
    try:
        with pygrib.open(tmp) as gr:
            msg = gr[1]
            vals = np.asarray(msg.values, dtype="float64")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return vals.ravel()[idx]


def one_task(day, cycle, fxx, member, lats, lons):
    url = BASE.format(day=day, cycle=cycle, member=member, fxx=fxx)
    rng = apcp_range(url)
    if rng is None:
        return None
    blob = http(url, byte_range=rng)
    if blob is None:
        return None
    idx = ensure_grid(blob, lats, lons)
    vals = read_at_stations(blob, idx) / MM_PER_INCH     # kg m-2 is millimetres of water
    bucket_end = (dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
                  + dt.timedelta(hours=cycle + fxx))
    return bucket_end, vals


def fetch_member(member, days, codes, lats, lons, workers):
    """One member over the whole window. Cached to a pickle so a restart costs nothing."""
    cache = os.path.join(CACHE, f"gefs_{member}_buckets.pkl")
    all_tasks = [(d, c, f) for d in days for c in CYCLES for f in LEADS]

    def bucket_end(d, c, f):
        return (dt.datetime.strptime(d, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
                + dt.timedelta(hours=c + f))

    if os.path.exists(cache):
        have = pd.read_pickle(cache)
        want = {bucket_end(d, c, f) for d, c, f in all_tasks}
        gaps = sorted(want - set(have.index))
        if not gaps:
            log(f"{member}: cached and complete ({len(have)} buckets)")
            return have
        log(f"{member}: cached with {len(gaps)} gap(s), repairing")
        tasks = [(d, c, f) for d, c, f in all_tasks if bucket_end(d, c, f) in set(gaps)]
        rows = {t: have.loc[t].to_numpy() for t in have.index}
        return _sweep(member, tasks, rows, codes, lats, lons, max(2, workers // 3), cache)

    tasks = list(all_tasks)
    rows = {}
    return _sweep(member, tasks, rows, codes, lats, lons, workers, cache)


def _sweep(member, tasks, rows, codes, lats, lons, workers, cache):
    misses = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_task, d, c, f, member, lats, lons): (d, c, f)
                for d, c, f in tasks}
        for fut in as_completed(futs):
            done += 1
            try:
                got = fut.result()
            except Exception as e:
                misses += 1
                if misses <= 3:
                    log(f"  {member}: {type(e).__name__}: {e}")
                continue
            if got is None:
                misses += 1
                continue
            rows[got[0]] = got[1]
            if done % 200 == 0:
                log(f"  {member}: {done}/{len(tasks)} records, {misses} missing")

    if not rows:
        log(f"{member}: NOTHING fetched")
        return None
    df = pd.DataFrame.from_dict(rows, orient="index", columns=codes).sort_index()
    df.index.name = "bucket_end_utc"
    df.to_pickle(cache)
    log(f"{member}: {len(df)} buckets, {misses} missing -> {cache}")
    return df


def to_qpf_pickle(member, buckets):
    """Three-hour buckets spread evenly over their twelve 15-minute steps, in the members' format."""
    grid = pd.date_range(START_DATE, END_DATE, freq="15min", tz="UTC", inclusive="left")
    steps = BUCKET_HOURS * 4
    out = pd.DataFrame(index=grid)
    for code in buckets.columns:
        ser = buckets[code].astype("float64")
        # a bucket labelled by its END time applies to the three hours before it
        shifted = ser.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=BUCKET_HOURS)
        q = shifted.reindex(grid, method="ffill", limit=steps - 1) / float(steps)
        out[f"{code}_rain_in_qpf"] = q.astype("float32")
    dst = os.path.join(QPF, f"qpf_gefs_{member}.pkl")
    out.to_pickle(dst)
    arr = out.to_numpy()
    log(f"{member}: wrote {dst}  shape {out.shape}  finite {np.isfinite(arr).mean():.1%}  "
        f"total {np.nansum(arr):.1f} in")
    return dst, float(np.isfinite(arr).mean()), float(np.nansum(arr))


def main():
    # Declared before any use, because the argparse defaults below read these module values and
    # Python forbids touching a name in a function before its global statement.
    global QPF, CACHE, START_DATE, END_DATE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", default="geavg",
                    help="comma separated, e.g. geavg or gec00,gep01,gep02")
    ap.add_argument("--workers", type=int, default=12)
    # The output directory, the cache and the date window are arguments so a different period can be
    # fetched without overwriting the January to June archive, which this script writes to by name
    # and with no existence check. The defaults reproduce the original fetch exactly.
    ap.add_argument("--qpf-dir", default=QPF)
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    args = ap.parse_args()

    QPF = args.qpf_dir
    CACHE = args.cache_dir
    START_DATE = args.start
    END_DATE = args.end
    for directory in (QPF, CACHE):
        os.makedirs(directory, exist_ok=True)
    log("writing rainfall pickles to", QPF)
    log("window", START_DATE, "to", END_DATE)

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    codes, lats, lons = station_table()
    days = [d.strftime("%Y%m%d")
            for d in pd.date_range(START_DATE, END_DATE, freq="D", inclusive="left")]
    log(f"{len(codes)} stations, {len(days)} days, {len(members)} member(s), "
        f"{len(days) * len(CYCLES) * len(LEADS)} records each")

    report = []
    for member in members:
        buckets = fetch_member(member, days, codes, lats, lons, args.workers)
        if buckets is None:
            continue
        path, finite, total = to_qpf_pickle(member, buckets)
        report.append({"member": member, "path": path, "buckets": int(len(buckets)),
                       "finite_fraction": round(finite, 4),
                       "total_inches_all_stations": round(total, 3),
                       "grid_km_mean": _grid.get("km_mean"), "grid_km_max": _grid.get("km_max")})

    if report:
        dst = os.path.join(OUT, "gefs_fetch_report.csv")
        old = pd.read_csv(dst) if os.path.exists(dst) else None
        new = pd.DataFrame(report)
        if old is not None:
            new = pd.concat([old[~old["member"].isin(new["member"])], new], ignore_index=True)
        new.to_csv(dst, index=False)
        with open(os.path.join(OUT, "gefs_fetch_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        log("wrote " + dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
