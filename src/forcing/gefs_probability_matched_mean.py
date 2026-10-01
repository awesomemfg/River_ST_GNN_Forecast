#!/usr/bin/env python3
"""
Build a probability-matched mean rainfall forcing from the individual GEFS members.

THE PROBLEM THIS FIXES
----------------------
Averaging 31 ensemble members produces a series with roughly the right total rainfall and entirely
the wrong intensity. Measured on the fetched data at the 41 rain gauges, over January to June 2026:

    GEFS ensemble mean   34.9 inches per station,  raining 41.1 % of the time, heaviest 15 min 0.071 in
    HRRR (deterministic) 35.4 inches per station,  raining  3.9 % of the time, heaviest 15 min 2.268 in

Nearly the same water, delivered as a constant drizzle instead of a few downpours. Members disagree
about WHEN and WHERE rain falls, so averaging them smears every peak across all the times any member
was wet. A river model driven by rainfall intensity cannot produce a crest from that, so the
ensemble mean is expected to under-forecast rises no matter how good the underlying ensemble is.

WHAT A PROBABILITY-MATCHED MEAN IS
----------------------------------
The standard fix, and the reason the operational HREF ensemble publishes `pmmn` rather than a plain
mean. It keeps the ensemble mean's PATTERN, which is its skilful part, and replaces its VALUES with
the intensity distribution the members actually predicted:

    1. take the ensemble mean series and note the rank order of its values
    2. pool every member's values, sort them, and thin them back to the original length
    3. write the pooled values onto the mean's ranks, largest to largest

The output rains as often and as hard as a real member, but at the times the ensemble collectively
favoured. Applied here along the time axis at each rain gauge, which is the point-series analogue of
the spatial procedure HREF uses.

This is not a correction invented for this project. It is the accepted treatment for exactly this
failure, and it makes the GEFS member a fair test of the ensemble rather than a test of averaging.

OUTPUT
------
    qpf/qpf_gefs_pmm.pkl    a drop-in forcing pickle, same format as every other member

Run:
    conda run -n operational python gefs_probability_matched_mean.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

EXP = "project/Experiments/QPF_ENSEMBLE_20260831"
CACHE = os.path.join(EXP, "gefs", "cache")
QPF = os.path.join(EXP, "qpf")
OUT = os.path.join(EXP, "outputs")

START_DATE = "2026-01-01"
END_DATE = "2026-07-02"
BUCKET_HOURS = 3


def log(*a):
    print("[pmm]", *a, flush=True)


def member_files():
    """Individual members only. The published ensemble mean is not a member of itself."""
    fs = sorted(glob.glob(os.path.join(CACHE, "gefs_ge[cp]*_buckets.pkl")))
    return [f for f in fs if "geavg" not in os.path.basename(f)]


def probability_match(mean_series, pooled):
    """Give `mean_series` the intensity distribution of `pooled`, keeping its rank order.

    pooled holds every member's value at every time. Sorting it and thinning by the member count
    recovers a distribution of the original length, which is then written onto the mean's ranks.
    """
    t = len(mean_series)
    finite = np.isfinite(pooled)
    vals = np.sort(pooled[finite])[::-1]
    if len(vals) == 0:
        return mean_series
    # thin the pooled distribution back to t values, preserving its shape
    take = np.linspace(0, len(vals) - 1, t).round().astype(int)
    target = vals[take]                       # descending, length t

    out = np.full(t, np.nan, dtype="float64")
    ok = np.isfinite(mean_series)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(mean_series[idx])[::-1]]     # ranks of the mean, largest first
    out[order] = target[:len(order)]
    return out


def main():
    # Declared before any use, because the argparse defaults below read these module values and
    # Python forbids touching a name in a function before its global statement.
    global CACHE, QPF, OUT, START_DATE, END_DATE

    parser = argparse.ArgumentParser(description=__doc__)
    # This script writes qpf_gefs_pmm.pkl by name with no existence check, so a second period needs
    # its own directories or it destroys the first. The defaults reproduce the original January to
    # June build exactly.
    parser.add_argument("--cache-dir", default=CACHE)
    parser.add_argument("--qpf-dir", default=QPF)
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument(
        "--hrrr-reference",
        default="",
        help="deterministic HRRR pickle for the intensity comparison row; defaults to the one "
             "beside the members",
    )
    arguments = parser.parse_args()

    CACHE = arguments.cache_dir
    QPF = arguments.qpf_dir
    OUT = arguments.out_dir
    START_DATE = arguments.start
    END_DATE = arguments.end
    for directory in (QPF, OUT):
        os.makedirs(directory, exist_ok=True)
    log("members from", CACHE)
    log("writing to", QPF)
    log("window", START_DATE, "to", END_DATE)

    files = member_files()
    if len(files) < 5:
        raise SystemExit(f"[FATAL] only {len(files)} member cache files found in {CACHE}. "
                         f"Run gefs_fetch.py for the individual members first.")
    log(f"{len(files)} individual members")

    frames = [pd.read_pickle(f) for f in files]
    index = frames[0].index
    for fr in frames[1:]:
        index = index.intersection(fr.index)
    index = index.sort_values()
    cols = list(frames[0].columns)
    log(f"{len(index)} buckets common to all members, {len(cols)} stations")

    stack = np.stack([fr.reindex(index)[cols].to_numpy(dtype="float64") for fr in frames], axis=0)
    log(f"stack {stack.shape}  (members, buckets, stations)")

    ens_mean = np.nanmean(stack, axis=0)
    pmm = np.full_like(ens_mean, np.nan)
    for j in range(stack.shape[2]):
        pmm[:, j] = probability_match(ens_mean[:, j], stack[:, :, j].ravel())

    buckets = pd.DataFrame(pmm, index=index, columns=cols)
    buckets.index.name = "bucket_end_utc"
    buckets.to_pickle(os.path.join(CACHE, "gefs_pmm_buckets.pkl"))

    grid = pd.date_range(START_DATE, END_DATE, freq="15min", tz="UTC", inclusive="left")
    steps = BUCKET_HOURS * 4
    out = pd.DataFrame(index=grid)
    for code in cols:
        ser = buckets[code].astype("float64")
        shifted = ser.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=BUCKET_HOURS)
        q = shifted.reindex(grid, method="ffill", limit=steps - 1) / float(steps)
        out[f"{code}_rain_in_qpf"] = q.astype("float32")
    dst = os.path.join(QPF, "qpf_gefs_pmm.pkl")
    out.to_pickle(dst)

    def describe(arr, label):
        a = arr[np.isfinite(arr)]
        return {
            "forcing": label,
            "total_inches_all_stations": round(float(np.nansum(arr)), 1),
            "per_station_inches": round(float(np.nansum(arr)) / arr.shape[-1], 1),
            "wet_fraction_pct": round(100 * float(np.mean(a > 0.0001)), 1),
            "p99_9": round(float(np.percentile(a, 99.9)), 4),
            "max": round(float(a.max()), 3),
        }

    mean_q = pd.read_pickle(os.path.join(QPF, "qpf_gefs_geavg.pkl")).to_numpy()
    rep = [describe(mean_q, "GEFS ensemble mean"), describe(out.to_numpy(), "GEFS pmm")]
    hrrr = arguments.hrrr_reference or os.path.join(QPF, "qpf_gfs_hrrr.pkl")
    if os.path.exists(hrrr):
        rep.append(describe(pd.read_pickle(hrrr).to_numpy(), "HRRR (deterministic reference)"))
    t = pd.DataFrame(rep)
    log("\n" + t.to_string(index=False))
    t.to_csv(os.path.join(OUT, "gefs_pmm_intensity_check.csv"), index=False)
    with open(os.path.join(OUT, "gefs_pmm_intensity_check.json"), "w") as f:
        json.dump({"members_used": [os.path.basename(x) for x in files], "table": rep}, f, indent=2)
    log("wrote " + dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
