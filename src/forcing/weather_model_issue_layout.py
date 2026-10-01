"""Rewrite a valid-time rain-forecast record into the per-origin layout read by lstm_inference_before_fix.py.

The original evaluator's archived_qpf mode reads one rain-forecast value per 15 min valid time
(columns <code>_rain_in_qpf). Putting the same record into the per-origin layout lets 04 be checked
against the original evaluator on identical rain: for origin t0, step k holds the value valid at
t0 + (k + 1) x 15 min. Missing values stay NaN.

Output npz: origins_utc, rain_in [O, 96, C] (inches), rain_codes, source.
"""
import argparse
import os

import numpy as np
import pandas as pd

EXP = "project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
parser = argparse.ArgumentParser()
parser.add_argument("--qpf-pickle", default="project/Experiments/"
                                            "REFORECAST_2026H1/<SERVER>/archived_qpf_previous_day1.pkl")
parser.add_argument("--out", default=os.path.join(EXP, "validation", "eval_vs_original",
                                                  "archived_qpf_previous_day1_issue_layout.npz"))
# The origin window is an argument so a longer record can be converted without touching the January
# to June layouts, whose hashes are recorded in the runs that used them. The defaults reproduce the
# original January to June conversion exactly.
parser.add_argument("--origin-start", default="2026-01-01 00:00")
parser.add_argument("--origin-end", default="2026-06-30 23:00")
args = parser.parse_args()

q = pd.read_pickle(args.qpf_pickle).sort_index()
if q.index.tz is not None:
    q.index = q.index.tz_convert("UTC").tz_localize(None)
columns = [c for c in q.columns if c.endswith("_rain_in_qpf")]
codes = [c.removesuffix("_rain_in_qpf") for c in columns]
values = q[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
origins = pd.date_range(args.origin_start, args.origin_end, freq="h")
positions = q.index.get_indexer(origins)
rain = np.full((len(origins), 96, len(codes)), np.nan, np.float32)
complete = 0
for o, p in enumerate(positions):
    if p < 0 or p + 96 >= len(q):
        continue
    expected = pd.date_range(origins[o] + pd.Timedelta(minutes=15), periods=96, freq="15min")
    if not q.index[p + 1:p + 97].equals(expected):
        continue
    rain[o] = values[p + 1:p + 97]
    complete += 1
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, origins_utc=np.asarray([t.isoformat() for t in origins]), rain_in=rain,
                    rain_codes=np.asarray(codes), source=args.qpf_pickle)
print("[qpf-layout] origins", len(origins), "with a full 24 h record", complete, "codes", len(codes),
      "NaN fraction", round(float(np.isnan(rain).mean()), 5))
print("[qpf-layout] wrote", args.out)
print("QPF_LAYOUT_DONE", flush=True)
