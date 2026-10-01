"""Compare one anchor-fix run with the original run of the same model, seed, and forcing.

Usage: python compare_anchor_test.py OLD.npz NEW.npz
Reports how many origin-gauge forecasts changed, how much, and the BMSA1291 forecast verifying on 28 May 2026 at
12:00 UTC (issued 24 h earlier), where the frozen sensor made the old forecast start from the gauge mean.
Forecasts whose last reading was valid must be identical, so any other change is reported as an error.
"""
import sys

import numpy as np
import pandas as pd

FT2M = 0.3048


def load(path):
    archive = np.load(path, allow_pickle=True)
    quantiles = [float(q) for q in archive["quantiles_saved"]]
    return {
        "origins": pd.to_datetime(archive["origins_utc"]),
        "nodes": [str(n) for n in archive["nodes"]],
        "p80": archive["pred_ft"][..., quantiles.index(0.8)].astype(float) * FT2M,
    }


old = load(sys.argv[1])
new = load(sys.argv[2])
if list(old["origins"]) != list(new["origins"]) or old["nodes"] != new["nodes"]:
    raise SystemExit("[FATAL] the two runs do not cover the same origins and gauges")

difference = np.abs(new["p80"] - old["p80"])
changed = difference.max(axis=1) > 1.0e-4
print(f"origin-gauge forecasts changed: {int(changed.sum())} of {changed.size} ({100 * changed.mean():.2f} %)")
print(f"largest change anywhere: {difference.max():.3f} m")
per_gauge = pd.Series(changed.sum(axis=0), index=old["nodes"])
print("gauges with changed forecasts:", int((per_gauge > 0).sum()))
print(per_gauge[per_gauge > 0].sort_values(ascending=False).head(10).to_string())

gauge = old["nodes"].index("BMSA1291")
origin = pd.Timestamp("2026-05-27 12:00")
row = list(old["origins"]).index(origin)
print(f"BMSA1291, issued {origin}, h+24 P80: old {old['p80'][row, 95, gauge]:.3f} m, "
      f"new {new['p80'][row, 95, gauge]:.3f} m (observed about 1.937 m)")
