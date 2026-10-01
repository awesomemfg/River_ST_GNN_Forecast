#!/usr/bin/env python3
"""
Rebuild Figure 7 and Table 4 of Section 4.2 from the frozen_stgnn run, so the whole section rests
on one evaluation instead of two.

Why the section had two runs
    out_obs_pre2026 is one arm of the graph-ranking experiment. Its job was to pick a graph, and it
    is the leakage-safe observation lead-lag arm of that comparison. It belongs to Table 2.
    frozen_stgnn is the reforecast built for the head-to-head against the cleanly retrained Bi-LSTM.

    Both train to 31 December 2025 on the same matrix and score the same 4,344 origins at the same
    51 in-parish gauges. They are two independently trained instances of the same configuration, so
    they differ only by training noise: 0.149 against 0.148 m at h+24, well inside the 0.008 m
    seed-only floor measured in NODE_SUBSAMPLE_20260813. Neither is more correct than the other.

    frozen_stgnn is chosen because it is the only one of the two that also carries the Bi-LSTM at
    every lead and the exactly matched three-model comparison at h+24. Building the section on it
    means the paragraph, the table, the figure and the head-to-head all come from one run.

Panel recipe is copied unchanged from
    revision_2026_07_17/scripts/recipe_revised_paper_figures.py
so only the data source moves. Same panels, same colours, same limits, same ordering.

Run:
    conda run -n operational python recipe_graph_figure_and_table.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FROZEN = ("project/hpc/Experiments/"
          "REFORECAST_LSTM_CLEAN_20260729/frozen_stgnn/reforecast_2026H1_lead_metrics.csv")
FIGSCRIPT = ("project/manuscript/"
             "revision_2026_07_17/scripts/recipe_revised_paper_figures.py")

OUT = ("project/manuscript/"
       "20260811_Revision/Work/verified")
FIGNAME = "fig07_reforecast_frozen_stgnn_51gauges"

FORECAST_COLOR = "#D1495B"
GRID_COLOR = "#808080"
LEADS = [3.0, 6.0, 12.0, 24.0]

# Copied verbatim from recipe_revised_paper_figures.py so the rebuilt figure carries the same type
# as every other figure in the paper. Without this the panels fall back to the matplotlib sans
# default and Figure 7 no longer matches its neighbours.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.0,
        "savefig.dpi": 300,
    }
)


def empirical_cdf(values):
    """Sorted finite values and their empirical cumulative probabilities."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    v = np.sort(v)
    return v, np.arange(1, len(v) + 1) / len(v)


def inparish():
    src = open(FIGSCRIPT).read()
    blk = re.search(r"IN_PARISH_GAUGES = \[(.*?)\]", src, re.S).group(1)
    return [t.strip().strip('",') for t in blk.split() if t.strip().strip('",')]


def main():
    os.makedirs(OUT, exist_ok=True)
    ip = inparish()

    d = pd.read_csv(FROZEN)
    d = d[d["node"].isin(ip)]
    if d["node"].nunique() != 51:
        raise SystemExit(f"[FATAL] expected 51 gauges, found {d['node'].nunique()}")

    lead_summary = (d.groupby("lead_hours")[["RMSE_m", "MAE_m", "NSE", "Pearson_r"]]
                     .median()
                     .rename(columns={"RMSE_m": "median_RMSE_m", "MAE_m": "median_MAE_m",
                                      "NSE": "median_NSE", "Pearson_r": "median_Pearson_r"})
                     .reset_index())

    h24 = d[np.isclose(d["lead_hours"], 24.0)]
    if len(h24) != 51:
        raise SystemExit(f"[FATAL] expected 51 h+24 rows, found {len(h24)}")

    rv, rc = empirical_cdf(h24["RMSE_m"])
    nv, nc = empirical_cdf(h24["NSE"])
    med_r = float(h24["RMSE_m"].median())
    med_n = float(h24["NSE"].median())

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12.2, 7.8))

    a = ax[0, 0]
    a.plot(lead_summary["lead_hours"], lead_summary["median_RMSE_m"],
           color="#003F5C", linewidth=2.4, label="Median RMSE")
    a.plot(lead_summary["lead_hours"], lead_summary["median_MAE_m"],
           color=FORECAST_COLOR, linewidth=2.4, label="Median MAE")
    a.set_title("(a) Error across forecast lead times", loc="left")
    a.set_xlabel("Forecast lead (h)")
    a.set_ylabel("Stage error (m)")
    a.set_xlim(0.0, 24.0)
    a.legend(loc="upper left", framealpha=0.92)
    a.grid(color=GRID_COLOR, alpha=0.22)

    b = ax[0, 1]
    b.plot(lead_summary["lead_hours"], lead_summary["median_NSE"],
           color="#007C91", linewidth=2.4, label="Median NSE")
    b.plot(lead_summary["lead_hours"], lead_summary["median_Pearson_r"],
           color="#F0A830", linewidth=2.4, label="Median Pearson r")
    b.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    b.set_title("(b) Deterministic skill", loc="left")
    b.set_xlabel("Forecast lead (h)")
    b.set_ylabel("Skill")
    b.set_xlim(0.0, 24.0)
    b.set_ylim(-0.10, 1.05)
    b.legend(loc="lower left", framealpha=0.92)
    b.grid(color=GRID_COLOR, alpha=0.22)

    c = ax[1, 0]
    c.plot(rv, rc, color="#003F5C", linewidth=2.4)
    c.axvline(med_r, color=FORECAST_COLOR, linestyle="--", linewidth=1.4,
              label=f"Median = {med_r:.3f} m")
    c.set_title("(c) h+24 RMSE distribution", loc="left")
    c.set_xlabel("h+24 RMSE (m)")
    c.set_ylabel("Gauge ECDF")
    c.set_ylim(0.0, 1.02)
    c.legend(loc="lower right", framealpha=0.92)
    c.grid(color=GRID_COLOR, alpha=0.22)

    e = ax[1, 1]
    e.plot(nv, nc, color="#007C91", linewidth=2.4)
    e.axvline(med_n, color=FORECAST_COLOR, linestyle="--", linewidth=1.4,
              label=f"Median = {med_n:.3f}")
    e.set_title("(d) h+24 NSE distribution", loc="left")
    e.set_xlabel("h+24 NSE")
    e.set_ylabel("Gauge ECDF")
    e.set_ylim(0.0, 1.02)
    e.legend(loc="lower right", framealpha=0.92)
    e.grid(color=GRID_COLOR, alpha=0.22)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{FIGNAME}.{ext}"), dpi=300, bbox_inches="tight")

    tab = (d[d["lead_hours"].isin(LEADS)]
           .groupby("lead_hours")[["RMSE_m", "MAE_m", "NSE", "Pearson_r", "KGE"]]
           .median()
           .round(3))
    tab.to_csv(os.path.join(OUT, "table4_frozen_stgnn.csv"))

    print("TABLE 4, rebuilt from frozen_stgnn (51 in-parish gauges, 4,344 origins)")
    print(f"  {'lead':>6}{'RMSE_m':>10}{'MAE_m':>10}{'NSE':>10}{'Correlation':>13}{'KGE':>10}")
    for hz in LEADS:
        r = tab.loc[hz]
        print(f"  {hz:>4.0f} h{r['RMSE_m']:>10.3f}{r['MAE_m']:>10.3f}{r['NSE']:>10.3f}"
              f"{r['Pearson_r']:>13.3f}{r['KGE']:>10.3f}")
    print()
    print(f"  Figure 7 panel (c) median line : {med_r:.3f} m")
    print(f"  Figure 7 panel (d) median line : {med_n:.3f}")
    print(f"  gauges with NSE > 0 at h+24    : {int((h24['NSE'] > 0).sum())} of {len(h24)}")
    print(f"  gauges with h+24 RMSE <= 0.25 m: {int((h24['RMSE_m'] <= 0.25).sum())} of {len(h24)}")
    dn = float(tab.loc[3.0, "NSE"] - tab.loc[24.0, "NSE"])
    dr = float(tab.loc[3.0, "Pearson_r"] - tab.loc[24.0, "Pearson_r"])
    print(f"  NSE fall {dn:.3f} vs correlation fall {dr:.3f} -> "
          f"'NSE declined faster' {'holds' if dn > dr else 'DOES NOT HOLD'}")
    er = (float(tab.loc[6.0, "RMSE_m"] - tab.loc[3.0, "RMSE_m"]) / 3.0,
          float(tab.loc[24.0, "RMSE_m"] - tab.loc[12.0, "RMSE_m"]) / 12.0)
    em = (float(tab.loc[6.0, "MAE_m"] - tab.loc[3.0, "MAE_m"]) / 3.0,
          float(tab.loc[24.0, "MAE_m"] - tab.loc[12.0, "MAE_m"]) / 12.0)
    print(f"  RMSE decelerates {er[0] / er[1]:.2f}x vs MAE {em[0] / em[1]:.2f}x -> "
          f"'RMSE faster early' {'holds' if er[0] / er[1] > em[0] / em[1] else 'DOES NOT HOLD'}")
    print()
    print(f"[out] {os.path.join(OUT, FIGNAME + '.png')}")
    print(f"[out] {os.path.join(OUT, 'table4_frozen_stgnn.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
