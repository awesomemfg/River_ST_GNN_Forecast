"""Shared helpers for the per-gauge spatial-skill maps (RMSE / correlation / KGE / NSE).

Single source of truth: analysis/efficiency_per_gauge.csv, which carries RMSE_24h, corr_24h, NSE_24h and
KGE_24h for BOTH the deployed ST-GNN and the Bi-LSTM shadow, per in-parish gauge. The archive span shown
in titles is read from per_cycle_median.csv (data-driven, not hardcoded). All maps use the figure_01
river basemap. Median + mean are annotated on every map (no fabricated numbers).
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from _mapbase import make_basemap, load_xy, is_external, basin, source, BASIN_COL, add_cities
from _units import FT2M
import matplotlib.pyplot as plt

AN = "project/manuscript/analysis"   # 68-gauge basis (MBSA4570 excluded); built by rebuild_analysis_no4570.py
EFF = os.path.join(AN, "efficiency_per_gauge.csv")        # NSE / KGE (only source), both models
PGS = os.path.join(AN, "per_gauge_stats.csv")             # delivered 410-cycle archive RMSE / corr (body-consistent)
PPG = os.path.join(AN, "paired_per_gauge.csv")            # paired ST-GNN-Bi-LSTM RMSE difference (body-consistent)
PCM = os.path.join(AN, "per_cycle_median.csv")
DST = "project/manuscript/figure_library_outputs"


def archive_span():
    """Human-readable archive span from per_cycle_median.csv issue times, e.g. '7-22 June 2026'."""
    t = pd.to_datetime(pd.read_csv(PCM)["issue_time_local"])
    a, b = t.min(), t.max()
    mon = b.strftime("%B %Y")
    if a.month == b.month and a.year == b.year:
        return f"{a.day}-{b.day} {mon}", len(t)
    return f"{a.strftime('%d %b')}-{b.strftime('%d %b %Y')}", len(t)


def _eff(model):
    e = pd.read_csv(EFF)
    return e[e["model"] == model].set_index("gauge")


def load_stgnn_skill():
    """Per-gauge ST-GNN skill: RMSE+correlation from the delivered archive (per_gauge_stats, body-consistent),
    KGE+NSE from efficiency_per_gauge (the only source). Indexed by in-parish gauge, with (E,N)."""
    xy = load_xy()
    pgs = pd.read_csv(PGS)
    pgs = pgs[pgs["is_apg"] == True].set_index("gauge")
    eff = _eff("ST-GNN")
    rows = {}
    for g in pgs.index:
        if g not in xy:
            continue
        rows[g] = {
            "rmse": float(pgs.loc[g, "mean_rmse_24h"]) * FT2M,   # feet -> metres
            "corr": float(pgs.loc[g, "median_corr_24h"]),
            "kge": float(eff.loc[g, "KGE_24h"]) if g in eff.index else np.nan,
            "nse": float(eff.loc[g, "NSE_24h"]) if g in eff.index else np.nan,
            "E": xy[g][0], "N": xy[g][1],
        }
    return pd.DataFrame(rows).T


def load_diff():
    """Per-gauge ST-GNN minus Bi-LSTM: RMSE difference from paired_per_gauge (body-consistent 45/51);
    correlation/KGE/NSE differences from efficiency_per_gauge. Negative RMSE / positive others = ST-GNN better."""
    xy = load_xy()
    ppg = pd.read_csv(PPG).set_index("gauge")
    st, bl = _eff("ST-GNN"), _eff("BiLSTM shadow")
    rows = {}
    for g in ppg.index:
        if g not in xy or is_external(g):
            continue
        d = {"rmse": float(ppg.loc[g, "delta_mean_rmse_24h"]) * FT2M, "E": xy[g][0], "N": xy[g][1]}
        if g in st.index and g in bl.index:
            d["corr"] = float(st.loc[g, "corr_24h"]) - float(bl.loc[g, "corr_24h"])
            d["kge"] = float(st.loc[g, "KGE_24h"]) - float(bl.loc[g, "KGE_24h"])
            d["nse"] = float(st.loc[g, "NSE_24h"]) - float(bl.loc[g, "NSE_24h"])
        else:
            d["corr"] = d["kge"] = d["nse"] = np.nan
        rows[g] = d
    return pd.DataFrame(rows).T


def load_model_comparison():
    """Per-gauge side-by-side ST-GNN and Bi-LSTM values plus ST-GNN minus Bi-LSTM differences.
    RMSE comes from paired_per_gauge.csv so it matches the manuscript's 45/51 mapped-gauge delta-RMSE count.
    Correlation, KGE, and NSE come from efficiency_per_gauge.csv, which carries both models."""
    xy = load_xy()
    ppg = pd.read_csv(PPG).set_index("gauge")
    st = _eff("ST-GNN")
    bl = _eff("BiLSTM shadow")
    rows = {}
    for g in ppg.index:
        if g not in xy or is_external(g):
            continue
        if g not in st.index or g not in bl.index:
            continue
        row = {
            "rmse_stgnn": float(ppg.loc[g, "mean_rmse_24h_stgnn"]) * FT2M,
            "rmse_bilstm": float(ppg.loc[g, "mean_rmse_24h_bilstm"]) * FT2M,
            "rmse_diff": float(ppg.loc[g, "delta_mean_rmse_24h"]) * FT2M,
            "corr_stgnn": float(st.loc[g, "corr_24h"]),
            "corr_bilstm": float(bl.loc[g, "corr_24h"]),
            "corr_diff": float(st.loc[g, "corr_24h"]) - float(bl.loc[g, "corr_24h"]),
            "kge_stgnn": float(st.loc[g, "KGE_24h"]),
            "kge_bilstm": float(bl.loc[g, "KGE_24h"]),
            "kge_diff": float(st.loc[g, "KGE_24h"]) - float(bl.loc[g, "KGE_24h"]),
            "nse_stgnn": float(st.loc[g, "NSE_24h"]),
            "nse_bilstm": float(bl.loc[g, "NSE_24h"]),
            "nse_diff": float(st.loc[g, "NSE_24h"]) - float(bl.loc[g, "NSE_24h"]),
            "E": xy[g][0],
            "N": xy[g][1],
        }
        rows[g] = row
    return pd.DataFrame(rows).T


def _scatter_metric(ax, df, vals, cmap, vmin, vmax, s=150):
    return ax.scatter(df["E"], df["N"], c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=s,
                      edgecolor="black", linewidth=0.8, zorder=8)


def add_reference_cities(ax):
    """Add the two local city labels requested for the spatial skill/difference maps."""
    add_cities(ax, ["Donaldsonville", "Gonzales"], dot=24, fs=9.6, dy=-1.0)


def limits_for(vals, higher_better, diff=False):
    """Return (vmin, vmax) robust limits for the colour scale."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if diff:
        lim = float(np.percentile(np.abs(v), 90)) or 1e-3
        return -lim, lim
    if higher_better:
        lo = min(0.0, float(np.percentile(v, 5)))
        return lo, 1.0
    return 0.0, float(np.percentile(v, 95))


def annotate_stats(ax, vals, unit=""):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    txt = "median = %.2f%s\nmean = %.2f%s\nN = %d gauges" % (np.median(v), unit, np.mean(v), unit, len(v))
    ax.text(0.985, 0.985, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.92}, zorder=22)
