#!/usr/bin/env python3
"""
Figure 13, rebuilt for the ensemble. The drawing recipe is taken VERBATIM from the paper's own
figure script,

    manuscript/20260826_Final_V2/figure_revision_scripts/
        fig13_flood_safety_larger_fonts.py

with the bars() function copied unchanged: same serif typeface and sizes, same 1x3 layout and figure
size, same bar widths and offsets, same difference annotation above each pair, same axis limits and
grid, same colours, same PNG and PDF output.

Only the DATA changes, in the two ways Sect. 5.1 changed: postprocessing is gone, and the forecast
case is the ensemble mean of six weather models rather than one archived forecast.

WHY THIS NEEDS THE FULL TRAJECTORIES
------------------------------------
Figure 13 scores the PEAK of each 24 h window, and the peak of the ensemble mean is not the mean of
the members' peaks. Averaging six members that place a crest at slightly different times produces a
flatter trajectory whose maximum is lower than any individual member's. So the ensemble trajectory
has to be formed first and its peak taken afterwards, which is why every member was re-run with
full 96-step trajectories rather than reusing the h+24 endpoint.

THE PEAK DEFINITION IS COPIED FROM THE EVALUATOR, NOT REINVENTED
----------------------------------------------------------------
Taken from eval_reforecast_2026H1_perfectQPF.py lines 917-960:

    observed peak      the maximum observed level in the window, needing at least 4 valid steps
    predicted peak     the maximum predicted level within +/-16 steps (+/-4 h) of the observed peak,
                       not the maximum over the whole window
    peak_bias_m        (predicted peak - observed peak) * 0.3048
    peak_absolute_error_m   the absolute value of the same, in metres
    observed_rise_m    (observed peak - stage at issue time) * 0.3048

The local +/-4 h search is the part that would be easy to get wrong, and getting it wrong would
quietly change every number. So the same computation is first run on a SINGLE member and checked
against that member's own origin_window_metrics.csv, which the evaluator wrote. The figure is only
drawn if that check passes.

Run:
    conda run -n operational python recipe_ensemble_crest_error.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = "project/Experiments/QPF_ENSEMBLE_20260831"
TRAJ = os.path.join(EXP, "evaluations_paper_traj")
PAPEV = os.path.join(EXP, "evaluations_paper")
OUTD = os.path.join(EXP, "figures")
FIGSCRIPT = ("project/manuscript/"
             "revision_2026_07_17/scripts/recipe_revised_paper_figures.py")

MEMBERS = [("gfs_hrrr", "HRRR"), ("gfs_global", "GFS"), ("ncep_nbm_conus", "NBM"),
           ("ecmwf_ifs025", "ECMWF IFS"), ("gem_global", "GEM"), ("jma_seamless", "JMA")]

# verbatim from the native script
WINDOW = 24
RISE_THRESHOLD_M = 0.1524
UNDERCALL_M = 0.1524
PEAK_SEARCH_RADIUS_STEPS = 16          # eval_reforecast_2026H1_perfectQPF.py:921
FT2M = 0.3048
STRATA = [("0.15–0.30\nm", 0.1524, 0.3048),
          ("0.30–0.61\nm", 0.3048, 0.6096),
          (">0.61\nm", 0.6096, np.inf)]

COL_OBS = "#1f77b4"
COL_QPF = "#d62728"
COL_MEM = "#7f7f7f"
GRID = "#808080"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 14.0,
    "axes.titlesize": 17.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 16.0,
    "legend.fontsize": 13.0,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "savefig.dpi": 300,
})


def inparish():
    blk = re.search(r"IN_PARISH_GAUGES = \[(.*?)\]", open(FIGSCRIPT).read(), re.S).group(1)
    return {t.strip().strip('",') for t in blk.split() if t.strip().strip('",')}


def load_cube(tag):
    """Trajectories as [origin, lead_step, node] in feet, plus the axes."""
    f = os.path.join(TRAJ, tag, "reforecast_2026H1_trajectories.parquet")
    d = pd.read_parquet(f, columns=["t0_utc", "lead_step", "node", "observed_ft",
                                    "predicted_ft"])
    origins = np.sort(d["t0_utc"].unique())
    nodes = np.sort(d["node"].unique())
    steps = np.sort(d["lead_step"].unique())
    oi = pd.Index(origins).get_indexer(d["t0_utc"])
    ni = pd.Index(nodes).get_indexer(d["node"])
    si = pd.Index(steps).get_indexer(d["lead_step"])
    shape = (len(origins), len(steps), len(nodes))
    pred = np.full(shape, np.nan, dtype=np.float32)
    obs = np.full(shape, np.nan, dtype=np.float32)
    pred[oi, si, ni] = d["predicted_ft"].to_numpy(dtype=np.float32)
    obs[oi, si, ni] = d["observed_ft"].to_numpy(dtype=np.float32)
    return origins, steps, nodes, obs, pred


def peaks(obs, pred):
    """
    The evaluator's peak metrics, copied from eval_reforecast_2026H1_perfectQPF.py:917-960.
    Returns peak_bias_m and peak_absolute_error_m as [origin, node] arrays.
    """
    n_o, n_s, n_n = obs.shape
    bias = np.full((n_o, n_n), np.nan, dtype=np.float64)
    aerr = np.full((n_o, n_n), np.nan, dtype=np.float64)
    for i in range(n_o):
        o_win = obs[i]
        p_win = pred[i]
        valid = np.isfinite(o_win)
        enough = valid.sum(axis=0) >= 4
        for j in np.where(enough)[0]:
            os_ = np.where(valid[:, j], o_win[:, j], np.nan)
            ps_ = np.where(valid[:, j], p_win[:, j], np.nan)
            k = int(np.nanargmax(os_))
            o_peak = float(os_[k])
            a = max(0, k - PEAK_SEARCH_RADIUS_STEPS)
            b = min(n_s, k + PEAK_SEARCH_RADIUS_STEPS + 1)
            seg = ps_[a:b]
            if not np.isfinite(seg).any():
                continue
            p_peak = float(seg[int(np.nanargmax(seg))])
            bias[i, j] = (p_peak - o_peak) * FT2M
            aerr[i, j] = abs(p_peak - o_peak) * FT2M
    return bias, aerr


def owm(tag_dir):
    f = os.path.join(PAPEV, tag_dir, "reforecast_2026H1_origin_window_metrics.csv")
    d = pd.read_csv(f, usecols=["t0_utc", "gauge", "window_hours", "observed_rise_m",
                                "peak_absolute_error_m", "peak_bias_m"])
    return d[d["window_hours"] == WINDOW].set_index(["t0_utc", "gauge"])


def bars(ax, t, col_a, col_b, lab_a, lab_b, ya, yb, ylabel, title, pct):
    """Copied unchanged from fig13_flood_safety_larger_fonts.py."""
    x = np.arange(len(t))
    ax.bar(x - 0.19, t[ya], 0.36, color=col_a, label=lab_a)
    ax.bar(x + 0.19, t[yb], 0.36, color=col_b, label=lab_b)
    for i in range(len(t)):
        d = t[yb].iloc[i] - t[ya].iloc[i]
        top = max(t[ya].iloc[i], t[yb].iloc[i])
        txt = f"{d:+.1f}%" if pct else f"{d:+.3f} m"
        ax.annotate(txt, (x[i], top), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=13.5, color="0.2", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(t["stratum"], fontsize=12.5)
    ax.set_xlabel("Observed 24 h stage rise")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(t[[ya, yb]].to_numpy().max() * 1.22, 1e-6))
    ax.set_title(title, loc="left")
    ax.grid(color=GRID, axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.93)


def stratify(rise, a_bias, a_aerr, b_bias, b_aerr):
    rows = []
    for name, lo, hi in STRATA:
        m = (rise >= max(lo, RISE_THRESHOLD_M)) & (rise < hi)
        m &= np.isfinite(a_bias) & np.isfinite(b_bias)
        if not m.any():
            continue
        rows.append({"stratum": name, "n": int(m.sum()),
                     "peak_a": float(np.nanmean(a_aerr[m])),
                     "peak_b": float(np.nanmean(b_aerr[m])),
                     "miss_a": float((a_bias[m] < -UNDERCALL_M).mean() * 100.0),
                     "miss_b": float((b_bias[m] < -UNDERCALL_M).mean() * 100.0)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTD, exist_ok=True)
    keep = inparish()

    origins, steps, nodes, obs, _ = load_cube("out_ptraj_obs")
    print(f"grid: {len(origins)} origins x {len(steps)} steps x {len(nodes)} nodes")

    # ---- validation: reproduce a member's own peak metrics from its trajectory ------------
    _, _, _, o_h, p_h = load_cube("out_ptraj_gfs_hrrr_qpf")
    b_h, a_h = peaks(o_h, p_h)
    ref = owm("out_pap_gfs_hrrr_qpf")
    # the window-metrics CSV writes t0 as "YYYY-MM-DD HH:MM:SS"; the trajectory parquet carries it
    # as a timestamp, so it is formatted the same way before joining rather than compared loosely
    origin_str = pd.to_datetime(pd.Index(origins)).strftime("%Y-%m-%d %H:%M:%S")
    idx = pd.MultiIndex.from_product([pd.Index(origin_str), pd.Index(nodes)],
                                     names=["t0_utc", "gauge"])
    mine = pd.DataFrame({"peak_bias_m": b_h.reshape(-1),
                         "peak_absolute_error_m": a_h.reshape(-1)}, index=idx)
    j = mine.join(ref, how="inner", rsuffix="_ref")
    both = j["peak_bias_m"].notna() & j["peak_bias_m_ref"].notna()
    dmax = float((j.loc[both, "peak_bias_m"] - j.loc[both, "peak_bias_m_ref"]).abs().max())
    print(f"[check] peak_bias reproduced on {int(both.sum()):,} cells, "
          f"largest difference {dmax:.3e} m")
    if dmax > 1e-6:
        raise SystemExit("[FATAL] peak computation does not reproduce the evaluator's own output")
    print("[check] PASS — the peak definition matches the evaluator exactly")

    # ---- observed-rainfall case and the ensemble ------------------------------------------
    _, _, _, o_o, p_o = load_cube("out_ptraj_obs")
    b_obs, a_obs = peaks(o_o, p_o)

    stack = np.zeros_like(p_o, dtype=np.float32)
    cnt = np.zeros_like(p_o, dtype=np.float32)
    per_member = {}
    for key, label in MEMBERS:
        _, _, _, om, pm = load_cube(f"out_ptraj_{key}_qpf")
        per_member[label] = peaks(om, pm)
        good = np.isfinite(pm)
        stack[good] += pm[good]
        cnt[good] += 1.0
        print(f"  stacked {label}")
    ens = np.where(cnt > 0, stack / np.maximum(cnt, 1), np.nan)
    b_ens, a_ens = peaks(o_o, ens)

    rise = owm("out_pap_gfs_hrrr_qpf")["observed_rise_m"].reindex(idx).to_numpy().reshape(
        len(origins), len(nodes))

    keep_mask = np.isin(nodes, sorted(keep))
    def sub(x):
        return x[:, keep_mask].reshape(-1)
    rise_v = sub(rise)

    rain = stratify(rise_v, sub(b_obs), sub(a_obs), sub(b_ens), sub(a_ens))

    # ---- draw ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.9))
    bars(axes[0], rain, COL_OBS, COL_QPF, "Observed rainfall", "Ensemble mean",
         "peak_a", "peak_b", "Mean error in the peak level (m)",
         "(a) Accuracy of the crest", pct=False)
    bars(axes[1], rain, COL_OBS, COL_QPF, "Observed rainfall", "Ensemble mean",
         "miss_a", "miss_b", "Crests missed (%)",
         "(b) How often the crest is missed", pct=True)
    # The native script forces the axis to 0-100 after bars() has already annotated at bar top +5
    # points. With a bar at 77 % the label lands on the bar edge and becomes unreadable, so the
    # headroom is widened. Bars, colours and values are untouched; only the empty space above them.
    axes[1].set_ylim(0, 112)

    # panel (c) replaces the old postprocessing panel: every member against the ensemble mean,
    # which is where the ensemble's weakness at the crest actually shows
    ax = axes[2]
    labels = ["HRRR", "GFS", "NBM", "ECMWF", "GEM", "JMA", "Ensemble mean"]
    x = np.arange(len(labels))
    top = STRATA[-1]
    m = (rise_v >= top[1]) & (rise_v < top[2])
    vals = []
    for _, lab in MEMBERS:
        bb = sub(per_member[lab][0])
        vals.append(float((bb[m & np.isfinite(bb)] < -UNDERCALL_M).mean() * 100.0))
    be = sub(b_ens)
    vals.append(float((be[m & np.isfinite(be)] < -UNDERCALL_M).mean() * 100.0))
    cols = [COL_MEM] * len(MEMBERS) + [COL_QPF]
    ax.bar(x, vals, 0.62, color=cols)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}%", (x[i], v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=12.5, color="0.2", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.0, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Crests missed (%)")
    ax.set_xlabel(f"Rises above 0.61 m  (n = {int(m.sum()):,})")
    ax.set_ylim(0, 112)
    ax.set_title("(c) The average is no better than its members", loc="left")
    ax.grid(color=GRID, axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTD, f"fig13_ensemble.{ext}"), bbox_inches="tight")
    plt.close(fig)

    print()
    print("RAINFALL SOURCE, rising windows only, in-parish gauges, 24 h")
    print(f"  {'stratum':<16}{'n':>8}{'peak obs':>10}{'peak ens':>10}"
          f"{'miss obs':>10}{'miss ens':>10}{'extra pts':>11}")
    for _, r in rain.iterrows():
        print(f"  {r['stratum'][:12]:<16}{int(r['n']):>8,}{r['peak_a']:>10.3f}{r['peak_b']:>10.3f}"
              f"{r['miss_a']:>9.1f}%{r['miss_b']:>9.1f}%{r['miss_b'] - r['miss_a']:>+11.1f}")
    print()
    print(f"MISSED CRESTS ON RISES ABOVE 0.61 m  (n = {int(m.sum()):,})")
    for lab, v in zip(labels, vals):
        print(f"  {lab.replace(chr(10), ' '):<16}{v:>7.1f}%")
    rain.to_csv(os.path.join(EXP, "outputs", "fig13_ensemble_strata.csv"), index=False)
    pd.DataFrame({"forcing": [l.replace("\n", " ") for l in labels],
                  "missed_pct_gt061": vals}).to_csv(
        os.path.join(EXP, "outputs", "fig13_members_gt061.csv"), index=False)
    print()
    print(f"[out] {OUTD}/fig13_ensemble.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
