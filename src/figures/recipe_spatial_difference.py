"""Fig 13 / Fig S4 (4.3.2) - ST-GNN minus Bi-LSTM per-gauge skill difference across Ascension Parish, on the
figure_01 river basemap. Green always means ST-GNN is better (lower RMSE, or higher correlation/KGE/NSE).

  Fig 13 (main):  fig15_spatial_diff.png - Delta RMSE from paired_per_gauge.csv (body-consistent count).
  Fig S4 (supp):  fig15a/b/c + fig15_spatial_panel.png - Delta correlation / KGE / NSE from efficiency_per_gauge.csv.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _mapbase import make_basemap
from _spatialmap import load_diff, load_model_comparison, archive_span, _scatter_metric, add_reference_cities, DST
from _units import label_axes_m
import matplotlib.pyplot as plt

df = load_diff()
cmp = load_model_comparison()
span, _ = archive_span()
gauges = list(df.index)

# key -> (pretty, higher_is_better, cmap, unit)
SPEC = {
    "rmse": ("0-24 h RMSE", False, "RdYlGn_r", " m"),
    "corr": ("0-24 h correlation", True, "RdYlGn", ""),
    "kge": ("0-24 h KGE", True, "RdYlGn", ""),
    "nse": ("0-24 h NSE", True, "RdYlGn", ""),
}
NAME = {"corr": "fig15a_spatial_dcorr", "kge": "fig15b_spatial_dkge", "nse": "fig15c_spatial_dnse"}
TRIPTYCH_NAME = {
    "rmse": "fig15a_i_rmse_triptych",
    "corr": "fig15a_ii_corr_triptych",
    "kge": "fig15a_iii_kge_triptych",
    "nse": "fig15a_iv_nse_triptych",
}
TRIPTYCH_SPEC = {
    "rmse": {
        "pretty": "0-24 h RMSE",
        "unit": "m",
        "absolute_cmap": "RdYlGn_r",
        "diff_cmap": "RdYlGn_r",
        "higher_is_better": False,
        "abs_vmin": 0.0,
        "abs_vmax": None,
    },
    "corr": {
        "pretty": "0-24 h correlation",
        "unit": "",
        "absolute_cmap": "RdYlGn",
        "diff_cmap": "RdYlGn",
        "higher_is_better": True,
        "abs_vmin": 0.0,
        "abs_vmax": 1.0,
    },
    "kge": {
        "pretty": "0-24 h KGE",
        "unit": "",
        "absolute_cmap": "RdYlGn",
        "diff_cmap": "RdYlGn",
        "higher_is_better": True,
        "abs_vmin": -0.5,
        "abs_vmax": 1.0,
    },
    "nse": {
        "pretty": "0-24 h NSE",
        "unit": "",
        "absolute_cmap": "RdYlGn",
        "diff_cmap": "RdYlGn",
        "higher_is_better": True,
        "abs_vmin": -1.0,
        "abs_vmax": 1.0,
    },
}


def draw(ax, key, cb=True):
    pretty, hib, cmap, unit = SPEC[key]
    d = df[key].to_numpy(dtype=float)
    fin = np.isfinite(d)
    lim = float(np.percentile(np.abs(d[fin]), 90)) or 1e-3
    better = int(np.sum(d[fin] > 0)) if hib else int(np.sum(d[fin] < 0))
    n = int(np.sum(fin))
    make_basemap(None, gauges, ax=ax, river_alpha=0.34)
    sc = _scatter_metric(ax, df, d, cmap, -lim, lim, s=160)
    add_reference_cities(ax)
    ax.text(0.985, 0.985, "ST-GNN better at %d of %d gauges" % (better, n), transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.92}, zorder=22)
    if cb:
        cb_ = ax.figure.colorbar(sc, ax=ax, shrink=0.56, pad=0.02)
        cb_.set_label(f"ST-GNN $-$ Bi-LSTM {pretty}{unit}; green = ST-GNN better", fontsize=11)
    return pretty, better, n


def _finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _absolute_limits(key, spec):
    if spec["abs_vmax"] is not None:
        return spec["abs_vmin"], spec["abs_vmax"]
    vals = np.concatenate([
        _finite(cmp[f"{key}_stgnn"].to_numpy(dtype=float)),
        _finite(cmp[f"{key}_bilstm"].to_numpy(dtype=float)),
    ])
    return spec["abs_vmin"], float(np.percentile(vals, 95))


def _draw_triptych_panel(ax, key, column, title, cmap, vmin, vmax, colorbar_label, gauges_for_extent, s=150):
    make_basemap(None, gauges_for_extent, ax=ax, river_alpha=0.34)
    vals = cmp[column].to_numpy(dtype=float)
    sc = _scatter_metric(ax, cmp, vals, cmap, vmin, vmax, s=s)
    add_reference_cities(ax)
    ax.set_title(title, pad=4)
    cb_ = ax.figure.colorbar(sc, ax=ax, shrink=0.52, pad=0.015)
    cb_.set_label(colorbar_label, fontsize=10)
    return sc


def draw_triptych(key):
    spec = TRIPTYCH_SPEC[key]
    abs_vmin, abs_vmax = _absolute_limits(key, spec)
    diff = cmp[f"{key}_diff"].to_numpy(dtype=float)
    fin = np.isfinite(diff)
    diff_lim = float(np.percentile(np.abs(diff[fin]), 90)) or 1e-3
    better = int(np.sum(diff[fin] > 0)) if spec["higher_is_better"] else int(np.sum(diff[fin] < 0))
    n = int(np.sum(fin))
    fig, axes = plt.subplots(1, 3, figsize=(20.8, 4.9))
    gauges_for_extent = list(cmp.index)
    _draw_triptych_panel(
        axes[0],
        key,
        f"{key}_stgnn",
        f"(a) ST-GNN {spec['pretty']}",
        spec["absolute_cmap"],
        abs_vmin,
        abs_vmax,
        f"ST-GNN {spec['pretty']} {spec['unit']}".strip(),
        gauges_for_extent,
    )
    _draw_triptych_panel(
        axes[1],
        key,
        f"{key}_bilstm",
        f"(b) Bi-LSTM {spec['pretty']}",
        spec["absolute_cmap"],
        abs_vmin,
        abs_vmax,
        f"Bi-LSTM {spec['pretty']} {spec['unit']}".strip(),
        gauges_for_extent,
    )
    _draw_triptych_panel(
        axes[2],
        key,
        f"{key}_diff",
        f"(c) ST-GNN $-$ Bi-LSTM difference",
        spec["diff_cmap"],
        -diff_lim,
        diff_lim,
        f"Difference in {spec['pretty']} {spec['unit']}; green = ST-GNN better".strip(),
        gauges_for_extent,
        s=160,
    )
    axes[2].text(0.985, 0.985, "ST-GNN better at %d of %d gauges" % (better, n), transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=10.0,
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.92}, zorder=22)
    fig.suptitle(f"ST-GNN and Bi-LSTM per-gauge {spec['pretty']} across Ascension Parish - hourly forecasts, {span}",
                 fontsize=16, y=0.905)
    fig.tight_layout(rect=[0, 0, 1, 0.900])
    out = f"{DST}/{TRIPTYCH_NAME[key]}"
    fig.savefig(out + ".png", dpi=240, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", out + ".png", f"(ST-GNN better at {better}/{n})")


# ---- Fig 13 (main): Delta RMSE (paired_per_gauge, body-consistent) ----
fig, ax = plt.subplots(figsize=(8.8, 8.2))
pretty, better, n = draw(ax, "rmse")
ax.set_title(f"ST-GNN vs Bi-LSTM per-gauge {pretty}\nhourly forecasts, {span}  "
             f"(ST-GNN better at {better} of {n} mapped gauges)")
fig.tight_layout()
fig.savefig(f"{DST}/fig15_spatial_diff.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{DST}/fig15_spatial_diff.pdf", bbox_inches="tight")
plt.close(fig)
print("[SAVED] fig15_spatial_diff.png", f"(RMSE: ST-GNN better at {better}/{n})")

# ---- Fig S4 (supp): Delta correlation / KGE / NSE (efficiency) ----
for key in ["corr", "kge", "nse"]:
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    pretty, better, n = draw(ax, key)
    ax.set_title(f"ST-GNN vs Bi-LSTM per-gauge {pretty}\nhourly forecasts, {span}  "
                 f"(ST-GNN better at {better} of {n} gauges)")
    fig.tight_layout()
    fig.savefig(f"{DST}/{NAME[key]}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{DST}/{NAME[key]}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", NAME[key] + ".png")

fig, axes = plt.subplots(1, 3, figsize=(20.5, 5.3))
for ax, key, tg in zip(axes, ["corr", "kge", "nse"], ["(a)", "(b)", "(c)"]):
    pretty, better, n = draw(ax, key)
    ax.set_title(f"{tg} {pretty}")
fig.suptitle(f"ST-GNN minus Bi-LSTM per-gauge efficiency across Ascension Parish - hourly forecasts, {span}",
             fontsize=16, y=0.905)
fig.tight_layout(rect=[0, 0, 1, 0.900])
fig.savefig(f"{DST}/fig15_spatial_panel.png", dpi=240, bbox_inches="tight")
fig.savefig(f"{DST}/fig15_spatial_panel.pdf", bbox_inches="tight")
print("[SAVED] fig15_spatial_panel.png")

fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.8))
for ax, key, tg in zip(axes.ravel(), ["rmse", "corr", "kge", "nse"], ["(a)", "(b)", "(c)", "(d)"]):
    pretty, better, n = draw(ax, key)
    label_axes_m(ax)
    ax.set_xlabel("Easting (NAD83 Louisiana South StatePlane, m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{tg} {pretty}")
fig.suptitle(f"ST-GNN minus Bi-LSTM per-gauge skill across Ascension Parish - hourly forecasts, {span}",
             fontsize=17, y=0.905)
fig.tight_layout(rect=[0, 0, 1, 0.935])
fig.savefig(f"{DST}/fig15_spatial_panel_4.png", dpi=240, bbox_inches="tight")
fig.savefig(f"{DST}/fig15_spatial_panel_4.pdf", bbox_inches="tight")
print("[SAVED] fig15_spatial_panel_4.png")

for key in ["rmse", "corr", "kge", "nse"]:
    draw_triptych(key)

plt.show()
