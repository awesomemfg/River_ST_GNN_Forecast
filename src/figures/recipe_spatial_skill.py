"""Fig 9 series (4.2.3) - delivered ST-GNN per-gauge skill across Ascension Parish, on the figure_01 river
basemap. One map per metric (RMSE / correlation / KGE / NSE) + a 2x2 montage, each with median+mean
annotated and the archive timeframe in the title.

Sources (kept consistent with the manuscript): RMSE + correlation from the delivered 410-cycle archive
(per_gauge_stats.csv); KGE + NSE from efficiency_per_gauge.csv (the only source for those).
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _mapbase import make_basemap
from _spatialmap import load_stgnn_skill, archive_span, _scatter_metric, annotate_stats, add_reference_cities, DST
import matplotlib.pyplot as plt

df = load_stgnn_skill()
span, ncyc = archive_span()
gauges = list(df.index)

# key -> (column, colormap, vmin, vmax, label, unit)
SPEC = {
    "rmse": ("rmse", "RdYlGn_r", 0.0, float(np.percentile(df["rmse"], 95)), "mean 0-24 h RMSE (m)", " m"),
    "corr": ("corr", "RdYlGn", 0.0, 1.0, "median 0-24 h correlation", ""),
    "kge": ("kge", "RdYlGn", -0.5, 1.0, "0-24 h KGE", ""),
    "nse": ("nse", "RdYlGn", -1.0, 1.0, "0-24 h NSE", ""),
}
NAME = {"rmse": "fig11_spatial_skill", "corr": "fig11a_spatial_corr",
        "kge": "fig11b_spatial_kge", "nse": "fig11c_spatial_nse"}
PRETTY = {"rmse": "mean 0-24 h RMSE", "corr": "median 0-24 h correlation",
          "kge": "0-24 h KGE", "nse": "0-24 h NSE"}


def draw(ax, key, cb=True):
    col, cmap, vmin, vmax, label, unit = SPEC[key]
    vals = df[col].to_numpy(dtype=float)
    make_basemap(None, gauges, ax=ax, river_alpha=0.34)
    sc = _scatter_metric(ax, df, vals, cmap, vmin, vmax)
    add_reference_cities(ax)
    annotate_stats(ax, vals, unit=unit)
    if cb:
        cb_ = ax.figure.colorbar(sc, ax=ax, shrink=0.56, pad=0.02)
        cb_.set_label(f"ST-GNN {label}", fontsize=12)
    return sc


for key in ["rmse", "corr", "kge", "nse"]:
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    draw(ax, key)
    ax.set_title(f"Delivered ST-GNN forecast skill across Ascension Parish: {PRETTY[key]}\n"
                 f"hourly forecasts, {span}  ({len(gauges)} in-parish gauges)")
    fig.tight_layout()
    out = f"{DST}/{NAME[key]}"
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", out + ".png")

fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.8))
for ax, key, tg in zip(axes.ravel(), ["rmse", "corr", "kge", "nse"], ["(a)", "(b)", "(c)", "(d)"]):
    draw(ax, key, cb=True)
    ax.set_title(f"{tg} {PRETTY[key]}")
fig.suptitle(f"Delivered ST-GNN per-gauge skill across Ascension Parish - hourly forecasts, {span}",
             fontsize=17, y=0.905)
fig.tight_layout(rect=[0, 0, 1, 0.935])
fig.savefig(f"{DST}/fig11_spatial_panel.png", dpi=240, bbox_inches="tight")
fig.savefig(f"{DST}/fig11_spatial_panel.pdf", bbox_inches="tight")
print("[SAVED]", f"{DST}/fig11_spatial_panel.png")
